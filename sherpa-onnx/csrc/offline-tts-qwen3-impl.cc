// sherpa-onnx/csrc/offline-tts-qwen3-impl.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-qwen3-impl.h"

#include <algorithm>
#include <array>
#include <chrono>  // NOLINT
#include <cmath>
#include <fstream>
#include <future>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "nlohmann/json.hpp"

namespace sherpa_onnx {

namespace {
inline void Fp32ToBf16(const float *src, uint16_t *dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    union { float f; uint32_t u; } tmp;
    tmp.f = src[i];
    dst[i] = static_cast<uint16_t>(tmp.u >> 16);
  }
}
}  // namespace

OfflineTtsQwen3Impl::OfflineTtsQwen3Impl(const OfflineTtsConfig &config)
    : config_(config),
      model_(config.model),
      tokenizer_(config.model.qwen3.tokenizer_dir) {
  if (config.model.debug) {
    SHERPA_ONNX_LOGE("Qwen3-TTS model loaded successfully.");
  }
}

#if __ANDROID_API__ >= 9
OfflineTtsQwen3Impl::OfflineTtsQwen3Impl(AAssetManager *mgr,
                                         const OfflineTtsConfig &config)
    : config_(config),
      model_(mgr, config.model),
      tokenizer_(mgr, config.model.qwen3.tokenizer_dir) {
  if (config.model.debug) {
    SHERPA_ONNX_LOGE("Qwen3-TTS model loaded successfully (Android).");
  }
}
#endif

#if __OHOS__
OfflineTtsQwen3Impl::OfflineTtsQwen3Impl(NativeResourceManager *mgr,
                                         const OfflineTtsConfig &config)
    : config_(config),
      model_(mgr, config.model),
      tokenizer_(mgr, config.model.qwen3.tokenizer_dir) {
  if (config.model.debug) {
    SHERPA_ONNX_LOGE("Qwen3-TTS model loaded successfully (HarmonyOS).");
  }
}
#endif

GeneratedAudio OfflineTtsQwen3Impl::Generate(
    const std::string &text, int64_t sid, float speed,
    GeneratedAudioCallback callback) const {
  GenerationConfig gen_config;
  gen_config.sid = sid;
  gen_config.speed = speed;
  return Generate(text, gen_config, callback);
}

// ---------------------------------------------------------------------------

std::vector<float> OfflineTtsQwen3Impl::DecodeFrames(
    const std::vector<std::vector<int64_t>> &codes, bool use_stream) const {
  if (codes.empty()) return {};

  const auto &cfg = model_.GetConfig();
  int32_t num_frames = static_cast<int32_t>(codes.size());
  int32_t num_code_groups = cfg.num_code_groups;
  auto allocator = model_.Allocator();

  std::array<int64_t, 3> shape = {1, num_frames, num_code_groups};
  Ort::Value audio_codes =
      Ort::Value::CreateTensor(allocator, shape.data(), shape.size(),
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  auto *dst = audio_codes.GetTensorMutableData<int64_t>();
  for (int32_t t = 0; t < num_frames; ++t)
    for (int32_t g = 0; g < num_code_groups; ++g)
      dst[t * num_code_groups + g] = codes[t][g];

  auto result =
      use_stream ? model_.RunTokenizer12hzDecodeStream(std::move(audio_codes))
                 : model_.RunTokenizer12hzDecode(std::move(audio_codes));

  const float *audio_data = result.audio_values.GetTensorData<float>();
  auto audio_shape = result.audio_values.GetTensorTypeAndShapeInfo().GetShape();
  int32_t total = 1;
  for (auto s : audio_shape) total *= static_cast<int32_t>(s);

  int32_t valid = total;
  if (result.lengths.IsTensor()) {
    auto *len = result.lengths.GetTensorData<int64_t>();
    valid = static_cast<int32_t>(len[0]);
    if (valid > total) {
      SHERPA_ONNX_LOGE("Warning: lengths (%d) > buffer (%d), clamping", valid,
                       total);
      valid = total;
    }
  }
  return std::vector<float>(audio_data, audio_data + valid);
}

// ---------------------------------------------------------------------------

GeneratedAudio OfflineTtsQwen3Impl::Generate(
    const std::string &text, const GenerationConfig &gen_config,
    GeneratedAudioCallback callback) const {
  const auto &cfg = model_.GetConfig();
  const int32_t num_code_groups = cfg.num_code_groups;
  const int32_t D = cfg.hidden_size;

  const float temperature = gen_config.GetExtraFloat("temperature", 0.9f);
  const int32_t top_k = gen_config.GetExtraInt("top_k", 50);
  const float top_p = gen_config.GetExtraFloat("top_p", 1.0f);
  const float rep_penalty =
      gen_config.GetExtraFloat("repetition_penalty", 1.05f);
  const int32_t max_new_tokens = gen_config.GetExtraInt("max_new_tokens", 2048);
  const float sub_temperature =
      gen_config.GetExtraFloat("sub_temperature", 0.9f);
  const int32_t sub_top_k = gen_config.GetExtraInt("sub_top_k", 50);
  const float sub_top_p = gen_config.GetExtraFloat("sub_top_p", 1.0f);

  // chunk_frames=0  → batch mode: generate all frames, decode once, optional
  //                   final callback.
  // chunk_frames>0 + tokenizer12hz_decode_stream loaded
  //               → true streaming: fire per-chunk decode every chunk_frames
  //                 AR steps using the fast stream decoder; first-chunk
  //                 latency ≈ AR time for chunk_frames frames + fast decode.
  // chunk_frames>0 + only batch decoder loaded
  //               → fake streaming: fire progress-only callbacks (0 audio)
  //               every
  //                 chunk_frames AR steps, then decode all frames once with the
  //                 batch decoder and deliver audio in chunk_frames-sized
  //                 pieces.
  //
  // NOTE: the community tokenizer12hz_decode*.onnx was traced with
  // max_codes_length=1024, so every call costs ~16 s on M-series CPU regardless
  // of input length.  Calling it once per chunk would be 10-40× slower than a
  // single batch decode.  Only use the batch decoder once, at the end.
  const int32_t chunk_frames = gen_config.GetExtraInt("chunk_frames", 0);
  const bool streaming = (chunk_frames > 0) && (callback != nullptr);
  const bool has_stream_decoder = model_.HasTokenizer12hzDecodeStream();
  // context_frames: previous codec frames prepended for the stream decoder to
  // give left context at chunk boundaries.
  // context=0  → fast, corr≈0.83 vs batch
  // context=50 → trace_T=75, corr=1.00 vs batch (default)
  const int32_t context_frames = gen_config.GetExtraInt("context_frames", 50);
  // Codec frames kept from the previous chunk to use as left context.
  std::vector<std::vector<int64_t>> context_codes;
  // overlap_frames: audio crossfade window at chunk boundaries (codec frames).
  const int32_t overlap_frames = gen_config.GetExtraInt("overlap_frames", 2);
  const int32_t kSamplesPerFrame = 1920;
  const int32_t crossfade_samples =
      (streaming && has_stream_decoder) ? overlap_frames * kSamplesPerFrame : 0;
  // Holds the last crossfade_samples from the previous streaming chunk so
  // they can be blended with the start of the next chunk (overlap-add).
  std::vector<float> held_tail;

  auto t_start = std::chrono::steady_clock::now();

  // ------------------------------------------------------------------
  // Step 1: Tokenize
  // ------------------------------------------------------------------
  const std::string formatted_text =
      "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
  const auto input_ids = tokenizer_.Encode(formatted_text);
  if (config_.model.debug)
    SHERPA_ONNX_LOGE("Text tokenized: %d tokens", (int)input_ids.size());

  // ------------------------------------------------------------------
  // Step 2: Build prefill embeddings
  // ------------------------------------------------------------------
  const std::vector<int64_t> special_ids = {
      cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id};
  auto special_embed = RunTextProjectHelper(special_ids);  // [1,3,D]

  const std::vector<int64_t> role_ids(input_ids.begin(), input_ids.begin() + 3);
  auto role_embed = RunTextProjectHelper(role_ids);  // [1,3,D]

  const std::vector<int64_t> codec_prefix_ids = {
      cfg.codec_nothink_id, cfg.codec_think_bos_id, cfg.codec_think_eos_id,
      cfg.codec_pad_id, cfg.codec_bos_id};
  auto codec_prefix_embed = RunCodecEmbedHelper(codec_prefix_ids);  // [1,5,D]

  const int32_t text_start = 3;
  const int32_t text_end = static_cast<int32_t>(input_ids.size()) - 5;
  std::vector<int64_t> body_text_ids;
  if (text_end > text_start)
    body_text_ids.assign(input_ids.begin() + text_start,
                         input_ids.begin() + text_end);

  const float *special_data = special_embed.GetTensorData<float>();
  const float *role_data = role_embed.GetTensorData<float>();
  const float *codec_prefix_data = codec_prefix_embed.GetTensorData<float>();

  const bool non_streaming =
      gen_config.GetExtraInt("non_streaming_mode", 0) != 0;

  // trailing_text_hidden
  std::vector<std::vector<float>> trailing;
  const std::vector<float> tts_pad_vec(special_data + 2 * D,
                                       special_data + 3 * D);
  if (non_streaming) {
    trailing.emplace_back(tts_pad_vec);
  } else {
    if (body_text_ids.size() > 1) {
      const std::vector<int64_t> trail_ids(body_text_ids.begin() + 1,
                                           body_text_ids.end());
      auto te = RunTextProjectHelper(trail_ids);
      const float *td = te.GetTensorData<float>();
      const int32_t n = static_cast<int32_t>(trail_ids.size());
      for (int32_t i = 0; i < n; ++i)
        trailing.emplace_back(td + i * D, td + (i + 1) * D);
    }
    trailing.emplace_back(special_data + 1 * D, special_data + 2 * D);
  }

  // Prefill layout
  const int32_t prefill_len =
      non_streaming ? 6 + static_cast<int32_t>(body_text_ids.size()) + 1 + 1
                    : 8;
  std::vector<float> prefill_data(prefill_len * D, 0.0f);
  int32_t pos = 0;

  for (int32_t i = 0; i < 3; ++i, ++pos)
    for (int32_t d = 0; d < D; ++d)
      prefill_data[pos * D + d] = role_data[i * D + d];

  for (int32_t i = 0; i < 3; ++i, ++pos)
    for (int32_t d = 0; d < D; ++d)
      prefill_data[pos * D + d] =
          special_data[2 * D + d] + codec_prefix_data[i * D + d];

  if (non_streaming) {
    // all body text + codec_pad
    for (size_t i = 0; i < body_text_ids.size(); ++i, ++pos) {
      auto text_embed = RunTextProjectHelper({body_text_ids[i]});
      const float *td = text_embed.GetTensorData<float>();
      for (int32_t d = 0; d < D; ++d)
        prefill_data[pos * D + d] = td[d] + codec_prefix_data[3 * D + d];
    }
    // tts_eos + codec_pad
    for (int32_t d = 0; d < D; ++d)
      prefill_data[pos * D + d] =
          special_data[1 * D + d] + codec_prefix_data[3 * D + d];
    ++pos;
    // tts_pad + codec_bos
    for (int32_t d = 0; d < D; ++d)
      prefill_data[pos * D + d] =
          special_data[2 * D + d] + codec_prefix_data[4 * D + d];
    ++pos;
  } else {
    // tts_bos + codec_pad
    for (int32_t d = 0; d < D; ++d)
      prefill_data[pos * D + d] =
          special_data[0 * D + d] + codec_prefix_data[3 * D + d];
    ++pos;
    // text[0] + codec_bos
    auto first_embed = RunTextProjectHelper({body_text_ids[0]});
    const float *fd = first_embed.GetTensorData<float>();
    for (int32_t d = 0; d < D; ++d)
      prefill_data[pos * D + d] = fd[d] + codec_prefix_data[4 * D + d];
    ++pos;
  }

  if (config_.model.debug) {
    SHERPA_ONNX_LOGE("prefill_len=%d, trailing=%d, non_streaming=%d",
                     prefill_len, (int)trailing.size(), (int)non_streaming);

    // Print prefill tokens
    SHERPA_ONNX_LOGE("=== Prefill Tokens ===");
    int32_t p = 0;
    SHERPA_ONNX_LOGE("Pos %d-%d: role tokens [%d, %d, %d]", p, p + 2,
                     (int)role_ids[0], (int)role_ids[1], (int)role_ids[2]);
    p += 3;
    SHERPA_ONNX_LOGE("Pos %d: tts_pad(%d) + codec_nothink(%d)", p,
                     (int)cfg.tts_pad_token_id, (int)cfg.codec_nothink_id);
    ++p;
    SHERPA_ONNX_LOGE("Pos %d: tts_pad(%d) + codec_think_bos(%d)", p,
                     (int)cfg.tts_pad_token_id, (int)cfg.codec_think_bos_id);
    ++p;
    SHERPA_ONNX_LOGE("Pos %d: tts_pad(%d) + codec_think_eos(%d)", p,
                     (int)cfg.tts_pad_token_id, (int)cfg.codec_think_eos_id);
    ++p;
    if (non_streaming) {
      for (size_t i = 0; i < body_text_ids.size(); ++i, ++p) {
        SHERPA_ONNX_LOGE("Pos %d: text[%d](%d) + codec_pad(%d)", p, (int)i,
                         (int)body_text_ids[i], (int)cfg.codec_pad_id);
      }
      SHERPA_ONNX_LOGE("Pos %d: tts_eos(%d) + codec_pad(%d)", p,
                       (int)cfg.tts_eos_token_id, (int)cfg.codec_pad_id);
      ++p;
      SHERPA_ONNX_LOGE("Pos %d: tts_pad(%d) + codec_bos(%d)", p,
                       (int)cfg.tts_pad_token_id, (int)cfg.codec_bos_id);
      ++p;
    } else {
      SHERPA_ONNX_LOGE("Pos %d: tts_bos(%d) + codec_pad(%d)", p,
                       (int)cfg.tts_bos_token_id, (int)cfg.codec_pad_id);
      ++p;
      SHERPA_ONNX_LOGE("Pos %d: text[0](%d) + codec_bos(%d)", p,
                       (int)body_text_ids[0], (int)cfg.codec_bos_id);
      ++p;
    }
    SHERPA_ONNX_LOGE("======================");

    // Write prefill embeddings to file
    std::string debug_path = "qwen3_tts_prefill_debug.bin";
    std::ofstream ofs(debug_path, std::ios::binary);
    if (ofs) {
      ofs.write(reinterpret_cast<const char *>(prefill_data.data()),
                prefill_data.size() * sizeof(float));
      SHERPA_ONNX_LOGE(
          "Prefill embeddings written to %s [%d floats, shape=1x%dx%d]",
          debug_path.c_str(), (int)prefill_data.size(), prefill_len, D);
    } else {
      SHERPA_ONNX_LOGE("Failed to write prefill debug file: %s",
                       debug_path.c_str());
    }

    // Write tts_pad_vec to file (for non-streaming decode)
    std::string pad_vec_path = "tts_pad_vec.bin";
    std::ofstream pad_ofs(pad_vec_path, std::ios::binary);
    if (pad_ofs) {
      int32_t hidden = D;
      pad_ofs.write(reinterpret_cast<const char *>(&hidden), sizeof(hidden));
      pad_ofs.write(reinterpret_cast<const char *>(tts_pad_vec.data()),
                    tts_pad_vec.size() * sizeof(float));
      SHERPA_ONNX_LOGE("tts_pad_vec written to %s [hidden=%d]",
                       pad_vec_path.c_str(), hidden);
    } else {
      SHERPA_ONNX_LOGE("Failed to write tts_pad_vec file: %s",
                       pad_vec_path.c_str());
    }

    // Write trailing_text_hiddens to file (for streaming decode)
    // Shape: [trailing_len, D], float32
    if (!non_streaming) {
      std::string trail_path = "trailing_text_hiddens.bin";
      std::ofstream trail_ofs(trail_path, std::ios::binary);
      if (trail_ofs) {
        for (const auto &vec : trailing) {
          trail_ofs.write(reinterpret_cast<const char *>(vec.data()),
                          vec.size() * sizeof(float));
        }
        SHERPA_ONNX_LOGE(
            "trailing_text_hiddens written to %s [%d x %d] (streaming)",
            trail_path.c_str(), (int)trailing.size(), D);
      } else {
        SHERPA_ONNX_LOGE("Failed to write trailing_text_hiddens file: %s",
                         trail_path.c_str());
      }
    }

    // ── BFloat16 export for ax-llm debug ────────────────────────────────────
    // prefill_embeds_bf16.bin
    {
      std::string bf16_path = "prefill_embeds_bf16.bin";
      std::ofstream ofs(bf16_path, std::ios::binary);
      if (ofs) {
        std::vector<uint16_t> bf16_buf(prefill_data.size());
        Fp32ToBf16(prefill_data.data(), bf16_buf.data(), prefill_data.size());
        ofs.write(reinterpret_cast<const char *>(bf16_buf.data()),
                  bf16_buf.size() * sizeof(uint16_t));
        SHERPA_ONNX_LOGE(
            "prefill_embeds_bf16.bin written [%d x %d] bf16",
            prefill_len, D);
      } else {
        SHERPA_ONNX_LOGE("Failed to write %s", bf16_path.c_str());
      }
    }

    // tts_pad_vec_bf16.bin
    {
      std::string bf16_path = "tts_pad_vec_bf16.bin";
      std::ofstream ofs(bf16_path, std::ios::binary);
      if (ofs) {
        int32_t hidden = D;
        ofs.write(reinterpret_cast<const char *>(&hidden), sizeof(hidden));
        std::vector<uint16_t> bf16_buf(D);
        Fp32ToBf16(tts_pad_vec.data(), bf16_buf.data(), D);
        ofs.write(reinterpret_cast<const char *>(bf16_buf.data()),
                  D * sizeof(uint16_t));
        SHERPA_ONNX_LOGE("tts_pad_vec_bf16.bin written [hidden=%d] bf16", D);
      } else {
        SHERPA_ONNX_LOGE("Failed to write %s", bf16_path.c_str());
      }
    }

    // trailing_text_hiddens_bf16.bin (streaming only)
    if (!non_streaming) {
      std::string bf16_path = "trailing_text_hiddens_bf16.bin";
      std::ofstream ofs(bf16_path, std::ios::binary);
      if (ofs) {
        for (const auto &vec : trailing) {
          std::vector<uint16_t> bf16_buf(D);
          Fp32ToBf16(vec.data(), bf16_buf.data(), D);
          ofs.write(reinterpret_cast<const char *>(bf16_buf.data()),
                    D * sizeof(uint16_t));
        }
        SHERPA_ONNX_LOGE(
            "trailing_text_hiddens_bf16.bin written [%d x %d] bf16 (streaming)",
            (int)trailing.size(), D);
      } else {
        SHERPA_ONNX_LOGE("Failed to write %s", bf16_path.c_str());
      }
    }

    // meta.json
    {
      nlohmann::json j;
      j["S"] = prefill_len;
      j["hidden_size"] = D;
      j["audio_token_id"] = cfg.tts_bos_token_id;
      j["streaming"] = !non_streaming;
      if (!non_streaming) {
        j["trailing_len"] = static_cast<int>(trailing.size());
      }
      std::ofstream ofs("meta.json");
      if (ofs) {
        ofs << j.dump(2);
        SHERPA_ONNX_LOGE("meta.json written");
      } else {
        SHERPA_ONNX_LOGE("Failed to write meta.json");
      }
    }
  }

  auto allocator = model_.Allocator();

  std::array<int64_t, 3> ps = {1, prefill_len, D};
  Ort::Value prefill_embeds = Ort::Value::CreateTensor(
      allocator, ps.data(), ps.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::copy(prefill_data.begin(), prefill_data.end(),
            prefill_embeds.GetTensorMutableData<float>());

  std::array<int64_t, 2> ms = {1, prefill_len};
  Ort::Value attn_mask = Ort::Value::CreateTensor(
      allocator, ms.data(), ms.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  std::fill(attn_mask.GetTensorMutableData<int64_t>(),
            attn_mask.GetTensorMutableData<int64_t>() + prefill_len, 1LL);

  // ------------------------------------------------------------------
  // Step 3: Talker prefill
  // ------------------------------------------------------------------
  auto t_prefill_start = std::chrono::steady_clock::now();
  auto pr =
      model_.RunTalkerPrefill(std::move(prefill_embeds), std::move(attn_mask));
  auto t_prefill_end = std::chrono::steady_clock::now();

  Ort::Value logits = std::move(pr.logits);
  Ort::Value last_hidden = std::move(pr.last_hidden);
  Qwen3TalkerState talker_state = std::move(pr.state);
  int64_t total_seq_len = prefill_len;

  const int32_t suppress_start = cfg.talker_vocab_size - 1024;
  const int32_t suppress_end = cfg.talker_vocab_size;
  constexpr int32_t kMinNewTokens = 2;

  bool stop_requested = false;

  // ------------------------------------------------------------------
  // Step 4: AR generate loop
  // ------------------------------------------------------------------
  std::vector<std::vector<int64_t>> all_codes;
  std::vector<int64_t> generated_primary;

  // Async stream decode: background thread decodes chunk N while main thread
  // generates chunk N+1, overlapping the two phases.
  std::future<std::vector<float>> pending_future;
  float pending_progress = 0.0f;
  int32_t pending_ctx_size = 0;
  std::vector<float> pending_prev_held_tail;
  // Accumulates all audio delivered via callback; used as return value to
  // avoid a redundant full re-decode at the end.
  std::vector<float> accumulated_audio;

  // Helper: deliver n samples via callback and append to accumulated_audio.
  auto deliver = [&](const float *s, int32_t n, float p) -> bool {
    if (s && n > 0) accumulated_audio.insert(accumulated_audio.end(), s, s + n);
    return callback(s, n, p) != 0;
  };

  // Wait for the pending async decode, apply crossfade, and deliver via
  // callback.  Returns false if the callback requested stop.
  auto flush_pending = [&]() -> bool {
    if (!pending_future.valid()) return true;
    auto raw = pending_future.get();
    pending_future = {};
    const int32_t ctx_audio = pending_ctx_size * kSamplesPerFrame;
    std::vector<float> samples;
    if (ctx_audio > 0 && ctx_audio < static_cast<int32_t>(raw.size())) {
      samples.assign(raw.begin() + ctx_audio, raw.end());
    } else {
      samples = std::move(raw);
    }
    if (!pending_prev_held_tail.empty() && crossfade_samples > 0) {
      const int32_t cf =
          std::min(crossfade_samples,
                   static_cast<int32_t>(std::min(pending_prev_held_tail.size(),
                                                 samples.size())));
      for (int32_t i = 0; i < cf; ++i) {
        const float t = static_cast<float>(i + 1) / (cf + 1);
        samples[i] = pending_prev_held_tail[i] * (1.0f - t) + samples[i] * t;
      }
    }
    if (crossfade_samples > 0 &&
        static_cast<int32_t>(samples.size()) > crossfade_samples) {
      held_tail.assign(samples.end() - crossfade_samples, samples.end());
      return deliver(samples.data(),
                     static_cast<int32_t>(samples.size()) - crossfade_samples,
                     pending_progress);
    } else {
      held_tail.clear();
      return deliver(samples.data(), static_cast<int32_t>(samples.size()),
                     pending_progress);
    }
  };

  auto t_gen_start = std::chrono::steady_clock::now();

  for (int32_t step = 0; step < max_new_tokens && !stop_requested; ++step) {
    // 4a: Sample primary code
    const int64_t primary_code = SampleFromLogits(
        logits, cfg.talker_vocab_size, temperature, top_k, top_p, rep_penalty,
        generated_primary, suppress_start, suppress_end, cfg.codec_eos_token_id,
        step < kMinNewTokens);

    if (primary_code == cfg.codec_eos_token_id) {
      if (config_.model.debug) SHERPA_ONNX_LOGE("EOS at step %d", step);
      break;
    }

    generated_primary.push_back(primary_code);
    if (config_.model.debug && step < 3)
      SHERPA_ONNX_LOGE("step %d: primary=%d", step, (int)primary_code);

    // 4b: Primary embedding
    auto primary_embed = RunCodecEmbedHelper({primary_code});

    // 4c: Residual codes via code predictor
    std::vector<int64_t> frame_codes(num_code_groups);
    frame_codes[0] = primary_code;

    const float *lh_data = last_hidden.GetTensorData<float>();
    const int32_t lh_seq = static_cast<int32_t>(
        last_hidden.GetTensorTypeAndShapeInfo().GetShape()[1]);
    const float *pe_data = primary_embed.GetTensorData<float>();

    std::vector<float> cp_ctx(2 * D);
    std::copy(lh_data + (lh_seq - 1) * D, lh_data + lh_seq * D, cp_ctx.begin());
    std::copy(pe_data, pe_data + D, cp_ctx.begin() + D);

    std::vector<float> codec_sum(D);
    std::copy(pe_data, pe_data + D, codec_sum.begin());

    for (int32_t j = 0; j < num_code_groups - 1; ++j) {
      const int32_t cp_len = static_cast<int32_t>(cp_ctx.size()) / D;
      std::array<int64_t, 3> cp_shape = {1, cp_len, D};
      Ort::Value cp_emb =
          Ort::Value::CreateTensor(allocator, cp_shape.data(), cp_shape.size(),
                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      std::copy(cp_ctx.begin(), cp_ctx.end(),
                cp_emb.GetTensorMutableData<float>());

      std::array<int64_t, 1> gs_shape = {1};
      Ort::Value gen_step =
          Ort::Value::CreateTensor(allocator, gs_shape.data(), gs_shape.size(),
                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      gen_step.GetTensorMutableData<int64_t>()[0] = j;

      auto cp_logits =
          model_.RunCodePredictor(std::move(cp_emb), std::move(gen_step));
      const int64_t res_code =
          SampleFromLogits(cp_logits, cfg.code_predictor_vocab_size,
                           sub_temperature, sub_top_k, sub_top_p, 1.0f, {});
      frame_codes[j + 1] = res_code;

      std::array<int64_t, 2> rid_shape = {1, 1};
      Ort::Value rid = Ort::Value::CreateTensor(
          allocator, rid_shape.data(), rid_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      rid.GetTensorMutableData<int64_t>()[0] = res_code;

      Ort::Value gs2 =
          Ort::Value::CreateTensor(allocator, gs_shape.data(), gs_shape.size(),
                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      gs2.GetTensorMutableData<int64_t>()[0] = j;

      auto res_emb =
          model_.RunCodePredictorEmbed(std::move(rid), std::move(gs2));
      const float *rd = res_emb.GetTensorData<float>();
      cp_ctx.insert(cp_ctx.end(), rd, rd + D);
      for (int32_t d = 0; d < D; ++d) codec_sum[d] += rd[d];
    }

    all_codes.push_back(frame_codes);

    // 4d: Build next talker input
    const std::vector<float> &txt_hidden =
        (step < static_cast<int32_t>(trailing.size())) ? trailing[step]
                                                       : tts_pad_vec;
    std::vector<float> next_in(D);
    for (int32_t d = 0; d < D; ++d) next_in[d] = codec_sum[d] + txt_hidden[d];

    std::array<int64_t, 3> ne_shape = {1, 1, D};
    Ort::Value next_emb =
        Ort::Value::CreateTensor(allocator, ne_shape.data(), ne_shape.size(),
                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::copy(next_in.begin(), next_in.end(),
              next_emb.GetTensorMutableData<float>());

    total_seq_len++;
    std::array<int64_t, 2> nm_shape = {1, total_seq_len};
    Ort::Value new_mask =
        Ort::Value::CreateTensor(allocator, nm_shape.data(), nm_shape.size(),
                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::fill(new_mask.GetTensorMutableData<int64_t>(),
              new_mask.GetTensorMutableData<int64_t>() + total_seq_len, 1LL);

    auto dr = model_.RunTalkerDecode(std::move(next_emb), std::move(new_mask),
                                     std::move(talker_state));
    logits = std::move(dr.logits);
    last_hidden = std::move(dr.last_hidden);
    talker_state = std::move(dr.state);

    if (config_.model.debug && (step + 1) % 100 == 0)
      SHERPA_ONNX_LOGE("Generated %d frames", step + 1);

    // ------------------------------------------------------------------
    // Streaming: when a chunk is complete, either decode it immediately
    // (if tokenizer12hz_decode_stream is loaded) or fire a progress-only
    // callback (when only the slow batch decoder is available).
    // ------------------------------------------------------------------
    if (streaming &&
        static_cast<int32_t>(all_codes.size()) % chunk_frames == 0) {
      const float progress =
          static_cast<float>(all_codes.size()) / max_new_tokens;

      if (has_stream_decoder) {
        // Flush the decode that was started for the PREVIOUS chunk.  This
        // typically completes in near-zero time because the background decode
        // ran concurrently with the AR generation steps just performed.
        if (!flush_pending()) stop_requested = true;

        if (!stop_requested) {
          const int32_t chunk_start =
              static_cast<int32_t>(all_codes.size()) - chunk_frames;
          std::vector<std::vector<int64_t>> chunk(
              all_codes.begin() + chunk_start, all_codes.end());

          // Prepend context frames for left-context at boundaries.
          std::vector<std::vector<int64_t>> ctx_chunk;
          int32_t ctx_size = 0;
          if (context_frames > 0 && !context_codes.empty()) {
            const int32_t use = std::min(
                context_frames, static_cast<int32_t>(context_codes.size()));
            ctx_size = use;
            ctx_chunk.insert(ctx_chunk.end(), context_codes.end() - use,
                             context_codes.end());
          }
          ctx_chunk.insert(ctx_chunk.end(), chunk.begin(), chunk.end());

          // Update context for next chunk.
          if (context_frames > 0) {
            const int32_t keep = std::min(
                context_frames, static_cast<int32_t>(all_codes.size()));
            context_codes.assign(all_codes.end() - keep, all_codes.end());
          }

          // Snapshot held_tail and metadata for when this future resolves.
          pending_prev_held_tail = held_tail;
          pending_progress = progress;
          pending_ctx_size = ctx_size;

          // Launch async decode — runs concurrently with the next AR chunk.
          pending_future = std::async(
              std::launch::async, [this, ctx = std::move(ctx_chunk)]() {
                return DecodeFrames(ctx, /*use_stream=*/true);
              });
        }
      } else {
        // No stream decoder: progress-only callback.  Audio will be delivered
        // after the AR loop via a single batch decode + chunked delivery.
        if (callback(nullptr, 0, progress) == 0) stop_requested = true;
      }
    }
  }

  auto t_gen_end = std::chrono::steady_clock::now();

  if (config_.model.debug) {
    SHERPA_ONNX_LOGE("Generated %d frames (num_code_groups=%d):",
                     (int)all_codes.size(), num_code_groups);
    for (size_t f = 0; f < all_codes.size(); ++f) {
      std::string line = "Frame " + std::to_string(f) + ": ";
      for (size_t g = 0; g < all_codes[f].size(); ++g) {
        line += std::to_string(all_codes[f][g]) + " ";
      }
      SHERPA_ONNX_LOGE("%s", line.c_str());
    }
  }

  // Flush the async decode for the last AR chunk (started but not yet
  // delivered because there were no further chunks to trigger flush_pending).
  if (streaming && has_stream_decoder) {
    if (pending_future.valid()) {
      if (!stop_requested) {
        if (!flush_pending()) stop_requested = true;
      } else {
        pending_future.get();  // join the thread; discard result
      }
    }
  }

  // ------------------------------------------------------------------
  // Step 5: Decode audio and deliver via callback.
  //
  // Two streaming sub-modes:
  //   A) has_stream_decoder: chunks already delivered above; only a tail
  //      (< chunk_frames frames) needs decoding + delivery.
  //   B) !has_stream_decoder: no per-chunk decode was done; decode all
  //      frames once with the batch decoder and deliver in chunks.
  //
  // Batch mode: decode all frames once, then single callback delivery.
  // ------------------------------------------------------------------
  GeneratedAudio audio;
  audio.sample_rate = SampleRate();

  auto t_decode_start = std::chrono::steady_clock::now();

  // For mode A (stream decoder), delivered_frames counts what was sent.
  // For mode B (!has_stream_decoder), nothing was delivered yet → 0.
  const int32_t delivered_frames =
      (streaming && has_stream_decoder)
          ? (static_cast<int32_t>(all_codes.size()) / chunk_frames) *
                chunk_frames
          : 0;
  const int32_t remaining =
      static_cast<int32_t>(all_codes.size()) - delivered_frames;

  if (!streaming || has_stream_decoder) {
    // Batch mode or stream-decoder mode: decode the tail (or all for batch).
    if (remaining > 0) {
      std::vector<std::vector<int64_t>> tail(
          all_codes.begin() + delivered_frames, all_codes.end());
      // Use the fast stream decoder for the tail when available (same quality,
      // much faster than the fixed trace_T=1024 batch decoder).
      audio.samples = DecodeFrames(tail, /*use_stream=*/has_stream_decoder);
    }
  } else {
    // Fake-streaming mode (streaming but no stream decoder): decode all at
    // once.
    audio.samples = DecodeFrames(all_codes, /*use_stream=*/false);
  }
  auto t_decode_end = std::chrono::steady_clock::now();

  if (config_.model.debug) {
    const float decode_ms =
        std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start)
            .count();
    SHERPA_ONNX_LOGE("decode: %.0f ms, %d frames, %d samples", decode_ms,
                     remaining, (int)audio.samples.size());
  }

  // ------------------------------------------------------------------
  // Step 6: Deliver audio via callback.
  // ------------------------------------------------------------------
  if (callback) {
    if (streaming && !has_stream_decoder) {
      // Mode B: deliver all decoded audio in chunk_frames-sized pieces.
      const int32_t total_s = static_cast<int32_t>(audio.samples.size());
      const int32_t chunk_s = chunk_frames * kSamplesPerFrame;
      for (int32_t off = 0; off < total_s && !stop_requested; off += chunk_s) {
        const int32_t n = std::min(chunk_s, total_s - off);
        const float prog = static_cast<float>(off + n) / total_s;
        if (callback(audio.samples.data() + off, n, prog) == 0)
          stop_requested = true;
      }
    } else if (!audio.samples.empty() || !held_tail.empty()) {
      // Mode A (stream decoder tail) or batch mode: deliver with crossfade.
      if (!held_tail.empty()) {
        if (!audio.samples.empty()) {
          // Blend held_tail with start of tail audio
          const int32_t cf = std::min(
              crossfade_samples, static_cast<int32_t>(std::min(
                                     held_tail.size(), audio.samples.size())));
          for (int32_t i = 0; i < cf; ++i) {
            const float t = static_cast<float>(i + 1) / (cf + 1);
            audio.samples[i] = held_tail[i] * (1.0f - t) + audio.samples[i] * t;
          }
          deliver(audio.samples.data(), cf, 1.0f);
          if (static_cast<int32_t>(audio.samples.size()) > cf)
            deliver(audio.samples.data() + cf,
                    static_cast<int32_t>(audio.samples.size()) - cf, 1.0f);
        } else {
          deliver(held_tail.data(), static_cast<int32_t>(held_tail.size()),
                  1.0f);
        }
        held_tail.clear();
      } else if (!audio.samples.empty()) {
        deliver(audio.samples.data(),
                static_cast<int32_t>(audio.samples.size()), 1.0f);
      }
    }
  }

  // Return value: for stream-decoder mode, use the audio already delivered via
  // callback (no extra decode needed).  For fake-streaming / batch, use
  // audio.samples as set by Step 5.
  if (streaming && has_stream_decoder && delivered_frames > 0) {
    audio.samples = std::move(accumulated_audio);
  } else if (streaming && !has_stream_decoder) {
    // audio.samples already contains the full decode from mode B above.
  } else if (!streaming && remaining == 0) {
    audio.samples.clear();
  }

  // ------------------------------------------------------------------
  // Performance report
  // ------------------------------------------------------------------
  {
    auto t_end = std::chrono::steady_clock::now();
    const int32_t num_frames = static_cast<int32_t>(all_codes.size());
    const float audio_dur =
        static_cast<float>(audio.samples.size()) / SampleRate();

    auto ms = [](std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b) {
      return std::chrono::duration<float, std::milli>(b - a).count();
    };

    const float prefill_ms = ms(t_prefill_start, t_prefill_end);
    const float gen_ms = ms(t_gen_start, t_gen_end);
    const float total_ms = ms(t_start, t_end);
    const float rtf = (audio_dur > 0) ? (total_ms / 1000.0f / audio_dur) : 0;
    const float fps = (gen_ms > 0) ? (num_frames * 1000.0f / gen_ms) : 0;

    SHERPA_ONNX_LOGE(
        "%s | audio=%.2fs %d frames | "
        "total=%.0fms (prefill=%.0fms generate=%.0fms) | "
        "RTF=%.2f | %.1ffps",
        streaming ? "streaming" : "batch", audio_dur, num_frames, total_ms,
        prefill_ms, gen_ms, rtf, fps);
  }

  return audio;
}

// ---------------------------------------------------------------------------

Ort::Value OfflineTtsQwen3Impl::RunTextProjectHelper(
    const std::vector<int64_t> &ids) const {
  auto allocator = model_.Allocator();
  std::array<int64_t, 2> shape = {1, static_cast<int64_t>(ids.size())};
  Ort::Value input =
      Ort::Value::CreateTensor(allocator, shape.data(), shape.size(),
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  std::copy(ids.begin(), ids.end(), input.GetTensorMutableData<int64_t>());
  return model_.RunTextProject(std::move(input));
}

Ort::Value OfflineTtsQwen3Impl::RunCodecEmbedHelper(
    const std::vector<int64_t> &ids) const {
  auto allocator = model_.Allocator();
  std::array<int64_t, 2> shape = {1, static_cast<int64_t>(ids.size())};
  Ort::Value input =
      Ort::Value::CreateTensor(allocator, shape.data(), shape.size(),
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  std::copy(ids.begin(), ids.end(), input.GetTensorMutableData<int64_t>());
  return model_.RunCodecEmbed(std::move(input));
}

int64_t OfflineTtsQwen3Impl::SampleFromLogits(
    const Ort::Value &logits_tensor, int32_t vocab_size, float temperature,
    int32_t top_k, float top_p, float repetition_penalty,
    const std::vector<int64_t> &generated_ids, int32_t suppress_start,
    int32_t suppress_end, int64_t suppress_exception, bool suppress_eos) const {
  const auto logits_shape =
      logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
  const float *logits_data = logits_tensor.GetTensorData<float>();

  int32_t total = 1;
  for (auto s : logits_shape) total *= static_cast<int32_t>(s);
  const int32_t V = total >= vocab_size ? vocab_size : total;
  const float *src = logits_data + (total - V);

  std::vector<float> buf(src, src + V);

  if (suppress_start >= 0 && suppress_end > suppress_start)
    for (int32_t i = suppress_start; i < std::min(suppress_end, V); ++i)
      if (i != static_cast<int32_t>(suppress_exception)) buf[i] = -1e9f;

  if (suppress_eos && suppress_exception >= 0 && suppress_exception < V)
    buf[static_cast<int32_t>(suppress_exception)] = -1e9f;

  if (repetition_penalty > 1.0f)
    for (auto id : generated_ids)
      if (id >= 0 && id < V)
        buf[id] = buf[id] > 0 ? buf[id] / repetition_penalty
                              : buf[id] * repetition_penalty;

  if (temperature < 1e-6f)
    return static_cast<int64_t>(std::max_element(buf.begin(), buf.end()) -
                                buf.begin());

  for (auto &v : buf) v /= temperature;

  if (top_k > 0 && top_k < V) {
    std::vector<float> tmp(buf.begin(), buf.end());
    std::partial_sort(tmp.begin(), tmp.begin() + top_k, tmp.end(),
                      std::greater<float>());
    const float thr = tmp[top_k - 1];
    for (auto &v : buf)
      if (v < thr) v = -1e9f;
  }

  const float max_v = *std::max_element(buf.begin(), buf.end());
  float sum = 0;
  for (auto &v : buf) {
    v = std::exp(v - max_v);
    sum += v;
  }
  for (auto &v : buf) v /= sum;

  if (top_p < 1.0f && top_p > 0.0f) {
    std::vector<std::pair<float, int32_t>> pi(V);
    for (int32_t i = 0; i < V; ++i) pi[i] = {buf[i], i};
    std::sort(pi.begin(), pi.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    float cum = 0;
    int32_t cut = V;
    for (int32_t i = 0; i < V; ++i) {
      cum += pi[i].first;
      if (cum >= top_p) {
        cut = i + 1;
        break;
      }
    }
    for (int32_t i = cut; i < V; ++i) buf[pi[i].second] = 0.0f;
    float ns = 0;
    for (auto v : buf) ns += v;
    if (ns > 0)
      for (auto &v : buf) v /= ns;
  }

  thread_local std::mt19937 rng(std::random_device{}());
  return static_cast<int64_t>(
      std::discrete_distribution<int32_t>(buf.begin(), buf.end())(rng));
}

}  // namespace sherpa_onnx
