// sherpa-onnx/csrc/offline-tts-qwen3-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_IMPL_H_

#include <algorithm>
#include <chrono>  // NOLINT
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/funasr-nano-tokenizer.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-qwen3-model.h"

namespace sherpa_onnx {

class OfflineTtsQwen3Impl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsQwen3Impl(const OfflineTtsConfig &config)
      : config_(config),
        model_(config.model),
        tokenizer_(config.model.qwen3.tokenizer_dir) {
    SHERPA_ONNX_LOGE("Qwen3-TTS model loaded successfully.");
  }

#if __ANDROID_API__ >= 9
  OfflineTtsQwen3Impl(AAssetManager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(mgr, config.model),
        tokenizer_(mgr, config.model.qwen3.tokenizer_dir) {
    SHERPA_ONNX_LOGE("Qwen3-TTS model loaded successfully (Android).");
  }
#endif

#if __OHOS__
  OfflineTtsQwen3Impl(NativeResourceManager *mgr,
                      const OfflineTtsConfig &config)
      : config_(config),
        model_(mgr, config.model),
        tokenizer_(mgr, config.model.qwen3.tokenizer_dir) {
    SHERPA_ONNX_LOGE("Qwen3-TTS model loaded successfully (HarmonyOS).");
  }
#endif

  int32_t SampleRate() const override { return 24000; }

  int32_t NumSpeakers() const override { return 1; }

  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override {
    GenerationConfig gen_config;
    gen_config.sid = sid;
    gen_config.speed = speed;
    return Generate(text, gen_config, callback);
  }

  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback = nullptr) const override {
    const auto &cfg = model_.GetConfig();
    int32_t num_code_groups = cfg.num_code_groups;  // 16
    int32_t D = cfg.hidden_size;                    // 1024

    // Generation parameters from extra config
    float temperature =
        gen_config.GetExtraFloat("temperature", 0.9f);
    int32_t top_k = gen_config.GetExtraInt("top_k", 50);
    float top_p = gen_config.GetExtraFloat("top_p", 1.0f);
    float repetition_penalty =
        gen_config.GetExtraFloat("repetition_penalty", 1.05f);
    int32_t max_new_tokens =
        gen_config.GetExtraInt("max_new_tokens", 2048);
    float sub_temperature =
        gen_config.GetExtraFloat("sub_temperature", 0.9f);
    int32_t sub_top_k = gen_config.GetExtraInt("sub_top_k", 50);
    float sub_top_p = gen_config.GetExtraFloat("sub_top_p", 1.0f);

    auto t_start = std::chrono::steady_clock::now();

    // Step 1: Tokenize text
    // Format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    std::string formatted_text =
        "<|im_start|>assistant\n" + text +
        "<|im_end|>\n<|im_start|>assistant\n";

    auto input_ids = tokenizer_.Encode(formatted_text);

    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("Text tokenized: %d tokens", (int)input_ids.size());
    }

    // Step 2: Get text embeddings via text_project
    // Role prefix = first 3 tokens: <|im_start|>, assistant, \n
    // Remaining text = tokens[3:-5]  (last 5 are <|im_end|>\n<|im_start|>assistant\n)
    // Trailing suffix = last 5 tokens

    // Get special token embeddings: tts_bos, tts_eos, tts_pad
    std::vector<int64_t> special_ids = {cfg.tts_bos_token_id,
                                        cfg.tts_eos_token_id,
                                        cfg.tts_pad_token_id};
    auto special_embed = RunTextProjectHelper(special_ids);  // [1, 3, D]
    // special_embed[0] = tts_bos, [1] = tts_eos, [2] = tts_pad

    // Get role prefix embeddings (first 3 tokens)
    std::vector<int64_t> role_ids(input_ids.begin(), input_ids.begin() + 3);
    auto role_embed = RunTextProjectHelper(role_ids);  // [1, 3, D]

    // Build codec prefix: [nothink, think_bos, think_eos, pad, bos]
    // (no speaker embedding for basic TTS, no language specification → auto)
    std::vector<int64_t> codec_prefix_ids = {
        cfg.codec_nothink_id, cfg.codec_think_bos_id,
        cfg.codec_think_eos_id, cfg.codec_pad_id, cfg.codec_bos_id};

    auto codec_prefix_embed =
        RunCodecEmbedHelper(codec_prefix_ids);  // [1, 5, D]

    // Build text embeddings (tokens 3 to end-5 = body text)
    int32_t text_start = 3;
    int32_t text_end = static_cast<int32_t>(input_ids.size()) - 5;
    std::vector<int64_t> body_text_ids;
    if (text_end > text_start) {
      body_text_ids.assign(input_ids.begin() + text_start,
                           input_ids.begin() + text_end);
    }

    // Extract raw float pointers from Ort::Values
    auto special_data = special_embed.GetTensorData<float>();
    auto role_data = role_embed.GetTensorData<float>();
    auto codec_prefix_data = codec_prefix_embed.GetTensorData<float>();

    // Streaming mode (default, matches reference non_streaming_mode=False):
    // Prefill: [role(3)] + [tts_pad+nothink, tts_pad+think_bos,
    //           tts_pad+think_eos, tts_bos+pad] + [text[3]+codec_bos]
    // = 3 + 4 + 1 = 8 positions
    // trailing_text_hidden: [text[4], text[5], ..., text[-6], tts_eos]
    // During decode: next_input = codec_sum + trailing_text_hidden[step]
    //   (or tts_pad if step >= trailing_text_hidden.size())

    int32_t prefill_len = 3 + 3 + 1 + 1;  // role + codec_prefix + tts_bos+pad + text[3]+codec_bos

    // Allocate prefill embedding buffer [1, prefill_len, D]
    std::vector<float> prefill_data(prefill_len * D, 0.0f);

    int32_t pos = 0;

    // Role prefix (3 positions) - just text embeddings, no codec
    for (int32_t i = 0; i < 3; ++i) {
      for (int32_t d = 0; d < D; ++d) {
        prefill_data[pos * D + d] = role_data[i * D + d];
      }
      pos++;
    }

    // Codec prefix 3 tokens: tts_pad + codec_{nothink, think_bos, think_eos}
    for (int32_t i = 0; i < 3; ++i) {
      for (int32_t d = 0; d < D; ++d) {
        prefill_data[pos * D + d] =
            special_data[2 * D + d] +      // tts_pad
            codec_prefix_data[i * D + d];   // codec prefix[i]
      }
      pos++;
    }

    // tts_bos + codec_pad
    for (int32_t d = 0; d < D; ++d) {
      prefill_data[pos * D + d] =
          special_data[0 * D + d] +      // tts_bos
          codec_prefix_data[3 * D + d];  // codec_pad
    }
    pos++;

    // text[3] + codec_bos (first body text token + codec_bos)
    {
      std::vector<int64_t> first_text_ids = {body_text_ids[0]};
      auto first_text_embed = RunTextProjectHelper(first_text_ids);
      auto first_text_data = first_text_embed.GetTensorData<float>();

      for (int32_t d = 0; d < D; ++d) {
        prefill_data[pos * D + d] =
            first_text_data[d] +             // text[3]
            codec_prefix_data[4 * D + d];    // codec_bos
      }
      pos++;
    }

    // Build trailing_text_hidden for streaming mode:
    // [text[4], text[5], ..., text[-6], tts_eos_embed]
    std::vector<std::vector<float>> trailing_text_hidden;
    if (body_text_ids.size() > 1) {
      std::vector<int64_t> trailing_ids(body_text_ids.begin() + 1,
                                         body_text_ids.end());
      auto trailing_embed = RunTextProjectHelper(trailing_ids);
      auto trailing_data = trailing_embed.GetTensorData<float>();
      int32_t n_trailing = static_cast<int32_t>(trailing_ids.size());
      for (int32_t i = 0; i < n_trailing; ++i) {
        trailing_text_hidden.emplace_back(trailing_data + i * D,
                                          trailing_data + (i + 1) * D);
      }
    }
    // Append tts_eos_embed
    trailing_text_hidden.emplace_back(special_data + 1 * D,
                                      special_data + 2 * D);

    // tts_pad_embed for use after trailing text is exhausted
    std::vector<float> tts_pad_vec(special_data + 2 * D,
                                   special_data + 3 * D);

    if (config_.model.debug) {
      SHERPA_ONNX_LOGE(
          "Streaming prefill: %d positions, trailing_text: %d steps",
          prefill_len, static_cast<int32_t>(trailing_text_hidden.size()));
    }

    // Create Ort::Value for prefill embeddings
    std::array<int64_t, 3> prefill_shape = {1, prefill_len, D};
    auto allocator = model_.Allocator();

    Ort::Value prefill_embeds = Ort::Value::CreateTensor(
        allocator, prefill_shape.data(), prefill_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::copy(prefill_data.begin(), prefill_data.end(),
              prefill_embeds.GetTensorMutableData<float>());

    // Create attention mask (all 1s)
    std::array<int64_t, 2> mask_shape = {1, prefill_len};
    Ort::Value attention_mask = Ort::Value::CreateTensor(
        allocator, mask_shape.data(), mask_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto mask_data = attention_mask.GetTensorMutableData<int64_t>();
    std::fill(mask_data, mask_data + prefill_len, 1);

    // Step 3: Run talker prefill
    auto t_prefill_start = std::chrono::steady_clock::now();
    auto prefill_result =
        model_.RunTalkerPrefill(std::move(prefill_embeds),
                                std::move(attention_mask));
    auto t_prefill_end = std::chrono::steady_clock::now();

    // Step 4: Autoregressive generation loop
    // All generated codes: vector of [num_code_groups] per frame
    std::vector<std::vector<int64_t>> all_codes;
    std::vector<int64_t> generated_primary_codes;

    auto logits = std::move(prefill_result.logits);
    auto last_hidden = std::move(prefill_result.last_hidden);
    Qwen3TalkerState talker_state = std::move(prefill_result.state);

    int64_t total_seq_len = prefill_len;

    // suppress_tokens: suppress [vocab_size-1024, vocab_size) except eos
    // This prevents generating invalid control tokens
    int32_t suppress_start = cfg.talker_vocab_size - 1024;
    int32_t suppress_end = cfg.talker_vocab_size;
    int32_t min_new_tokens = 2;

    auto t_gen_start = std::chrono::steady_clock::now();
    for (int32_t step = 0; step < max_new_tokens; ++step) {
      // 4a: Sample primary code from logits
      int64_t primary_code = SampleFromLogits(
          logits, cfg.talker_vocab_size, temperature, top_k, top_p,
          repetition_penalty, generated_primary_codes,
          suppress_start, suppress_end, cfg.codec_eos_token_id,
          step < min_new_tokens);

      // Check for EOS
      if (primary_code == cfg.codec_eos_token_id) {
        if (config_.model.debug) {
          SHERPA_ONNX_LOGE("EOS at step %d", step);
        }
        break;
      }

      generated_primary_codes.push_back(primary_code);

      if (config_.model.debug && step < 3) {
        SHERPA_ONNX_LOGE("Step %d: primary_code=%d", step, (int)primary_code);
      }

      // 4b: Get primary code embedding
      auto primary_embed = RunCodecEmbedHelper({primary_code});  // [1, 1, D]

      // 4c: Run code predictor to get residual codes
      std::vector<int64_t> frame_codes(num_code_groups);
      frame_codes[0] = primary_code;

      // Build code predictor context: [last_hidden, primary_embed]
      auto lh_data = last_hidden.GetTensorData<float>();
      auto lh_shape = last_hidden.GetTensorTypeAndShapeInfo().GetShape();
      int32_t lh_seq = static_cast<int32_t>(lh_shape[1]);

      auto pe_data = primary_embed.GetTensorData<float>();

      // Accumulating embeddings for code predictor
      // Start with [last_hidden_last_pos, primary_embed]
      std::vector<float> cp_context(2 * D);
      // Copy last position of last_hidden
      std::copy(lh_data + (lh_seq - 1) * D, lh_data + lh_seq * D,
                cp_context.begin());
      // Copy primary embed
      std::copy(pe_data, pe_data + D, cp_context.begin() + D);

      // Sum of all codec embeddings (for building next talker input)
      std::vector<float> codec_sum(D, 0.0f);
      for (int32_t d = 0; d < D; ++d) {
        codec_sum[d] = pe_data[d];  // start with primary embed
      }

      for (int32_t j = 0; j < num_code_groups - 1; ++j) {
        // Create inputs_embeds for code predictor
        int32_t cp_seq_len = static_cast<int32_t>(cp_context.size()) / D;
        std::array<int64_t, 3> cp_shape = {1, cp_seq_len, D};
        Ort::Value cp_embeds = Ort::Value::CreateTensor(
            allocator, cp_shape.data(), cp_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        std::copy(cp_context.begin(), cp_context.end(),
                  cp_embeds.GetTensorMutableData<float>());

        // Create generation_step tensor
        std::array<int64_t, 1> gs_shape = {1};
        Ort::Value gen_step = Ort::Value::CreateTensor(
            allocator, gs_shape.data(), gs_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        gen_step.GetTensorMutableData<int64_t>()[0] = j;

        // Run code predictor
        auto cp_logits = model_.RunCodePredictor(std::move(cp_embeds),
                                                 std::move(gen_step));

        // Sample residual code
        int64_t residual_code = SampleFromLogits(
            cp_logits, cfg.code_predictor_vocab_size,
            sub_temperature, sub_top_k, sub_top_p, 1.0f, {});

        frame_codes[j + 1] = residual_code;

        // Get residual code embedding and append to context
        std::array<int64_t, 2> rid_shape = {1, 1};
        Ort::Value rid_tensor = Ort::Value::CreateTensor(
            allocator, rid_shape.data(), rid_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        rid_tensor.GetTensorMutableData<int64_t>()[0] = residual_code;

        Ort::Value gs_tensor = Ort::Value::CreateTensor(
            allocator, gs_shape.data(), gs_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        gs_tensor.GetTensorMutableData<int64_t>()[0] = j;

        auto res_embed = model_.RunCodePredictorEmbed(
            std::move(rid_tensor), std::move(gs_tensor));
        auto res_data = res_embed.GetTensorData<float>();

        // Append to code predictor context
        cp_context.insert(cp_context.end(), res_data, res_data + D);

        // Add to codec sum
        for (int32_t d = 0; d < D; ++d) {
          codec_sum[d] += res_data[d];
        }
      }

      all_codes.push_back(frame_codes);

      // 4d: Build next talker input = codec_sum + trailing_text_hidden[step]
      //   (or tts_pad if past the trailing text)
      const std::vector<float> &text_hidden =
          (step < static_cast<int32_t>(trailing_text_hidden.size()))
              ? trailing_text_hidden[step]
              : tts_pad_vec;
      std::vector<float> next_input(D);
      for (int32_t d = 0; d < D; ++d) {
        next_input[d] = codec_sum[d] + text_hidden[d];
      }

      // Create input embedding for talker decode
      std::array<int64_t, 3> next_shape = {1, 1, D};
      Ort::Value next_embeds = Ort::Value::CreateTensor(
          allocator, next_shape.data(), next_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      std::copy(next_input.begin(), next_input.end(),
                next_embeds.GetTensorMutableData<float>());

      // Growing attention mask
      total_seq_len++;
      std::array<int64_t, 2> new_mask_shape = {1, total_seq_len};
      Ort::Value new_mask = Ort::Value::CreateTensor(
          allocator, new_mask_shape.data(), new_mask_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      auto new_mask_data = new_mask.GetTensorMutableData<int64_t>();
      std::fill(new_mask_data, new_mask_data + total_seq_len, 1);

      // Run talker decode
      auto decode_result = model_.RunTalkerDecode(
          std::move(next_embeds), std::move(new_mask),
          std::move(talker_state));

      logits = std::move(decode_result.logits);
      last_hidden = std::move(decode_result.last_hidden);
      talker_state = std::move(decode_result.state);

      // Progress callback
      if (callback) {
        float progress = static_cast<float>(step + 1) / max_new_tokens;
        if (callback(nullptr, 0, progress) == 0) {
          break;
        }
      }

      if (config_.model.debug && (step + 1) % 100 == 0) {
        SHERPA_ONNX_LOGE("Generated %d frames...", step + 1);
      }
    }

    auto t_gen_end = std::chrono::steady_clock::now();

    if (all_codes.empty()) {
      GeneratedAudio audio;
      audio.sample_rate = SampleRate();
      return audio;
    }

    // Step 5: Decode codec tokens to audio waveform
    int32_t num_frames = static_cast<int32_t>(all_codes.size());
    std::array<int64_t, 3> codes_shape = {1, num_frames, num_code_groups};

    Ort::Value audio_codes = Ort::Value::CreateTensor(
        allocator, codes_shape.data(), codes_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto codes_data = audio_codes.GetTensorMutableData<int64_t>();

    for (int32_t t = 0; t < num_frames; ++t) {
      for (int32_t g = 0; g < num_code_groups; ++g) {
        codes_data[t * num_code_groups + g] = all_codes[t][g];
      }
    }

    auto t_decode_start = std::chrono::steady_clock::now();
    auto decode_result =
        model_.RunTokenizer12hzDecode(std::move(audio_codes));
    auto t_decode_end = std::chrono::steady_clock::now();

    // Extract audio samples using lengths output
    auto audio_data = decode_result.audio_values.GetTensorData<float>();
    auto audio_shape =
        decode_result.audio_values.GetTensorTypeAndShapeInfo().GetShape();
    int32_t total_samples = 1;
    for (auto s : audio_shape) {
      total_samples *= static_cast<int32_t>(s);
    }

    // Use lengths output to determine actual valid samples
    int32_t num_samples = total_samples;
    if (decode_result.lengths.IsTensor()) {
      auto lengths_data = decode_result.lengths.GetTensorData<int64_t>();
      num_samples = static_cast<int32_t>(lengths_data[0]);
      // Clamp to buffer size to prevent buffer overflow
      if (num_samples > total_samples) {
        SHERPA_ONNX_LOGE(
            "Warning: lengths output (%d) exceeds buffer size (%d), clamping",
            num_samples, total_samples);
        num_samples = total_samples;
      }
      if (config_.model.debug) {
        SHERPA_ONNX_LOGE(
            "Decoder output: total=%d, valid=%d (lengths output)",
            total_samples, num_samples);
      }
    }

    GeneratedAudio audio;
    audio.sample_rate = SampleRate();
    audio.samples.assign(audio_data, audio_data + num_samples);

    auto t_end = std::chrono::steady_clock::now();

    // Print performance metrics
    {
      float audio_dur = static_cast<float>(num_samples) / SampleRate();

      auto ms = [](std::chrono::steady_clock::time_point a,
                    std::chrono::steady_clock::time_point b) -> float {
        return std::chrono::duration<float, std::milli>(b - a).count();
      };

      float prefill_ms = ms(t_prefill_start, t_prefill_end);
      float gen_ms = ms(t_gen_start, t_gen_end);
      float decode_ms = ms(t_decode_start, t_decode_end);
      float total_ms = ms(t_start, t_end);
      float total_s = total_ms / 1000.0f;
      float rtf = (audio_dur > 0) ? (total_s / audio_dur) : 0.0f;
      float frames_per_sec =
          (gen_ms > 0) ? (num_frames * 1000.0f / gen_ms) : 0.0f;

      SHERPA_ONNX_LOGE(
          "Audio duration: %.2f s, %d frames (%d codebooks)\n"
          "  Elapsed: %.0f ms  (prefill: %.0f ms, generate: %.0f ms, "
          "decode: %.0f ms)\n"
          "  RTF: %.4f  (%.2fx realtime)\n"
          "  Throughput: %.1f frames/s  (%.1f tokens/s)",
          audio_dur, num_frames, num_code_groups,
          total_ms, prefill_ms, gen_ms, decode_ms,
          rtf, (rtf > 0 ? 1.0f / rtf : 0.0f),
          frames_per_sec, frames_per_sec * num_code_groups);
    }

    if (callback) {
      callback(audio.samples.data(), num_samples, 1.0f);
    }

    return audio;
  }

 private:
  // Helper: run text_project on a vector of token IDs
  Ort::Value RunTextProjectHelper(const std::vector<int64_t> &ids) const {
    auto allocator = model_.Allocator();
    int64_t seq_len = static_cast<int64_t>(ids.size());
    std::array<int64_t, 2> shape = {1, seq_len};

    Ort::Value input = Ort::Value::CreateTensor(
        allocator, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::copy(ids.begin(), ids.end(),
              input.GetTensorMutableData<int64_t>());

    return model_.RunTextProject(std::move(input));
  }

  // Helper: run codec_embed on a vector of token IDs
  Ort::Value RunCodecEmbedHelper(const std::vector<int64_t> &ids) const {
    auto allocator = model_.Allocator();
    int64_t seq_len = static_cast<int64_t>(ids.size());
    std::array<int64_t, 2> shape = {1, seq_len};

    Ort::Value input = Ort::Value::CreateTensor(
        allocator, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::copy(ids.begin(), ids.end(),
              input.GetTensorMutableData<int64_t>());

    return model_.RunCodecEmbed(std::move(input));
  }

  // Sample a token from logits with temperature, top-k, top-p, rep penalty
  // suppress_start/end: suppress tokens in [start, end) except exception
  // suppress_eos: if true, also suppress the eos token (for min_new_tokens)
  int64_t SampleFromLogits(
      const Ort::Value &logits_tensor, int32_t vocab_size,
      float temperature, int32_t top_k, float top_p,
      float repetition_penalty,
      const std::vector<int64_t> &generated_ids,
      int32_t suppress_start = -1, int32_t suppress_end = -1,
      int64_t suppress_exception = -1,
      bool suppress_eos = false) const {
    auto logits_shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const float *logits_data = logits_tensor.GetTensorData<float>();

    // For tensors of shape [1, seq_len, V], take last position
    // For tensors of shape [1, V], take directly
    int32_t total = 1;
    for (auto s : logits_shape) total *= static_cast<int32_t>(s);
    int32_t actual_vocab = total >= vocab_size ? vocab_size : total;

    // Point to the last vocab_size elements
    const float *last_logits = logits_data + (total - actual_vocab);

    // Copy logits to mutable buffer
    std::vector<float> logits_buf(last_logits, last_logits + actual_vocab);

    // Suppress invalid tokens (control tokens except eos)
    if (suppress_start >= 0 && suppress_end > suppress_start) {
      for (int32_t i = suppress_start;
           i < std::min(suppress_end, actual_vocab); ++i) {
        if (i != static_cast<int32_t>(suppress_exception)) {
          logits_buf[i] = -1e9f;
        }
      }
    }

    // Suppress EOS for min_new_tokens
    if (suppress_eos && suppress_exception >= 0 &&
        suppress_exception < actual_vocab) {
      logits_buf[static_cast<int32_t>(suppress_exception)] = -1e9f;
    }

    // Apply repetition penalty
    if (repetition_penalty > 1.0f && !generated_ids.empty()) {
      for (auto id : generated_ids) {
        if (id >= 0 && id < actual_vocab) {
          if (logits_buf[id] > 0) {
            logits_buf[id] /= repetition_penalty;
          } else {
            logits_buf[id] *= repetition_penalty;
          }
        }
      }
    }

    // Temperature
    if (temperature < 1e-6f) {
      // Greedy
      return static_cast<int64_t>(
          std::max_element(logits_buf.begin(), logits_buf.end()) -
          logits_buf.begin());
    }

    for (auto &v : logits_buf) v /= temperature;

    // Top-k filtering
    if (top_k > 0 && top_k < actual_vocab) {
      // Find the top_k-th largest value
      std::vector<float> sorted_logits(logits_buf.begin(), logits_buf.end());
      std::partial_sort(sorted_logits.begin(),
                        sorted_logits.begin() + top_k,
                        sorted_logits.end(), std::greater<float>());
      float threshold = sorted_logits[top_k - 1];

      for (auto &v : logits_buf) {
        if (v < threshold) v = -1e9f;
      }
    }

    // Softmax
    float max_val = *std::max_element(logits_buf.begin(), logits_buf.end());
    float sum_exp = 0.0f;
    for (auto &v : logits_buf) {
      v = std::exp(v - max_val);
      sum_exp += v;
    }
    for (auto &v : logits_buf) v /= sum_exp;

    // Top-p (nucleus) filtering
    if (top_p < 1.0f && top_p > 0.0f) {
      // Create index-probability pairs and sort by probability
      std::vector<std::pair<float, int32_t>> prob_idx(actual_vocab);
      for (int32_t i = 0; i < actual_vocab; ++i) {
        prob_idx[i] = {logits_buf[i], i};
      }
      std::sort(prob_idx.begin(), prob_idx.end(),
                [](const auto &a, const auto &b) {
                  return a.first > b.first;
                });

      float cumsum = 0.0f;
      int32_t cutoff = actual_vocab;
      for (int32_t i = 0; i < actual_vocab; ++i) {
        cumsum += prob_idx[i].first;
        if (cumsum >= top_p) {
          cutoff = i + 1;
          break;
        }
      }

      // Zero out tokens beyond cutoff
      for (int32_t i = cutoff; i < actual_vocab; ++i) {
        logits_buf[prob_idx[i].second] = 0.0f;
      }

      // Renormalize
      float new_sum = 0.0f;
      for (auto v : logits_buf) new_sum += v;
      if (new_sum > 0.0f) {
        for (auto &v : logits_buf) v /= new_sum;
      }
    }

    // Sample from the distribution
    thread_local std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int32_t> dist(logits_buf.begin(),
                                             logits_buf.end());
    return static_cast<int64_t>(dist(gen));
  }

  OfflineTtsConfig config_;
  OfflineTtsQwen3Model model_;
  mutable FunASRNanoTokenizer tokenizer_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_IMPL_H_
