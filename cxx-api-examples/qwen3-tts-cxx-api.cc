// cxx-api-examples/qwen3-tts-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

// This file shows how to use sherpa-onnx CXX API
// for TTS with Qwen3-TTS.
//
// clang-format off
/*
Usage

Please download the Qwen3-TTS ONNX models first:

  https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

./qwen3-tts-cxx-api

 */
// clang-format on

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

// Streaming callback: receives audio chunks as they are decoded.
// samples/num_samples are non-null in streaming mode.
// Return 1 to continue, 0 to stop.
// arg points to a std::vector<float> that accumulates all chunks.
static int32_t StreamingCallback(const float *samples, int32_t num_samples,
                                 float progress, void *arg) {
  fprintf(stderr, "chunk: %d samples  progress: %.1f%%\n", num_samples,
          progress * 100);
  if (samples && num_samples > 0 && arg) {
    auto *acc = reinterpret_cast<std::vector<float> *>(arg);
    acc->insert(acc->end(), samples, samples + num_samples);
  }
  return 1;
}

int32_t main(int32_t argc, char *argv[]) {
  using namespace sherpa_onnx::cxx;  // NOLINT

  OfflineTtsConfig config;
  std::string model_dir = "./qwen3-tts-0.6b-int8";

  config.model.qwen3.text_project = model_dir + "/text_project_q.onnx";
  config.model.qwen3.codec_embed = model_dir + "/codec_embed_q.onnx";
  config.model.qwen3.code_predictor_embed =
      model_dir + "/code_predictor_embed_q.onnx";
  config.model.qwen3.code_predictor = model_dir + "/code_predictor_q.onnx";
  config.model.qwen3.talker_prefill = model_dir + "/talker_prefill_q.onnx";
  config.model.qwen3.talker_decode = model_dir + "/talker_decode_q.onnx";
  config.model.qwen3.tokenizer12hz_decode =
      model_dir + "/tokenizer12hz_decode_q.onnx";
  // Optional: streaming-optimized decoder exported with a small trace_T
  // (e.g. python3 export-onnx.py --streaming-chunk-frames=25).
  // When set, streaming mode decodes each chunk as it is generated
  // (true real-time first-chunk delivery).  Without it, streaming mode
  // fires progress-only callbacks during AR generation and delivers all
  // audio at once after a single batch decode (same quality as batch).
  config.model.qwen3.tokenizer12hz_decode_stream =
      model_dir + "/tokenizer12hz_decode_stream.onnx";
  config.model.qwen3.tokenizer_dir = model_dir + "/tokenizer";
  config.model.num_threads = 4;
  config.model.provider = "cpu";
  config.model.debug = 0;

  std::string text = "今天天气真好，阳光明媚，适合出去走走。";
  auto tts = OfflineTts::Create(config);

  // ------------------------------------------------------------------
  // Test 1: Batch mode  (chunk_frames not set → decode all at once)
  // ------------------------------------------------------------------
  fprintf(stderr, "\n=== Batch mode ===\n");
  {
    GenerationConfig cfg;
    cfg.extra["max_new_tokens"] = "2048";
    GeneratedAudio audio = tts.Generate(text, cfg);
    WriteWave("./qwen3-batch.wav", {audio.samples, audio.sample_rate});
    fprintf(stderr, "Saved to ./qwen3-batch.wav  (%zu samples)\n",
            audio.samples.size());
  }

  // ------------------------------------------------------------------
  // Test 2: Streaming mode  (chunk_frames=25 → ~2s per chunk)
  // ------------------------------------------------------------------
  fprintf(stderr, "\n=== Streaming mode (chunk_frames=25, overlap=2) ===\n");
  {
    GenerationConfig cfg;
    cfg.extra["max_new_tokens"] = "2048";
    cfg.extra["chunk_frames"] = "25";
    cfg.extra["overlap_frames"] = "2";
    std::vector<float> callback_samples;
    GeneratedAudio audio =
        tts.Generate(text, cfg, StreamingCallback, &callback_samples);
    WriteWave("./qwen3-streaming.wav", {audio.samples, audio.sample_rate});
    fprintf(stderr, "Saved to ./qwen3-streaming.wav  (%zu samples)\n",
            audio.samples.size());
    // Save the raw callback audio (streaming decoder output, stitched chunks)
    WriteWave("./qwen3-streaming-callback.wav",
              {callback_samples, audio.sample_rate});
    fprintf(stderr, "Saved to ./qwen3-streaming-callback.wav  (%zu samples)\n",
            callback_samples.size());
  }

  fprintf(stderr, "\nInput text: %s\n", text.c_str());
  return 0;
}
