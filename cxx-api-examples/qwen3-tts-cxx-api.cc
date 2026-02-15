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

#include "sherpa-onnx/c-api/cxx-api.h"

static int32_t ProgressCallback(const float *samples, int32_t num_samples,
                                float progress, void *arg) {
  fprintf(stderr, "Progress: %.3f%%\n", progress * 100);
  // return 1 to continue generating
  // return 0 to stop generating
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
  // speaker_encoder omitted for basic TTS (no voice cloning)
  config.model.qwen3.tokenizer12hz_decode =
      model_dir + "/tokenizer12hz_decode_q.onnx";
  config.model.qwen3.tokenizer_dir = model_dir + "/tokenizer";

  config.model.num_threads = 4;
  config.model.provider = "cpu";

  // If you don't want to see debug messages, please set it to 0
  config.model.debug = 1;

  std::string filename = "./generated-qwen3-tts-cxx.wav";
  std::string text = "今天天气真好，阳光明媚，适合出去走走。";

  auto tts = OfflineTts::Create(config);
  GenerationConfig cfg;
  cfg.speed = 1.0;
  cfg.extra["max_new_tokens"] = "2048";

  // For zero-shot voice cloning, provide a reference audio
  // std::string reference_audio_file = "./reference.wav";
  // Wave wave = ReadWave(reference_audio_file);
  // cfg.reference_audio = std::move(wave.samples);
  // cfg.reference_sample_rate = wave.sample_rate;
  // cfg.extra["max_reference_audio_len"] = "10";

#if 0
  // If you don't want to use a callback, then please enable this branch
  GeneratedAudio audio = tts.Generate(text, cfg);
#else
  GeneratedAudio audio = tts.Generate(text, cfg, ProgressCallback);
#endif

  WriteWave(filename, {audio.samples, audio.sample_rate});

  fprintf(stderr, "Input text is: %s\n", text.c_str());
  fprintf(stderr, "Saved to: %s\n", filename.c_str());

  return 0;
}
