// cxx-api-examples/funasr-nano-cxx-api.cc
//
// Copyright (c)  2025  zengyw
//
// This file demonstrates how to use FunASR-nano with streaming ASR output.
//
// clang-format off
//
// Usage:
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
// tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
// rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
//
// ./funasr-nano-cxx-api [wave_file]
//
// clang-format on

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

#include "sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int32_t main(int32_t argc, char *argv[]) {
  // Configure the recognizer
  sherpa_onnx::OfflineRecognizerConfig config;
  config.model_config.num_threads = 2;
  config.model_config.debug = false;
  config.model_config.provider = "cpu";

  // FunASR-nano model paths
  // clang-format off
  config.model_config.funasr_nano.encoder_adaptor = "./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx";
  config.model_config.funasr_nano.llm = "./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx";
  config.model_config.funasr_nano.embedding = "./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx";
  config.model_config.funasr_nano.tokenizer = "./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B";

  // Use default user_prompt="语音转写：" (matches FunASR-nano training format)
  // Disable repetition control for content with natural repetition (songs):
  config.model_config.funasr_nano.repetition_penalty = 1.0f;
  config.model_config.funasr_nano.no_repeat_ngram_size = 0;
  // clang-format on

  // Validate configuration
  if (!config.Validate()) {
    std::cerr << "Invalid config. Please check your model paths.\n";
    return -1;
  }

  std::cout << "Loading model...\n";
  sherpa_onnx::OfflineRecognizerFunASRNanoImpl recognizer(config);
  std::cout << "Model loaded.\n\n";

  // Read audio file - use command line arg or default
  std::string wave_filename =
      "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav";

  if (argc > 1) {
    wave_filename = argv[1];
  }

  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(wave_filename, &sample_rate, &is_ok);

  if (!is_ok) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  float duration = samples.size() / static_cast<float>(sample_rate);
  std::cout << "Audio: " << wave_filename << "\n";
  std::cout << "Duration: " << duration << "s\n\n";

  // Create stream and accept waveform
  auto stream = recognizer.CreateStream();
  stream->AcceptWaveform(sample_rate, samples.data(), samples.size());

  const auto begin = std::chrono::steady_clock::now();

  // Test streaming API with callback
  std::cout << "Testing streaming API:\n>>> ";
  int32_t token_count = 0;
  auto streaming_callback = [&token_count](const std::string &text,
                                           int64_t token_id,
                                           bool is_final) -> bool {
    std::cout << text << std::flush;
    ++token_count;
    if (is_final) {
      std::cout << std::endl;
    }
    return true;  // Continue generation
  };

  recognizer.DecodeStreamWithCallback(stream.get(), streaming_callback);

  auto result = stream->GetResult();

  const auto end = std::chrono::steady_clock::now();
  const float elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.0f;

  std::cout << "\n\n";
  std::cout << "Tokens: " << token_count << "\n";
  std::cout << "Time: " << elapsed << "s\n";
  std::cout << "RTF: " << elapsed / duration << "\n";

  return 0;
}
