// sherpa-onnx/csrc/offline-tts-qwen3-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_IMPL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "sherpa-onnx/csrc/funasr-nano-tokenizer.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-qwen3-model.h"

namespace sherpa_onnx {

class OfflineTtsQwen3Impl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsQwen3Impl(const OfflineTtsConfig &config);

#if __ANDROID_API__ >= 9
  OfflineTtsQwen3Impl(AAssetManager *mgr, const OfflineTtsConfig &config);
#endif

#if __OHOS__
  OfflineTtsQwen3Impl(NativeResourceManager *mgr,
                      const OfflineTtsConfig &config);
#endif

  int32_t SampleRate() const override { return 24000; }

  int32_t NumSpeakers() const override { return 1; }

  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override;

  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback = nullptr) const override;

  GeneratedAudio Generate(
      const std::string &text, const std::string &prompt_text,
      const std::vector<float> &prompt_samples, int32_t sample_rate,
      float speed = 1.0, int32_t num_steps = 4,
      GeneratedAudioCallback callback = nullptr) const override;

 private:
  void InitMelBanks();

  // Compute mel spectrogram for speaker encoder.
  // Input: samples @ 24kHz mono float32 in [-1, 1]
  // Output: mels [num_frames, 128] row-major
  void ComputeMelSpectrogram(const std::vector<float> &samples,
                             std::vector<float> *mels) const;

  // Extract speaker embedding from 24kHz audio samples.
  Ort::Value ExtractSpeakerEmbedding(
      const std::vector<float> &samples_24k) const;

  // Encode reference audio to codec tokens using tokenizer12hz_encode.
  // Returns [1, T, 16] int64 codec codes.
  Ort::Value EncodeReferenceAudio(const std::vector<float> &samples_24k) const;

  // Build voice-clone prefill embeddings and trailing hiddens.
  // ref_text empty  -> x-vector only mode
  // ref_text non-empty -> ICL mode
  GeneratedAudio GenerateVoiceClone(
      const std::string &text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback) const;

  Ort::Value RunTextProjectHelper(const std::vector<int64_t> &ids) const;
  Ort::Value RunCodecEmbedHelper(const std::vector<int64_t> &ids) const;

  // Shared AR generation + decode from a prebuilt prefill.
  GeneratedAudio RunGenerationFromPrefill(
      const std::vector<float> &prefill_data, int32_t prefill_len,
      const std::vector<std::vector<float>> &trailing,
      const std::vector<float> &tts_pad_vec,
      const GenerationConfig &gen_config,
      GeneratedAudioCallback callback) const;

  // Decode a batch of codec frames to audio samples.
  std::vector<float> DecodeFrames(
      const std::vector<std::vector<int64_t>> &codes,
      bool use_stream = false) const;

  int64_t SampleFromLogits(
      const Ort::Value &logits_tensor, int32_t vocab_size, float temperature,
      int32_t top_k, float top_p, float repetition_penalty,
      const std::vector<int64_t> &generated_ids, int32_t suppress_start = -1,
      int32_t suppress_end = -1, int64_t suppress_exception = -1,
      bool suppress_eos = false) const;

  OfflineTtsConfig config_;
  OfflineTtsQwen3Model model_;
  mutable FunASRNanoTokenizer tokenizer_;

  std::unique_ptr<knf::MelBanks> mel_banks_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_QWEN3_IMPL_H_
