// sherpa-onnx/csrc/axera/offline-qwen3-asr-model-axera.h
//
// Copyright (c)  2026  M5Stack Technology CO LTD

#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_QWEN3_ASR_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_QWEN3_ASR_MODEL_AXERA_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineQwen3ASRModelAxera {
 public:
  explicit OfflineQwen3ASRModelAxera(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineQwen3ASRModelAxera(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineQwen3ASRModelAxera();

  /** Run the conv_frontend axmodel.
   *
   * @param input_features  A tensor of shape (N, T, C). Mel features, float32.
   * @return Return conv output of shape (N, T', C'), float32.
   */
  Ort::Value ForwardConvFrontend(Ort::Value input_features);

  /** Run the encoder axmodel.
   *
   * @param conv_output  A tensor of shape (N, T, C). Conv frontend output, float32.
   * @param feature_attention_mask  A tensor of shape (N, T) containing
   * attention mask, bool.
   * @return Return audio features of shape (N, T', hidden_size), float32.
   */
  Ort::Value ForwardEncoder(Ort::Value conv_output,
                            Ort::Value feature_attention_mask);

  /** Run the LLM model (stub for Axera; real decode is done via ax-llm).
   *
   * These interfaces exist only to keep the same API surface as
   * OfflineQwen3ASRModel so that OfflineRecognizerQwen3ASRImpl can
   * compile against both backends.  In Phase-1 they are never called
   * because GenerateText returns early for provider==axera.
   */
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLM(Ort::Value input_ids, Ort::Value audio_features,
             Ort::Value attention_mask, const Ort::Value &cache_position,
             const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv);

  std::vector<std::pair<Ort::Value, Ort::Value>> CreateEmptyKVCache(
      int64_t batch);

  void ApplyKvDeltaInplace(
      std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
      const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
      const Ort::Value &cache_position);

  int32_t GetMaxTotalLen() const;
  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_QWEN3_ASR_MODEL_AXERA_H_
