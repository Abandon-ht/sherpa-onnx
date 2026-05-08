// sherpa-onnx/csrc/axcl/offline-qwen3-asr-model-axcl.h
//
// Copyright (c)  2026  M5Stack Technology CO LTD

#ifndef SHERPA_ONNX_CSRC_AXCL_OFFLINE_QWEN3_ASR_MODEL_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_OFFLINE_QWEN3_ASR_MODEL_AXCL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineQwen3ASRModelAxcl {
 public:
  explicit OfflineQwen3ASRModelAxcl(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineQwen3ASRModelAxcl(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineQwen3ASRModelAxcl();

  Ort::Value ForwardConvFrontend(Ort::Value input_features);

  Ort::Value ForwardEncoder(Ort::Value conv_output,
                            Ort::Value feature_attention_mask);

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

  bool InitDecoder(const std::string &model_dir, int32_t max_new_tokens);
  void ReleaseDecoder();

  std::string DecodeFromEmbed(const std::vector<unsigned short> &combined_embed,
                              int32_t seq_len, int32_t hidden_size,
                              int32_t max_new_tokens) const;

  std::vector<float> GetTextEmbedding(
      const std::vector<int64_t> &input_ids) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_OFFLINE_QWEN3_ASR_MODEL_AXCL_H_
