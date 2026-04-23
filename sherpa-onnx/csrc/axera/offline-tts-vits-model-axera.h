// sherpa-onnx/csrc/axera/offline-tts-vits-model-axera.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_TTS_VITS_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_TTS_VITS_MODEL_AXERA_H_

#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model-meta-data.h"

namespace sherpa_onnx {

class OfflineTtsVitsModelAxera {
 public:
  ~OfflineTtsVitsModelAxera();

  explicit OfflineTtsVitsModelAxera(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsVitsModelAxera(Manager *mgr, const OfflineTtsModelConfig &config);

  Ort::Value Run(Ort::Value x, int64_t sid = 0, float speed = 1.0);

  Ort::Value Run(Ort::Value x, Ort::Value tones, int64_t sid = 0,
                 float speed = 1.0) const;

  const OfflineTtsVitsModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_TTS_VITS_MODEL_AXERA_H_
