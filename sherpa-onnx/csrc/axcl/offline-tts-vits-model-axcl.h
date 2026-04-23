// sherpa-onnx/csrc/axcl/offline-tts-vits-model-axcl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_VITS_MODEL_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_VITS_MODEL_AXCL_H_

#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model-meta-data.h"

namespace sherpa_onnx {

class OfflineTtsVitsModelAxcl {
 public:
  ~OfflineTtsVitsModelAxcl();

  explicit OfflineTtsVitsModelAxcl(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsVitsModelAxcl(Manager *mgr, const OfflineTtsModelConfig &config);

  Ort::Value Run(Ort::Value x, int64_t sid = 0, float speed = 1.0);

  Ort::Value Run(Ort::Value x, Ort::Value tones, int64_t sid = 0,
                 float speed = 1.0) const;

  const OfflineTtsVitsModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_VITS_MODEL_AXCL_H_
