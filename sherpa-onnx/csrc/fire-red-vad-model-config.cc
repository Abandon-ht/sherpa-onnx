// sherpa-onnx/csrc/fire-red-vad-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/fire-red-vad-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void FireRedVadModelConfig::Register(ParseOptions *po) {
  po->Register("fire-red-vad-model", &model,
               "Path to the fire-red-vad model.onnx file");

  po->Register("fire-red-vad-threshold", &threshold,
               "Threshold for fire-red vad model");

  po->Register("fire-red-vad-min-silence-duration", &min_silence_duration,
               "Minimum silence duration in seconds");

  po->Register("fire-red-vad-min-speech-duration", &min_speech_duration,
               "Minimum speech duration in seconds");

  po->Register("fire-red-vad-window-size", &window_size,
               "Window size in samples for fire-red vad model");
}

bool FireRedVadModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide a fire-red-vad model");
    return false;
  }

  if (!(model.empty() || EndsWith(model, ".onnx") ||
        EndsWith(model, ".int8.onnx"))) {
    SHERPA_ONNX_LOGE("fire-red-vad model should end with .onnx or .int8.onnx");
    return false;
  }

  return true;
}

std::string FireRedVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "FireRedVadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "window_size=" << window_size << ")";

  return os.str();
}

}  // namespace sherpa_onnx
