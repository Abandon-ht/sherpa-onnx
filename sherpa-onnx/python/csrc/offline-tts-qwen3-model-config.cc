// sherpa-onnx/python/csrc/offline-tts-qwen3-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-tts-qwen3-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-qwen3-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsQwen3ModelConfig(py::module *m) {
  using PyClass = OfflineTtsQwen3ModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsQwen3ModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &>(),
           py::arg("text_project"), py::arg("codec_embed"),
           py::arg("code_predictor_embed"), py::arg("code_predictor"),
           py::arg("talker_prefill"), py::arg("talker_decode"),
           py::arg("speaker_encoder"), py::arg("tokenizer12hz_encode"),
           py::arg("tokenizer12hz_decode"), py::arg("tokenizer_dir"))
      .def_readwrite("text_project", &PyClass::text_project)
      .def_readwrite("codec_embed", &PyClass::codec_embed)
      .def_readwrite("code_predictor_embed", &PyClass::code_predictor_embed)
      .def_readwrite("code_predictor", &PyClass::code_predictor)
      .def_readwrite("talker_prefill", &PyClass::talker_prefill)
      .def_readwrite("talker_decode", &PyClass::talker_decode)
      .def_readwrite("speaker_encoder", &PyClass::speaker_encoder)
      .def_readwrite("tokenizer12hz_encode", &PyClass::tokenizer12hz_encode)
      .def_readwrite("tokenizer12hz_decode", &PyClass::tokenizer12hz_decode)
      .def_readwrite("tokenizer_dir", &PyClass::tokenizer_dir)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
