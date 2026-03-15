// sherpa-onnx/csrc/offline-tts-qwen3-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-qwen3-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsQwen3ModelConfig::Register(ParseOptions *po) {
  po->Register("qwen3-text-project", &text_project,
               "Path to Qwen3-TTS text_project ONNX model");
  po->Register("qwen3-codec-embed", &codec_embed,
               "Path to Qwen3-TTS codec_embed ONNX model");
  po->Register("qwen3-code-predictor-embed", &code_predictor_embed,
               "Path to Qwen3-TTS code_predictor_embed ONNX model");
  po->Register("qwen3-code-predictor", &code_predictor,
               "Path to Qwen3-TTS code_predictor ONNX model");
  po->Register("qwen3-talker-prefill", &talker_prefill,
               "Path to Qwen3-TTS talker_prefill ONNX model");
  po->Register("qwen3-talker-decode", &talker_decode,
               "Path to Qwen3-TTS talker_decode ONNX model");
  po->Register("qwen3-speaker-encoder", &speaker_encoder,
               "Path to Qwen3-TTS speaker_encoder ONNX model");
  po->Register("qwen3-tokenizer12hz-encode", &tokenizer12hz_encode,
               "Path to Qwen3-TTS tokenizer12hz_encode ONNX model");
  po->Register("qwen3-tokenizer12hz-decode", &tokenizer12hz_decode,
               "Path to Qwen3-TTS tokenizer12hz_decode ONNX model (batch)");
  po->Register(
      "qwen3-tokenizer12hz-decode-stream", &tokenizer12hz_decode_stream,
      "Path to streaming-optimized tokenizer12hz_decode ONNX model. "
      "Exported with small fixed max_codes_length for low-latency chunk "
      "decode. Optional; if empty, batch decoder is used for all modes.");
  po->Register("qwen3-tokenizer-dir", &tokenizer_dir,
               "Directory containing Qwen3-TTS tokenizer files");
}

bool OfflineTtsQwen3ModelConfig::Validate() const {
  if (talker_decode.empty()) {
    SHERPA_ONNX_LOGE("Please provide --qwen3-talker-decode");
    return false;
  }

  if (!FileExists(talker_decode)) {
    SHERPA_ONNX_LOGE("--qwen3-talker-decode '%s' does not exist",
                     talker_decode.c_str());
    return false;
  }

  if (talker_prefill.empty()) {
    SHERPA_ONNX_LOGE("Please provide --qwen3-talker-prefill");
    return false;
  }

  if (!FileExists(talker_prefill)) {
    SHERPA_ONNX_LOGE("--qwen3-talker-prefill '%s' does not exist",
                     talker_prefill.c_str());
    return false;
  }

  if (text_project.empty()) {
    SHERPA_ONNX_LOGE("Please provide --qwen3-text-project");
    return false;
  }

  if (!FileExists(text_project)) {
    SHERPA_ONNX_LOGE("--qwen3-text-project '%s' does not exist",
                     text_project.c_str());
    return false;
  }

  if (codec_embed.empty()) {
    SHERPA_ONNX_LOGE("Please provide --qwen3-codec-embed");
    return false;
  }

  if (!FileExists(codec_embed)) {
    SHERPA_ONNX_LOGE("--qwen3-codec-embed '%s' does not exist",
                     codec_embed.c_str());
    return false;
  }

  if (code_predictor.empty()) {
    SHERPA_ONNX_LOGE("Please provide --qwen3-code-predictor");
    return false;
  }

  if (!FileExists(code_predictor)) {
    SHERPA_ONNX_LOGE("--qwen3-code-predictor '%s' does not exist",
                     code_predictor.c_str());
    return false;
  }

  if (code_predictor_embed.empty()) {
    SHERPA_ONNX_LOGE("Please provide --qwen3-code-predictor-embed");
    return false;
  }

  if (!FileExists(code_predictor_embed)) {
    SHERPA_ONNX_LOGE("--qwen3-code-predictor-embed '%s' does not exist",
                     code_predictor_embed.c_str());
    return false;
  }

  // speaker_encoder is optional (only needed for voice cloning)
  if (!speaker_encoder.empty() && !FileExists(speaker_encoder)) {
    SHERPA_ONNX_LOGE("--qwen3-speaker-encoder '%s' does not exist",
                     speaker_encoder.c_str());
    return false;
  }

  // tokenizer12hz_encode is optional (only needed for voice cloning)
  if (!tokenizer12hz_encode.empty() && !FileExists(tokenizer12hz_encode)) {
    SHERPA_ONNX_LOGE("--qwen3-tokenizer12hz-encode '%s' does not exist",
                     tokenizer12hz_encode.c_str());
    return false;
  }

  if (tokenizer12hz_decode.empty()) {
    SHERPA_ONNX_LOGE("Please provide --qwen3-tokenizer12hz-decode");
    return false;
  }

  if (!FileExists(tokenizer12hz_decode)) {
    SHERPA_ONNX_LOGE("--qwen3-tokenizer12hz-decode '%s' does not exist",
                     tokenizer12hz_decode.c_str());
    return false;
  }

  return true;
}

std::string OfflineTtsQwen3ModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsQwen3ModelConfig(";
  os << "text_project=\"" << text_project << "\", ";
  os << "codec_embed=\"" << codec_embed << "\", ";
  os << "code_predictor_embed=\"" << code_predictor_embed << "\", ";
  os << "code_predictor=\"" << code_predictor << "\", ";
  os << "talker_prefill=\"" << talker_prefill << "\", ";
  os << "talker_decode=\"" << talker_decode << "\", ";
  os << "speaker_encoder=\"" << speaker_encoder << "\", ";
  os << "tokenizer12hz_encode=\"" << tokenizer12hz_encode << "\", ";
  os << "tokenizer12hz_decode=\"" << tokenizer12hz_decode << "\", ";
  os << "tokenizer12hz_decode_stream=\"" << tokenizer12hz_decode_stream
     << "\", ";
  os << "tokenizer_dir=\"" << tokenizer_dir << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
