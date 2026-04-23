// sherpa-onnx/csrc/axera/offline-tts-vits-model-axera.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/offline-tts-vits-model-axera.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "ax_engine_api.h"  // NOLINT
#include "ax_sys_api.h"     // NOLINT
#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"
#include "sherpa-onnx/csrc/axera/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

struct ModelPaths {
  std::string encoder;
  std::string decoder;
};

static std::string Basename(const std::string &path) {
  auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }

  return path.substr(pos + 1);
}

static bool EndsWith(const std::string &s, const std::string &suffix) {
  if (suffix.size() > s.size()) {
    return false;
  }

  return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static int32_t ExtractIntegerAfterTag(const std::string &s,
                                      const std::string &tag) {
  auto pos = s.find(tag);
  if (pos == std::string::npos) {
    return -1;
  }

  pos += tag.size();
  int32_t value = 0;
  bool found = false;
  while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
    found = true;
    value = value * 10 + (s[pos] - '0');
    ++pos;
  }

  return found ? value : -1;
}

static ModelPaths ParseModelPaths(const std::string &model_str) {
  std::vector<std::string> files;
  SplitStringToVector(model_str, ",", false, &files);

  if (files.empty()) {
    SHERPA_ONNX_LOGE("Please provide --vits-model");
    SHERPA_ONNX_EXIT(-1);
  }

  ModelPaths paths;
  int32_t best_bucket = -1;

  for (const auto &f : files) {
    if (!FileExists(f)) {
      SHERPA_ONNX_LOGE("Model file '%s' does not exist", f.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::string name = Basename(f);
    if (EndsWith(name, ".onnx") && paths.encoder.empty()) {
      paths.encoder = f;
      continue;
    }

    if (EndsWith(name, ".axmodel")) {
      int32_t b = ExtractIntegerAfterTag(name, "-b");
      if (b <= 0) {
        b = ExtractIntegerAfterTag(name, "b");
      }

      if (b > best_bucket) {
        best_bucket = b;
        paths.decoder = f;
      } else if (paths.decoder.empty()) {
        paths.decoder = f;
      }
    }
  }

  if (paths.encoder.empty() || paths.decoder.empty()) {
    SHERPA_ONNX_LOGE(
        "For Melo Axera 2-stage pipeline, --vits-model must contain one "
        "encoder ONNX and one decoder AXMODEL. Given: %s",
        model_str.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  return paths;
}

class AxeraModel {
 public:
  AxeraModel(const void *cpu_buf, size_t buf_len_in_bytes) {
    Init(cpu_buf, buf_len_in_bytes);
  }

  ~AxeraModel() {
    if (io_data_.pInputs || io_data_.pOutputs) {
      FreeIO(&io_data_);
    }

    if (handle_) {
      AX_ENGINE_DestroyHandle(handle_);
    }
  }

  std::vector<std::string> InputTensorNames() const { return input_names_; }

  std::vector<std::string> OutputTensorNames() const { return output_names_; }

  bool HasTensor(const std::string &name) const {
    return input_name_to_index_.count(name) || output_name_to_index_.count(name);
  }

  std::vector<int32_t> TensorShape(const std::string &name) const {
    auto it = input_name_to_index_.find(name);
    if (it != input_name_to_index_.end()) {
      return input_tensor_shapes_[it->second];
    }

    auto it2 = output_name_to_index_.find(name);
    if (it2 != output_name_to_index_.end()) {
      return output_tensor_shapes_[it2->second];
    }

    SHERPA_ONNX_LOGE("Found no tensor with name: '%s'", name.c_str());
    return {};
  }

  template <typename T>
  bool SetInputTensorData(const std::string &name, const T *p, int32_t n) {
    auto it = input_name_to_index_.find(name);
    if (it == input_name_to_index_.end()) {
      SHERPA_ONNX_LOGE("Found no input tensor with name: '%s'", name.c_str());
      return false;
    }

    int32_t i = it->second;
    size_t expected_size = io_info_->pInputs[i].nSize;
    size_t given_size = n * sizeof(T);
    if (expected_size != given_size) {
      SHERPA_ONNX_LOGE(
          "Input tensor '%s' size mismatch. Expected %zu bytes, got %zu bytes",
          name.c_str(), expected_size, given_size);
      return false;
    }

    std::memcpy(io_data_.pInputs[i].pVirAddr, p, expected_size);
    return true;
  }

  std::vector<float> GetOutputTensorData(const std::string &name) {
    auto it = output_name_to_index_.find(name);
    if (it == output_name_to_index_.end()) {
      SHERPA_ONNX_LOGE("Found no output tensor with name: '%s'", name.c_str());
      return {};
    }

    int32_t i = it->second;
    const auto &out_meta = io_info_->pOutputs[i];
    auto &out_buf = io_data_.pOutputs[i];

    AX_SYS_MinvalidateCache(out_buf.phyAddr, out_buf.pVirAddr, out_meta.nSize);

    size_t out_elems = out_meta.nSize / sizeof(float);
    std::vector<float> out(out_elems);
    std::memcpy(out.data(), out_buf.pVirAddr, out_meta.nSize);
    return out;
  }

  bool Run() {
    int ret = AX_ENGINE_RunSync(handle_, &io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed, ret = %d", ret);
      return false;
    }

    return true;
  }

 private:
  void Init(const void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, false, &handle_);
    InitInputOutputAttrs(handle_, false, &io_info_);
    PrepareIO(io_info_, &io_data_, false);

    input_tensor_shapes_.reserve(io_info_->nInputSize);
    input_names_.reserve(io_info_->nInputSize);
    for (uint32_t i = 0; i < io_info_->nInputSize; ++i) {
      const auto &in = io_info_->pInputs[i];
      std::string name = in.pName;
      input_name_to_index_[name] = i;
      input_tensor_shapes_.emplace_back(in.pShape, in.pShape + in.nShapeSize);
      input_names_.push_back(name);
    }

    output_tensor_shapes_.reserve(io_info_->nOutputSize);
    output_names_.reserve(io_info_->nOutputSize);
    for (uint32_t i = 0; i < io_info_->nOutputSize; ++i) {
      const auto &out = io_info_->pOutputs[i];
      std::string name = out.pName;
      output_name_to_index_[name] = i;
      output_tensor_shapes_.emplace_back(out.pShape,
                                         out.pShape + out.nShapeSize);
      output_names_.push_back(name);
    }
  }

 private:
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *io_info_ = nullptr;
  AX_ENGINE_IO_T io_data_{};

  std::unordered_map<std::string, int32_t> input_name_to_index_;
  std::unordered_map<std::string, int32_t> output_name_to_index_;
  std::vector<std::vector<int32_t>> input_tensor_shapes_;
  std::vector<std::vector<int32_t>> output_tensor_shapes_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace

class OfflineTtsVitsModelAxera::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init(config, [](const std::string &filename) { return ReadFile(filename); });
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init(config, [mgr](const std::string &filename) { return ReadFile(mgr, filename); });
  }

  Ort::Value Run(Ort::Value x, int64_t sid, float speed) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    auto x_shape = x.GetTensorTypeAndShapeInfo().GetShape();
    if (x_shape.size() != 2 || x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Expected x to be int64[1, L]");
      SHERPA_ONNX_EXIT(-1);
    }

    int64_t len = x_shape[1];
    std::vector<int64_t> zero_tones(len, 0);

    std::array<int64_t, 2> tones_shape = {1, len};
    Ort::Value tones = Ort::Value::CreateTensor(
        memory_info, zero_tones.data(), zero_tones.size(), tones_shape.data(),
        tones_shape.size());

    return Run(std::move(x), std::move(tones), sid, speed);
  }

  Ort::Value Run(Ort::Value x, Ort::Value tones, int64_t sid,
                 float speed) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (encoder_input_names_ptr_.size() != 7) {
      SHERPA_ONNX_LOGE(
          "Encoder input count mismatch. Expected 7 inputs "
          "(x,x_lengths,tones,sid,noise_scale,length_scale,noise_scale_w), "
          "but got %zu",
          encoder_input_names_ptr_.size());
      for (size_t i = 0; i != encoder_input_names_.size(); ++i) {
        SHERPA_ONNX_LOGE("  encoder input[%zu]: %s", i,
                         encoder_input_names_[i].c_str());
      }
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta_data_.num_speakers == 1) {
      sid = meta_data_.speaker_id;
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> x_shape = x.GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> tones_shape = tones.GetTensorTypeAndShapeInfo().GetShape();

    if (x_shape.size() != 2 || tones_shape.size() != 2 || x_shape[0] != 1 ||
        tones_shape[0] != 1 || x_shape[1] != tones_shape[1]) {
      SHERPA_ONNX_LOGE("x and tones should have shape int64[1, L]");
      SHERPA_ONNX_EXIT(-1);
    }

    int64_t len = x_shape[1];
    int64_t len_shape = 1;

    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &len, 1, &len_shape, 1);

    int64_t scale_shape = 1;
    float noise_scale = config_.vits.noise_scale;
    float length_scale = config_.vits.length_scale;
    float noise_scale_w = config_.vits.noise_scale_w;

    if (speed != 1 && speed > 0) {
      length_scale = 1.0f / speed;
    }

    Ort::Value noise_scale_tensor =
        Ort::Value::CreateTensor(memory_info, &noise_scale, 1, &scale_shape, 1);
    Ort::Value length_scale_tensor =
        Ort::Value::CreateTensor(memory_info, &length_scale, 1, &scale_shape, 1);
    Ort::Value noise_scale_w_tensor = Ort::Value::CreateTensor(
        memory_info, &noise_scale_w, 1, &scale_shape, 1);
    Ort::Value sid_tensor =
        Ort::Value::CreateTensor(memory_info, &sid, 1, &scale_shape, 1);

    std::vector<Ort::Value> inputs;
    inputs.reserve(7);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    inputs.push_back(std::move(tones));
    inputs.push_back(std::move(sid_tensor));
    inputs.push_back(std::move(noise_scale_tensor));
    inputs.push_back(std::move(length_scale_tensor));
    inputs.push_back(std::move(noise_scale_w_tensor));

    auto out = encoder_sess_->Run({}, encoder_input_names_ptr_.data(),
                                  inputs.data(), inputs.size(),
                                  encoder_output_names_ptr_.data(),
                                  encoder_output_names_ptr_.size());

    if (out.size() < 2) {
      SHERPA_ONNX_LOGE("Encoder output size expected >= 2. Given: %zu", out.size());
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int64_t> z_shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    if (z_shape.size() != 3 || z_shape[1] != 192) {
      SHERPA_ONNX_LOGE("Unexpected z_p shape from encoder");
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t t = static_cast<int32_t>(z_shape[2]);
    const float *z_ptr = out[0].GetTensorData<float>();
    const float *g_ptr = out[1].GetTensorData<float>();

    std::vector<float> audio;

    for (int32_t offset = 0; offset < t; offset += bucket_len_) {
      int32_t cur_t = std::min(bucket_len_, t - offset);
      std::vector<float> z_chunk(192 * bucket_len_, 0.0f);
      for (int32_t c = 0; c != 192; ++c) {
        const float *src = z_ptr + c * t + offset;
        float *dst = z_chunk.data() + c * bucket_len_;
        std::copy(src, src + cur_t, dst);
      }

      if (!decoder_model_->SetInputTensorData(decoder_z_name_, z_chunk.data(),
                                              z_chunk.size()) ||
          !decoder_model_->SetInputTensorData(decoder_g_name_, g_ptr,
                                              g_num_elements_)) {
        SHERPA_ONNX_LOGE("Failed to set decoder inputs");
        SHERPA_ONNX_EXIT(-1);
      }

      if (!decoder_model_->Run()) {
        SHERPA_ONNX_LOGE("Failed to run decoder axmodel");
        SHERPA_ONNX_EXIT(-1);
      }

      std::vector<float> y = decoder_model_->GetOutputTensorData(decoder_audio_name_);
      int32_t keep = cur_t * upsample_factor_;
      if (keep > static_cast<int32_t>(y.size())) {
        SHERPA_ONNX_LOGE("Decoder output too short. keep=%d, got=%d", keep,
                         static_cast<int32_t>(y.size()));
        SHERPA_ONNX_EXIT(-1);
      }

      audio.insert(audio.end(), y.begin(), y.begin() + keep);
    }

    std::array<int64_t, 3> out_shape = {1, 1, static_cast<int64_t>(audio.size())};
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value out_tensor =
        Ort::Value::CreateTensor<float>(allocator, out_shape.data(), out_shape.size());
    float *out_ptr = out_tensor.GetTensorMutableData<float>();
    if (!audio.empty()) {
      std::copy(audio.begin(), audio.end(), out_ptr);
    }

    return out_tensor;
  }

  const OfflineTtsVitsModelMetaData &GetMetaData() const { return meta_data_; }

 private:
  template <typename ReadBytes>
  void Init(const OfflineTtsModelConfig &config, ReadBytes &&read_bytes) {
    if (config.provider != "axera") {
      SHERPA_ONNX_LOGE(
          "This model only supports axera provider. Please use provider=axera");
      SHERPA_ONNX_EXIT(-1);
    }

    auto paths = ParseModelPaths(config.vits.model);

    auto encoder_buf = read_bytes(paths.encoder);
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, encoder_buf.data(), encoder_buf.size(), sess_opts_);
    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);
    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    auto decoder_buf = read_bytes(paths.decoder);
    decoder_model_ =
        std::make_unique<AxeraModel>(decoder_buf.data(), decoder_buf.size());

    decoder_z_name_ = decoder_model_->HasTensor("z_p")
                          ? "z_p"
                          : decoder_model_->InputTensorNames().front();

    if (decoder_model_->HasTensor("g")) {
      decoder_g_name_ = "g";
    } else {
      auto in_names = decoder_model_->InputTensorNames();
      decoder_g_name_ = in_names.size() > 1 ? in_names[1] : in_names.front();
    }

    decoder_audio_name_ = decoder_model_->HasTensor("audio")
                              ? "audio"
                              : decoder_model_->OutputTensorNames().front();

    auto z_shape = decoder_model_->TensorShape(decoder_z_name_);
    auto g_shape = decoder_model_->TensorShape(decoder_g_name_);
    auto y_shape = decoder_model_->TensorShape(decoder_audio_name_);

    if (z_shape.size() < 3 || z_shape[1] != 192) {
      SHERPA_ONNX_LOGE("Invalid decoder z_p shape");
      SHERPA_ONNX_EXIT(-1);
    }

    bucket_len_ = z_shape[2];

    if (g_shape.size() < 3 || g_shape[1] != 256 || g_shape[2] != 1) {
      SHERPA_ONNX_LOGE("Invalid decoder g shape");
      SHERPA_ONNX_EXIT(-1);
    }

    g_num_elements_ = g_shape[0] * g_shape[1] * g_shape[2];

    if (y_shape.size() < 3 || y_shape[2] <= 0 || y_shape[2] % bucket_len_ != 0) {
      SHERPA_ONNX_LOGE("Invalid decoder audio shape");
      SHERPA_ONNX_EXIT(-1);
    }

    upsample_factor_ = y_shape[2] / bucket_len_;

    InitMetaData();

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Axera Melo 2-stage model loaded");
      SHERPA_ONNX_LOGE("  encoder: %s", paths.encoder.c_str());
      SHERPA_ONNX_LOGE("  decoder: %s", paths.decoder.c_str());
      SHERPA_ONNX_LOGE("  bucket_len: %d", bucket_len_);
      SHERPA_ONNX_LOGE("  upsample_factor: %d", upsample_factor_);
    }
  }

  void InitMetaData() {
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();

    if (config_.debug) {
      std::ostringstream os;
      os << "---axera melo encoder model---\n";
      PrintModelMetadata(os, meta_data);
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;
    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.add_blank, "add_blank",
                                            0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.speaker_id, "speaker_id",
                                            0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 0);
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_speakers, "n_speakers");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.punctuations,
                                                "punctuation", "");
    SHERPA_ONNX_READ_META_DATA_STR(meta_data_.language, "language");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.voice, "voice", "");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.frontend, "frontend",
                                                "");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.jieba, "jieba", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.blank_id, "blank_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.bos_id, "bos_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.eos_id, "eos_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.use_eos_bos,
                                            "use_eos_bos", 1);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.pad_id, "pad_id", 0);

    std::string comment;
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(comment, "comment", "");
    if (comment.find("piper") != std::string::npos) {
      meta_data_.is_piper = true;
    }
    if (comment.find("coqui") != std::string::npos) {
      meta_data_.is_coqui = true;
    }
    if (comment.find("icefall") != std::string::npos) {
      meta_data_.is_icefall = true;
    }
    if (comment.find("melo") != std::string::npos) {
      meta_data_.is_melo_tts = true;
    }
  }

 private:
  mutable std::mutex mutex_;

  AxEngineGuard ax_engine_guard_;
  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;
  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::unique_ptr<AxeraModel> decoder_model_;
  std::string decoder_z_name_;
  std::string decoder_g_name_;
  std::string decoder_audio_name_;

  int32_t bucket_len_ = 0;
  int32_t upsample_factor_ = 0;
  int32_t g_num_elements_ = 0;

  OfflineTtsVitsModelMetaData meta_data_;
};

OfflineTtsVitsModelAxera::~OfflineTtsVitsModelAxera() = default;

OfflineTtsVitsModelAxera::OfflineTtsVitsModelAxera(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsVitsModelAxera::OfflineTtsVitsModelAxera(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

Ort::Value OfflineTtsVitsModelAxera::Run(Ort::Value x, int64_t sid,
                                         float speed) {
  return impl_->Run(std::move(x), sid, speed);
}

Ort::Value OfflineTtsVitsModelAxera::Run(Ort::Value x, Ort::Value tones,
                                         int64_t sid, float speed) const {
  return impl_->Run(std::move(x), std::move(tones), sid, speed);
}

const OfflineTtsVitsModelMetaData &OfflineTtsVitsModelAxera::GetMetaData() const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineTtsVitsModelAxera::OfflineTtsVitsModelAxera(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsVitsModelAxera::OfflineTtsVitsModelAxera(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
