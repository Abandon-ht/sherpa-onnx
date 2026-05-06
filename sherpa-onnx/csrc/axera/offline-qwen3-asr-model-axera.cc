// sherpa-onnx/csrc/axera/offline-qwen3-asr-model-axera.cc
//
// Copyright (c)  2026  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axera/offline-qwen3-asr-model-axera.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
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

namespace sherpa_onnx {

namespace {

constexpr int32_t kMelDim = 128;

void ValidateAxModelInputShape(AX_ENGINE_IO_INFO_T *io_info,
                               const char *name,
                               AX_ENGINE_DATA_TYPE_T expected_dtype,
                               const std::vector<int64_t> &expected_shape) {
  if (!io_info || io_info->nInputSize == 0) {
    SHERPA_ONNX_LOGE("%s: no inputs in axmodel", name);
    SHERPA_ONNX_EXIT(-1);
  }
  const auto &in0 = io_info->pInputs[0];
  if (in0.nShapeSize != expected_shape.size()) {
    SHERPA_ONNX_LOGE("%s: input rank mismatch, expected %zu, got %u", name,
                     expected_shape.size(), in0.nShapeSize);
    SHERPA_ONNX_EXIT(-1);
  }
  for (AX_U8 i = 0; i < in0.nShapeSize; ++i) {
    if (in0.pShape[i] != expected_shape[i]) {
      SHERPA_ONNX_LOGE("%s: input shape[%d] mismatch, expected %ld, got %d",
                       name, i, expected_shape[i], in0.pShape[i]);
      SHERPA_ONNX_EXIT(-1);
    }
  }
  if (in0.eDataType != expected_dtype) {
    SHERPA_ONNX_LOGE("%s: input dtype mismatch", name);
    SHERPA_ONNX_EXIT(-1);
  }
}

void ValidateAxModelOutputShape(AX_ENGINE_IO_INFO_T *io_info,
                                const char *name,
                                AX_ENGINE_DATA_TYPE_T expected_dtype,
                                const std::vector<int64_t> &expected_shape) {
  if (!io_info || io_info->nOutputSize == 0) {
    SHERPA_ONNX_LOGE("%s: no outputs in axmodel", name);
    SHERPA_ONNX_EXIT(-1);
  }
  const auto &out0 = io_info->pOutputs[0];
  if (out0.nShapeSize != expected_shape.size()) {
    SHERPA_ONNX_LOGE("%s: output rank mismatch, expected %zu, got %u", name,
                     expected_shape.size(), out0.nShapeSize);
    SHERPA_ONNX_EXIT(-1);
  }
  for (AX_U8 i = 0; i < out0.nShapeSize; ++i) {
    if (out0.pShape[i] != expected_shape[i]) {
      SHERPA_ONNX_LOGE("%s: output shape[%d] mismatch, expected %ld, got %d",
                       name, i, expected_shape[i], out0.pShape[i]);
      SHERPA_ONNX_EXIT(-1);
    }
  }
  if (out0.eDataType != expected_dtype) {
    SHERPA_ONNX_LOGE("%s: output dtype mismatch", name);
    SHERPA_ONNX_EXIT(-1);
  }
}

}  // namespace

class OfflineQwen3ASRModelAxera::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    const auto &c = config_.qwen3_asr;

    // conv_frontend
    {
      auto buf = ReadFile(c.conv_frontend);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read conv_frontend: %s",
                         c.conv_frontend.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      InitAxModel(buf.data(), buf.size(), "conv_frontend", &conv_handle_,
                  &conv_io_info_, &conv_io_data_);

      ValidateAxModelInputShape(
          conv_io_info_, "conv_frontend", AX_ENGINE_DT_FLOAT32, {1, 3000, 128});
      ValidateAxModelOutputShape(
          conv_io_info_, "conv_frontend", AX_ENGINE_DT_FLOAT32, {1, 390, 896});

      conv_input_bytes_ = conv_io_info_->pInputs[0].nSize;
      conv_output_bytes_ = conv_io_info_->pOutputs[0].nSize;
    }

    // encoder
    {
      auto buf = ReadFile(c.encoder);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read encoder: %s", c.encoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      InitAxModel(buf.data(), buf.size(), "encoder", &encoder_handle_,
                  &encoder_io_info_, &encoder_io_data_);

      if (encoder_io_info_->nInputSize != 2) {
        SHERPA_ONNX_LOGE("encoder must have 2 inputs, got %u",
                         encoder_io_info_->nInputSize);
        SHERPA_ONNX_EXIT(-1);
      }
      ValidateAxModelInputShape(
          encoder_io_info_, "encoder[0]", AX_ENGINE_DT_FLOAT32, {1, 390, 896});
      // input1 is uint8 [1, 390]
      {
        const auto &in1 = encoder_io_info_->pInputs[1];
        if (in1.nShapeSize != 2 || in1.pShape[0] != 1 || in1.pShape[1] != 390) {
          SHERPA_ONNX_LOGE(
              "encoder input1 shape mismatch, expected [1,390], got [%d,%d]",
              in1.pShape[0], in1.pShape[1]);
          SHERPA_ONNX_EXIT(-1);
        }
      }
      ValidateAxModelOutputShape(
          encoder_io_info_, "encoder", AX_ENGINE_DT_FLOAT32, {1, 390, 1024});

      encoder_input0_bytes_ = encoder_io_info_->pInputs[0].nSize;
      encoder_input1_bytes_ = encoder_io_info_->pInputs[1].nSize;
      encoder_output_bytes_ = encoder_io_info_->pOutputs[0].nSize;
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Qwen3ASR Axera model init done.");
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    const auto &c = config_.qwen3_asr;

    {
      auto buf = ReadFile(mgr, c.conv_frontend);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read conv_frontend: %s",
                         c.conv_frontend.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      InitAxModel(buf.data(), buf.size(), "conv_frontend", &conv_handle_,
                  &conv_io_info_, &conv_io_data_);
    }

    {
      auto buf = ReadFile(mgr, c.encoder);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read encoder: %s", c.encoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      InitAxModel(buf.data(), buf.size(), "encoder", &encoder_handle_,
                  &encoder_io_info_, &encoder_io_data_);
    }
  }

  ~Impl() {
    FreeIO(&conv_io_data_);
    FreeIO(&encoder_io_data_);
    if (conv_handle_) AX_ENGINE_DestroyHandle(conv_handle_);
    if (encoder_handle_) AX_ENGINE_DestroyHandle(encoder_handle_);
  }

  Ort::Value ForwardConvFrontend(Ort::Value input_features) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto info = input_features.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    auto elem_type =
        static_cast<ONNXTensorElementDataType>(info.GetElementType());

    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      SHERPA_ONNX_LOGE(
          "ForwardConvFrontend: expected float32 input, got elem_type=%d",
          static_cast<int>(elem_type));
      SHERPA_ONNX_EXIT(-1);
    }
    if (shape.size() != 3 || shape[0] != 1 || shape[2] != kMelDim) {
      SHERPA_ONNX_LOGE(
          "ForwardConvFrontend: input shape must be [1,T,%d], got [%ld,%ld,"
          "%ld]",
          kMelDim, shape[0], shape[1], shape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    int64_t in_frames = shape[1];
    const float *src = input_features.GetTensorData<float>();

    // Pad or truncate to fixed shape [1, 3000, 128]
    std::vector<float> padded(conv_input_bytes_ / sizeof(float), 0.0f);
    int64_t copy_frames = std::min<int64_t>(in_frames, 3000);
    std::memcpy(padded.data(), src,
                static_cast<size_t>(copy_frames) * kMelDim * sizeof(float));

    std::memcpy(conv_io_data_.pInputs[0].pVirAddr, padded.data(),
                conv_input_bytes_);

    auto ret = AX_ENGINE_RunSync(conv_handle_, &conv_io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync(conv_frontend) failed, ret=%d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_meta = conv_io_info_->pOutputs[0];
    std::vector<int64_t> out_shape(out_meta.nShapeSize);
    for (AX_U8 i = 0; i < out_meta.nShapeSize; ++i) {
      out_shape[i] = out_meta.pShape[i];
    }

    Ort::Value output = Ort::Value::CreateTensor(
        allocator_, out_shape.data(), out_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(output.GetTensorMutableData<float>(),
                conv_io_data_.pOutputs[0].pVirAddr, out_meta.nSize);

    return output;
  }

  Ort::Value ForwardEncoder(Ort::Value conv_output,
                            Ort::Value feature_attention_mask) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto out_info = conv_output.GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();
    auto out_type =
        static_cast<ONNXTensorElementDataType>(out_info.GetElementType());

    if (out_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      SHERPA_ONNX_LOGE("ForwardEncoder: conv_output must be float32");
      SHERPA_ONNX_EXIT(-1);
    }
    if (out_shape.size() != 3 || out_shape[0] != 1 || out_shape[1] != 390 ||
        out_shape[2] != 896) {
      SHERPA_ONNX_LOGE(
          "ForwardEncoder: conv_output shape must be [1,390,896], got [%ld,%ld,"
          "%ld]",
          out_shape[0], out_shape[1], out_shape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    auto mask_info = feature_attention_mask.GetTensorTypeAndShapeInfo();
    auto mask_shape = mask_info.GetShape();
    auto mask_type =
        static_cast<ONNXTensorElementDataType>(mask_info.GetElementType());

    if (mask_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      SHERPA_ONNX_LOGE(
          "ForwardEncoder: feature_attention_mask must be bool, got %d",
          static_cast<int>(mask_type));
      SHERPA_ONNX_EXIT(-1);
    }
    if (mask_shape.size() != 2 || mask_shape[0] != 1 || mask_shape[1] != 390) {
      SHERPA_ONNX_LOGE(
          "ForwardEncoder: mask shape must be [1,390], got [%ld,%ld]",
          mask_shape[0], mask_shape[1]);
      SHERPA_ONNX_EXIT(-1);
    }

    // input0: float32 [1, 390, 896]
    std::memcpy(encoder_io_data_.pInputs[0].pVirAddr,
                conv_output.GetTensorData<float>(), encoder_input0_bytes_);

    // input1: uint8 [1, 390] — ONNX bool is 1 byte per element
    std::memcpy(encoder_io_data_.pInputs[1].pVirAddr,
                feature_attention_mask.GetTensorData<bool>(),
                encoder_input1_bytes_);

    auto ret = AX_ENGINE_RunSync(encoder_handle_, &encoder_io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync(encoder) failed, ret=%d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_meta = encoder_io_info_->pOutputs[0];
    std::vector<int64_t> encoder_out_shape(out_meta.nShapeSize);
    for (AX_U8 i = 0; i < out_meta.nShapeSize; ++i) {
      encoder_out_shape[i] = out_meta.pShape[i];
    }

    Ort::Value output = Ort::Value::CreateTensor(
        allocator_, encoder_out_shape.data(), encoder_out_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(output.GetTensorMutableData<float>(),
                encoder_io_data_.pOutputs[0].pVirAddr, out_meta.nSize);

    return output;
  }

  // Phase-1 stubs: never called when provider==axera because
  // OfflineRecognizerQwen3ASRImpl::GenerateText returns early.
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLM(Ort::Value /*input_ids*/, Ort::Value /*audio_features*/,
             Ort::Value /*attention_mask*/, const Ort::Value & /*cache_position*/,
             const std::vector<std::pair<Ort::Value, Ort::Value>> & /*cache_kv*/) {
    SHERPA_ONNX_LOGE(
        "ForwardLLM is not implemented for Axera backend. "
        "This should not be called in Phase-1.");
    SHERPA_ONNX_EXIT(-1);
    return {};
  }

  std::vector<std::pair<Ort::Value, Ort::Value>> CreateEmptyKVCache(
      int64_t /*batch*/) {
    SHERPA_ONNX_LOGE(
        "CreateEmptyKVCache is not implemented for Axera backend. "
        "This should not be called in Phase-1.");
    SHERPA_ONNX_EXIT(-1);
    return {};
  }

  void ApplyKvDeltaInplace(
      std::vector<std::pair<Ort::Value, Ort::Value>> * /*cache_kv*/,
      const std::vector<std::pair<Ort::Value, Ort::Value>> & /*kv_delta*/,
      const Ort::Value & /*cache_position*/) {
    SHERPA_ONNX_LOGE(
        "ApplyKvDeltaInplace is not implemented for Axera backend. "
        "This should not be called in Phase-1.");
    SHERPA_ONNX_EXIT(-1);
  }

  int32_t GetMaxTotalLen() const { return 512; }

  OrtAllocator *Allocator() { return allocator_; }

 private:
  void InitAxModel(void *model_data, size_t model_data_length,
                   const char *name, AX_ENGINE_HANDLE *handle,
                   AX_ENGINE_IO_INFO_T **io_info, AX_ENGINE_IO_T *io_data) {
    InitContext(model_data, model_data_length, config_.debug, handle);
    InitInputOutputAttrs(*handle, config_.debug, io_info);
    PrepareIO(*io_info, io_data, config_.debug);
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Axera model %s init ok", name);
    }
  }

 private:
  OfflineModelConfig config_;
  AxEngineGuard ax_engine_guard_;

  std::mutex mutex_;

  AX_ENGINE_HANDLE conv_handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *conv_io_info_ = nullptr;
  AX_ENGINE_IO_T conv_io_data_;
  AX_U32 conv_input_bytes_ = 0;
  AX_U32 conv_output_bytes_ = 0;

  AX_ENGINE_HANDLE encoder_handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *encoder_io_info_ = nullptr;
  AX_ENGINE_IO_T encoder_io_data_;
  AX_U32 encoder_input0_bytes_ = 0;
  AX_U32 encoder_input1_bytes_ = 0;
  AX_U32 encoder_output_bytes_ = 0;

  Ort::AllocatorWithDefaultOptions allocator_;
};

OfflineQwen3ASRModelAxera::OfflineQwen3ASRModelAxera(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineQwen3ASRModelAxera::OfflineQwen3ASRModelAxera(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineQwen3ASRModelAxera::~OfflineQwen3ASRModelAxera() = default;

Ort::Value OfflineQwen3ASRModelAxera::ForwardConvFrontend(
    Ort::Value input_features) {
  return impl_->ForwardConvFrontend(std::move(input_features));
}

Ort::Value OfflineQwen3ASRModelAxera::ForwardEncoder(
    Ort::Value conv_output, Ort::Value feature_attention_mask) {
  return impl_->ForwardEncoder(std::move(conv_output),
                               std::move(feature_attention_mask));
}

std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
OfflineQwen3ASRModelAxera::ForwardLLM(
    Ort::Value input_ids, Ort::Value audio_features, Ort::Value attention_mask,
    const Ort::Value &cache_position,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv) {
  return impl_->ForwardLLM(std::move(input_ids), std::move(audio_features),
                           std::move(attention_mask), cache_position, cache_kv);
}

std::vector<std::pair<Ort::Value, Ort::Value>>
OfflineQwen3ASRModelAxera::CreateEmptyKVCache(int64_t batch) {
  return impl_->CreateEmptyKVCache(batch);
}

void OfflineQwen3ASRModelAxera::ApplyKvDeltaInplace(
    std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
    const Ort::Value &cache_position) {
  impl_->ApplyKvDeltaInplace(cache_kv, kv_delta, cache_position);
}

int32_t OfflineQwen3ASRModelAxera::GetMaxTotalLen() const {
  return impl_->GetMaxTotalLen();
}

OrtAllocator *OfflineQwen3ASRModelAxera::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineQwen3ASRModelAxera::OfflineQwen3ASRModelAxera(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineQwen3ASRModelAxera::OfflineQwen3ASRModelAxera(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
