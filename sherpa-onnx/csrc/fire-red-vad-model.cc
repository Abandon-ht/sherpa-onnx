// sherpa-onnx/csrc/fire-red-vad-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/fire-red-vad-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "Eigen/Dense"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class FireRedVadModel::Impl {
 public:
  explicit Impl(const VadModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(config.fire_red_vad.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(mgr, config.fire_red_vad.model);
    Init(buf.data(), buf.size());
  }

  float Run(const float *samples, int32_t n) {
    // Compute FBank features using OnlineFbank
    fbank_->AcceptWaveform(sample_rate_, samples, n);

    // Get the feature for the current frame
    if (fbank_->NumFramesReady() == 0) {
      return 0.0f;
    }

    const float *feat = fbank_->GetFrame(fbank_->NumFramesReady() - 1);

    // Run inference with the current frame's features
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape = {1, 1, feat_dim_};

    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, const_cast<float *>(feat),
                                 feat_dim_, x_shape.data(), x_shape.size());

    // Prepare inputs: feat + caches
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + num_cache_tensors_);

    inputs.push_back(std::move(x));

    for (int32_t i = 0; i < num_cache_tensors_; ++i) {
      inputs.push_back(std::move(caches_[i]));
    }

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    // Get probabilities and update caches
    float prob = out[0].GetTensorData<float>()[0];

    // Update caches (outputs start from index 1)
    for (int32_t i = 0; i < num_cache_tensors_; ++i) {
      caches_[i] = std::move(out[i + 1]);
    }

    return prob;
  }

  void Reset() {
    triggered_ = false;
    current_sample_ = 0;
    temp_start_ = 0;
    temp_end_ = 0;

    // Reset feature extractor
    fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);

    // Reset caches
    ResetCache();
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      exit(-1);
    }

    float prob = Run(samples, n);

    return ProcessProbability(prob);
  }

  bool ProcessProbability(float prob) {
    float threshold = config_.fire_red_vad.threshold;

    current_sample_ += window_shift_;

    if (prob > threshold && temp_end_ != 0) {
      temp_end_ = 0;
    }

    if (prob > threshold && temp_start_ == 0) {
      // start speaking, but we require that it must satisfy
      // min_speech_duration
      temp_start_ = current_sample_;
      return false;
    }

    if (prob > threshold && temp_start_ != 0 && !triggered_) {
      if (current_sample_ - temp_start_ < min_speech_samples_) {
        return false;
      }

      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && !triggered_) {
      // silence
      temp_start_ = 0;
      temp_end_ = 0;
      return false;
    }

    float neg_threshold;
    if (config_.fire_red_vad.neg_threshold < 0) {
      neg_threshold = std::max(threshold - 0.15f, 0.01f);
    } else {
      neg_threshold = std::max(config_.fire_red_vad.neg_threshold, 0.01f);
    }

    if ((prob > neg_threshold) && triggered_) {
      // speaking
      return true;
    }

    if ((prob > threshold) && !triggered_) {
      // start speaking
      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && triggered_) {
      // stop to speak
      if (temp_end_ == 0) {
        temp_end_ = current_sample_;
      }

      if (current_sample_ - temp_end_ < min_silence_samples_) {
        // continue speaking
        return true;
      }
      // stopped speaking
      temp_start_ = 0;
      temp_end_ = 0;
      triggered_ = false;
      return false;
    }

    return false;
  }

  std::vector<float> ComputeProbs(const float *features, int32_t num_frames,
                                  int32_t feat_dim) {
    if (feat_dim != feat_dim_) {
      SHERPA_ONNX_LOGE("Feature dimension mismatch. Expected %d, got %d",
                       feat_dim_, feat_dim);
      return {};
    }

    if (num_frames <= 0) {
      return {};
    }

    // Apply CMVN
    std::vector<float> normalized_features(num_frames * feat_dim_);
    {
      using RowMajorMat =
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      Eigen::Map<RowMajorMat> x(normalized_features.data(), num_frames,
                                feat_dim_);

      Eigen::Map<const Eigen::RowVectorXf> mean(cmvn_mean_.data(), feat_dim_);
      Eigen::Map<const Eigen::RowVectorXf> inv_std(cmvn_inv_stddev_.data(),
                                                   feat_dim_);
      Eigen::Map<const RowMajorMat> feat(features, num_frames, feat_dim);

      x.array() =
          (feat.array().rowwise() - mean.array()).rowwise() * inv_std.array();
    }

    // Run inference with cache
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape = {1, num_frames, feat_dim_};

    Ort::Value x = Ort::Value::CreateTensor(
        memory_info, normalized_features.data(), normalized_features.size(),
        x_shape.data(), x_shape.size());

    // Prepare inputs: feat + caches
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + num_cache_tensors_);

    inputs.push_back(std::move(x));

    for (int32_t i = 0; i < num_cache_tensors_; ++i) {
      inputs.push_back(std::move(caches_[i]));
    }

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    // Get probabilities and update caches
    const float *probs_data = out[0].GetTensorData<float>();
    int64_t probs_size = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> probs(probs_data, probs_data + probs_size);

    // Update caches (outputs start from index 1)
    for (int32_t i = 0; i < num_cache_tensors_; ++i) {
      caches_[i] = std::move(out[i + 1]);
    }

    // Squeeze batch dimension
    std::vector<float> result(num_frames);
    for (int32_t i = 0; i < num_frames; ++i) {
      result[i] = probs[i * 1];  // shape is (1, num_frames, 1)
    }

    return result;
  }

  void ResetCache() {
    caches_.clear();
    caches_.reserve(num_cache_tensors_);

    for (int32_t i = 0; i < num_cache_tensors_; ++i) {
      std::array<int64_t, 3> shape{1, cache_dim_, max_cache_len_};
      Ort::Value cache = Ort::Value::CreateTensor<float>(
          allocator_, shape.data(), shape.size());
      Fill<float>(&cache, 0);
      caches_.push_back(std::move(cache));
    }
  }

  int32_t WindowShift() const { return window_shift_; }

  int32_t WindowSize() const { return window_size_; }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = sample_rate_ * s;
  }

  void SetThreshold(float threshold) {
    config_.fire_red_vad.threshold = threshold;
  }

  int32_t FeatDim() const { return feat_dim_; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %lld", sample_rate_);
      exit(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.fire_red_vad.min_silence_duration;
    min_speech_samples_ =
        sample_rate_ * config_.fire_red_vad.min_speech_duration;

    window_shift_ = config_.fire_red_vad.window_size;
    window_size_ = config_.fire_red_vad.window_size;

    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // First call Check() to get feat_dim_
    Check();

    // Now initialize OnlineFbank with the correct feat_dim
    InitFbank();

    Reset();
  }

  void InitFbank() {
    fbank_opts_.frame_opts.dither = 0;
    fbank_opts_.frame_opts.snip_edges = true;
    fbank_opts_.mel_opts.num_bins = feat_dim_;

    fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);
  }

  void Check() {
    // Get metadata from model
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(model_type, "model_type");

    if (!model_type.empty() && model_type != "fire-red-stream-vad-streaming") {
      SHERPA_ONNX_LOGE(
          "Expect model type fire-red-stream-vad-streaming, given '%s'",
          model_type.c_str());
      exit(-1);
    }

    // Read metadata
    SHERPA_ONNX_READ_META_DATA(feat_dim_, "feat_dim");
    SHERPA_ONNX_READ_META_DATA(num_cache_tensors_, "num_cache_tensors");
    SHERPA_ONNX_READ_META_DATA(max_cache_len_, "max_cache_len");
    SHERPA_ONNX_READ_META_DATA(cache_dim_, "cache_dim");

    if (config_.debug) {
      SHERPA_ONNX_LOGE("feat_dim: %d", feat_dim_);
      SHERPA_ONNX_LOGE("num_cache_tensors: %d", num_cache_tensors_);
      SHERPA_ONNX_LOGE("max_cache_len: %d", max_cache_len_);
      SHERPA_ONNX_LOGE("cache_dim: %d", cache_dim_);
    }

    // Read CMVN
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(cmvn_mean_, "cmvn_mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(cmvn_inv_stddev_, "cmvn_inv_stddev");

    if (static_cast<int32_t>(cmvn_mean_.size()) != feat_dim_) {
      SHERPA_ONNX_LOGE("Incorrect cmvn_mean size. Given %d, expected %d",
                       static_cast<int32_t>(cmvn_mean_.size()), feat_dim_);
      exit(-1);
    }

    if (static_cast<int32_t>(cmvn_inv_stddev_.size()) != feat_dim_) {
      SHERPA_ONNX_LOGE("Incorrect cmvn_inv_stddev size. Given %d, expected %d",
                       static_cast<int32_t>(cmvn_inv_stddev_.size()),
                       feat_dim_);
      exit(-1);
    }

    // Check input/output names
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Input names:");
      for (size_t i = 0; i < input_names_.size(); ++i) {
        SHERPA_ONNX_LOGE("  %d: %s", static_cast<int32_t>(i),
                         input_names_[i].c_str());
      }
      SHERPA_ONNX_LOGE("Output names:");
      for (size_t i = 0; i < output_names_.size(); ++i) {
        SHERPA_ONNX_LOGE("  %d: %s", static_cast<int32_t>(i),
                         output_names_[i].c_str());
      }
    }
  }

 private:
  VadModelConfig config_;

  knf::FbankOptions fbank_opts_;
  std::unique_ptr<knf::OnlineFbank> fbank_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  std::vector<Ort::Value> caches_;
  int32_t num_cache_tensors_ = 0;
  int32_t max_cache_len_ = 0;
  int32_t cache_dim_ = 0;
  int32_t feat_dim_ = 0;

  std::vector<float> cmvn_mean_;
  std::vector<float> cmvn_inv_stddev_;

  int64_t sample_rate_;
  int32_t min_silence_samples_;
  int32_t min_speech_samples_;

  int32_t window_shift_ = 160;  // 10ms at 16kHz
  int32_t window_size_ = 160;

  // VAD state machine
  bool triggered_ = false;
  int32_t current_sample_ = 0;
  int32_t temp_start_ = 0;
  int32_t temp_end_ = 0;
};

FireRedVadModel::FireRedVadModel(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
FireRedVadModel::FireRedVadModel(Manager *mgr, const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

FireRedVadModel::~FireRedVadModel() = default;

void FireRedVadModel::Reset() { return impl_->Reset(); }

bool FireRedVadModel::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

int32_t FireRedVadModel::FeatDim() const { return impl_->FeatDim(); }

int32_t FireRedVadModel::WindowSize() const { return impl_->WindowSize(); }

int32_t FireRedVadModel::WindowShift() const { return impl_->WindowShift(); }

int32_t FireRedVadModel::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}

int32_t FireRedVadModel::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

void FireRedVadModel::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void FireRedVadModel::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
}

float FireRedVadModel::Compute(const float *samples, int32_t n) {
  return impl_->Run(samples, n);
}

std::vector<float> FireRedVadModel::ComputeProbs(const float *features,
                                                 int32_t num_frames,
                                                 int32_t feat_dim) {
  return impl_->ComputeProbs(features, num_frames, feat_dim);
}

#if __ANDROID_API__ >= 9
template FireRedVadModel::FireRedVadModel(AAssetManager *mgr,
                                          const VadModelConfig &config);
#endif

#if __OHOS__
template FireRedVadModel::FireRedVadModel(NativeResourceManager *mgr,
                                          const VadModelConfig &config);
#endif

}  // namespace sherpa_onnx
