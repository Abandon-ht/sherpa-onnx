// sherpa-onnx/csrc/axcl/offline-qwen3-asr-model-axcl.cc
//
// Copyright (c)  2026  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axcl/offline-qwen3-asr-model-axcl.h"

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

#include "sherpa-onnx/csrc/axcl/axcl-model.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

#if __has_include("runner/LLM.hpp")
#include "runner/LLM.hpp"  // ax-llm
#define SHERPA_ONNX_HAS_AX_LLM 1
#else
#define SHERPA_ONNX_HAS_AX_LLM 0
#endif

namespace sherpa_onnx {

namespace {

constexpr int32_t kMelDim = 128;

inline unsigned short float_to_bfloat16(float f) {
  union {
    float f;
    uint32_t u;
  } conv;
  conv.f = f;
  if ((conv.u & 0x00010000) != 0) {
    conv.u += 0x00010000;
  }
  return static_cast<unsigned short>(conv.u >> 16);
}

inline float bfloat16_to_float(unsigned short v) {
  union {
    float f;
    uint32_t u;
  } conv;
  conv.u = static_cast<uint32_t>(v) << 16;
  return conv.f;
}

}  // namespace

class OfflineQwen3ASRModelAxcl::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    const auto &c = config_.qwen3_asr;

    // conv_frontend
    {
      conv_model_ = std::make_unique<AxclModel>(c.conv_frontend);
      if (!conv_model_->IsInitialized()) {
        SHERPA_ONNX_LOGE("Failed to initialize conv_frontend: %s",
                         c.conv_frontend.c_str());
        SHERPA_ONNX_EXIT(-1);
      }

      auto input_names = conv_model_->InputTensorNames();
      auto output_names = conv_model_->OutputTensorNames();
      if (input_names.empty() || output_names.empty()) {
        SHERPA_ONNX_LOGE("conv_frontend has no inputs or outputs");
        SHERPA_ONNX_EXIT(-1);
      }

      auto input_shape = conv_model_->TensorShape(input_names[0]);
      if (input_shape.size() != 3 || input_shape[0] != 1 ||
          input_shape[1] != 3000 || input_shape[2] != 128) {
        SHERPA_ONNX_LOGE(
            "conv_frontend input shape mismatch, expected [1,3000,128], got [%d,%d,%d]",
            input_shape[0], input_shape[1], input_shape[2]);
        SHERPA_ONNX_EXIT(-1);
      }

      auto output_shape = conv_model_->TensorShape(output_names[0]);
      if (output_shape.size() != 3 || output_shape[0] != 1 ||
          output_shape[1] != 390 || output_shape[2] != 896) {
        SHERPA_ONNX_LOGE(
            "conv_frontend output shape mismatch, expected [1,390,896], got [%d,%d,%d]",
            output_shape[0], output_shape[1], output_shape[2]);
        SHERPA_ONNX_EXIT(-1);
      }

      conv_input_bytes_ = conv_model_->TensorSizeInBytes(input_names[0]);
      conv_output_bytes_ = conv_model_->TensorSizeInBytes(output_names[0]);
    }

    // encoder
    {
      encoder_model_ = std::make_unique<AxclModel>(c.encoder);
      if (!encoder_model_->IsInitialized()) {
        SHERPA_ONNX_LOGE("Failed to initialize encoder: %s", c.encoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }

      auto input_names = encoder_model_->InputTensorNames();
      auto output_names = encoder_model_->OutputTensorNames();
      if (input_names.size() != 2 || output_names.empty()) {
        SHERPA_ONNX_LOGE("encoder must have 2 inputs and at least 1 output, got %zu inputs",
                         input_names.size());
        SHERPA_ONNX_EXIT(-1);
      }

      auto input0_shape = encoder_model_->TensorShape(input_names[0]);
      if (input0_shape.size() != 3 || input0_shape[0] != 1 ||
          input0_shape[1] != 390 || input0_shape[2] != 896) {
        SHERPA_ONNX_LOGE(
            "encoder input0 shape mismatch, expected [1,390,896], got [%d,%d,%d]",
            input0_shape[0], input0_shape[1], input0_shape[2]);
        SHERPA_ONNX_EXIT(-1);
      }

      auto input1_shape = encoder_model_->TensorShape(input_names[1]);
      if (input1_shape.size() != 2 || input1_shape[0] != 1 ||
          input1_shape[1] != 390) {
        SHERPA_ONNX_LOGE(
            "encoder input1 shape mismatch, expected [1,390], got [%d,%d]",
            input1_shape[0], input1_shape[1]);
        SHERPA_ONNX_EXIT(-1);
      }

      auto output_shape = encoder_model_->TensorShape(output_names[0]);
      if (output_shape.size() != 3 || output_shape[0] != 1 ||
          output_shape[1] != 390 || output_shape[2] != 1024) {
        SHERPA_ONNX_LOGE(
            "encoder output shape mismatch, expected [1,390,1024], got [%d,%d,%d]",
            output_shape[0], output_shape[1], output_shape[2]);
        SHERPA_ONNX_EXIT(-1);
      }

      encoder_input0_bytes_ = encoder_model_->TensorSizeInBytes(input_names[0]);
      encoder_input1_bytes_ = encoder_model_->TensorSizeInBytes(input_names[1]);
      encoder_output_bytes_ = encoder_model_->TensorSizeInBytes(output_names[0]);
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Qwen3ASR Axcl model init done.");
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
      conv_model_ = std::make_unique<AxclModel>(buf.data(), buf.size());
      if (!conv_model_->IsInitialized()) {
        SHERPA_ONNX_LOGE("Failed to initialize conv_frontend from memory");
        SHERPA_ONNX_EXIT(-1);
      }
    }

    {
      auto buf = ReadFile(mgr, c.encoder);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read encoder: %s", c.encoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      encoder_model_ = std::make_unique<AxclModel>(buf.data(), buf.size());
      if (!encoder_model_->IsInitialized()) {
        SHERPA_ONNX_LOGE("Failed to initialize encoder from memory");
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  ~Impl() { ReleaseDecoder(); }

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
          "ForwardConvFrontend: input shape must be [1,T,%d], got [%ld,%ld,%ld]",
          kMelDim, shape[0], shape[1], shape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    int64_t in_frames = shape[1];
    const float *src = input_features.GetTensorData<float>();

    std::vector<float> padded(conv_input_bytes_ / sizeof(float), 0.0f);
    int64_t copy_frames = std::min<int64_t>(in_frames, 3000);
    std::memcpy(padded.data(), src,
                static_cast<size_t>(copy_frames) * kMelDim * sizeof(float));

    conv_model_->SetInputTensorData(conv_model_->InputTensorNames()[0],
                                    padded.data(),
                                    static_cast<int32_t>(padded.size()));

    if (!conv_model_->Run()) {
      SHERPA_ONNX_LOGE("conv_frontend Run() failed");
      SHERPA_ONNX_EXIT(-1);
    }

    auto out_data =
        conv_model_->GetOutputTensorData(conv_model_->OutputTensorNames()[0]);
    auto out_shape =
        conv_model_->TensorShape(conv_model_->OutputTensorNames()[0]);
    std::vector<int64_t> out_shape_int64(out_shape.begin(), out_shape.end());

    Ort::Value output = Ort::Value::CreateTensor(
        allocator_, out_shape_int64.data(), out_shape_int64.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(output.GetTensorMutableData<float>(), out_data.data(),
                out_data.size() * sizeof(float));

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
          "ForwardEncoder: conv_output shape must be [1,390,896], got [%ld,%ld,%ld]",
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

    encoder_model_->SetInputTensorData(
        encoder_model_->InputTensorNames()[0],
        conv_output.GetTensorData<float>(),
        encoder_input0_bytes_ / static_cast<int32_t>(sizeof(float)));

    encoder_model_->SetInputTensorData(
        encoder_model_->InputTensorNames()[1],
        reinterpret_cast<const uint8_t *>(feature_attention_mask.GetTensorData<bool>()),
        encoder_input1_bytes_ / static_cast<int32_t>(sizeof(uint8_t)));

    if (!encoder_model_->Run()) {
      SHERPA_ONNX_LOGE("encoder Run() failed");
      SHERPA_ONNX_EXIT(-1);
    }

    auto out_data =
        encoder_model_->GetOutputTensorData(encoder_model_->OutputTensorNames()[0]);
    auto encoder_out_shape =
        encoder_model_->TensorShape(encoder_model_->OutputTensorNames()[0]);
    std::vector<int64_t> out_shape_int64(encoder_out_shape.begin(), encoder_out_shape.end());

    Ort::Value output = Ort::Value::CreateTensor(
        allocator_, out_shape_int64.data(), out_shape_int64.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(output.GetTensorMutableData<float>(), out_data.data(),
                out_data.size() * sizeof(float));

    return output;
  }

  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLM(Ort::Value /*input_ids*/, Ort::Value /*audio_features*/,
             Ort::Value /*attention_mask*/, const Ort::Value & /*cache_position*/,
             const std::vector<std::pair<Ort::Value, Ort::Value>> & /*cache_kv*/) {
    SHERPA_ONNX_LOGE(
        "ForwardLLM is not implemented for Axcl backend. "
        "This should not be called in Phase-1.");
    SHERPA_ONNX_EXIT(-1);
    return {};
  }

  std::vector<std::pair<Ort::Value, Ort::Value>> CreateEmptyKVCache(
      int64_t /*batch*/) {
    SHERPA_ONNX_LOGE(
        "CreateEmptyKVCache is not implemented for Axcl backend. "
        "This should not be called in Phase-1.");
    SHERPA_ONNX_EXIT(-1);
    return {};
  }

  void ApplyKvDeltaInplace(
      std::vector<std::pair<Ort::Value, Ort::Value>> * /*cache_kv*/,
      const std::vector<std::pair<Ort::Value, Ort::Value>> & /*kv_delta*/,
      const Ort::Value & /*cache_position*/) {
    SHERPA_ONNX_LOGE(
        "ApplyKvDeltaInplace is not implemented for Axcl backend. "
        "This should not be called in Phase-1.");
    SHERPA_ONNX_EXIT(-1);
  }

  int32_t GetMaxTotalLen() const { return 512; }

  OrtAllocator *Allocator() { return allocator_; }

  bool InitDecoder(const std::string &model_dir, int32_t max_new_tokens) {
#if SHERPA_ONNX_HAS_AX_LLM
    if (decoder_inited_) {
      return true;
    }

    llm_ = std::make_unique<LLM>();
    if (!llm_) {
      SHERPA_ONNX_LOGE("Failed to create LLM instance");
      return false;
    }

    LLMAttrType attr;
    if (!LoadLlmConfig(model_dir, attr)) {
      SHERPA_ONNX_LOGE("Failed to load LLM config from %s", model_dir.c_str());
      llm_.reset();
      return false;
    }

    if (!llm_->Init(attr)) {
      SHERPA_ONNX_LOGE("LLM::Init failed");
      llm_.reset();
      return false;
    }

    if (!LoadEmbedTable(attr.filename_tokens_embed, attr.tokens_embed_num,
                        attr.tokens_embed_size)) {
      SHERPA_ONNX_LOGE("Failed to load embed table from %s",
                       attr.filename_tokens_embed.c_str());
      llm_->Deinit();
      llm_.reset();
      return false;
    }

    embed_token_num_ = attr.tokens_embed_num;
    embed_hidden_size_ = attr.tokens_embed_size;
    decoder_inited_ = true;

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Qwen3ASR Axcl decoder init done. embed_num=%d hidden=%d",
                       embed_token_num_, embed_hidden_size_);
    }
    return true;
#else
    SHERPA_ONNX_LOGE("ax-llm is not available at compile time");
    return false;
#endif
  }

  void ReleaseDecoder() {
#if SHERPA_ONNX_HAS_AX_LLM
    if (llm_) {
      llm_->Deinit();
      llm_.reset();
    }
    embed_data_.clear();
    embed_token_num_ = 0;
    embed_hidden_size_ = 0;
    decoder_inited_ = false;
#endif
  }

  std::string DecodeFromEmbed(const std::vector<unsigned short> &combined_embed,
                              int32_t seq_len, int32_t hidden_size,
                              int32_t max_new_tokens) const {
#if SHERPA_ONNX_HAS_AX_LLM
    if (!decoder_inited_ || !llm_) {
      SHERPA_ONNX_LOGE("DecodeFromEmbed: decoder not initialized");
      return "";
    }

    if (seq_len <= 0 || hidden_size <= 0 ||
        static_cast<int32_t>(combined_embed.size()) != seq_len * hidden_size) {
      SHERPA_ONNX_LOGE(
          "DecodeFromEmbed: invalid shape, combined_embed.size=%zu, "
          "expected seq_len*hidden_size=%d",
          combined_embed.size(), seq_len * hidden_size);
      return "";
    }

    llm_->ResetKVCache();

    std::vector<unsigned short> embed_copy = combined_embed;
    std::string result = llm_->Run(embed_copy, max_new_tokens);
    return result;
#else
    SHERPA_ONNX_LOGE("ax-llm is not available at compile time");
    return "";
#endif
  }

  bool IsDecoderInited() const { return decoder_inited_; }

  void LookupEmbedding(const std::vector<int64_t> &input_ids,
                       std::vector<float> *out_embed) const {
    out_embed->clear();
    if (embed_data_.empty() || embed_hidden_size_ <= 0) {
      return;
    }

    out_embed->reserve(input_ids.size() * embed_hidden_size_);
    for (int64_t id : input_ids) {
      if (id < 0 || id >= embed_token_num_) {
        out_embed->insert(out_embed->end(), embed_hidden_size_, 0.0f);
        continue;
      }
      size_t offset =
          static_cast<size_t>(id) * static_cast<size_t>(embed_hidden_size_);
      for (int32_t h = 0; h < embed_hidden_size_; ++h) {
        out_embed->push_back(bfloat16_to_float(embed_data_[offset + h]));
      }
    }
  }

 private:
#if SHERPA_ONNX_HAS_AX_LLM
  bool LoadLlmConfig(const std::string &model_dir, LLMAttrType &attr) {
    std::string cfg_path = model_dir + "/config.json";
    auto buf = ReadFile(cfg_path);
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("config.json not found in %s", model_dir.c_str());
      return false;
    }

    std::string json_str(buf.begin(), buf.end());

    auto parse_string_field = [&](const std::string &key,
                                  std::string *out) -> bool {
      size_t pos = json_str.find("\"" + key + "\"");
      if (pos == std::string::npos) return false;
      pos = json_str.find(':', pos + key.size() + 2);
      if (pos == std::string::npos) return false;
      pos++;
      while (pos < json_str.size() &&
             (json_str[pos] == ' ' || json_str[pos] == '\t')) {
        pos++;
      }
      if (pos >= json_str.size() || json_str[pos] != '"') return false;
      pos++;
      size_t end = json_str.find('"', pos);
      if (end == std::string::npos) return false;
      *out = json_str.substr(pos, end - pos);
      return true;
    };

    auto parse_int_field = [&](const std::string &key, int *out) -> bool {
      size_t pos = json_str.find("\"" + key + "\"");
      if (pos == std::string::npos) return false;
      pos = json_str.find(':', pos + key.size() + 2);
      if (pos == std::string::npos) return false;
      pos++;
      while (pos < json_str.size() &&
             (json_str[pos] == ' ' || json_str[pos] == '\t' ||
              json_str[pos] == '\n' || json_str[pos] == '\r')) {
        pos++;
      }
      if (pos >= json_str.size()) return false;
      *out = std::atoi(json_str.c_str() + pos);
      return true;
    };

    auto parse_bool_field = [&](const std::string &key, bool *out) -> bool {
      size_t pos = json_str.find("\"" + key + "\"");
      if (pos == std::string::npos) return false;
      pos = json_str.find(':', pos + key.size() + 2);
      if (pos == std::string::npos) return false;
      pos++;
      while (pos < json_str.size() &&
             (json_str[pos] == ' ' || json_str[pos] == '\t' ||
              json_str[pos] == '\n' || json_str[pos] == '\r')) {
        pos++;
      }
      if (pos >= json_str.size()) return false;
      std::string val;
      while (pos < json_str.size() &&
             (json_str[pos] == 't' || json_str[pos] == 'r' ||
              json_str[pos] == 'u' || json_str[pos] == 'e' ||
              json_str[pos] == 'f' || json_str[pos] == 'a' ||
              json_str[pos] == 'l' || json_str[pos] == 's')) {
        val.push_back(json_str[pos]);
        pos++;
      }
      *out = (val == "true");
      return true;
    };

    std::string template_filename;
    std::string filename_post;
    std::string url_tokenizer;
    std::string filename_tokens_embed;
    std::string post_config_path;
    std::string tokenizer_type;
    int axmodel_num = 0;
    int tokens_embed_num = 0;
    int tokens_embed_size = 0;
    bool use_mmap_load_embed = false;
    int full_attention_interval = 0;

    if (!parse_string_field("template_filename_axmodel", &template_filename))
      return false;
    if (!parse_string_field("filename_post_axmodel", &filename_post))
      return false;
    if (!parse_string_field("url_tokenizer_model", &url_tokenizer))
      return false;
    if (!parse_string_field("filename_tokens_embed", &filename_tokens_embed))
      return false;

    parse_string_field("post_config_path", &post_config_path);
    parse_string_field("tokenizer_type", &tokenizer_type);
    parse_int_field("axmodel_num", &axmodel_num);
    parse_int_field("tokens_embed_num", &tokens_embed_num);
    parse_int_field("tokens_embed_size", &tokens_embed_size);
    parse_bool_field("use_mmap_load_embed", &use_mmap_load_embed);
    parse_bool_field("b_use_mmap_load_embed", &use_mmap_load_embed);
    parse_int_field("full_attention_interval", &full_attention_interval);

    attr.template_filename_axmodel =
        ResolvePath(model_dir, template_filename);
    attr.filename_post_axmodel = ResolvePath(model_dir, filename_post);
    attr.url_tokenizer_model = ResolvePath(model_dir, url_tokenizer);
    attr.filename_tokens_embed = ResolvePath(model_dir, filename_tokens_embed);
    attr.post_config_path = ResolvePath(model_dir, post_config_path);
    attr.tokenizer_type = tokenizer_type.empty() ? std::string("Qwen3") : tokenizer_type;
    attr.axmodel_num = axmodel_num;
    attr.tokens_embed_num = tokens_embed_num;
    attr.tokens_embed_size = tokens_embed_size;
    attr.b_use_mmap_load_embed = use_mmap_load_embed;
    attr.full_attention_interval = full_attention_interval;

    attr.b_bos = true;
    attr.b_eos = false;

    return true;
  }

  bool LoadEmbedTable(const std::string &path, int token_num, int hidden_size) {
    auto buf = ReadFile(path);
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("Failed to read embed table: %s", path.c_str());
      return false;
    }

    size_t expected = static_cast<size_t>(token_num) *
                      static_cast<size_t>(hidden_size) * sizeof(unsigned short);
    if (buf.size() != expected) {
      SHERPA_ONNX_LOGE(
          "Embed table size mismatch: %s expected %zu bytes, got %zu",
          path.c_str(), expected, buf.size());
      return false;
    }

    embed_data_.resize(token_num * hidden_size);
    std::memcpy(embed_data_.data(), buf.data(), expected);
    return true;
  }

  static std::string ResolvePath(const std::string &base,
                                  const std::string &p) {
    if (p.empty()) return p;
    if (p.rfind("http://", 0) == 0 || p.rfind("https://", 0) == 0)
      return p;
    if (p.front() == '/') return p;
    return base + "/" + p;
  }
#endif

 private:
  OfflineModelConfig config_;
  std::mutex mutex_;

  std::unique_ptr<AxclModel> conv_model_;
  std::unique_ptr<AxclModel> encoder_model_;

  int32_t conv_input_bytes_ = 0;
  int32_t conv_output_bytes_ = 0;

  int32_t encoder_input0_bytes_ = 0;
  int32_t encoder_input1_bytes_ = 0;
  int32_t encoder_output_bytes_ = 0;

  Ort::AllocatorWithDefaultOptions allocator_;

#if SHERPA_ONNX_HAS_AX_LLM
  mutable std::unique_ptr<LLM> llm_;
#endif
  mutable bool decoder_inited_ = false;
  std::vector<unsigned short> embed_data_;
  int32_t embed_token_num_ = 0;
  int32_t embed_hidden_size_ = 0;
};

OfflineQwen3ASRModelAxcl::OfflineQwen3ASRModelAxcl(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineQwen3ASRModelAxcl::OfflineQwen3ASRModelAxcl(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineQwen3ASRModelAxcl::~OfflineQwen3ASRModelAxcl() = default;

Ort::Value OfflineQwen3ASRModelAxcl::ForwardConvFrontend(
    Ort::Value input_features) {
  return impl_->ForwardConvFrontend(std::move(input_features));
}

Ort::Value OfflineQwen3ASRModelAxcl::ForwardEncoder(
    Ort::Value conv_output, Ort::Value feature_attention_mask) {
  return impl_->ForwardEncoder(std::move(conv_output),
                               std::move(feature_attention_mask));
}

std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
OfflineQwen3ASRModelAxcl::ForwardLLM(
    Ort::Value input_ids, Ort::Value audio_features, Ort::Value attention_mask,
    const Ort::Value &cache_position,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv) {
  return impl_->ForwardLLM(std::move(input_ids), std::move(audio_features),
                           std::move(attention_mask), cache_position, cache_kv);
}

std::vector<std::pair<Ort::Value, Ort::Value>>
OfflineQwen3ASRModelAxcl::CreateEmptyKVCache(int64_t batch) {
  return impl_->CreateEmptyKVCache(batch);
}

void OfflineQwen3ASRModelAxcl::ApplyKvDeltaInplace(
    std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
    const Ort::Value &cache_position) {
  impl_->ApplyKvDeltaInplace(cache_kv, kv_delta, cache_position);
}

int32_t OfflineQwen3ASRModelAxcl::GetMaxTotalLen() const {
  return impl_->GetMaxTotalLen();
}

OrtAllocator *OfflineQwen3ASRModelAxcl::Allocator() const {
  return impl_->Allocator();
}

bool OfflineQwen3ASRModelAxcl::InitDecoder(const std::string &model_dir,
                                            int32_t max_new_tokens) {
  return impl_->InitDecoder(model_dir, max_new_tokens);
}

void OfflineQwen3ASRModelAxcl::ReleaseDecoder() { impl_->ReleaseDecoder(); }

std::string OfflineQwen3ASRModelAxcl::DecodeFromEmbed(
    const std::vector<unsigned short> &combined_embed, int32_t seq_len,
    int32_t hidden_size, int32_t max_new_tokens) const {
  return impl_->DecodeFromEmbed(combined_embed, seq_len, hidden_size,
                                max_new_tokens);
}

std::vector<float> OfflineQwen3ASRModelAxcl::GetTextEmbedding(
    const std::vector<int64_t> &input_ids) const {
  std::vector<float> result;
  impl_->LookupEmbedding(input_ids, &result);
  return result;
}

#if __ANDROID_API__ >= 9
template OfflineQwen3ASRModelAxcl::OfflineQwen3ASRModelAxcl(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineQwen3ASRModelAxcl::OfflineQwen3ASRModelAxcl(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
