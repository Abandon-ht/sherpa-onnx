# MeloTTS模型导出与推理分析

## 目录
1. [两种模型导出方式对比](#两种模型导出方式对比)
2. [默认导出模型的输入输出](#默认导出模型的输入输出)
3. [C++推理代码分析](#c推理代码分析)
4. [模型预处理与输入输出兼容性](#模型预处理与输入输出兼容性)

---

## 两种模型导出方式对比

### 1. 方式一：单一完整模型导出 (`export-onnx.py`)

#### 特点
- **导出模式**：将完整的VITS模型导出为单一的ONNX文件
- **模型结构**：端到端的完整推理模型，包含编码器、生成器等所有组件
- **适用场景**：直接推理、完整流程推理

#### 导出流程
```python
# 1. 使用ModelWrapper包装模型
class ModelWrapper(torch.nn.Module):
    def forward(self, x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w):
        # 准备BERT和日文BERT特征（填充为零）
        bert = torch.zeros(x.shape[0], 1024, x.shape[1], dtype=torch.float32)
        ja_bert = torch.zeros(x.shape[0], 768, x.shape[1], dtype=torch.float32)
        # 构建语言ID张量
        lang_id = torch.zeros_like(x)
        lang_id[:, 1::2] = self.lang_id
        # 调用模型推理
        return self.model.model.infer(
            x=x, x_lengths=x_lengths, sid=sid, tone=tones, 
            language=lang_id, bert=bert, ja_bert=ja_bert,
            noise_scale=noise_scale, noise_scale_w=noise_scale_w, 
            length_scale=length_scale
        )[0]

# 2. 导出ONNX模型
torch.onnx.export(
    torch_model,
    (x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w),
    "model.onnx",
    opset_version=18,
    input_names=["x", "x_lengths", "tones", "sid", "noise_scale", "length_scale", "noise_scale_w"],
    output_names=["y"],
    dynamic_axes={
        "x": {0: "N", 1: "L"},
        "x_lengths": {0: "N"},
        "tones": {0: "N", 1: "L"},
        "y": {0: "N", 1: "S", 2: "T"},
    }
)

# 3. 添加元数据
meta_data = {
    "model_type": "melo-vits",
    "comment": "melo",
    "version": 2,
    "language": "Chinese + English",
    "add_blank": 1,
    "n_speakers": 1,
    "jieba": 1,
    "sample_rate": 44100,
    "bert_dim": 1024,
    "ja_bert_dim": 768,
    "speaker_id": 1,
    "lang_id": 3,
    "tone_start": 0,
    "url": "https://github.com/myshell-ai/MeloTTS",
    "license": "MIT license"
}
```

#### 优点
- ✅ 模型完整，推理速度快
- ✅ 部署简单，只需一个文件
- ✅ 易于集成到应用程序
- ✅ 元数据完整，便于识别模型类型

#### 缺点
- ❌ 模型文件较大
- ❌ 不适合分阶段量化
- ❌ 难以对不同部分进行性能优化

---

### 2. 方式二：子图分割模型导出 (`export_subgraphs_and_calib.py`)

#### 特点
- **导出模式**：将完整模型分割为两个子图（Subgraph A 和 Subgraph B）
- **模型结构**：
  - **Subgraph A（编码器部分）**：文本处理和特征提取
  - **Subgraph B（生成器部分）**：波形生成
- **适用场景**：分阶段推理、模型优化、量化压缩、边缘设备部署（如AXERA芯片）

#### 子图分割定义

```python
# Subgraph A: 编码器阶段
SUBGRAPH_A_INPUTS = ["x", "x_lengths", "tones", "sid", "noise_scale", "length_scale", "noise_scale_w"]
SUBGRAPH_A_OUTPUTS = ["/Mul_10_output_0", "/Unsqueeze_6_output_0"]
# 输出中间张量

# Subgraph B: 解码器/生成器阶段  
SUBGRAPH_B_INPUTS = ["/Mul_10_output_0", "/Unsqueeze_6_output_0"]
SUBGRAPH_B_OUTPUTS = ["y"]
# 接收Subgraph A的中间输出
```

#### 导出流程

```
完整模型 (model.onnx)
    ↓
    ├─→ onnx.utils.extract_model() ─→ subgraph_a.onnx
    │   (输入: x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w)
    │   (输出: /Mul_10_output_0, /Unsqueeze_6_output_0)
    │
    └─→ onnx.utils.extract_model() ─→ subgraph_b.onnx
        (输入: /Mul_10_output_0, /Unsqueeze_6_output_0)
        (输出: y)
```

#### 校准数据生成

```python
def make_subgraph_a_inputs(vocab_size, sid_value, length):
    """生成Subgraph A的输入"""
    return {
        "x": np.random.randint(0, vocab_size, size=(1, length), dtype=np.int64),
        "x_lengths": np.array([length], dtype=np.int64),
        "tones": np.zeros((1, length), dtype=np.int64),
        "sid": np.array([sid_value], dtype=np.int64),
        "noise_scale": np.array([0.667], dtype=np.float32),
        "length_scale": np.array([1.0], dtype=np.float32),
        "noise_scale_w": np.array([0.8], dtype=np.float32),
    }

# 收集校准数据
for sample_id in range(num_samples):  # 默认12个样本
    # 为Subgraph A生成多种长度的输入
    for length in length_buckets:  # [32, 48, 64, 80, 96, 128, 160, 192, 224, 256]
        a_inputs = make_subgraph_a_inputs(vocab_size, sid_value, length)
        
        # 保存Subgraph A的输入
        for name, value in a_inputs.items():
            save_sample(sample_root_a, name, value, sample_id)
        
        # 运行Subgraph A获取中间输出
        a_outs = sess_a.run(SUBGRAPH_A_OUTPUTS, a_inputs)
        
        # 保存Subgraph B的输入（即Subgraph A的输出）
        b_inputs = {
            SUBGRAPH_B_INPUTS[0]: a_outs[0],
            SUBGRAPH_B_INPUTS[1]: a_outs[1],
        }
        for name, value in b_inputs.items():
            save_sample(sample_root_b, name, value, sample_id)

# 打包为tar.gz格式用于量化工具
package_all_tensors(sample_root_a, SUBGRAPH_A_INPUTS)
package_all_tensors(sample_root_b, SUBGRAPH_B_INPUTS)
```

#### 生成的配置文件

为每个子图生成量化配置文件 (`config_subgraph_a_u16.json` 和 `config_subgraph_b_u16.json`)，其中包含：

```json
{
  "quant": {
    "input_configs": [
      {
        "tensor_name": "x",
        "calibration_dataset": "./x.tar.gz",
        "calibration_size": -1,
        "calibration_format": "Numpy"
      },
      // ... 其他输入张量配置
    ]
  }
}
```

#### 优点
- ✅ 支持分阶段优化和量化
- ✅ 适合边缘设备部署（AXERA等专用芯片）
- ✅ 生成校准数据集，支持INT8量化
- ✅ 可独立优化每个子图
- ✅ 便于性能分析和瓶颈识别

#### 缺点
- ❌ 推理流程复杂（需要依次运行两个子图）
- ❌ 需要保存中间张量，增加内存占用
- ❌ 部署时需要管理多个模型文件
- ❌ 量化工具支持有限

---

### 对比总结表

| 特性 | 单一模型 (export-onnx.py) | 子图分割 (export_subgraphs_and_calib.py) |
|------|------------------------|-------------------------------------|
| **模型数量** | 1个 | 2个 (subgraph_a + subgraph_b) |
| **推理方式** | 端到端一次执行 | 分阶段执行 (A→B) |
| **推理延迟** | 低 | 中等（多次调用开销） |
| **模型大小** | 大 | 较小（可分别量化） |
| **内存占用** | 中等 | 较高（需存储中间张量） |
| **量化支持** | 困难 | 完整（自带校准数据） |
| **边缘设备适配** | 一般 | 优秀 |
| **部署复杂度** | 低 | 高 |
| **性能优化** | 整体优化 | 局部优化 |

---

## 默认导出模型的输入输出

### 模型元数据

```
Model Type: melo-vits
Comment: melo (标识MeloTTS模型)
Version: 2
Language: Chinese + English
Sample Rate: 44100 Hz
License: MIT license
```

### 输入接口

| 输入名称 | 数据类型 | 形状 | 说明 |
|---------|--------|------|------|
| **x** | int64 | [N, L] | 音素/字符ID序列 |
| **x_lengths** | int64 | [N] | 序列长度 |
| **tones** | int64 | [N, L] | 声调信息 |
| **sid** | int64 | [1] | 说话人ID（对MeloTTS通常为1） |
| **noise_scale** | float32 | [1] | 噪声缩放因子（默认0.667） |
| **length_scale** | float32 | [1] | 长度缩放因子（默认1.0） |
| **noise_scale_w** | float32 | [1] | 对抗性噪声缩放（默认0.8） |

#### 输入说明详解

##### 1. x（音素/字符序列）
```
形状: [1, 长度]
数据类型: int64
范围: 0 到 vocab_size-1
说明: 经过文本编码的音素或字符ID，从VITS模型的符号表中获取
示例: [1, 256] 表示处理单个样本，最多256个音素
```

##### 2. x_lengths（序列长度）
```
形状: [1]
数据类型: int64
值: 实际序列长度
说明: 指定x中有效数据的长度
示例: [120] 表示前120个音素有效
用途: 变长序列处理
```

##### 3. tones（声调）
```
形状: [1, 长度]
数据类型: int64
范围: 0-5（表示不同的声调）
说明: 对应x中每个音素的声调标记
用法: 中文和日文需要；英文通常使用0
示例: 对中文文本，可能是 [1, 0, 0, 2, 3, ...] 表示各音素的声调
```

##### 4. sid（说话人ID）
```
形状: [1]
数据类型: int64
值: 0-n_speakers-1
说明: 标识使用的说话人（对MeloTTS固定为1）
注意: C++代码中对单说话人模型会自动设置为模型元数据中的speaker_id
```

##### 5. noise_scale（噪声缩放）
```
形状: [1]
数据类型: float32
范围: 0.0-2.0（典型值0.667）
说明: 控制生成过程中的随机性和多样性
作用: 值越大，生成的语音变化越大；值越小，生成越稳定
调整: 用于微调音质
```

##### 6. length_scale（长度缩放）
```
形状: [1]
数据类型: float32
范围: 0.5-2.0（默认1.0）
说明: 控制生成音频的长度/速度
计算: C++代码中 length_scale = 1.0 / speed
作用: >1 使音频变长（变慢）；<1 使音频变短（变快）
```

##### 7. noise_scale_w（对抗性噪声缩放）
```
形状: [1]
数据类型: float32
范围: 0.0-1.0（默认0.8）
说明: 控制对抗性训练中的噪声缩放
作用: 影响生成音频的自然度和稳定性
```

### 输出接口

| 输出名称 | 数据类型 | 形状 | 说明 |
|---------|--------|------|------|
| **y** | float32 | [N, S, T] | 生成的音频波形 |

#### 输出说明

```
形状: [1, 样本数, 时间步长]
数据类型: float32
范围: -1.0 到 1.0（音频采样值）
说明: 
  - 第一维: 批次大小（MeloTTS固定为1）
  - 第二维: 样本数量（通常等于音素长度×某个倍数）
  - 第三维: 时间步长（表示音频长度）
获取音频: 
  - 展平: output.flatten() 得到1维数组
  - 归一化: 已是[-1, 1]范围，可直接用于WAV编码
  - 采样率: 44100 Hz（从元数据获取）
```

### 动态轴定义

```
"x": {0: "N", 1: "L"}           # 批次和长度维度可变
"x_lengths": {0: "N"}           # 批次维度可变
"tones": {0: "N", 1: "L"}       # 批次和长度维度可变
"y": {0: "N", 1: "S", 2: "T"}   # 所有维度可变
```

这意味着模型支持不同长度的输入和输出，无需重新导出。

---

## C++推理代码分析

### 核心推理类结构

MeloTTS的C++推理实现位于 `offline-tts-vits-model.cc` 和 `offline-tts-vits-model-meta-data.h`。

### 元数据结构

```cpp
struct OfflineTtsVitsModelMetaData {
  // 基本参数
  int32_t sample_rate = 0;        // 采样率（44100）
  int32_t add_blank = 0;          // 是否添加blank符号
  int32_t num_speakers = 0;       // 说话人数量
  
  // 模型类型标识
  bool is_piper = false;
  bool is_coqui = false;
  bool is_icefall = false;
  bool is_melo_tts = false;       // MeloTTS标识
  
  // 中文处理
  int32_t jieba = 0;              // 是否使用jieba分词
  
  // 特殊符号ID
  int32_t blank_id = 0;
  int32_t bos_id = 0;             // 开始符号
  int32_t eos_id = 0;             // 结束符号
  int32_t use_eos_bos = 0;
  int32_t pad_id = 0;
  
  // MeloTTS特有
  int32_t speaker_id = 0;         // 说话人ID
  int32_t version = 0;            // 模型版本
  
  // 字符串信息
  std::string punctuations;       // 标点符号
  std::string language;           // 语言信息
  std::string voice;              // 嗓音信息
  std::string frontend;           // 前端处理方式
};
```

### 推理API

#### 1. 基础推理函数（非MeloTTS）

```cpp
Ort::Value Run(Ort::Value x, int64_t sid = 0, float speed = 1.0)
```

**参数说明：**
- `x`: 形状 [1, L] 的int64张量，表示音素ID序列
- `sid`: 说话人ID，对MeloTTS固定为1（由模型元数据决定）
- `speed`: 语速倍数（1.0为正常速度）

**处理流程：**
```
1. 检查batch size（必须为1）
2. 根据模型类型调用相应推理函数
   ├─ is_piper || is_coqui → RunVitsPiperOrCoqui()
   └─ else → RunVits()
3. 返回生成的音频张量
```

#### 2. MeloTTS推理函数

```cpp
Ort::Value Run(Ort::Value x, Ort::Value tones, int64_t sid = 0, 
               float speed = 1.0) const
```

**参数说明：**
- `x`: 形状 [1, L] 的int64张量
- `tones`: 形状 [1, L] 的int64张量，表示声调序列
- `sid`: 说话人ID（自动覆盖为模型元数据中的speaker_id）
- `speed`: 语速倍数

**MeloTTS特有处理：**
```cpp
// 对单说话人模型，强制使用模型指定的speaker_id
if (meta_data_.num_speakers == 1) {
    sid = meta_data_.speaker_id;  // 通常为1
}

// 构建输入向量（7个输入）
std::vector<Ort::Value> inputs;
inputs.push_back(std::move(x));                          // 音素ID
inputs.push_back(std::move(x_length));                   // 长度
inputs.push_back(std::move(tones));                      // 声调
inputs.push_back(std::move(sid_tensor));                 // 说话人ID
inputs.push_back(std::move(noise_scale_tensor));         // 噪声缩放
inputs.push_back(std::move(length_scale_tensor));        // 长度缩放
inputs.push_back(std::move(noise_scale_w_tensor));       // 对抗性噪声

// 执行推理
auto out = sess_->Run({}, input_names_ptr_.data(), inputs.data(), 
                      inputs.size(), output_names_ptr_.data(), 
                      output_names_ptr_.size());
```

### 核心推理流程详解

```cpp
// 1. 初始化
Ort::Env env_(ORT_LOGGING_LEVEL_ERROR);  // 创建ONNX运行环境
sess_ = std::make_unique<Ort::Session>(env_, model_path);  // 加载模型

// 2. 提取元数据
Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

// 3. 识别模型类型（根据comment字段）
if (comment.find("melo") != std::string::npos) {
    meta_data_.is_melo_tts = true;
    // 版本检查
    if (meta_data_.version < 2) {
        SHERPA_ONNX_EXIT(-1);  // 要求version >= 2
    }
}
```

### 速度控制实现

```cpp
float length_scale = config_.vits.length_scale;

if (speed != 1 && speed > 0) {
    length_scale = 1.0 / speed;
    // 例如: speed=2.0 → length_scale=0.5（音频缩短，速度加快）
    //      speed=0.5 → length_scale=2.0（音频延长，速度减慢）
}
```

### 关键的C++类和方法

#### OfflineTtsVitsModel 类

```cpp
class OfflineTtsVitsModel {
public:
    // 初始化（从文件）
    explicit OfflineTtsVitsModel(const OfflineTtsModelConfig &config);
    
    // 初始化（从内存管理器，用于Android/HarmonyOS）
    template <typename Manager>
    OfflineTtsVitsModel(Manager *mgr, const OfflineTtsModelConfig &config);
    
    // 基础推理API
    Ort::Value Run(Ort::Value x, int64_t sid = 0, float speed = 1.0);
    
    // MeloTTS推理API
    Ort::Value Run(Ort::Value x, Ort::Value tones, int64_t sid = 0, 
                   float speed = 1.0) const;
    
    // 获取元数据
    const OfflineTtsVitsModelMetaData &GetMetaData() const;
    
private:
    class Impl;  // 实现细节隐藏
    std::unique_ptr<Impl> impl_;
};
```

### 异常处理

```cpp
// 批次大小检查
if (x_shape[0] != 1) {
    SHERPA_ONNX_LOGE("Support only batch_size == 1. Given: %d",
                     static_cast<int32_t>(x_shape[0]));
    SHERPA_ONNX_EXIT(-1);
}

// MeloTTS版本检查
if (meta_data_.version < 2) {
    SHERPA_ONNX_LOGE(
        "Please download the latest MeloTTS model and retry. Current "
        "version: %d. Expected version: %d",
        meta_data_.version, expected_version);
    SHERPA_ONNX_EXIT(-1);
}
```

### 内存管理

- 使用 `Ort::MemoryInfo` 管理张量内存
- 使用 `std::move` 优化张量转移，避免拷贝
- 使用 `std::unique_ptr` 管理会话生命周期

---

## 模型预处理与输入输出兼容性

### 1. 文本预处理流程

#### 1.1 中文文本处理

```python
from pypinyin import lazy_pinyin, Style

def get_initial_final_tone(word: str) -> Tuple[List[str], List[str]]:
    """将中文字转换为音素和声调"""
    
    # 1. 分离声母、韵母和声调
    initials = lazy_pinyin(word, style=Style.INITIALS)          # 声母
    finals = lazy_pinyin(word, style=Style.FINALS_TONE3)        # 韵母+声调
    
    # 2. 处理韵母替换规则
    v_rep_map = {"uei": "ui", "iou": "iu", "uen": "un"}
    pinyin_rep_map = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
    single_rep_map = {"v": "yu", "e": "e", "i": "y", "u": "w"}
    
    # 3. 查找符号表并构建音素和声调列表
    phone = pinyin_to_symbol_map[pinyin]  # 获取音素
    ans_tone.append(tone)                 # 记录声调 [1-5]
    
    return ans_phone, ans_tone
```

示例转换：
```
输入文本: "爱芯"
↓ (pinyin转换)
爱 → a4 (声调4)
芯 → x1 (声调1)
↓ (音素转换)
爱 → [AE] + tone 4
芯 → [X] + tone 1
↓ (生成x向量)
x = [爱_id, 芯_id]
tones = [4, 1]
```

#### 1.2 英文文本处理

```python
from melo.text.english import eng_dict, refine_syllables

# 英文使用CMUDict词典
# 示例: "hello" → [H, EH1, L, OW]
phones, tones = refine_syllables(eng_dict["hello"])
tones = [t + language_tone_start_map["EN"] for t in tones]
# 结果: phones=[H,EH1,L,OW], tones全部为language_tone_start_map["EN"]
```

#### 1.3 文本到ID的映射

```python
# 构建词汇表
# tokens.txt 示例:
# <blank> 0
# <eos> 1
# a 2
# b 3
# ...
# AE 256
# ...

# 处理后的文本通过lookup table转换为ID序列
text_tokens = []
for phone in phonemes:
    text_tokens.append(token_to_id[phone])
```

### 2. C++端模型输入准备

#### 2.1 构建ONNX张量

```cpp
// 从Python接收的数据结构
struct TextInput {
    std::vector<int64_t> phoneme_ids;      // 音素ID序列
    std::vector<int64_t> tones;            // 声调序列
    int64_t speaker_id;                    // 说话人ID
    float speed;                           // 语速倍数
    float noise_scale;                     // 噪声缩放
    float length_scale;                    // 长度缩放
};

// 构建ONNX输入张量
auto memory_info = Ort::MemoryInfo::CreateCpu(
    OrtDeviceAllocator, OrtMemTypeDefault);

// x: [1, len]
std::vector<int64_t> x_shape = {1, (int64_t)phoneme_ids.size()};
Ort::Value x = Ort::Value::CreateTensor<int64_t>(
    memory_info, phoneme_ids.data(), phoneme_ids.size(), 
    x_shape.data(), x_shape.size());

// x_lengths: [1]
int64_t len = phoneme_ids.size();
std::vector<int64_t> len_shape = {1};
Ort::Value x_length = Ort::Value::CreateTensor<int64_t>(
    memory_info, &len, 1, len_shape.data(), len_shape.size());

// tones: [1, len]
std::vector<int64_t> tones_shape = {1, (int64_t)tones.size()};
Ort::Value tones_tensor = Ort::Value::CreateTensor<int64_t>(
    memory_info, tones.data(), tones.size(),
    tones_shape.data(), tones_shape.size());

// 标量张量
int64_t sid = meta_data_.speaker_id;
float noise_scale = 0.667f;
float length_scale = (speed != 1.0f && speed > 0) ? (1.0f / speed) : 1.0f;

std::vector<int64_t> scalar_shape = {1};
Ort::Value sid_tensor = Ort::Value::CreateTensor<int64_t>(
    memory_info, &sid, 1, scalar_shape.data(), scalar_shape.size());
Ort::Value noise_scale_tensor = Ort::Value::CreateTensor<float>(
    memory_info, &noise_scale, 1, scalar_shape.data(), scalar_shape.size());
```

#### 2.2 张量形状对齐

```cpp
// 输入张量形状对齐规则

// 1. x张量
输入需求: [1, L] 其中L为音素长度
示例: 60个音素 → shape [1, 60]

// 2. x_lengths张量
输入需求: [1]
值: 60（对应x的第二维长度）

// 3. tones张量
输入需求: [1, L]
与x相同的长度
值范围: 0-5（0表示无声调，1-4表示声调，5表示特殊标记）

// 4. 标量张量（sid, noise_scale等）
输入需求: [1]
值: 单个数值
```

### 3. 输出处理与音频生成

#### 3.1 输出张量格式

```cpp
// 模型输出: y
形状: [1, S, T]
数据类型: float32
范围: [-1.0, 1.0]

// 具体含义:
// [0]: 批次维度（固定1）
// [1]: 样本维度（通常 = len(phonemes) * hop_length / sample_rate）
// [2]: 时间步维度（实际音频长度）
```

#### 3.2 音频处理管道

```cpp
// 1. 获取输出张量
Ort::Value audio_tensor = model.Run(x, tones, sid, speed);

// 2. 获取输出形状和数据
auto info = audio_tensor.GetTensorTypeAndShapeInfo();
std::vector<int64_t> shape = info.GetShape();  // [1, S, T]
float* audio_data = audio_tensor.GetTensorMutableData<float>();
int64_t audio_length = shape[1] * shape[2];

// 3. 展平音频为1D数组
std::vector<float> audio_samples(audio_data, audio_data + audio_length);

// 4. 转换为int16格式（WAV编码）
std::vector<int16_t> pcm_samples(audio_length);
for (int i = 0; i < audio_length; i++) {
    float sample = audio_samples[i];
    // 限制范围在[-1, 1]
    sample = std::max(-1.0f, std::min(1.0f, sample));
    // 转换为int16
    pcm_samples[i] = static_cast<int16_t>(sample * 32767.0f);
}

// 5. 保存为WAV文件
save_wav_file("output.wav", pcm_samples, 44100, 16);  // 采样率从元数据获取
```

### 4. 兼容性检查

#### 4.1 模型版本兼容性

| 检查项 | 要求 | 处理 |
|------|------|------|
| **版本** | >= 2 | 旧模型需重新导出 |
| **comment字段** | 包含"melo" | 用于识别模型类型 |
| **说话人数** | = 1 | 单说话人模型 |
| **采样率** | 44100 | 从元数据读取 |
| **BERT维度** | 1024 | 内部处理 |
| **日文BERT维度** | 768 | 内部处理 |

#### 4.2 输入数据兼容性检查

```cpp
// 批次检查
if (x_shape[0] != 1) {
    error("Batch size must be 1");
    return;
}

// 形状一致性检查
if (x_shape[1] != tones_shape[1]) {
    error("x and tones must have same length");
    return;
}

// 值范围检查
if (x_lengths[0] != x_shape[1]) {
    error("x_lengths must match x length");
    return;
}

// 参数范围检查
if (speed <= 0) {
    error("Speed must be positive");
    return;
}
```

#### 4.3 跨语言兼容性

| 语言 | 声调 | 特殊处理 |
|------|------|--------|
| **中文(ZH)** | 1-4 | 需要jieba分词 |
| **英文(EN)** | 0 | CMUDict词典 |
| **日文(JP)** | 0-3 | 需要日文BERT |
| **韩文(KR)** | 0 | 专用韩文处理 |
| **法文(FR)** | 0 | 专用法文处理 |
| **西班牙文(ES)** | 0 | 专用西班牙文处理 |

### 5. 子图推理的兼容性

对于使用子图分割模型的场景：

#### 5.1 Subgraph A 的输入输出

```
输入 (同完整模型):
  - x: [1, L]
  - x_lengths: [1]
  - tones: [1, L]
  - sid: [1]
  - noise_scale: [1]
  - length_scale: [1]
  - noise_scale_w: [1]

输出 (中间表示):
  - /Mul_10_output_0: [1, 192, L]    # 潜在表示
  - /Unsqueeze_6_output_0: [1, 256, 1]  # 说话人嵌入
```

#### 5.2 Subgraph B 的输入输出

```
输入 (来自Subgraph A的输出):
  - /Mul_10_output_0: [1, 192, L]
  - /Unsqueeze_6_output_0: [1, 256, 1]

输出 (最终音频):
  - y: [1, S, T]  # 生成的音频
```

#### 5.3 兼容性要求

```
✓ Subgraph A的输出形状必须与Subgraph B的输入形状完全匹配
✓ 两个子图的量化精度必须一致（通常都是INT8）
✓ 张量命名规则必须严格遵守（/Mul_10_output_0 等）
✓ 中间张量的存储格式必须保持一致
```

---

## 总结与建议

### 部署建议

1. **完整PC/服务器部署**：使用 `export-onnx.py` 导出的单一模型
   - 推理速度快
   - 部署简单

2. **AXERA边缘设备部署**：使用 `export_subgraphs_and_calib.py` 导出的子图模型
   - 每个子图可独立量化
   - 充分利用硬件加速
   - 配备完整的校准数据集

3. **性能优化**：
   - 微调 `noise_scale` (0.5-1.0) 和 `length_scale` (0.5-1.5) 以获得最佳音质
   - 使用 `speed` 参数进行语速调整
   - 在实时系统中考虑缓冲和流式处理

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| 推理失败 | 模型版本过旧 | 重新导出使用最新版MeloTTS |
| 音频质量差 | 参数设置不当 | 调整noise_scale和length_scale |
| 内存溢出 | 输入序列过长 | 分块处理或增加硬件内存 |
| 量化精度下降 | 校准数据不足 | 使用更多或更多样的校准数据 |

---

## 参考资源

- **MeloTTS官方仓库**: https://github.com/myshell-ai/MeloTTS
- **ONNX模型标准**: https://onnx.ai/
- **Sherpa-ONNX项目**: https://github.com/k2-fsa/sherpa-onnx
- **AXERA芯片支持**: 根据axera目录下的配置文件
