# C++ 推理代码兼容性分析

> 本文档分析 Sherpa-ONNX C++ 代码（`offline-tts-vits-model.cc`）与 AXERA `convert.py` 导出模型格式的兼容性，并列出关键差异。

---

## 1. Sherpa-ONNX C++ 接口规范

### 1.1 MeloTTS 推理入口

```cpp
// offline-tts-vits-model.h
Ort::Value Run(Ort::Value x, Ort::Value tones, int64_t sid = 0, float speed = 1.0) const;
```

### 1.2 Run() 内部构建的 7 个输入张量

```cpp
// 按照 ONNX 模型的输入顺序排列
std::vector<Ort::Value> inputs;
inputs.push_back(x);                // [1] x            int64  [1, L]
inputs.push_back(x_length);         // [2] x_lengths    int64  [1]
inputs.push_back(tones);            // [3] tones         int64  [1, L]
inputs.push_back(sid_tensor);       // [4] sid           int64  [1]
inputs.push_back(noise_scale_tensor);  // [5] noise_scale  float32 [1]
inputs.push_back(length_scale_tensor); // [6] length_scale float32 [1]
inputs.push_back(noise_scale_w_tensor);// [7] noise_scale_w float32 [1]
```

### 1.3 C++ 期望的模型接口规范

```
输入  (7 个):
  x           int64[1, L]     - 音素 ID，2D 含 batch
  x_lengths   int64[1]        - 序列长度  
  tones       int64[1, L]     - 声调，2D 含 batch
  sid         int64[1]        - 说话人 ID
  noise_scale float32[1]      - 噪声缩放
  length_scale float32[1]     - 长度缩放
  noise_scale_w float32[1]    - 对抗性噪声缩放

输出  (1 个):
  y           float32[1, S, T] - 音频波形（动态形状）
```

### 1.4 模型识别逻辑

```cpp
// 通过 comment 元数据字段识别 MeloTTS
if (comment.find("melo") != std::string::npos) {
    meta_data_.is_melo_tts = true;
    // 要求 version >= 2
    if (meta_data_.version < 2) {
        SHERPA_ONNX_EXIT(-1);
    }
}

// 调用路径
if (meta_data_.is_melo_tts) {
    // 上层代码调用带 tones 参数的 Run 重载
    return Run(x, tones, sid, speed);
}
```

### 1.5 速度参数转换

```cpp
float length_scale = config_.vits.length_scale;  // 默认 1.0
if (speed != 1 && speed > 0) {
    length_scale = 1.0 / speed;  // speed=2.0 → length_scale=0.5
}
```

---

## 2. AXERA encoder.onnx 接口规范

### 2.1 AXERA encoder 期望的接口

```
输入  (8 个):
  phone       int32[phone_len]  - 音素 ID，1D 无 batch
  tone        int32[phone_len]  - 声调，1D 无 batch
  language    int32[phone_len]  - 语言 ID，1D 无 batch
  g           float32[1,256,1] - 说话人嵌入，外部传入
  noise_scale float32[1]
  noise_scale_w float32[1]
  length_scale float32[1]
  sdp_ratio   float32[1]

输出  (3 个):
  z_p          float32[1, 192, T] - 潜在表示
  pronoun_lens int32[phone_len]   - 每个音素的发音时长
  audio_len    int32[1]           - 总音频采样数
```

### 2.2 AXERA decoder.axmodel 期望的接口

```
输入  (2 个):
  z_p   float32[1, 192, 128]  - 固定形状！
  g     float32[1, 256, 1]    - 说话人嵌入

输出  (1 个):
  audio float32[1, 1, 65536]  - 固定形状！(128 * 512 = 65536)
```

---

## 3. 兼容性分析

### 3.1 ❌ 不兼容项（共 8 项）

| 编号 | 不兼容点 | Sherpa C++ 期望 | AXERA 模型提供 | 影响 |
|------|---------|----------------|--------------|------|
| **I1** | 音素数据类型 | `int64` | `int32` | **严重**：张量类型不匹配，直接崩溃 |
| **I2** | 声调数据类型 | `int64` | `int32` | **严重**：同上 |
| **I3** | 语言ID数据类型 | `int64`（内部） | `int32` | **严重**：同上 |
| **I4** | 音素张量维度 | `[1, L]` 2D | `[phone_len]` 1D | **严重**：shape 不匹配 |
| **I5** | 声调张量维度 | `[1, L]` 2D | `[phone_len]` 1D | **严重**：shape 不匹配 |
| **I6** | 输入个数 | 7 个 | encoder: 8 个 | **严重**：顺序和数量完全不同 |
| **I7** | 说话人嵌入 g | 通过 `sid` 在内部查表 | 外部 `g` 直接传入 | **严重**：接口语义不同 |
| **I8** | 模型数量 | 1 个完整 ONNX | encoder.onnx + decoder.axmodel | **架构**：C++ 只管理 1 个会话 |

### 3.2 ⚠️ 功能缺失项（共 4 项）

| 编号 | 缺失功能 | AXERA 有 | Sherpa C++ 有 | 说明 |
|------|---------|---------|--------------|------|
| **M1** | `pronoun_lens` 输出 | ✅ | ❌ | AXERA 用于 overlap 音频裁剪 |
| **M2** | `audio_len` 输出 | ✅ | ❌ | AXERA 用于截断最后一块的 padding |
| **M3** | z_p 滑窗切片 | ✅ | ❌ | AXERA 核心推理循环 |
| **M4** | overlap 拼接 | ✅（`generate_slices`） | ❌ | AXERA 消除边界伪影 |

### 3.3 ✅ 兼容项（共 3 项）

| 编号 | 兼容项 | Sherpa 值 | AXERA 值 |
|------|-------|---------|---------|
| **C1** | noise_scale 参数名及类型 | `noise_scale` float32[1] | `noise_scale` float32[1] |
| **C2** | noise_scale_w 参数名及类型 | `noise_scale_w` float32[1] | `noise_scale_w` float32[1] |
| **C3** | length_scale 语义 | `1/speed` | `length_scale = 1/speed` |

---

## 4. 兼容性矩阵图

```
                    AXERA encoder.onnx    AXERA decoder.axmodel
                    ──────────────────    ──────────────────────
Sherpa 单会话架构      ❌ 不兼容              ❌ 不兼容
Sherpa x (int64)     ❌ phone 是 int32      N/A
Sherpa x (shape)     ❌ 需要 2D 但 1D       N/A
Sherpa 7 输入顺序     ❌ 需要 8 输入         N/A
Sherpa 无 g 输入      ❌ 需要 g 输入         ❌ 需要 g 输入
Sherpa 输出 y[1,S,T]  N/A                   ❌ 输出 audio[1,1,65536]
Sherpa 动态 shape     N/A                   ❌ 固定 shape
```

---

## 5. 详细不兼容原因剖析

### 5.1 数据类型冲突（int64 vs int32）

**Sherpa C++ 代码：**
```cpp
// 传入 Run() 的 x 张量已经是 int64
Ort::Value Run(Ort::Value x, Ort::Value tones, ...);

// 内部构建 x_lengths 也是 int64
int64_t len = x_shape[1];
Ort::Value x_length = Ort::Value::CreateTensor<int64_t>(...);

// 构建 sid 也是 int64
int64_t sid_val = ...;
Ort::Value sid_tensor = Ort::Value::CreateTensor<int64_t>(...);
```

**AXERA 模型要求（convert.py）：**
```python
phones = torch.zeros(phone_len, dtype=torch.int32)   # int32!
tones = torch.randint(1, 5, size=(phone_len,), dtype=torch.int32)  # int32!
lang_ids = torch.zeros(phone_len, dtype=torch.int32) + 3           # int32!
```

若传入 int64 给 AXERA encoder，ONNX Runtime 会报 `Type mismatch` 错误。

### 5.2 输入顺序和数量不兼容

**Sherpa C++ 构建的输入顺序（7个）：**
```
0: x           int64[1,L]
1: x_lengths   int64[1]
2: tones       int64[1,L]
3: sid         int64[1]
4: noise_scale float32[1]
5: length_scale float32[1]
6: noise_scale_w float32[1]
```

**AXERA encoder 期望的输入顺序（8个）：**
```
0: phone        int32[phone_len]
1: tone         int32[phone_len]
2: language     int32[phone_len]
3: g            float32[1,256,1]
4: noise_scale  float32[1]
5: noise_scale_w float32[1]
6: length_scale float32[1]
7: sdp_ratio    float32[1]
```

注意：同名参数 `noise_scale` / `length_scale` / `noise_scale_w` 的**顺序也不同**（5,6,7 vs 4,6,5）！

### 5.3 说话人嵌入 g 的架构差异

**Sherpa 方案：**
```
sid (int64) 输入模型 → model.emb_g(sid) → g [1,256,1] → 流向 encoder + decoder
```
g 在 ONNX 计算图内部，对用户透明。

**AXERA 方案：**
```
sid → Python: g = model.emb_g(torch.IntTensor([sid])).unsqueeze(-1)
           → g.tofile("g-zh.bin")      # 预计算保存
                ↓
encoder(phone, tone, lang, g, ...)     # g 作为外部输入
decoder(z_p, g)                        # g 再次作为外部输入
```
g 在模型外部，必须显式传递给两个模型。Sherpa C++ 没有这种机制。

### 5.4 模型分段架构不兼容

Sherpa C++ 只维护一个 `Ort::Session`，一次 `sess_->Run()` 完成从文本到音频：

```cpp
auto out = sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
return std::move(out[0]);  // 直接返回音频
```

AXERA 需要两个独立的推理引擎：
```cpp
OnnxWrapper encoder;         // ONNX Runtime (CPU)
EngineWrapper decoder_model; // AXERA Engine API (NPU)
```

---

## 6. 兼容性改造方案

### 方案 1（最小改动）：ONNX 层面适配

在导出 encoder ONNX 时，添加适配层将接口调整为 Sherpa 期望的格式：

```python
class SherpaCompatWrapper(torch.nn.Module):
    """将 AXERA 编码器接口适配为 Sherpa C++ 期望的接口"""
    def forward(self, x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w):
        # 内部处理类型转换
        phone = x[0].int()           # int64[L] → int32[L]  
        tone = tones[0].int()        # int64[L] → int32[L]
        lang = torch.zeros_like(phone) + lang_id
        g = emb_g(sid).unsqueeze(-1)  # 内部计算 g
        # ... 调用 enc_forward
        z_p, pronoun_lens, audio_len = self.enc_forward(phone, tone, lang, g, ...)
        # 对 z_p 进行切片推理 decoder (但 decoder 在 NPU 上...)
        # 这里无法直接集成 NPU decoder
```

**局限**：decoder 在 NPU 上，无法集成到 ONNX 计算图中

### 方案 2（推荐）：新增 C++ 推理类

保持 AXERA 的 encoder+decoder 双模型架构，在 sherpa-onnx 中新增 `OfflineTtsAxeraMeloModel` 类：

```cpp
class OfflineTtsAxeraMeloModel {
public:
    // 专门用于 AXERA NPU 的 MeloTTS 推理
    Ort::Value Run(Ort::Value x,          // int32[1, L]
                   Ort::Value tones,       // int32[1, L]
                   int64_t sid,
                   float speed) const;
private:
    OnnxWrapper encoder_;      // encoder.onnx on CPU
    AxeraEngine decoder_;      // decoder.axmodel on NPU
    std::vector<float> g_;     // preloaded from g.bin
    
    // z_p 切片 + overlap 处理
    std::vector<float> RunDecoderWithOverlap(...);
};
```

配置项：
```cpp
struct OfflineTtsAxeraMeloConfig {
    std::string encoder_model;   // encoder-zh.onnx
    std::string decoder_model;   // decoder-zh.axmodel
    std::string g_bin;           // g-zh.bin
    int dec_len = 128;           // decoder 固定长度
    float noise_scale = 0.667f;
    float length_scale = 1.0f;
    float noise_scale_w = 0.8f;
    float sdp_ratio = 0.2f;
};
```

### 方案 3（兼容现有 Sherpa API）：导出适配版 ONNX

导出一个"桥接"版 model.onnx，接口与 Sherpa 完全相同，但内部只包含编码器部分（decoder 单独 axmodel）：

```python
class BridgeEncoder(torch.nn.Module):
    """
    接口: Sherpa 风格 (x int64[1,L], tones int64[1,L], sid int64[1], ...)
    内部: enc_forward (phone int32[L], tone int32[L], language int32[L], g, ...)
    输出: 适配 Sherpa Run() 的形式
    """
    def forward(self, x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w):
        phone = x[0].int()   # [1,L] → [L]，并 cast int32
        tone = tones[0].int()
        language = torch.zeros_like(phone) + self.lang_id
        g = self.emb_g(sid[0].int()).unsqueeze(-1).unsqueeze(0)  # [1,256,1]
        z_p, pronoun_lens, audio_len = self.model.enc_forward(
            phone, tone, language, g, 
            noise_scale[0], noise_scale_w[0], length_scale[0], 0.2
        )
        return z_p, pronoun_lens, audio_len, g
```

**输出**：z_p + pronoun_lens + audio_len + g，供 C++ 端再调用 axmodel decoder。  
这需要对 Sherpa C++ 层做有限修改，仅在识别到 AXERA 模型时调用 decoder axmodel。

---

## 7. 推荐兼容路径

```
短期（不改 C++）:
  使用 AXERA 提供的 melotts.cpp 直接部署 → 无需 Sherpa C++

中期（扩展 Sherpa）:
  1. 在 sherpa-onnx 增加 AxeraTtsMeloModel 类
  2. 复用 Sherpa 的文本前处理（TextFrontend）
  3. 新增 OfflineTtsConfig.axera 配置项

长期（统一接口）:
  设计 HybridTtsModel 统一接口
  自动检测是否有 axmodel 文件，选择 NPU 或 CPU 后端
```

---

## 8. 兼容性检查清单

在将 AXERA 模型与 Sherpa C++ 集成之前，需要检查的项目：

- [ ] **INT32 → INT64 适配**：导出时或运行时转换数据类型
- [ ] **1D → 2D 张量**：phone/tone/lang 需要 unsqueeze(0) 添加 batch dim
- [ ] **sid → g 路径**：确保 g 在 encoder 调用前可用
- [ ] **8 输入 → 7 输入**：sdp_ratio 参数需要在适配层中处理
- [ ] **语言 ID 构建**：language 张量不在 Sherpa 接口中，需要基于模型语言类型内部生成
- [ ] **输入名称映射**：phone/tone/language vs x/tones（名称不同）
- [ ] **解码器切片循环**：z_p 的切片和拼接逻辑需要在 C++ 中实现
- [ ] **Overlap 处理**：word2ph 信息需要从编码器传递到切片逻辑
- [ ] **音频截断**：audio_len 需要用于截断最后一块的 padding 部分
- [ ] **g.bin 加载**：在初始化时加载 g.bin 文件
