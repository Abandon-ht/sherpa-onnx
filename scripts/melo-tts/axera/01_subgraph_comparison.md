# 两种切子图方式深度对比

> 本文档对比 AXERA `convert.py` 方案与 Sherpa-ONNX `export_subgraphs_and_calib.py` 方案的详细差异。

---

## 1. 架构概览

### 方案 A：AXERA convert.py（Encoder + Decoder 分离）

```
文本输入
  │
  ▼ (Python前处理: lexicon.convert + intersperse)
phone[phone_len] / tone[phone_len] / language[phone_len]
  │
  ▼
encoder.onnx  ← g.bin (预计算的说话人嵌入) ← noise/scale 参数
  │
  ├─ z_p [1, 192, T_audio]     (潜在表示，动态长度)
  ├─ pronoun_lens [phone_len]  (每个音素的发音时长)
  └─ audio_len [1]             (总音频采样数)
  │
  ▼ (z_p 按 dec_len=128 切片，必要时 zero-pad)
decoder.axmodel (NPU)  ← g.bin
  │
  └─ audio_chunk [1, 1, 65536]  (固定形状输出)
  │
  ▼ (多块拼接 + overlap 裁剪)
完整音频波形
```

### 方案 B：Sherpa-ONNX export_subgraphs_and_calib.py（Subgraph A/B 分离）

```
文本输入
  │
  ▼ (Python前处理: text → phone_ids, tones)
x[1,L] / x_lengths[1] / tones[1,L] / sid[1] / scale 参数
  │
  ▼
subgraph_a.onnx (CPU)   (从完整模型ONNX节点切出)
  │
  ├─ /Mul_10_output_0   [1, 192, L]   (中间潜在表示)
  └─ /Unsqueeze_6_output_0 [1, 256, 1] (说话人嵌入)
  │
  ▼
subgraph_b.onnx (CPU)
  │
  └─ y [1, S, T]  (完整音频，动态形状)
```

---

## 2. 切分策略对比

| 维度 | AXERA convert.py | Sherpa-ONNX subgraph |
|------|-----------------|---------------------|
| **切分依据** | 模型语义边界（编码器/解码器） | ONNX 计算图内部节点名称 |
| **切分粒度** | 粗粒度（高层API） | 细粒度（内部张量节点） |
| **自定义程度** | 高（修改模型代码实现） | 低（依赖特定节点名） |
| **可移植性** | 好（跨模型版本稳定） | 差（节点名随训练改变） |
| **目标后端** | encoder: CPU ONNX / decoder: AXERA NPU | 纯 CPU ONNX（量化标定用） |

---

## 3. 模型接口详细对比

### 3.1 Encoder 输入接口

| 特性 | AXERA encoder | Sherpa subgraph_a | 差异说明 |
|------|--------------|-------------------|--------|
| **音素 ID** | `phone` int32[phone_len] | `x` int64[1,L] | 类型: int32 vs int64；维度: 1D vs 2D |
| **声调** | `tone` int32[phone_len] | `tones` int64[1,L] | 同上 |
| **语言 ID** | `language` int32[phone_len] | *(内部计算)* | AXERA需外部传入；sherpa在wrapper内生成 |
| **说话人嵌入** | `g` float32[1,256,1] | *(通过sid在模型内部计算)* | AXERA预计算g.bin；sherpa传sid给模型 |
| **噪声缩放** | `noise_scale` float32[1] | `noise_scale` float32[1] | 相同 |
| **长度缩放** | `length_scale` float32[1] | `length_scale` float32[1] | 相同 |
| **噪声_w** | `noise_scale_w` float32[1] | `noise_scale_w` float32[1] | 相同 |
| **SDP比率** | `sdp_ratio` float32[1] | *(无，固定=0)* | AXERA暴露此参数；sherpa固化 |
| **批次长度** | *(无，phone无batch dim)* | `x_lengths` int64[1] | AXERA在模型内隐式处理 |
| **输入总数** | **8** | **7** | 差 1 |

### 3.2 Encoder 输出接口

| 输出名 | AXERA encoder | Sherpa subgraph_a | 差异 |
|-------|--------------|-------------------|------|
| **潜在表示** | `z_p` float32[B,192,T] | `/Mul_10_output_0` float32[1,192,L] | 语义相同，命名不同 |
| **说话人嵌入** | *(无，g在输入中)* | `/Unsqueeze_6_output_0` float32[1,256,1] | Sherpa输出g；AXERA输入g |
| **发音时长** | `pronoun_lens` int32[phone_len] | *(无)* | AXERA额外输出，用于overlap裁剪 |
| **音频长度** | `audio_len` int32[1] | *(无)* | AXERA额外输出，用于裁剪最后一块 |

### 3.3 Decoder 输入/输出接口

| 特性 | AXERA decoder | Sherpa subgraph_b |
|------|--------------|-------------------|
| **输入 z_p** | `z_p` float32[1,192,**128**] — **固定形状** | `/Mul_10_output_0` float32[1,192,L] — **动态形状** |
| **输入 g** | `g` float32[1,256,1] | *(已包含在上游输出中)* |
| **输出音频** | `audio` float32[1,1,**65536**] — **固定形状** | `y` float32[1,S,T] — **动态形状** |

---

## 4. 推理流程对比

### AXERA 推理流程（C++）

```
1. 加载 encoder.onnx (ONNX Runtime, CPU)
2. 加载 decoder.axmodel (AXERA Engine, NPU)
3. 从 g.bin 读取说话人嵌入 g[256]
4. 对文本分句处理 (split_sentence)
5. 对每个句子:
   a. Lexicon.convert: 文本 → phones, tones, word2ph
   b. intersperse: 插入 blank token
   c. 构建 langids (全 3 for ZH)
   d. encoder.Run(phones, tones, langids, g, noise_scale, noise_scale_w, length_scale, sdp_ratio)
      → z_p, pronoun_lens, audio_len
   e. calc_word2pronoun(word2ph, pronoun_lens) → 每个词的发音帧数
   f. generate_slices(word2pronoun, dec_len) → 有 overlap 的切片策略
   g. 对每个 zp_slice:
      - 裁切/填充 z_p 到 [1, 192, 128]
      - decoder.Run(zp_slice, g) → audio_chunk[1, 1, 65536]
      - 按 overlap 策略裁剪音频头尾
      - 追加到 wavlist
6. 保存为 WAV 文件
```

### Sherpa-ONNX 推理流程（C++）

```
1. 加载 model.onnx (ONNX Runtime, CPU)
2. 文本前处理: 音素ID + 声调
3. model.Run(x, tones, sid, speed)
   内部:
   a. 构建输入: x[1,L], x_lengths[1], tones[1,L], sid[1], 3个scale
   b. sess_->Run(...)  →  y[1,S,T]
4. 返回音频张量
```

---

## 5. 关键技术差异

### 5.1 说话人嵌入 (g) 的处理方式

**AXERA 方案：**
```python
# 训练时预计算并保存
g = model.emb_g(torch.IntTensor([speaker_id])).unsqueeze(-1)  # [1, 256, 1]
g.numpy().astype(np.float32).tofile("g-zh.bin")

# 推理时从文件读取
# C++:  fread(g.data(), sizeof(float), g.size(), fp);
```

优点：encoder 和 decoder 共享同一个 g，且 g 是固定常量，不参与推理计算图

**Sherpa 方案：**
```python
# g 在模型内部由 sid 查表得到
# ModelWrapper.forward():
#   → model.infer(sid=sid, ...)
#   → g = self.emb_g(sid).unsqueeze(-1)  # 在 ONNX 计算图内部
```

优点：标准 ONNX 接口，不需要额外文件

### 5.2 固定形状与动态形状

| 形状特性 | AXERA decoder | Sherpa subgraph_b |
|---------|--------------|-------------------|
| NPU 兼容 | ✅ 是（固定形状是 NPU 必要条件） | ❌ 否 |
| 推理轮次 | ceil(T_audio / 128) 次 | 1 次 |
| 内存效率 | 高（固定小缓冲区） | 低（全长音频一次性分配） |
| 实现复杂度 | 高（需要 z_p 切片 + 音频拼接） | 低（直接运行） |

### 5.3 数据类型差异

| 张量 | AXERA | Sherpa |
|------|-------|--------|
| phone/x | int32 | int64 |
| tone/tones | int32 | int64 |
| language/lang_id | int32 | int64 |

**影响**：Sherpa-ONNX C++ 代码使用 `int64` 构建张量（`Ort::Value::CreateTensor<int64_t>`），与 AXERA encoder 不兼容

### 5.4 音频拼接的 Overlap 策略

AXERA 实现了精巧的基于词边界的 overlap 裁剪：

```cpp
// generate_slices(): 按词为单位生成有 overlap 的切片
// 每个切片可能与前一切片重叠 2 个词
// 拼接时:
//   - 裁掉当前切片的开头（重叠部分）
//   - 裁掉当前切片的结尾（下一切片的重叠部分）
// 效果: 有效消除拼接边界处的音质下降
```

Sherpa 无此处理（单次完整推理不需要）。

---

## 6. 适用场景对比

| 场景 | AXERA convert.py | Sherpa subgraph |
|------|-----------------|----------------|
| AXERA NPU 部署 | ✅ 主要目标 | ❌ 不适合 |
| PC/服务器 CPU 部署 | ⚠️ 可用，但复杂 | ✅ 简单 |
| INT8 量化 | ✅ decoder 专门优化 | ✅ 生成校准数据 |
| 实时流式处理 | ✅ 滑动窗口适合 | ❌ 需要等整句完成 |
| 跨平台移植 | ⚠️ C++ 依赖 AXERA SDK | ✅ 纯 ONNX Runtime |
| 长句处理 | ✅ 分块无内存压力 | ⚠️ 长音频内存大 |

---

## 7. 两种方案的模型文件对比

### AXERA 方案文件结构

```
models/
├── encoder-zh.onnx       # 编码器 (~50MB)，运行在 CPU
├── decoder-zh.axmodel    # 解码器 (~100MB)，运行在 NPU  
├── g-zh_mix_en.bin       # 说话人嵌入 (256 * 4 = 1KB)
├── lexicon.txt           # 词典（音素查表）
└── tokens.txt            # 音素符号表
```

### Sherpa 方案文件结构

```
zh_en/
├── model.onnx            # 完整模型 (~150MB)，CPU 推理
├── tokens.txt            # 音素符号表
└── lexicon.txt           # 词典
（subgraph_a.onnx / subgraph_b.onnx 为临时量化用途）
```

---

## 8. 总结

| 核心区别 | AXERA 方案 | Sherpa 方案 |
|---------|----------|-----------|
| **切分目的** | NPU 加速部署 | 量化校准数据生成 |
| **切分点** | enc_forward / flow_dec_forward（语义边界） | ONNX 计算图内部节点 |
| **形状策略** | decoder 强制固定形状 | 全动态形状 |
| **g 来源** | 外部预计算文件 | 模型内部 emb_g 查表 |
| **数据类型** | int32 | int64 |
| **推理次数** | 多次（decoder 滑窗） | 1 次 |
| **C++ 复杂度** | 高（需处理切片、overlap） | 低（直接 Run） |
| **音质策略** | 有 overlap 消除拼接缺陷 | 无需处理 |
