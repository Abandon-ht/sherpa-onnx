# 04 — MeloTTS 二阶段多桶 NPU 部署设计（sherpa-onnx 接口兼容版）

本文档描述满足以下三项约束的最终设计方案：

> 1. **尽量多在 NPU 上运行**（减少 CPU 计算）
> 2. **尽量少的拼接逻辑**（去除 pronoun_lens / audio_len 预测、overlap 拼接）
> 3. **尽量复用 sherpa-onnx 的前后处理代码**（C++ 侧无需改动文本前端）

---

## 一、为什么从三阶段退到二阶段？

`export_3stage_npu.py`（已存档，不再推荐）将模型切分为：

```
encoder(CPU) → flow(NPU) → generator(NPU)
```

运行时仍需对 `z_p` 进行逐 `dec_len` 切片并拼接，且依赖 `pronoun_lens` / `audio_len` 来控制边界，违反约束 2。

**二阶段方案**将 `flow + generator` 合并为单一 decoder NPU 模型，搭配**桶策略（bucketing）**处理可变长度：

```
encoder(CPU) → decoder(NPU, flow+gen 合并, 固定桶形状)
```

运行时只需：选桶 + 一次右侧 pad + 一次尾部 trim。

---

## 二、encoder：sherpa-onnx 接口兼容性

### 2.1 接口设计演进

旧 AXERA 风格（`convert.py` 沿用）：

| 输入名        | 类型           | 说明                  |
|-------------|--------------|---------------------|
| phone       | int32[L]     | 1D，无 batch dim       |
| tone        | int32[L]     | 1D                   |
| language    | int32[L]     | 1D                   |
| g           | float32[1,256,1] | 外部预计算           |
| noise_scale | float32[1]   |                      |
| noise_scale_w | float32[1] |                      |
| length_scale | float32[1]  |                      |
| sdp_ratio   | float32[1]   |                      |

sherpa-onnx 风格（`offline-tts-vits-model.cc Run()` 构建）：

| 输入名          | 类型           | 说明                  |
|---------------|--------------|---------------------|
| x             | int64[1, L]  | batch=1，int64        |
| x_lengths     | int64[1]     | 序列长度               |
| tones         | int64[1, L]  | batch=1，int64        |
| sid           | int64[1]     | 说话人 ID             |
| noise_scale   | float32[1]   |                      |
| length_scale  | float32[1]   | = 1.0/speed（sherpa 语义）|
| noise_scale_w | float32[1]   |                      |

### 2.2 兼容性结论

不兼容是**设计选择**，不是结构限制。`SherpaCompatEncoder` wrapper 在 ONNX 图内部处理：

```python
phone    = x[0].to(torch.int32)         # int64[1,L] → int32[L]
tone     = tones[0].to(torch.int32)     # int64[1,L] → int32[L]
language = torch.zeros_like(phone) + lang_id   # int32[L]，per-language 常量
g        = self.model.emb_g(sid).unsqueeze(-1)  # int64[1] → float32[1,256,1]
# sdp_ratio: register_buffer 固定 0.0
```

所有转换对 C++ 推理层完全透明。

### 2.3 为什么 `emb_g(sid)` 可以在 ONNX 图内？

- `nn.Embedding` 接受 int64 LongTensor，ONNX `Gather` 算子支持 int64 indices。
- 旧方案用外部 `g.bin` 文件是为了避免将 speaker embedding lookup 放入图，本方案将其折叠进图，
  去掉了 C++ 侧加载 `g.bin` 并手动构造 `g` 张量的逻辑。

---

## 三、decoder：固定形状多桶策略

### 3.1 为什么不用动态轴？

AXERA NPU（AX620E / AX650N）经 `pulsar2 build` 编译后要求完全静态形状。
动态形状 ONNX 只能在 ONNX Runtime（CPU）上运行，无法直接转换为 axmodel。

### 3.2 桶策略

| 桶编号 | z_p 形状         | audio 输出形状         | 最大可处理 z_len |
|:----:|:---------------:|:--------------------:|:------------:|
| B=256  | [1, 192, 256]  | [1, 1, 131072]  | 256          |
| B=512  | [1, 192, 512]  | [1, 1, 262144]  | 512          |
| B=1024 | [1, 192, 1024] | [1, 1, 524288]  | 1024         |
| B=1536 | [1, 192, 1536] | [1, 1, 786432]  | 1536         |

上采样因子 = 512（HiFiGAN: 8×8×4×2）。

选桶规则：`B = min(b for b in BUCKETS if b >= z_len)`

### 3.3 一次 pad + 一次 trim

```
z_p_padded = zero_pad_right(z_p, dim=2, target=B)
audio_full  = decoder_bB.Run(z_p_padded, g)          # shape [1,1,B*512]
audio_out   = audio_full[:, :, :z_len * 512]          # trim
```

- 无 overlap，无 pronoun_lens，无 audio_len 依赖
- z_p 右侧补零（静音），被 trim 掉，不影响有效音频质量

---

## 四、完整运行时流程（5 步）

```
┌─── C++ 侧（完全复用 sherpa 前端）──────────────────────────────────────┐
│ 1. tokens, tones = MeloTtsLexicon::ConvertTextToTokenIds(text)         │
│    x = AddBlank(tokens)        // int64[1,L]                           │
│    tones = AddBlank(tones)     // int64[1,L]                           │
│    length_scale = 1.0 / speed                                          │
│                                                                         │
│ 2. [z_p, g] = encoder_sess.Run({x, x_lengths, tones, sid, scales...}) │
│    z_len = z_p.shape[2]                                                 │
│                                                                         │
│ 3. B = select_bucket(z_len, buckets)                                   │
│                                                                         │
│ 4. z_p_padded = zero_pad_right(z_p, B)   // 仅右侧 pad dim=2          │
│                                                                         │
│ 5. audio = decoder_bB_sess.Run({z_p_padded, g})                        │
│    audio = audio[:, :, :z_len * 512]     // trim                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 五、与 sherpa C++ 代码的集成点

### 5.1 复用不变的部分

| 组件 | 文件 | 是否需要改动 |
|------|------|:-----------:|
| 文本前端（中文/英文 lexicon） | `offline-tts-vits-impl.h` `MeloTtsLexicon` | **否** |
| `AddBlank()` | `offline-tts-vits-impl.h` | **否** |
| `speed → length_scale` 转换 | `offline-tts-vits-model.cc:88` | **否** |
| 输出 audio tensor 提取 | `offline-tts-vits-impl.h Process()` | **否** |

### 5.2 需要替换的部分

原 `offline-tts-vits-model.cc Run()` 中：

```cpp
// 旧: 单个 sess_ 运行全图
auto out = sess_->Run(input_names_, inputs, output_names_);
```

新（伪代码）：

```cpp
// 新: encoder(CPU) + bucket decoder(NPU)
auto [z_p, g] = encoder_sess_->Run(x, x_lengths, tones, sid,
                                    noise_scale, length_scale, noise_scale_w);
int64_t z_len = z_p.shape[2];
int64_t B     = SelectBucket(z_len, buckets_);
auto z_p_pad  = PadRight(z_p, B);        // 一次 pad
auto audio    = decoder_sess_[B]->Run(z_p_pad, g);
audio         = audio.Slice(2, 0, z_len * 512);  // 一次 trim
```

---

## 六、与旧方案对比

| 维度 | 旧三阶段 (export_3stage_npu.py) | 本方案（export_2stage_npu.py） |
|------|:-----------------------------:|:----------------------------:|
| Encoder 输入接口 | AXERA 风格 int32[L]，与 sherpa 不兼容 | sherpa 风格 int64[1,L]，完全兼容 |
| g.bin 外部文件   | 需要（C++ 侧加载并构造 g 张量） | **不需要**（emb_g 在图内部） |
| NPU 模型数量     | 2 个 NPU 模型（flow + gen 分开） | **1 个** NPU 模型/桶（flow+gen 合并） |
| 运行时循环       | z_p 逐 dec_len 切片（有循环） | **无循环**，一次推理 |
| pronoun_lens 依赖 | 有（需要裁剪和填充计算） | **无** |
| audio 拼接逻辑   | overlap 窗口拼接 | 仅 trim，无拼接 |
| C++ 前端改动     | 需适配 int32[L] 输入构建 | **零改动**（直接复用 sherpa 7-input） |
| NPU 利用率       | flow + gen 各一个 NPU 推理/块 | **一个 NPU 推理/句** |

---

## 七、导出命令与输出文件

```bash
conda activate cosyvoice
cd /home/m5stack/Workspace/AXERA/melotts.axera/model_convert

python /home/m5stack/Workspace/kaldi/sherpa-onnx/scripts/melo-tts/axera/export_2stage_npu.py \
    -l ZH \
    --buckets 256,512,1024,1536 \
    --out-dir ./output_2stage_bucket
```

输出：

```
output_2stage_bucket/
├── encoder-sherpa-zh.onnx       # CPU 运行，sherpa 接口，动态 L/T
├── decoder-b256-zh.onnx         # NPU，固定形状 [1,192,256]→[1,1,131072]
├── decoder-b512-zh.onnx         # NPU，固定形状 [1,192,512]→[1,1,262144]
├── decoder-b1024-zh.onnx        # NPU，固定形状 [1,192,1024]→[1,1,524288]
├── decoder-b1536-zh.onnx        # NPU，固定形状 [1,192,1536]→[1,1,786432]
└── runtime_spec.json            # 桶列表、上采样因子、C++ 加载规格
```

可选：生成量化校准数据（使用真实文本的 encoder 输出 z_p 片段，而非随机数）：

```bash
python export_2stage_npu.py -l ZH --buckets 256,512,1024,1536 --calib
```

---

## 八、NPU 转换参考命令

```bash
# 以 B=512 桶为例（其他桶类同）
pulsar2 build \
    --input  output_2stage_bucket/decoder-b512-zh.onnx \
    --config quant_config.json \
    --output decoder-b512-zh.axmodel

# encoder 仍使用 ONNX Runtime (CPU)，无需转换 axmodel
```

---

*相关文件：*
- [export_2stage_npu.py](export_2stage_npu.py) — 本设计对应的导出脚本
- [03_new_split_design.md](03_new_split_design.md) — 三阶段设计（已存档）
- [02_cpp_compatibility.md](02_cpp_compatibility.md) — C++ 接口兼容性分析
