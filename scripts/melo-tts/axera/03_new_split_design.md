# 三阶段 NPU 切分方案设计文档

> 在 AXERA 两阶段方案基础上，通过将 decoder 进一步切分为 **flow** 和 **generator** 两个独立 NPU 子模型，最大化 NPU 利用率并支持分阶段优化。

---

## 1. 背景与动机

### 1.1 现有两阶段方案的局限

AXERA `convert.py` 的两阶段方案（encoder + decoder）已经将最重的 decoder 部分（flow + HiFiGAN）放到 NPU 上运行。但 decoder 内部的两个模块有明显的结构差异：

| 子模块 | 类型 | 特点 | 最优量化策略 |
|--------|------|------|-------------|
| Flow (`flow`) | Transformer Coupling Block | 注意力为主，参数精度敏感 | INT8 可能有精度损失，倾向 INT16 |
| Generator (`dec`) | HiFiGAN + ResBlocks | 卷积为主，参数量大 | INT8 兼容性好，压缩收益高 |

将两者合并在一个 axmodel 中，只能使用统一的量化策略。分离后可以对各自优化。

### 1.2 三阶段方案的优势

```
两阶段方案:
  encoder(CPU) → decoder.axmodel(NPU, flow+gen 合并)

三阶段方案:
  encoder(CPU) → flow.axmodel(NPU) → generator.axmodel(NPU)
```

**优势：**
1. **独立量化**：flow 可用 INT16，generator 可用 INT8
2. **精度调试**：可以分别验证 flow 和 generator 的精度
3. **灵活部署**：可以按需重编 flow 或 generator，不影响另一个
4. **性能分析**：可以分别统计 flow 和 generator 的 NPU 延迟
5. **双核并行**（AX650N 等支持双 NPU 核的芯片）：对不同帧并行推理

---

## 2. 模型架构分析

### 2.1 SynthesizerTrn 中的 flow 和 dec

```
SynthesizerTrn
├── enc_p (TextEncoder)          ─┐
├── sdp (StochasticDurationPredictor)  ├─ enc_forward()
├── dp  (DurationPredictor)      ─┘
│
├── flow (TransformerCouplingBlock)  ─── flow_forward()
│
└── dec  (Generator / HiFiGAN)       ─── dec_forward()
```

**两阶段 flow_dec_forward（AXERA 原始）：**
```python
def flow_dec_forward(self, z_p, g):
    y_mask = torch.ones(1, z_p.size(-1))
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    # HiFiGAN
    audio = self.dec(z * y_mask, g=g)
    return audio
```

**三阶段分离版：**
```python
def flow_forward(self, z_p, g):
    y_mask = torch.ones(1, 1, z_p.size(-1))
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    return z * y_mask

def dec_forward(self, z, g):
    return self.dec(z, g=g)
```

### 2.2 Flow 网络（TransformerCouplingBlock）

Flow 网络执行归一化流的逆变换（reverse=True）：

```
z_p [1, 192, 128]
    │
    ▼
TransformerCouplingBlock
    ├── CouplingLayer_0  (WN + Transformer)
    ├── Flip
    ├── CouplingLayer_1
    ├── Flip
    ├── CouplingLayer_2
    └── Flip
    │
    ▼
z [1, 192, 128]
```

特点：
- 输入输出同形状 `[1, 192, T]`
- 使用 y_mask 掩码（固定形状时全为 1）
- 参数量 ~30MB，推理延迟较高

### 2.3 HiFiGAN Generator（dec）

```
z [1, 192, 128]  + g [1, 256, 1]
    │
    ▼
Linear Conv + Upsample stack
    ├── ConvTranspose1d (stride=8) → [1, 512, 1024]
    ├── ResBlocks × n
    ├── ConvTranspose1d (stride=8) → [1, 256, 8192]
    ├── ResBlocks × n
    ├── ConvTranspose1d (stride=4) → [1, 128, 32768]
    ├── ResBlocks × n
    ├── ConvTranspose1d (stride=2) → [1, 64, 65536]
    ├── ResBlocks × n
    └── Conv1d → Tanh
    │
    ▼
audio [1, 1, 65536]
```

特点：
- 纯卷积结构，对 INT8 量化友好
- 上采样 512x（8×8×4×2），参数量 ~50MB
- 推理延迟与序列长度成线性关系

---

## 3. 三阶段模型接口规范

### 3.1 Stage 1: encoder.onnx（CPU，动态 shape）

与 AXERA `convert.py` 中的 encoder 完全兼容。

```
输入 (8 个):
  phone        int32[phone_len]    - 音素 ID
  tone         int32[phone_len]    - 声调
  language     int32[phone_len]    - 语言 ID
  g            float32[1, 256, 1]  - 说话人嵌入
  noise_scale  float32[1]
  noise_scale_w float32[1]
  length_scale float32[1]
  sdp_ratio    float32[1]

输出 (3 个):
  z_p          float32[1, 192, T]  - 潜在表示（T 动态）
  pronoun_lens int32[phone_len]    - 发音帧数（用于 overlap 切片）
  audio_len    int32[1]            - 总音频采样数（用于截断 padding）

动态轴:
  phone/tone/language: dim 0 = "phone_len"
  z_p: dim 2 = "audio_len"
```

### 3.2 Stage 2: flow.onnx → flow.axmodel（NPU，固定 shape）

```
输入 (2 个):
  z_p   float32[1, 192, 128]    - 固定形状！（dec_len=128）
  g     float32[1, 256, 1]      - 说话人嵌入

输出 (1 个):
  z     float32[1, 192, 128]    - 固定形状！

注:
  - dec_len 必须在导出时固定（推荐 128）
  - y_mask 在 FlowWrapper 内部生成（全 1 张量，不作为输入）
  - 导出时 dynamic_axes 为空
```

### 3.3 Stage 3: generator.onnx → generator.axmodel（NPU，固定 shape）

```
输入 (2 个):
  z     float32[1, 192, 128]       - 固定形状！（来自 flow 输出）
  g     float32[1, 256, 1]         - 说话人嵌入

输出 (1 个):
  audio float32[1, 1, 65536]       - 固定形状！(128 × 512)

注:
  - 上采样倍率 = 8×8×4×2 = 512（由模型 ups 层参数决定）
  - 输出长度 = dec_len × upsample_factor
```

### 3.4 g.bin（预计算说话人嵌入）

```
格式: float32 二进制，共 256 个浮点数
大小: 256 × 4 = 1024 字节（1 KB）
生成: model.emb_g(torch.IntTensor([speaker_id])).numpy().astype(np.float32).tofile(path)
用途: 同时提供给 encoder、flow、generator 三个模型
```

---

## 4. 推理流程设计

### 4.1 三阶段 C++ 推理伪代码

```cpp
// 初始化
OnnxWrapper encoder;           encoder.Load("encoder-zh.onnx");
AxeraEngine flow_model;        flow_model.Load("flow-zh-d128.axmodel");
AxeraEngine gen_model;         gen_model.Load("generator-zh-d128.axmodel");

// 加载说话人嵌入
float g[1*256*1];
FILE* fp = fopen("g-zh.bin", "rb");
fread(g, sizeof(float), 256, fp);

// 对每个句子推理
void infer_sentence(phones, tones, langids) {

    // Stage 1: encoder (CPU, 动态)
    encoder.Run(phones, tones, langids, g, noise_scale, noise_scale_w, length_scale, sdp_ratio)
    → z_p[1, 192, T], pronoun_lens, audio_len

    // 计算切片策略（同 AXERA 原始方案）
    calc_word2pronoun(word2ph, pronoun_lens) → word2pronoun
    generate_slices(word2pronoun, dec_len=128) → pn_slices, zp_slices

    // Stage 2+3: flow → generator (NPU, 固定形状)
    for each zp_slice in zp_slices:
        // 裁切或 zero-pad 到 [1, 192, 128]
        z_p_chunk = slice_and_pad(z_p, zp_slice, dec_len)

        // Stage 2: flow
        flow_model.SetInput(z_p_chunk, 0)   // z_p [1,192,128]
        flow_model.SetInput(g, 1)            // g [1,256,1]
        flow_model.Run()
        flow_model.GetOutput(z_chunk, 0)     // z [1,192,128]

        // Stage 3: generator
        gen_model.SetInput(z_chunk, 0)       // z [1,192,128]
        gen_model.SetInput(g, 1)             // g [1,256,1]
        gen_model.Run()
        gen_model.GetOutput(audio_chunk, 0)  // audio [1,1,65536]

        // overlap 裁剪拼接（同原始方案）
        wavlist.append(trim(audio_chunk, pn_slice))
    
    // 截断最后一块 padding
    final_audio = wavlist.concat()[:audio_len]
}
```

### 4.2 与两阶段方案的 C++ 改动对比

```
两阶段 C++:
  decoder_model.SetInput(z_p_chunk, 0)
  decoder_model.SetInput(g, 1)
  decoder_model.Run()
  decoder_model.GetOutput(audio_chunk, 0)

三阶段 C++ (新增 flow 步骤):
  flow_model.SetInput(z_p_chunk, 0)
  flow_model.SetInput(g, 1)
  flow_model.Run()
  flow_model.GetOutput(z_chunk, 0)          // ← 新增

  gen_model.SetInput(z_chunk, 0)            // ← 使用 z_chunk 而非 z_p_chunk
  gen_model.SetInput(g, 1)
  gen_model.Run()
  gen_model.GetOutput(audio_chunk, 0)
```

**改动量：极小**，只需在现有 decoder 调用处插入一个 flow 调用。

---

## 5. 导出脚本使用说明

### 5.1 脚本位置

```
scripts/melo-tts/axera/export_3stage_npu.py
```

### 5.2 使用方法

```bash
# 激活 cosyvoice conda 环境
conda activate cosyvoice

# 进入 AXERA melotts 的 model_convert 目录
cd /home/m5stack/Workspace/AXERA/melotts.axera/model_convert

# 基本导出（中文，dec_len=128）
python /home/m5stack/Workspace/kaldi/sherpa-onnx/scripts/melo-tts/axera/export_3stage_npu.py \
    -l ZH -d 128 \
    --config config.json \
    --ckpt checkpoint.pth \
    --out-dir ./output_3stage

# 同时生成校准数据
python export_3stage_npu.py -l ZH -d 128 --calib --out-dir ./output_3stage

# 同时导出原始兼容版 decoder（验证等价性）
python export_3stage_npu.py -l ZH -d 128 --export-combined-decoder --out-dir ./output_3stage
```

### 5.3 输出文件说明

```
output_3stage/
├── encoder-zh.onnx              # Stage 1: CPU 编码器（与 AXERA convert.py 兼容）
├── flow-zh-d128.onnx            # Stage 2: flow 网络（待转 axmodel）
├── generator-zh-d128.onnx       # Stage 3: HiFiGAN 生成器（待转 axmodel）
├── g-zh.bin                     # 说话人嵌入（float32 × 256）
│
├── decoder-zh.onnx              # [可选] flow+gen 合并版（AXERA 兼容）
│
└── calib_3stage/                # [可选] 校准数据
    ├── encoder_calib/
    │   ├── phone/               # encoder 输入样本
    │   └── g/
    ├── flow_calib/
    │   ├── z_p/                 # flow 输入样本（来自 encoder 真实输出）
    │   ├── z_p.tar.gz
    │   ├── g/
    │   └── g.tar.gz
    └── generator_calib/
        ├── z/                   # generator 输入样本（来自 flow 真实输出）
        ├── z.tar.gz
        ├── g/
        └── g.tar.gz
```

### 5.4 转换为 axmodel

```bash
# 使用 AXERA Pulsar2 分别转换（可以使用不同量化策略）
pulsar2 build \
    --input flow-zh-d128.onnx \
    --config flow_quant_config.json \     # 可使用 INT16 量化
    --output flow-zh-d128.axmodel \
    --target AX650

pulsar2 build \
    --input generator-zh-d128.onnx \
    --config gen_quant_config.json \      # 可使用 INT8 量化
    --output generator-zh-d128.axmodel \
    --target AX650
```

---

## 6. 精度等价性验证

三阶段方案应与两阶段方案结果完全一致（在浮点精度范围内）：

```python
import torch
import numpy as np

# 加载两种方案
decoder_combined = load_onnx("decoder-zh.onnx")   # flow + gen 合并
flow_model = load_onnx("flow-zh-d128.onnx")
gen_model  = load_onnx("generator-zh-d128.onnx")

# 测试输入
z_p = np.random.randn(1, 192, 128).astype(np.float32)
g   = np.random.randn(1, 256, 1).astype(np.float32)

# 两阶段
audio_combined = decoder_combined.run(["audio"], {"z_p": z_p, "g": g})[0]

# 三阶段
z = flow_model.run(["z"], {"z_p": z_p, "g": g})[0]
audio_split = gen_model.run(["audio"], {"z": z, "g": g})[0]

# 验证等价性
max_diff = np.max(np.abs(audio_combined - audio_split))
print(f"最大绝对误差: {max_diff:.2e}")  # 预期 < 1e-5
assert max_diff < 1e-4, "精度验证失败！"
print("三阶段与两阶段输出等价 ✓")
```

---

## 7. 方案对比总结

| 维度 | 单阶段（Sherpa） | 两阶段（AXERA） | 三阶段（本方案） |
|------|-----------------|----------------|----------------|
| CPU 计算 | 全部（encoder+flow+gen） | encoder | encoder |
| NPU 计算 | 无 | flow+gen 合并 | flow + gen 分离 |
| 量化策略 | 统一或不量化 | 统一 INT8 | flow/gen 可独立设置 |
| 模型文件数 | 1 个 onnx | encoder + decoder | encoder + flow + gen |
| 校准数据 | 不需要 | decoder 1份 | flow 1份 + gen 1份 |
| C++ 复杂度 | 低 | 中 | 中（仅多一次 NPU 调用） |
| 调试便利性 | 低 | 中 | 高（可单独测试 flow/gen） |
| 精度控制 | 无法分开 | 无法分开 | **可以分开** |
| 适合场景 | 通用 PC 部署 | AXERA NPU 初次部署 | AXERA NPU 优化部署 |

---

## 8. 已知约束

1. **dec_len 固定**：flow 和 generator 都使用 `dec_len=128` 固定形状。若需要其他长度，需重新导出。
2. **y_mask 固化**：FlowWrapper 内部生成全 1 的 y_mask（假设 z_p 无 padding），适合已在 Python 端处理好的切片。
3. **g 重复传入**：g 向量需同时传给 encoder、flow、generator 三个模型（这与原始两阶段一致）。
4. **flow 的 Transformer 注意力**：部分 AXERA 芯片对 softmax + attention 的量化支持有限，flow 可能需要保持 FP16 或 INT16。
5. **校准数据顺序**：flow 的校准数据应来自 encoder 的真实输出（而非随机数），以保证校准质量。
