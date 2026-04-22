# MeloTTS 分析摘要

## 快速参考

### 两种导出方式的核心区别

| 方面 | 单一模型 (export-onnx.py) | 子图分割 (export_subgraphs_and_calib.py) |
|------|----------------------|-------------------------------------|
| **文件数量** | 1个 (model.onnx) | 3个 (subgraph_a.onnx, subgraph_b.onnx + config) |
| **推理流程** | 一次执行 | 分阶段 A→B |
| **适用场景** | PC/服务器 | AXERA等边缘设备 |
| **量化支持** | 困难 | 完整（含校准数据） |
| **部署复杂度** | 低 | 高 |

### 关键技术细节

#### 子图分割
- **Subgraph A (编码器)**
  - 输入: x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w
  - 输出: /Mul_10_output_0, /Unsqueeze_6_output_0

- **Subgraph B (生成器)**
  - 输入: Subgraph A的两个输出
  - 输出: 最终音频 (y)

#### 校准数据
- 12个样本 (可配置)
- 10个长度变量: [32, 48, 64, 80, 96, 128, 160, 192, 224, 256]
- 每个子图独立量化配置

### 模型输入/输出规范

#### 输入 (7个)
| 名称 | 类型 | 形状 | 范围 |
|------|------|------|------|
| x | int64 | [1, L] | 0 ~ vocab_size |
| x_lengths | int64 | [1] | - |
| tones | int64 | [1, L] | 0-5 |
| sid | int64 | [1] | 0 (MeloTTS中固定为模型元数据值) |
| noise_scale | float32 | [1] | 0.0-2.0 (default: 0.667) |
| length_scale | float32 | [1] | 0.5-2.0 (default: 1.0) |
| noise_scale_w | float32 | [1] | 0.0-1.0 (default: 0.8) |

#### 输出
| 名称 | 类型 | 形状 | 备注 |
|------|------|------|------|
| y | float32 | [1, S, T] | 音频采样值 [-1, 1] |

### C++ 推理关键API

```cpp
// 基础推理
Ort::Value Run(Ort::Value x, int64_t sid = 0, float speed = 1.0);

// MeloTTS推理 (带声调)
Ort::Value Run(Ort::Value x, Ort::Value tones, int64_t sid = 0, 
               float speed = 1.0) const;

// 关键特性
- 自动将单说话人模型的sid设置为meta_data_.speaker_id
- speed参数转换: length_scale = 1.0 / speed
- 强制batch_size = 1
```

### 文本预处理管道

```
中文文本 → 
  pinyin_to_symbol → 
    音素序列 + 声调 → 
      ID映射 → 
        [x_ids, tones] → 模型输入

英文文本 → 
  CMUDict词典 → 
    音素序列 + 声调(固定) → 
      ID映射 → 
        [x_ids, tones] → 模型输入
```

### 输出后处理

```
模型输出 [1, S, T] 
  → 展平为1D 
  → 限制[-1, 1]范围 
  → 转换int16 
  → 乘以32767 
  → 保存WAV (44100 Hz)
```

## 部署建议

### 选择标准

1. **使用单一模型** 当：
   - PC/服务器部署
   - 需要高推理速度
   - 存储空间充足

2. **使用子图模型** 当：
   - AXERA等专用芯片部署
   - 需要INT8量化
   - 存储空间受限
   - 需要分阶段优化

### 性能调优参数

| 参数 | 默认值 | 调整范围 | 效果 |
|------|------|--------|------|
| noise_scale | 0.667 | 0.5-1.0 | 音质稳定性 |
| length_scale | 1.0 | 0.5-1.5 | 音频长度/速度 |
| noise_scale_w | 0.8 | 0.6-1.0 | 自然度 |
| speed | 1.0 | 0.5-2.0 | 播放速度 |

## 文件位置

- **完整分析**: MeloTTS_Model_Export_Analysis.md
- **导出脚本**: ../export-onnx.py, ../export_subgraphs_and_calib.py
- **推理代码**: ../../csrc/offline-tts-vits-model.cc
- **模型配置**: zh_en/, en/, jp/, kr/, es/, fr/ 目录

## 注意事项

⚠️ **版本要求**: MeloTTS模型版本必须 >= 2
⚠️ **批次大小**: C++推理仅支持batch_size = 1
⚠️ **采样率**: 固定44100 Hz，从模型元数据读取
⚠️ **子图连接**: Subgraph A/B的张量命名必须完全匹配

---

更多详细信息参考: [完整分析文档](./MeloTTS_Model_Export_Analysis.md)
