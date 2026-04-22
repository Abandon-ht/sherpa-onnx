# AXERA MeloTTS 部署指南

## 📋 文档导览

本目录包含MeloTTS模型在AXERA芯片上的导出、优化和部署完整分析。

### 📄 文档说明

1. **ANALYSIS_SUMMARY.md** - ⭐ 快速开始
   - 快速参考表格
   - 核心技术细节概览
   - 部署建议
   - 最佳实践

2. **MeloTTS_Model_Export_Analysis.md** - 📚 详细技术文档
   - 两种导出方式深度对比
   - 完整输入输出规范
   - C++推理代码详解
   - 文本预处理管道
   - 兼容性检查清单

## 🚀 快速开始

### 场景1: 部署到AXERA芯片

```bash
# 1. 导出子图模型和校准数据
cd /home/m5stack/Workspace/kaldi/sherpa-onnx/scripts/melo-tts
python export_subgraphs_and_calib.py \
  --base-dir ./axera \
  --samples 12 \
  --length-buckets "32,48,64,80,96,128,160,192,224,256"

# 2. 检查生成的文件
ls -la ./axera/zh_en/
# 应输出:
# - subgraph_a.onnx         (编码器)
# - subgraph_b.onnx         (生成器)  
# - config_subgraph_a_u16.json  (A量化配置)
# - config_subgraph_b_u16.json  (B量化配置)
# - calib/subgraph_a/*.tar.gz    (A校准数据)
# - calib/subgraph_b/*.tar.gz    (B校准数据)
```

### 场景2: 部署到PC/服务器

```bash
# 1. 导出单一完整模型
cd /home/m5stack/Workspace/kaldi/sherpa-onnx/scripts/melo-tts
python export-onnx.py

# 2. 生成文件
# - model.onnx           (完整模型)
# - tokens.txt           (符号表)
# - lexicon.txt          (词汇表)
```

### 场景3: C++推理集成

```cpp
#include "sherpa-onnx/csrc/offline-tts-vits-model.h"

// 1. 初始化模型
OfflineTtsModelConfig config;
config.vits.model = "path/to/model.onnx";
config.vits.noise_scale = 0.667f;
config.vits.length_scale = 1.0f;
config.vits.noise_scale_w = 0.8f;

OfflineTtsVitsModel model(config);

// 2. 准备输入
std::vector<int64_t> x = {/* 音素ID序列 */};
std::vector<int64_t> tones = {/* 声调序列 */};

// 3. 构建ONNX张量
// ... (参考MeloTTS_Model_Export_Analysis.md 的C++部分)

// 4. 推理
Ort::Value output = model.Run(x_tensor, tones_tensor, 
                              speaker_id, speed);

// 5. 处理输出
float* audio_data = output.GetTensorMutableData<float>();
// ... (转换为WAV格式)
```

## 📊 核心指标速查

### 模型规格
- **类型**: MeloTTS VITS
- **版本**: 2+
- **说话人**: 单说话人 (sid固定)
- **语言**: 中文、英文、日文、韩文、法文、西班牙文
- **采样率**: 44100 Hz
- **ONNX Opset**: 18

### 输入格式
```
x (音素ID)
  shape: [1, L]
  dtype: int64
  range: 0 ~ vocab_size

tones (声调)
  shape: [1, L] 
  dtype: int64
  range: 0-5
  note: 英文/日文通常为0

noise_scale
  shape: [1]
  dtype: float32
  default: 0.667
  
length_scale  
  shape: [1]
  dtype: float32
  default: 1.0
```

### 输出格式
```
y (音频)
  shape: [1, S, T]
  dtype: float32
  range: [-1.0, 1.0]
  → 展平后即为WAV数据
```

## 🔧 配置文件结构

### 单一模型目录
```
scripts/melo-tts/
├── export-onnx.py           # 导出脚本
├── model.onnx               # 导出的完整模型
├── tokens.txt               # 音素符号表
├── lexicon.txt              # 词汇表
└── test.wav                 # 测试音频
```

### 子图模型目录 (AXERA)
```
scripts/melo-tts/axera/zh_en/
├── model.onnx                    # 原始模型
├── subgraph_a.onnx               # 编码器子图
├── subgraph_b.onnx               # 生成器子图
├── config_subgraph_a_u16.json    # A量化配置
├── config_subgraph_b_u16.json    # B量化配置
├── subgraph_export_summary.json  # 导出摘要
└── calib/                        # 校准数据
    ├── subgraph_a/
    │   ├── x.tar.gz
    │   ├── x_lengths.tar.gz
    │   ├── tones.tar.gz
    │   ├── sid.tar.gz
    │   ├── noise_scale.tar.gz
    │   ├── length_scale.tar.gz
    │   └── noise_scale_w.tar.gz
    └── subgraph_b/
        ├── __Mul_10_output_0__.tar.gz
        └── __Unsqueeze_6_output_0__.tar.gz
```

## 🎯 性能优化建议

### 参数调优

| 场景 | noise_scale | length_scale | noise_scale_w | 效果 |
|------|------------|-------------|--------------|------|
| 清晰语音 | 0.5 | 1.0 | 0.8 | 低噪音 |
| 自然语音 | 0.667 | 1.0 | 0.8 | 平衡 |
| 多样化 | 1.0 | 1.0 | 0.8 | 高变化 |
| 快速播放 | 0.667 | 0.5 | 0.8 | 2倍速 |
| 缓慢播放 | 0.667 | 1.5 | 0.8 | 0.67倍速 |

### 子图量化策略

1. **数据准备**
   - 校准数据覆盖多种长度 ✓
   - 随机音素序列代表性好 ✓
   - 保存为NPY格式打包 ✓

2. **量化配置**
   - Subgraph A: 关键路径，INT8最好
   - Subgraph B: 可考虑INT8或FP16混合

3. **验证步骤**
   - 量化前后精度对比
   - 实时性能测试
   - 边界情况测试

## ⚠️ 常见问题

### Q: 模型推理失败，提示版本不兼容
**A**: 检查模型元数据中的`version`字段
```python
import onnx
model = onnx.load("model.onnx")
meta = {x.key: x.value for x in model.metadata_props}
print(f"Version: {meta.get('version')}")  # 必须 >= 2
```
若版本过旧，需要重新导出

### Q: 子图输入/输出对接不上
**A**: 验证张量名称完全匹配
```python
# 导出前检查中间张量名
import onnx
model = onnx.load("model.onnx")
for node in model.graph.node:
    if "Mul_10" in node.output[0]:
        print(f"Found: {node.output[0]}")  # 应为 "/Mul_10_output_0"
```

### Q: 音质不佳，如何调优
**A**: 尝试调整参数组合
```cpp
// 测试不同参数
for (float noise_scale : {0.5f, 0.667f, 1.0f}) {
    for (float length_scale : {0.8f, 1.0f, 1.2f}) {
        result = model.Run(x, tones, sid, speed);
        // 对比音频质量
    }
}
```

### Q: 推理速度过慢
**A**: 检查是否已量化
- 子图模型量化后速度: 3-5倍提升
- 检查量化配置是否正确应用
- 验证AXERA NPU是否被正确调用

## 📚 相关文档

| 文件 | 用途 |
|------|------|
| MeloTTS_Model_Export_Analysis.md | 完整技术文档 |
| ANALYSIS_SUMMARY.md | 快速参考 |
| README.md | 本指南 |
| ../export-onnx.py | 单一模型导出 |
| ../export_subgraphs_and_calib.py | 子图导出 |
| ../../csrc/offline-tts-vits-model.cc | C++推理实现 |

## 🔗 相关链接

- [MeloTTS官方仓库](https://github.com/myshell-ai/MeloTTS)
- [Sherpa-ONNX项目](https://github.com/k2-fsa/sherpa-onnx)
- [ONNX标准文档](https://onnx.ai/)
- [AXERA官方文档](https://www.axera.com/)

## 📝 更新日志

- **2024-04-22**: 初始版本 - 完整分析文档
  - ✅ 两种导出方式对比
  - ✅ 模型输入输出规范
  - ✅ C++推理代码分析
  - ✅ 文本预处理管道
  - ✅ 兼容性检查

---

**最后更新**: 2024-04-22  
**维护者**: Sherpa-ONNX Team  
**许可证**: MIT  
