# MeloTTS 分析文档索引

> 📌 本文件是所有分析文档的导航索引，帮助快速找到所需信息

## 📑 文档清单

### 1️⃣ README.md （新手必读）
**位置**: `scripts/melo-tts/axera/README.md`

**内容**:
- 文档导览和使用指南
- 三种部署场景的快速命令
- 核心指标速查表
- 文件结构说明
- 性能优化建议
- 常见问题FAQ

**适用人群**: 首次接触MeloTTS的开发者

**快速链接**:
- [快速开始](#快速开始)
- [部署场景1:AXERA](#场景1-部署到axera芯片)
- [部署场景2:PC/服务器](#场景2-部署到pc服务器)
- [常见问题](#常见问题)

---

### 2️⃣ ANALYSIS_SUMMARY.md （中级参考）
**位置**: `scripts/melo-tts/axera/ANALYSIS_SUMMARY.md`

**内容**:
- 导出方式核心区别表
- 关键技术细节速查
- 模型输入输出规范表
- C++ API参考
- 文本预处理流程图
- 输出后处理步骤
- 部署决策矩阵
- 性能调优参数表

**适用人群**: 有基础了解、需要快速查阅的开发者

**重点表格**:
| 表格名称 | 用途 |
|---------|------|
| 两种导出方式对比 | 选择合适的导出方式 |
| 模型输入/输出规范 | 理解数据格式 |
| C++ API参考 | 集成推理代码 |
| 性能调优参数 | 优化音质和速度 |

---

### 3️⃣ MeloTTS_Model_Export_Analysis.md （深度技术文档）
**位置**: `scripts/melo-tts/axera/MeloTTS_Model_Export_Analysis.md`

**章节结构**:

#### 第一章：两种模型导出方式对比
- **方式一：单一完整模型导出 (export-onnx.py)**
  - 特点和适用场景
  - 完整导出流程（含代码示例）
  - 优点和缺点列表
  
- **方式二：子图分割模型导出 (export_subgraphs_and_calib.py)**
  - 特点和适用场景
  - 子图定义和分割流程
  - 校准数据生成详解
  - 优点和缺点列表

- **对比总结表**
  - 14个维度的详细对比

#### 第二章：默认导出模型的输入输出
- **模型元数据**
- **7个输入接口详解**
  - 各输入的数据类型、形状、范围、说明
  - 详细的使用示例
  
- **输出接口详解**
  - 输出形状和含义
  - 获取和使用方式

- **动态轴定义**

#### 第三章：C++推理代码分析
- **元数据结构详解**
- **两个推理API**
  - 基础推理函数
  - MeloTTS推理函数
  
- **推理流程详解**（带代码）
- **速度控制实现**
- **关键的C++类和方法**
- **异常处理**
- **内存管理**

#### 第四章：模型预处理与输入输出兼容性
- **文本预处理流程**
  - 中文处理（含转换示例）
  - 英文处理
  - 文本到ID的映射
  
- **C++端模型输入准备**
  - ONNX张量构建
  - 张量形状对齐规则
  
- **输出处理与音频生成**
  - 输出张量格式
  - 完整的音频处理管道（含代码）
  
- **兼容性检查**
  - 模型版本兼容性
  - 输入数据兼容性检查
  - 跨语言兼容性表
  
- **子图推理的兼容性**
  - Subgraph A的输入输出
  - Subgraph B的输入输出
  - 兼容性要求

**适用人群**: 需要深入理解技术细节、进行定制化开发的高级开发者

---

## 🎯 场景导航

### 场景1: "我想快速部署MeloTTS到AXERA芯片"
**推荐阅读顺序**:
1. README.md → [快速开始](#快速开始) → [场景1:AXERA](#场景1-部署到axera芯片)
2. ANALYSIS_SUMMARY.md → [选择标准](#部署建议) 
3. 查看具体语言目录 (zh_en/, en/, jp/ 等)

**关键文件**:
- `export_subgraphs_and_calib.py` - 导出脚本
- `config_subgraph_a_u16.json` - A量化配置
- `config_subgraph_b_u16.json` - B量化配置

---

### 场景2: "我需要理解输入输出数据格式"
**推荐阅读顺序**:
1. ANALYSIS_SUMMARY.md → [关键技术细节](#关键技术细节)
2. MeloTTS_Model_Export_Analysis.md → [第二章:默认导出模型的输入输出](#默认导出模型的输入输出)
3. 运行 `show-info.py` 查看实际模型信息

**关键表格**:
- 输入规范表（7个输入）
- 输出规范表
- 动态轴定义

---

### 场景3: "我需要集成C++推理代码"
**推荐阅读顺序**:
1. README.md → [场景3:C++推理集成](#场景3-c推理集成)
2. ANALYSIS_SUMMARY.md → [C++ 推理关键API](#c-推理关键api)
3. MeloTTS_Model_Export_Analysis.md → [第三章:C++推理代码分析](#c推理代码分析)
4. 查看 `offline-tts-vits-model.cc` 源码

**关键API**:
```cpp
// 基础推理
Ort::Value Run(Ort::Value x, int64_t sid = 0, float speed = 1.0);

// MeloTTS推理
Ort::Value Run(Ort::Value x, Ort::Value tones, int64_t sid = 0, 
               float speed = 1.0) const;
```

---

### 场景4: "我需要优化音质或性能"
**推荐阅读顺序**:
1. README.md → [性能优化建议](#性能优化建议)
2. ANALYSIS_SUMMARY.md → [性能调优参数表](#快速参考)
3. MeloTTS_Model_Export_Analysis.md → [模型预处理与输入输出兼容性](#模型预处理与输入输出兼容性)

**调优参数**:
| 参数 | 默认值 | 调整范围 | 效果 |
|------|------|--------|------|
| noise_scale | 0.667 | 0.5-1.0 | 音质稳定性 |
| length_scale | 1.0 | 0.5-1.5 | 音频长度/速度 |
| noise_scale_w | 0.8 | 0.6-1.0 | 自然度 |

---

### 场景5: "我遇到问题，需要调试"
**推荐阅读顺序**:
1. README.md → [常见问题](#常见问题)
2. ANALYSIS_SUMMARY.md → [注意事项](#注意事项)
3. MeloTTS_Model_Export_Analysis.md → [兼容性检查](#兼容性检查)

**常见问题**:
- Q: 模型推理失败，提示版本不兼容
- Q: 子图输入/输出对接不上
- Q: 音质不佳，如何调优
- Q: 推理速度过慢

---

## 📊 技术对比速查

### 两种导出方式的决策树

```
需要部署MeloTTS?
    │
    ├─ 目标设备是AXERA芯片? 
    │   ├─ YES → 使用 export_subgraphs_and_calib.py
    │   │       └─ 生成子图 + 校准数据
    │   │       └─ 转向 ANALYSIS_SUMMARY.md → 子图部分
    │   │
    │   └─ NO → 目标设备是PC/服务器?
    │       ├─ YES → 使用 export-onnx.py
    │       │       └─ 生成单一模型
    │       │       └─ 转向 README.md → 场景2
    │       │
    │       └─ NO → 咨询具体设备文档
    │
    └─ 需要量化?
        ├─ YES (INT8) → 使用 export_subgraphs_and_calib.py
        │               └─ 提供完整校准数据集
        │
        └─ NO → 使用 export-onnx.py
```

---

## 📈 信息密度对比

| 文档 | 阅读时间 | 深度 | 代码示例 | 表格数 | 适合场景 |
|------|---------|------|--------|-------|--------|
| README.md | 15分钟 | 浅 | 5个 | 8个 | 快速上手 |
| ANALYSIS_SUMMARY.md | 20分钟 | 中 | 2个 | 12个 | 快速查阅 |
| MeloTTS_Model_Export_Analysis.md | 60分钟 | 深 | 30+个 | 20+个 | 深度学习 |

---

## 🔗 文件位置映射

```
scripts/melo-tts/
├── README.md                          # 此目录使用说明
├── export-onnx.py                     # 单一模型导出脚本
├── export_subgraphs_and_calib.py      # 子图导出脚本
├── show-info.py                       # 显示模型信息
│
└── axera/                             # AXERA部署文件夹
    ├── README.md ⭐                   # 部署指南（必读）
    ├── ANALYSIS_SUMMARY.md ⭐         # 快速参考（必读）
    ├── MeloTTS_Model_Export_Analysis.md ⭐  # 详细文档
    ├── INDEX.md                       # 本文件
    │
    ├── zh_en/                         # 中英文模型
    │   ├── model.onnx                 # 原始模型
    │   ├── subgraph_a.onnx           # 编码器子图
    │   ├── subgraph_b.onnx           # 生成器子图
    │   ├── config_*.json             # 量化配置
    │   └── calib/                    # 校准数据
    │
    ├── en/, jp/, kr/, es/, fr/        # 其他语言模型
    └── test/                          # 测试用例
    
sherpa-onnx/csrc/
├── offline-tts-vits-model.h           # 推理类声明
├── offline-tts-vits-model.cc          # 推理实现（核心）
└── offline-tts-vits-model-meta-data.h # 元数据结构
```

---

## ✅ 学习路径建议

### 初级（1-2小时）
1. 阅读 README.md 完整版
2. 浏览 ANALYSIS_SUMMARY.md 快速参考
3. 运行一个示例导出
4. 查看生成的文件结构

### 中级（2-4小时）
1. 深入学习 ANALYSIS_SUMMARY.md 的所有表格
2. 学习 C++ API 部分
3. 理解文本预处理流程
4. 尝试调整推理参数

### 高级（4-8小时）
1. 完整阅读 MeloTTS_Model_Export_Analysis.md
2. 研究源代码 `offline-tts-vits-model.cc`
3. 理解ONNX模型结构
4. 进行定制化开发或优化

---

## 🎓 核心概念速记

### 关键术语

| 术语 | 定义 | 在文档中的位置 |
|------|------|---------------|
| **MeloTTS** | 多语言文本转语音模型 | README.md 开头 |
| **VITS** | 变分推理文本转语音架构 | ANALYSIS_SUMMARY.md |
| **子图分割** | 将大模型分成多个子模型 | MeloTTS_Model_Export_Analysis.md 第一章 |
| **校准数据** | 用于量化的代表性输入数据 | ANALYSIS_SUMMARY.md 或第一章 |
| **Subgraph A** | 编码器部分（文本→潜在表示） | ANALYSIS_SUMMARY.md |
| **Subgraph B** | 生成器部分（潜在表示→音频） | ANALYSIS_SUMMARY.md |
| **量化** | 将浮点模型转为整数（如INT8） | README.md 性能优化部分 |
| **声调** | 中文/日文等中的音调标记 | 第四章 文本预处理 |
| **噪声缩放** | 控制生成多样性的参数 | ANALYSIS_SUMMARY.md 参数表 |
| **长度缩放** | 控制音频长度/语速的参数 | ANALYSIS_SUMMARY.md 参数表 |

---

## 📞 更新和支持

- **最后更新**: 2024-04-22
- **版本**: 1.0
- **维护**: Sherpa-ONNX Team
- **许可证**: MIT

---

**开始探索**: 根据你的使用场景，选择合适的文档！ 🚀
