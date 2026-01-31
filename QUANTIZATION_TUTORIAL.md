# 模型量化实战指南 (Model Quantization Tutorial)

本文档旨在为初学者详细解读 `quantization/base.py` 中的代码，涵盖了 **PyTorch 动态量化**、**ONNX Runtime 量化** 以及 **BitsAndBytes** 加载三种主流技术。

---

## 1. 基础依赖导入 (Imports)

```python
1 import os
2 import time
3 import torch
4 import psutil
5 import numpy as np
6 from datasets import load_dataset
7 from evaluate import load
8 from transformers import AutoModelForSequenceClassification, AutoTokenizer
```

*   **1-5行**: 导入 Python 基础库。`psutil` 用于监控内存使用，`torch` 是深度学习框架，`numpy` 用于数值计算。
*   **6-8行**: 导入 Hugging Face 生态库。
    *   `load_dataset`: 用于加载标准数据集（如 AG News）。
    *   `evaluate`: 用于加载评估指标（如准确率 accuracy）。
    *   `AutoModel...`, `AutoTokenizer`: 自动根据模型名加载对应的模型结构和分词器。

```python
10 # Imports for Quantization
11 import onnx
12 import torch.ao.quantization
13 from torch.onnx import export
14 import onnxruntime as ort
15 from onnxruntime.quantization import quantize_dynamic, QuantType
```

*   **11-15行**: 导入量化相关库。
    *   `onnx`: 处理 ONNX 模型文件的标准库。
    *   `torch.ao.quantization`: PyTorch 原生的量化工具箱 (AO = Architecture Optimization)。
    *   `torch.onnx.export`: 将 PyTorch 模型转换为 ONNX 格式的函数。
    *   `onnxruntime`: 用于推理 ONNX 模型的引擎（通常比 PyTorch 原生推理更快）。
    *   `quantize_dynamic`: ONNX Runtime 提供的动态量化工具。

---

## 2. 环境配置与工具函数

```python
19 os.environ["OMP_NUM_THREADS"] = "1"
```
*   **19行**: 设置环境变量，限制 ONNX Runtime 的并发线程数为 1。这是为了防止在某些多核 CPU 上出现资源争抢导致性能下降。

```python
22 if os.uname().sysname == "Darwin" and os.uname().machine == "arm64":
23     torch.backends.quantized.engine = "qnnpack"
```
*   **22-23行**: 针对 Apple Silicon (M1/M2/M3) 的特殊优化。
    *   PyTorch 的量化引擎默认是 `fbgemm` (适用于 x86 Intel/AMD CPU)。
    *   在 ARM 架构（Mac）上，必须切换为 `qnnpack` 引擎，否则会报错。

```python
26 def get_memory_usage():
27     """获取当前进程的内存占用（MB）"""
28     process = psutil.Process()
29     return process.memory_info().rss / 1024 / 1024
```
*   **26-29行**: 辅助函数。使用 `psutil` 获取当前 Python 进程占用的物理内存 (RSS)，用于评估量化前后的内存节省情况。

```python
31 def measure_latency(model_fn, inputs, n_warmup=5, n_runs=20):
```
*   **31-47行**: 延迟测试函数。
    *   **Warmup (预热)**: 先跑几次推理不计时，让 CPU/GPU 缓存加载好，模型进入稳定状态。
    *   **Runs (正式运行)**: 运行多次取平均值，减少偶然误差。

---

## 3. 模型评估函数

```python
49 def evaluate_torch_model(model, tokenizer, dataset, metric):
```
*   **49-63行**: 评估 PyTorch 模型的准确率。
    *   `model.eval()`: 切换到评估模式（关闭 Dropout 等训练时特有的层）。
    *   `tokenizer(...)`: 将文本转换为模型能理解的数字 ID (Input IDs)。
    *   `torch.no_grad()`: **关键点**。告诉 PyTorch 不需要计算梯度（Gradient），这样可以大幅减少显存占用并加速推理。

```python
65 def evaluate_onnx_model(ort_session, tokenizer, dataset, metric):
```
*   **65-83行**: 评估 ONNX 模型的准确率。
    *   ONNX Runtime (ORT) 接收 `numpy` 数组作为输入，而不是 PyTorch Tensor。
    *   `ort_session.run(None, ort_inputs)`: 执行 ONNX 推理。

---

## 4. 实验准备 (Setup)

```python
88 model_name = "textattack/distilbert-base-uncased-AG-News"
89 tokenizer = AutoTokenizer.from_pretrained(model_name)
90 dataset = load_dataset("ag_news", split="test[:200]")
```
*   **88行**: 选用的模型。这是一个已经微调好的 DistilBERT 模型，用于新闻分类（AG News 数据集）。
*   **90行**: `split="test[:200]"` 表示只取测试集的前 200 条数据进行快速验证，避免等待太久。

---

## 5. Baseline (FP32 浮点模型)

```python
103 model_fp32 = AutoModelForSequenceClassification.from_pretrained(model_name)
```
*   **103行**: 加载原始模型。默认情况下，模型参数是 **FP32** (32位浮点数)，精度最高但体积最大。

---

## 6. PyTorch 动态量化 (Dynamic Quantization)

这是最简单的一种量化方法，通常用于 RNN/BERT 等 NLP 模型。

```python
126 model_int8 = torch.ao.quantization.quantize_dynamic(
127     model_fp32,
128     {torch.nn.Linear},
129     dtype=torch.qint8
130 )
```
*   **126行**: 核心代码。
*   **128行**: `{torch.nn.Linear}` 表示只对 **线性层 (Linear Layers)** 进行量化。Transformer 模型的大部分计算量都在线性层。
*   **129行**: `dtype=torch.qint8` 表示将权重从 FP32 压缩为 **Int8** (8位整数)。
*   **效果**: 模型体积通常减小 2-4 倍，推理速度提升 1.5-3 倍。

---

## 7. PyTorch FX Graph Mode Quantization (Static Int8)

这是 PyTorch 提供的高级量化模式，相比动态量化，它能提供更好的性能（理论上），但对模型结构的要求更严格。

```python
    # FX Quantization Setup
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    
    # Use transformers.utils.fx.symbolic_trace to handle HF model quirks
    traced_model = symbolic_trace(model_fx, input_names=["input_ids", "attention_mask"])
    
    # Prepare
    prepared_model = prepare_fx(traced_model, qconfig_mapping, example_inputs)
```

*   **FX Graph Mode**: 是一种全图量化模式，它会尝试“追踪”模型的整个执行流程。
*   **挑战**: 对于 Hugging Face 这种复杂的 Transformer 模型，普通的 FX Tracing 经常会遇到问题（如控制流错误、切片索引错误等）。
*   **代码现状**: 在脚本中，你会看到这部分被包裹在 `try-except` 中。如果遇到 `FX Quantization failed`，这是正常的。通常需要使用更高级的 `optimum` 库来完美处理 HF 模型的 FX 量化。

---

## 8. ONNX Runtime 量化

ONNX 是通用的模型交换格式，ONNX Runtime 是业界广泛使用的高性能推理引擎。

### 7.1 导出 ONNX 模型

```python
161 torch.onnx.export(
162     model_fp32,
163     (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
164     onnx_model_path,
...
167     dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, ...}
170     opset_version=17
171 )
```
*   **161行**: 将 PyTorch 模型“翻译”成 ONNX 格式。
*   **163行**: 必须提供一组假输入 (dummy inputs)，让 PyTorch 跑一遍模型来追踪计算图。
*   **167行**: `dynamic_axes` 非常重要。它告诉 ONNX：“输入数据的 Batch Size 和长度是可变的，不要写死”。如果不设置这个，导出的模型就只能处理固定长度的文本。

### 7.2 ONNX 预处理 (Shape 清理)

```python
176 model_onnx = onnx.load(onnx_model_path)
178 for input in model_onnx.graph.input:
...
182         dim.ClearField("dim_value")
```
*   **176-188行**: 这是一段“补丁”代码。在某些版本组合（Torch + ONNX）下，导出的模型会携带错误的固定尺寸信息，导致后续量化失败。这段代码手动清除了这些形状信息，让模型更灵活。

### 7.3 执行量化

```python
192 quantize_dynamic(
193     onnx_model_path,
194     onnx_quant_path,
195     weight_type=QuantType.QUInt8,
196     extra_options={"DisableShapeInference": True}
197 )
```
*   **192行**: 使用 ONNX Runtime 的量化工具。
*   **195行**: `QUInt8` (Unsigned Int8) 是目标数据类型。
*   **196行**: `DisableShapeInference` 是为了绕过 Mac 环境下的一个常见兼容性 Bug。

---

## 8. BitsAndBytes (8-bit Loading)

这是专门针对大模型（LLM）显存优化的技术，常用于显卡显存不足时运行大模型。

```python
233 model_bnb = AutoModelForSequenceClassification.from_pretrained(
234     model_name,
235     load_in_8bit=True,
236     device_map="auto"
237 )
```
*   **235行**: `load_in_8bit=True` 是核心。它在加载模型时直接将权重转换为 8-bit，大幅降低显存占用。
*   **限制**: 这个功能高度依赖 CUDA (NVIDIA 显卡)。在代码中你可以看到 `if torch.cuda.is_available():` 的判断，因此在 Mac 上这段代码会自动跳过。

---

## 总结

这份代码展示了一个完整的量化实验流程：
1.  **准备**: 加载模型和数据。
2.  **Baseline**: 测 FP32 的基准。
3.  **方案 A (Torch)**: 简单快捷，纯 Python 栈。
4.  **方案 B (FX Graph)**: 理论性能更优，但对动态图模型支持较难（容易报错）。
5.  **方案 C (ONNX)**: 工业界常用，通常更快，但流程复杂（导出 -> 量化 -> 推理）。
6.  **方案 D (BitsAndBytes)**: 针对 GPU 显存优化。

通过对比 `measure_latency` 和 `evaluate_...` 的结果，你可以直观地看到“空间换时间”或“精度换速度”的权衡 (Trade-off)。
