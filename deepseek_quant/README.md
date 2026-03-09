# DeepSeek-R1 14B 模型量化与 vLLM 部署指南 (llm-compressor)

本指南将帮助你在单张 NVIDIA RTX 4090 (24GB VRAM) 上运行 DeepSeek-R1 14B 模型。

由于 `AutoAWQ` 已被官方弃用且不再维护，本项目现已迁移至官方推荐的继任者 **`llm-compressor`**。

## 目录结构

- `quantize_deepseek.py`: 使用 `llm-compressor` 将模型量化为 4-bit AWQ 格式的脚本。
- `run_vllm_inference.py`: 使用 `vLLM` 加载量化模型并进行测试推理的脚本。
- `requirements.txt`: 依赖库列表 (已更新为 llm-compressor)。

## 1. 环境准备

确保你已经安装了 CUDA 12.1 或更高版本。

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

**关键依赖：**
- `llm-compressor`: vLLM 官方支持的模型压缩/量化工具。
- `vllm`: 高性能推理引擎。

## 2. 执行量化

运行以下命令下载原始模型并将其量化为 AWQ 格式：

```bash
python quantize_deepseek.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --quant_path ./DeepSeek-R1-Distill-Qwen-14B-AWQ
```

**量化过程说明：**
- 脚本会从 HuggingFace 下载模型。
- 使用 `neuralmagic/LLM_compression_calibration` 数据集进行校准（这是 AWQ 算法必需的）。
- 脚本配置为 **W4A16** (4-bit 权重, 16-bit 激活)，这是标准的 AWQ 配置，兼容 vLLM。
- 量化后的模型将保存为 `safetensors` 格式。

## 3. 运行推理

量化完成后，使用 vLLM 加载并运行模型：

```bash
python run_vllm_inference.py --model_path ./DeepSeek-R1-Distill-Qwen-14B-AWQ
```

或者直接启动 vLLM API 服务器：

```bash
vllm serve ./DeepSeek-R1-Distill-Qwen-14B-AWQ --quantization awq --dtype float16 --max-model-len 8192
```

## 显存占用预估

- **原始 FP16**: ~28GB (无法在 4090 上运行)
- **AWQ INT4**: ~8-9GB (模型权重) + KV Cache (剩余显存可用)
  - 在 24GB 显存下，你将有约 14-15GB 的空间用于上下文 (KV Cache)，这足以支持较长的对话历史。
