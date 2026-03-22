import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
from transformers import AutoConfig, AutoTokenizer

model_path = "./heretic-DeepSeek"

# 读取模型配置
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("=" * 50)
print(f"模型架构类型:     {config.model_type}")
print(f"参数量(隐层维度): {config.hidden_size}")
print(f"层数:             {config.num_hidden_layers}")
print(f"词表大小:         {config.vocab_size}")
print(f"Tokenizer词表:    {len(tokenizer)}")
print("=" * 50)

# 估算各部分显存占用
vocab    = config.vocab_size
hidden   = config.hidden_size
layers   = config.num_hidden_layers

# embedding层是BF16，不被量化
embed_mem_gb = (vocab * hidden * 2 * 2) / 1024**3  # embed_tokens + lm_head
# transformer层是4bit
layer_params = layers * hidden * hidden * 12  # 粗估
layer_mem_gb = (layer_params * 0.5) / 1024**3  # 4bit = 0.5 bytes/param

print(f"Embedding层显存估算 (BF16, 不被量化): {embed_mem_gb:.2f} GB")
print(f"Transformer层显存估算 (4-bit):        {layer_mem_gb:.2f} GB")
print(f"理论最低总显存:                       {embed_mem_gb + layer_mem_gb:.2f} GB")

# 检查磁盘上的权重格式
import os, glob
files = glob.glob(os.path.join(model_path, "*.safetensors")) + \
        glob.glob(os.path.join(model_path, "*.bin"))
total_size = sum(os.path.getsize(f) for f in files) / 1024**3
print(f"\n磁盘上模型文件总大小: {total_size:.2f} GB")
print(f"文件列表: {[os.path.basename(f) for f in files]}")
