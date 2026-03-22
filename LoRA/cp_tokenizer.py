import shutil
import os
from transformers import AutoTokenizer

base_model_path = "./heretic-DeepSeek"
merged_model_path = "./heretic-DeepSeek-lora-merge"
quant_path = "./heretic-DeepSeek-lora-awq"
# 需要完整替换的 tokenizer 文件列表
tokenizer_files = [
    "tokenizer.json",
    "tokenizer_config.json", 
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]
for fname in tokenizer_files:
    src = os.path.join(base_model_path, fname)
    dst = os.path.join(quant_path, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"量化目录已更新: {fname}")

# 最终验证
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
test = "你好，请用中文回答我。"
ids = tokenizer.encode(test, add_special_tokens=False)
print("量化目录 tokenizer 验证:", tokenizer.decode(ids))



# for fname in tokenizer_files:
#     src = os.path.join(base_model_path, fname)
#     dst = os.path.join(merged_model_path, fname)
#     if os.path.exists(src):
#         shutil.copy(src, dst)
#         print(f"已替换: {fname}")
#     else:
#         print(f"基础模型中不存在: {fname}（跳过）")

# # 替换后立即验证
# tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
# print("\n--- 验证 ---")
# print("tokenizer 类型:", type(tokenizer))

# test = "你好世界，这是中文测试。"
# ids = tokenizer.encode(test, add_special_tokens=False)
# print("token ids:", ids)
# for tid in ids:
#     print(f"  id={tid:6d}  →  '{tokenizer.decode([tid])}'")
