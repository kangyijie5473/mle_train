from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "../heretic-DeepSeek"   # 原始基础模型路径
lora_adapter_path = "../heretic-DeepSeek-lora"       # LoRA 适配器路径（含 adapter_config.json）
merged_model_path = "../heretic-DeepSeek-lora-merge" # 合并后保存路径

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    trust_remote_code=True,
)

# 加载 LoRA 适配器并合并
model = PeftModel.from_pretrained(model, lora_adapter_path)
model = model.merge_and_unload()  # 关键：将 lora_A @ lora_B 合并进基础权重

# 保存合并后的完整模型
tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=True)
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("LoRA 合并完成，保存至:", merged_model_path)
