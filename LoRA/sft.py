import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import gc
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="./heretic-DeepSeek")
    parser.add_argument("--data_path", type=str, default="./lora_dataset.json")
    args, _ = parser.parse_known_args()

    output_dir       = "./heretic-DeepSeek-lora"
    num_train_epochs = 3
    learning_rate    = 2e-4
    lora_r           = 8
    lora_alpha       = 16
    max_seq_length   = 512

    # 1. Tokenizer
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. 数据集
    print(f"加载数据集: {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    def format_prompt(example):
        instruction = example.get("instruction", "")
        input_text  = example.get("input", "")
        output      = example.get("output", "")
        if input_text:
            prompt = (
                f"以下是描述任务的指令，以及提供进一步上下文的输入。"
                f"请编写一个适当的回复来完成该请求。\n\n"
                f"### 指令:\n{instruction}\n\n"
                f"### 输入:\n{input_text}\n\n"
                f"### 回复:\n"
            )
        else:
            prompt = (
                f"以下是描述任务的指令。请编写一个适当的回复来完成该请求。\n\n"
                f"### 指令:\n{instruction}\n\n"
                f"### 回复:\n"
            )
        return {"text": prompt + output + tokenizer.eos_token}

    dataset = dataset.map(format_prompt)
    print(f"数据集大小: {len(dataset)}")

    # 3. 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
    )

    # 4. ============================================================
    #    [核心修复] 两阶段加载：先在 CPU 完成量化，再按需搬到 GPU
    #    这样 GPU 上永远不会出现 BF16 全量权重的峰值
    # ============================================================
    print("第一阶段：在 CPU 上完成模型量化（需要约 30GB CPU 内存，耗时较长）...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="cpu",           # 先全部加载到 CPU，在 CPU 上完成量化
        low_cpu_mem_usage=True,     # 逐层加载，避免 CPU 内存峰值翻倍
        trust_remote_code=True,
    )

    print("第二阶段：将量化后的 Transformer 层搬运到 GPU，Embedding 层留在 CPU...")

    # 把 Transformer 层逐层移到 GPU
    # embed_tokens 和 lm_head 是 BF16 大户（2.90GB），强制留在 CPU
    model.model.embed_tokens = model.model.embed_tokens.to("cpu")

    # 逐层将 decoder layers 移到 GPU
    for i, layer in enumerate(model.model.layers):
        layer.to("cuda:0")

    # norm 和 lm_head 的处理
    model.model.norm = model.model.norm.to("cuda:0")
    model.lm_head = model.lm_head.to("cpu")   # lm_head 也是 BF16，留 CPU

    model.config.use_cache = False

    # 验证显存占用
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"两阶段加载完成，GPU 显存占用: {gpu_mem:.2f} GB")
    # 预期：约 7~8GB（仅 Transformer 层的 4-bit 权重）

    gc.collect()
    torch.cuda.empty_cache()

    # 5. LoRA 配置
    print("配置 LoRA...")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen2 架构：只训练 attention 的 q/v，节省最多显存
        # 确认能跑后，可逐步加回 "k_proj", "o_proj"
        target_modules=["q_proj", "v_proj"],
    )

    # 6. 训练配置
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=5,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=True,
        dataloader_pin_memory=False,
    )

    # 7. Trainer
    print("初始化 Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    gc.collect()
    torch.cuda.empty_cache()

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"训练前 GPU 显存占用: {gpu_mem:.2f} GB")
    print("开始训练...")
    trainer.train()

    print(f"保存 LoRA 权重到 {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("完成！")


if __name__ == "__main__":
    main()
