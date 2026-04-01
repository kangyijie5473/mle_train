import argparse
import logging
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset
import os

# 设置日志
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def quantize_model(model_path, quant_path, n_samples=512, max_seq_len=2048, algo="awq"):
    """
    使用 llm-compressor (AutoAWQ 的继任者) 量化 DeepSeek-R1 14B 模型
    
    Args:
        model_path: 原始模型路径
        quant_path: 量化后模型保存路径
        n_samples: 校准样本数量
        max_seq_len: 最大序列长度
    """
    logger.info(f"正在加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 准备校准数据集
    # 使用 neuralmagic/LLM_compression_calibration 数据集作为校准源
    logger.info(f"正在准备校准数据集 (samples={n_samples})...")
    
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["text"]}],
                tokenize=False,
                add_generation_prompt=False
            )
        }

    # 加载数据集 (使用 streaming 模式避免下载整个数据集)
    ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train", streaming=True)
    ds = ds.take(n_samples).map(preprocess)
    logger.info("正在将流式数据加载到内存...")
    tokenized_samples = []
    for sample in ds:
        encoded = tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_samples.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    ds = Dataset.from_list(tokenized_samples)

    # 配置量化配方 (Recipe)
    # W4A16: 权重 4-bit, 激活 16-bit (标准的 AWQ 配置)
    # 忽略 lm_head 层以保持精度
    if algo == "awq":
        logger.info(f"正在加载模型用于 AWQ: {model_path}")
        model, tokenizer = load_model_and_tokenizer(model_path)
        logger.info("配置量化参数 (W4A16 AWQ)...")
        recipe_awq = QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
        logger.info("开始量化过程 (AWQ oneshot)...")
        oneshot(model=model, dataset=ds, recipe=recipe_awq, max_seq_length=max_seq_len, num_calibration_samples=n_samples)
        logger.info(f"保存量化模型到: {quant_path}")
        model.save_pretrained(quant_path, save_compressed=True)
        tokenizer.save_pretrained(quant_path)
        logger.info(f"量化完成: {quant_path}")
    if algo == "gptq":
        logger.info(f"正在加载模型用于 GPTQ: {model_path}")
        model, tokenizer = load_model_and_tokenizer(model_path)
        logger.info("配置量化参数 (W4A16 GPTQ)...")
        recipe_gptq = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], actorder="static", block_size=128, dampening_frac=0.001)
        logger.info("开始量化过程 (GPTQ oneshot)...")
        oneshot(model=model, dataset=ds, recipe=recipe_gptq, max_seq_length=max_seq_len, num_calibration_samples=n_samples)
        logger.info(f"保存量化模型到: {quant_path}")
        model.save_pretrained(quant_path, save_compressed=True)
        tokenizer.save_pretrained(quant_path)
        logger.info(f"量化完成: {quant_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-R1 14B 模型量化脚本 (llm-compressor)")
    parser.add_argument("--model_path", type=str, default="./DeepSeek-R1-14B", help="原始模型路径或 HuggingFace ID")
    parser.add_argument("--quant_path", type=str, default="./DeepSeek-R1-Distill-Qwen-14B-AWQ", help="量化后模型保存路径")
    parser.add_argument("--algo", type=str, choices=["awq", "gptq"], default="awq", help="量化算法")
    parser.add_argument("--n_samples", type=int, default=512, help="校准样本数量")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="校准时的最大序列长度")
    args = parser.parse_args()
    os.makedirs(args.quant_path, exist_ok=True)
    quantize_model(args.model_path, args.quant_path, args.n_samples, args.max_seq_len, args.algo)
