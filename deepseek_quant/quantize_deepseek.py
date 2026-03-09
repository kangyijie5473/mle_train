import argparse
import logging
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from datasets import load_dataset
import os

# 设置日志
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def quantize_model(model_path, quant_path, n_samples=512, max_seq_len=2048):
    """
    使用 llm-compressor (AutoAWQ 的继任者) 量化 DeepSeek-R1 14B 模型
    
    Args:
        model_path: 原始模型路径
        quant_path: 量化后模型保存路径
        n_samples: 校准样本数量
        max_seq_len: 最大序列长度
    """
    logger.info(f"正在加载模型: {model_path}")
    
    # 加载模型和分词器
    # 使用 trust_remote_code=True 因为 DeepSeek 模型可能需要
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype="auto",
        trust_remote_code=True
    )
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

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=["text", "id", "meta"]) # 移除不需要的列
    logger.info("正在将流式数据加载到内存...")
    ds = Dataset.from_list(list(ds)) 

    # 配置量化配方 (Recipe)
    # W4A16: 权重 4-bit, 激活 16-bit (标准的 AWQ 配置)
    # 忽略 lm_head 层以保持精度
    logger.info("配置量化参数 (W4A16 AWQ)...")
    
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16", 
        ignore=["lm_head"]
        )

    logger.info("开始量化过程 (oneshot)...")
    
    # 执行量化
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=n_samples,
    )

    logger.info(f"保存量化模型到: {quant_path}")
    
    # 保存模型 (使用 save_compressed=True 保存为 safetensors 格式)
    model.save_pretrained(quant_path, save_compressed=True)
    tokenizer.save_pretrained(quant_path)
    
    logger.info(f"量化完成! 模型已保存至 {quant_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-R1 14B 模型量化脚本 (llm-compressor)")
    parser.add_argument("--model_path", type=str, default="./DeepSeek-R1-14B", help="原始模型路径或 HuggingFace ID")
    parser.add_argument("--quant_path", type=str, default="./DeepSeek-R1-Distill-Qwen-14B-AWQ", help="量化后模型保存路径")
    parser.add_argument("--n_samples", type=int, default=512, help="校准样本数量")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="校准时的最大序列长度")

    args = parser.parse_args()
    
    # 确保保存目录存在
    os.makedirs(args.quant_path, exist_ok=True)
    
    quantize_model(args.model_path, args.quant_path, args.n_samples, args.max_seq_len)
