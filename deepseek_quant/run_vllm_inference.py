import argparse
import logging
from vllm import LLM, SamplingParams
import torch

# 设置日志
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def run_inference(model_path, quantization_type="awq"):
    """
    使用 vLLM 运行量化后的 DeepSeek-R1 14B 模型
    
    Args:
        model_path: 量化后的模型路径 (本地路径)
        quantization_type: 量化类型 (awq)
    """
    logger.info(f"正在加载 vLLM 模型: {model_path}，量化类型: {quantization_type}")
    
    # 初始化 vLLM
    # gpu_memory_utilization=0.9 确保大部分显存可用
    # max_model_len 可以根据需求调整，通常 DeepSeek 模型上下文较长
    llm = LLM(
        model=model_path,
        quantization=quantization_type,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        trust_remote_code=True,
        dtype="float16" # vLLM 在加载 AWQ 模型时通常使用 float16 进行计算
    )
    
    # 示例 prompt
    prompts = [
        "Please provide a Python function to solve the Fibonacci sequence.",
        "What is the capital of France?",
        "Explain quantum entanglement in simple terms."
    ]
    
    # 采样参数
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
    
    logger.info("开始生成...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-R1 14B 量化模型推理脚本 (vLLM)")
    parser.add_argument("--model_path", type=str, default="./DeepSeek-R1-Distill-Qwen-14B-AWQ", help="量化后模型路径")
    parser.add_argument("--quantization", type=str, default="awq", help="量化类型 (awq, gptq, squeeze, fp8)")

    args = parser.parse_args()
    
    run_inference(args.model_path, args.quantization)
