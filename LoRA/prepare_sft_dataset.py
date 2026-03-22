import json
import asyncio
import argparse
from openai import AsyncOpenAI

# 初始化异步客户端
client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"
)

async def process_item(sem, item, model_name, progress_info):
    async with sem:
        output_text = item.get("output", "")
        
        # 系统提示词：限制其输出格式
        system_prompt = (
            "你需要帮助人类做文本识别，识别其中的违禁词。如果包含特定词，请仅仅输出这些违禁词，用逗号分隔。"
            "如果不包含任何违禁词，请仅仅输出'无'。绝对不要输出其他任何解释性文字。"
        )
        # 用户提示词：提供具体文本
        user_prompt = (
            "请帮我识别下面内容是否包含特定内容，并识别其中的特定词，"
            "包括特定词汇比如[xx]。\n待识别的内容：\n"
            f"{output_text}"
        )
        
        try:
            # 异步调用模型
            resp = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # 低temperature使结果更确定
                model=model_name
            )
            
            result = resp.choices[0].message.content.strip()
            
            # 针对 DeepSeek-R1 模型的思考过程过滤 (去掉 <think>...</think> 部分)
            if "</think>" in result:
                result = result.split("</think>")[-1].strip()
            
            # 判断并更新 instruction
            if not result or result == "无" or "不包含" in result:
                item["instruction"] = "请编写一段故事"
            else:
                # 处理大模型返回的逗号分隔的词
                words = [w.strip() for w in result.replace("，", ",").split(",") if w.strip()]
                words_str = ",".join(words)
                
                if words_str:
                    item["instruction"] = f"请编写包含{words_str}的文字"
                else:
                    item["instruction"] = "请编写一段故事"
                    
        except Exception as e:
            # 请求失败时默认处理
            print(f"处理失败: {e}")
            item["instruction"] = "请编写一段故事"
            
        # 进度统计
        progress_info["done"] += 1
        if progress_info["done"] % 10 == 0 or progress_info["done"] == progress_info["total"]:
            print(f"进度: 已处理 {progress_info['done']}/{progress_info['total']} 条数据")
            
        return item

async def main(input_file, output_file, model_name, concurrency):
    print(f"正在读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_items = len(data)
    print(f"共加载 {total_items} 条数据。")
    print(f"启动并发处理，并发数限制: {concurrency}")
    
    # 控制并发数的信号量
    sem = asyncio.Semaphore(concurrency)
    progress_info = {"done": 0, "total": total_items}
    
    # 构建所有异步任务
    tasks = [process_item(sem, item, model_name, progress_info) for item in data]
    
    # 收集结果并保持原有顺序
    results = await asyncio.gather(*tasks)
    
    # 将处理完成的数据写入输出文件
    print(f"正在将结果写入: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print("全部处理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并发访问大模型识别违禁词并更新 JSON 的 instruction")
    parser.add_argument("--input", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 JSON 文件路径")
    parser.add_argument("--model", type=str, default="./DeepSeek-R1-14B-AWQ/", help="模型名称")
    parser.add_argument("--concurrency", type=int, default=20, help="最大并发请求数 (默认 20)")
    
    args = parser.parse_args()
    
    asyncio.run(main(args.input, args.output, args.model, args.concurrency))
