import argparse
import datetime as dt
import json
import os
import textwrap
import urllib.parse
import urllib.request
from typing import Any, Dict, List


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def ddg_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """使用 DuckDuckGo Instant Answer API 做免 key 搜索。"""
    params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
    url = f"https://api.duckduckgo.com/?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results: List[Dict[str, str]] = []
    if data.get("AbstractText"):
        results.append(
            {
                "title": data.get("Heading") or "Abstract",
                "snippet": data.get("AbstractText", ""),
                "url": data.get("AbstractURL", ""),
            }
        )

    def collect_related(items: List[Dict[str, Any]]) -> None:
        for it in items:
            if len(results) >= max_results:
                return
            # RelatedTopics 中既可能是条目，也可能是带 Topics 的分组
            if "Topics" in it and isinstance(it["Topics"], list):
                collect_related(it["Topics"])
                continue
            text = it.get("Text", "")
            first_url = it.get("FirstURL", "")
            if text:
                title = text.split(" - ")[0].strip()
                results.append({"title": title, "snippet": text, "url": first_url})

    related = data.get("RelatedTopics", [])
    if isinstance(related, list):
        collect_related(related)

    return results[:max_results]


def call_local_llm(prompt: str, model: str, endpoint: str, timeout: int = 60) -> str:
    """
    调用本地 LLM（默认兼容 Ollama /api/generate）。
    endpoint 示例: http://127.0.0.1:11434/api/generate
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    text = result.get("response", "")
    return text.strip()


class ReActMemoryAgent:
    def __init__(self, model: str, endpoint: str, memory_path: str, log_path: str):
        self.model = model
        self.endpoint = endpoint
        self.memory_path = memory_path
        self.log_path = log_path
        self.memory = read_json(memory_path, {"facts": [], "summaries": []})

    def _recent_memory_text(self, limit: int = 5) -> str:
        facts = self.memory.get("facts", [])[-limit:]
        summaries = self.memory.get("summaries", [])[-limit:]
        lines = ["近期记忆："]
        for i, f in enumerate(facts, 1):
            lines.append(f"{i}. {f}")
        for i, s in enumerate(summaries, 1):
            lines.append(f"S{i}. {s}")
        return "\n".join(lines)

    def _summarize_turn(self, user_query: str, final_answer: str) -> str:
        summary_prompt = textwrap.dedent(
            f"""
            请将这轮问答压缩为一句中文记忆（不超过35字），用于长期记忆检索。
            问题：{user_query}
            回答：{final_answer}
            只输出一句话，不要解释。
            """
        ).strip()
        try:
            return call_local_llm(summary_prompt, self.model, self.endpoint)
        except Exception:
            return f"用户问了：{user_query[:20]}；回答主题：{final_answer[:20]}"

    def answer(self, user_query: str, use_search: bool = True) -> Dict[str, Any]:
        # ReAct: Thought
        thought = "先结合历史记忆理解问题；若涉及事实/时效信息，调用搜索；再让本地LLM综合给出回答。"

        # ReAct: Action 1 - Search
        search_results: List[Dict[str, str]] = []
        observation_search = "未执行搜索"
        if use_search:
            try:
                search_results = ddg_search(user_query, max_results=5)
                observation_search = f"搜索返回 {len(search_results)} 条结果"
            except Exception as e:
                observation_search = f"搜索失败: {e}"

        # ReAct: Action 2 - Local LLM direct answer
        memory_text = self._recent_memory_text()
        search_text = "\n".join(
            [f"- {i+1}. {r['title']} | {r['snippet']} | {r['url']}" for i, r in enumerate(search_results)]
        )
        if not search_text:
            search_text = "(无可用搜索结果)"

        llm_prompt = textwrap.dedent(
            f"""
            你是本地助手。请结合用户问题、记忆和搜索信息，给出准确、简洁、结构化的中文回答。

            用户问题:
            {user_query}

            {memory_text}

            搜索结果:
            {search_text}

            输出要求:
            1) 先给直接答案
            2) 再给依据（最多3条）
            3) 若信息不确定，明确标注不确定点
            """
        ).strip()

        try:
            final_answer = call_local_llm(llm_prompt, self.model, self.endpoint)
            observation_llm = "本地LLM调用成功"
        except Exception as e:
            final_answer = (
                "本地LLM调用失败。以下是可用搜索结果摘要：\n"
                + "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results[:3]])
            )
            observation_llm = f"本地LLM调用失败: {e}"

        # 记忆更新
        summary = self._summarize_turn(user_query, final_answer)
        self.memory.setdefault("facts", []).append(f"Q: {user_query}")
        self.memory.setdefault("summaries", []).append(summary)
        self.memory["facts"] = self.memory["facts"][-100:]
        self.memory["summaries"] = self.memory["summaries"][-100:]
        write_json(self.memory_path, self.memory)

        record = {
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "query": user_query,
            "thought": thought,
            "action": ["search", "local_llm_synthesis"],
            "observation": [observation_search, observation_llm],
            "answer": final_answer,
            "summary": summary,
            "search_results": search_results,
        }
        append_jsonl(self.log_path, record)
        return record


def interactive_loop(agent: ReActMemoryAgent, use_search: bool) -> None:
    print("ReAct 记忆 Agent 已启动。输入 exit 退出。", flush=True)
    while True:
        user_query = input("\n你: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("已退出。")
            break

        record = agent.answer(user_query, use_search=use_search)
        print("\n助手:")
        print(record["answer"])
        print("\n[本轮摘要记忆]")
        print(record["summary"])


def main():
    parser = argparse.ArgumentParser(description="非常简单的 ReAct 对话 Agent（搜索 + 本地LLM + 记忆）")
    parser.add_argument("--model", type=str, default=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b"), help="本地LLM模型名")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("LOCAL_LLM_ENDPOINT", "http://127.0.0.1:11434/api/generate"),
        help="本地LLM接口地址（Ollama兼容）",
    )
    parser.add_argument(
        "--memory_path",
        type=str,
        default="/root/mle_train/agent/memory.json",
        help="记忆文件路径",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="/root/mle_train/agent/qa_log.jsonl",
        help="问答日志路径",
    )
    parser.add_argument("--no_search", action="store_true", help="禁用网页搜索")
    args = parser.parse_args()

    agent = ReActMemoryAgent(
        model=args.model,
        endpoint=args.endpoint,
        memory_path=args.memory_path,
        log_path=args.log_path,
    )
    interactive_loop(agent, use_search=not args.no_search)


if __name__ == "__main__":
    main()
