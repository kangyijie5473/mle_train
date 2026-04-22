import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from typing import Dict, List


@dataclass
class EvalCase:
    case_id: str
    question: str
    expected_keywords: List[str]
    category: str


@dataclass
class Variant:
    variant_id: str
    system_prompt_style: str
    use_search: bool
    use_memory: bool


class MockSearchTool:
    def search(self, query: str) -> List[str]:
        kb = {
            "python": "Python 是一种通用编程语言，常用于 AI、Web、自动化。",
            "all reduce": "all_reduce 会把各 rank 张量先规约（sum/max 等），再回传给所有 rank。",
            "gptq": "GPTQ 是后训练量化方法，常将权重量化到 int4 以减少显存。",
        }
        q = query.lower()
        hits = [v for k, v in kb.items() if k in q]
        return hits[:2]


class MockMemoryStore:
    def __init__(self) -> None:
        self.items = [
            "用户偏好：回答简洁并列出依据。",
            "用户关注：分布式训练与量化。",
        ]

    def recall(self, k: int = 2) -> List[str]:
        return self.items[-k:]


class MockLLM:
    """
    模拟 LLM 输出质量随 harness 配置变化而变化，用于演示 Harness Engineering。
    """

    def generate(self, question: str, prompt_style: str, tool_context: str, memory_context: str) -> str:
        q = question.lower()
        base = "直接答案："
        if "python" in q:
            answer = "Python 是编程语言。"
        elif "all reduce" in q:
            answer = "all_reduce 对所有 rank 做规约并返回。"
        elif "gptq" in q:
            answer = "GPTQ 是后训练量化方法，常见 int4。"
        else:
            answer = "这是通用回答。"

        # 不同 prompt style 模拟不同回答质量
        if prompt_style == "structured":
            answer = f"{answer} 依据：{tool_context or '无外部依据'}"
        if prompt_style == "cot-light":
            answer = f"{answer} 推理：先定义，再给用途。"
        if memory_context:
            answer = f"{answer} 记忆：{memory_context}"
        return f"{base}{answer}"


class HarnessEngine:
    def __init__(self) -> None:
        self.llm = MockLLM()
        self.search = MockSearchTool()
        self.memory = MockMemoryStore()

    @staticmethod
    def _norm(text: str) -> str:
        text = text.lower().strip()
        return re.sub(r"\s+", " ", text)

    def run_case(self, case: EvalCase, variant: Variant) -> Dict:
        t0 = time.perf_counter()
        search_hits = self.search.search(case.question) if variant.use_search else []
        memory_hits = self.memory.recall() if variant.use_memory else []

        answer = self.llm.generate(
            question=case.question,
            prompt_style=variant.system_prompt_style,
            tool_context=" | ".join(search_hits),
            memory_context=" | ".join(memory_hits),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        norm_answer = self._norm(answer)
        matched = [k for k in case.expected_keywords if self._norm(k) in norm_answer]
        recall = len(matched) / max(1, len(case.expected_keywords))

        return {
            "case_id": case.case_id,
            "category": case.category,
            "question": case.question,
            "answer": answer,
            "matched_keywords": matched,
            "keyword_recall": round(recall, 4),
            "passed": recall >= 1.0,
            "latency_ms": round(latency_ms, 2),
            "variant_id": variant.variant_id,
        }

    def run_experiment(self, cases: List[EvalCase], variants: List[Variant]) -> Dict:
        runs: List[Dict] = []
        for variant in variants:
            for case in cases:
                runs.append(self.run_case(case, variant))

        summary_by_variant: Dict[str, Dict] = {}
        for variant in variants:
            v_runs = [r for r in runs if r["variant_id"] == variant.variant_id]
            avg_recall = sum(r["keyword_recall"] for r in v_runs) / max(1, len(v_runs))
            pass_rate = sum(1 for r in v_runs if r["passed"]) / max(1, len(v_runs))
            avg_latency = sum(r["latency_ms"] for r in v_runs) / max(1, len(v_runs))
            summary_by_variant[variant.variant_id] = {
                "variant": asdict(variant),
                "num_cases": len(v_runs),
                "avg_keyword_recall": round(avg_recall, 4),
                "pass_rate": round(pass_rate, 4),
                "avg_latency_ms": round(avg_latency, 2),
            }

        leaderboard = sorted(
            summary_by_variant.values(),
            key=lambda x: (-x["avg_keyword_recall"], -x["pass_rate"], x["avg_latency_ms"]),
        )

        return {
            "time": datetime.now().isoformat(timespec="seconds"),
            "protocol": {
                "name": "harness_engineering_demo",
                "primary_metric": "avg_keyword_recall",
                "secondary_metric": "pass_rate",
                "tie_breaker": "avg_latency_ms(lower_is_better)",
            },
            "summary_by_variant": summary_by_variant,
            "leaderboard": leaderboard,
            "runs": runs,
        }


def build_cases() -> List[EvalCase]:
    return [
        EvalCase("c1_python", "请简单介绍 Python", ["python", "编程语言"], "general"),
        EvalCase("c2_allreduce", "什么是 all reduce？", ["all_reduce", "rank", "规约"], "distributed"),
        EvalCase("c3_gptq", "GPTQ 的作用是什么？", ["gptq", "量化", "int4"], "quantization"),
    ]


def build_variants() -> List[Variant]:
    variants: List[Variant] = []
    for style, use_search, use_memory in product(
        ["plain", "structured", "cot-light"],
        [False, True],
        [False, True],
    ):
        v_id = f"style={style}|search={int(use_search)}|memory={int(use_memory)}"
        variants.append(
            Variant(
                variant_id=v_id,
                system_prompt_style=style,
                use_search=use_search,
                use_memory=use_memory,
            )
        )
    return variants


def main():
    parser = argparse.ArgumentParser(description="Harness Engineering demo：实验矩阵 + 指标聚合 + 排行榜")
    parser.add_argument(
        "--report_path",
        type=str,
        default="/root/mle_train/agent/harness_engineering_report.json",
        help="报告输出路径",
    )
    args = parser.parse_args()

    engine = HarnessEngine()
    report = engine.run_experiment(build_cases(), build_variants())

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    best = report["leaderboard"][0]
    print("Harness Engineering 运行完成")
    print("最佳变体:", best["variant"]["variant_id"])
    print("指标:", {k: best[k] for k in ["avg_keyword_recall", "pass_rate", "avg_latency_ms"]})
    print("报告:", args.report_path)


if __name__ == "__main__":
    main()
