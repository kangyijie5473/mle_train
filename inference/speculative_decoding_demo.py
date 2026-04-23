from typing import Dict, List


class ToyLM:
    """
    一个极简语言模型：
    仅根据“最后一个 token”决定下一个 token（贪心）。
    """

    def __init__(self, name: str, transitions: Dict[str, str], eos_token: str = "<EOS>"):
        self.name = name
        self.transitions = transitions
        self.eos_token = eos_token

    def next_token(self, prefix: List[str]) -> str:
        last = prefix[-1]
        return self.transitions.get(last, self.eos_token)

    def draft(self, prefix: List[str], k: int) -> List[str]:
        out: List[str] = []
        state = list(prefix)
        for _ in range(k):
            nxt = self.next_token(state)
            out.append(nxt)
            state.append(nxt)
            if nxt == self.eos_token:
                break
        return out


def speculative_decode(
    target: ToyLM,
    draft_model: ToyLM,
    prompt: List[str],
    max_new_tokens: int = 12,
    draft_k: int = 3,
) -> List[str]:
    generated = list(prompt)
    new_count = 0
    step = 0

    print(f"Prompt: {' '.join(prompt)}")
    print(f"Target={target.name}, Draft={draft_model.name}, draft_k={draft_k}\n")

    while new_count < max_new_tokens:
        step += 1
        proposal = draft_model.draft(generated, draft_k)
        if not proposal:
            break

        print(f"[Step {step}] Draft proposal: {proposal}")
        accepted_all = True

        for i, token_d in enumerate(proposal):
            token_t = target.next_token(generated)
            if token_d == token_t:
                generated.append(token_d)
                new_count += 1
                print(f"  - accept  pos={i} token={token_d}")
                if token_d == target.eos_token or new_count >= max_new_tokens:
                    break
            else:
                # 拒绝 draft token，并回退为 target token
                accepted_all = False
                generated.append(token_t)
                new_count += 1
                print(
                    f"  - reject  pos={i} draft={token_d} target={token_t} -> fallback to target token"
                )
                break

        if generated[-1] == target.eos_token or new_count >= max_new_tokens:
            break

        # 如果本轮 proposal 全部被接受，按照经典 speculative decoding 追加一个 target bonus token
        if accepted_all:
            bonus = target.next_token(generated)
            generated.append(bonus)
            new_count += 1
            print(f"  - bonus from target: {bonus}")
            if bonus == target.eos_token:
                break

        print(f"  Current output: {' '.join(generated)}\n")

    return generated


def build_models() -> (ToyLM, ToyLM):
    # target 更“准确”
    target_transitions = {
        "<BOS>": "I",
        "I": "like",
        "like": "to",
        "to": "eat",
        "eat": "pizza",
        "pizza": "today",
        "today": "<EOS>",
    }

    # draft 更快但有偏差：I 后面给 love；eat 后面给 pasta
    draft_transitions = {
        "<BOS>": "I",
        "I": "love",
        "love": "to",
        "like": "to",
        "to": "eat",
        "eat": "pasta",
        "pasta": "today",
        "pizza": "today",
        "today": "<EOS>",
    }

    target = ToyLM(name="target_model", transitions=target_transitions)
    draft = ToyLM(name="draft_model", transitions=draft_transitions)
    return target, draft


def main():
    target, draft = build_models()
    prompt = ["<BOS>"]

    output = speculative_decode(
        target=target,
        draft_model=draft,
        prompt=prompt,
        max_new_tokens=12,
        draft_k=3,
    )

    print("\nFinal tokens:", output)
    print("Final text  :", " ".join(output))


if __name__ == "__main__":
    main()
