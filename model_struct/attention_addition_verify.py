"""
笔试题验证：用单层 dot-product attention（无缩放）实现两个数字的加法。

题目公式：A = softmax(Q @ K^T)，O = A @ V（无 /sqrt(d_k) 缩放）

输入：两个 one-hot 向量 x1=e_a, x2=e_b (各 10 维)，拼成 X ∈ R^{2×10}。
输出：O[0,0] = a + b。

构造：
  W_Q = W_K = 1·e_0^T  (10×10，每行只有第 0 列为 1)
    → Q, K 的每一行都是 e_0^T，与 a,b 无关
    → QK^T = [[1,1],[1,1]]，softmax 后每行均为 (0.5, 0.5)

  W_V 第 k 行第 0 列 = 2k，其余为 0
    → V = [[2a, 0, ...], [2b, 0, ...]]
    → O = 0.5 * V_a + 0.5 * V_b，第 0 维 = 0.5*(2a+2b) = a+b
"""

import torch


def build_weights():
    d = 10

    W_Q = torch.zeros(d, d)
    W_Q[:, 0] = 1.0

    W_K = torch.zeros(d, d)
    W_K[:, 0] = 1.0

    W_V = torch.zeros(d, d)
    for k in range(d):
        W_V[k, 0] = 2.0 * k

    return W_Q, W_K, W_V


def attention_forward(X, W_Q, W_K, W_V):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.transpose(-2, -1)
    attn = torch.softmax(scores, dim=-1)
    O = attn @ V
    return O, attn


def main():
    W_Q, W_K, W_V = build_weights()

    print("W_Q (= W_K):")
    print(W_Q)
    print("\nW_V:")
    print(W_V)

    max_err = 0.0
    for a in range(10):
        for b in range(10):
            x1 = torch.zeros(10)
            x1[a] = 1.0
            x2 = torch.zeros(10)
            x2[b] = 1.0
            X = torch.stack([x1, x2])  # (2, 10)

            O, attn = attention_forward(X, W_Q, W_K, W_V)
            result = O[0, 0].item()
            expected = a + b
            err = abs(result - expected)
            max_err = max(max_err, err)

            if err > 1e-5:
                print(f"FAIL: a={a}, b={b}, got {result:.4f}, expected {expected}")

    print(f"\nMax absolute error across all 100 pairs: {max_err:.2e}")
    if max_err < 1e-5:
        print("ALL 100 (a,b) pairs PASSED: O[0,0] == a + b")
    else:
        print("Some pairs FAILED.")

    print("\nExample: a=3, b=7")
    
    x1 = torch.zeros(10); x1[3] = 1.0
    x2 = torch.zeros(10); x2[7] = 1.0
    X = torch.stack([x1, x2])
    O, attn = attention_forward(X, W_Q, W_K, W_V)
    print(f"  Attention weights:\n{attn}")
    print(f"  Output O:\n{O}")
    print(f"  O[0,0] = {O[0,0].item():.1f}  (expected {3+7})")


if __name__ == "__main__":
    main()
