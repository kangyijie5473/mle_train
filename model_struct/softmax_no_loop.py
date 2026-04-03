"""
笔试题：实现 softmax，不使用任何 for/while 循环。
提供两个版本：
  1. softmax_numpy  — 纯 NumPy 向量化
  2. softmax_pure   — 纯 Python 内置函数（math.exp / max / sum / map），不依赖 NumPy
输入为 2D 矩阵 (N, D)，按行计算 softmax。
"""

import math
import numpy as np


# ── Version 1: NumPy 向量化 ──────────────────────────────────────────
def softmax_numpy(x: np.ndarray) -> np.ndarray:
    """数值稳定的按行 softmax，纯向量化实现（无循环）。"""
    x_stable = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ── Version 2: 纯 Python，无 NumPy，无 for/while ─────────────────────
def _softmax_row(row: list[float]) -> list[float]:
    """对单行做数值稳定 softmax，仅用内置函数。"""
    row_max = max(row)
    exp_vals = list(map(lambda v: math.exp(v - row_max), row))
    total = sum(exp_vals)
    return list(map(lambda e: e / total, exp_vals))


def softmax_pure(x: list[list[float]]) -> list[list[float]]:
    """纯 Python 按行 softmax，无 for/while 循环，无 NumPy。"""
    return list(map(_softmax_row, x))


def main():
    np.random.seed(42)

    # ── NumPy 版本测试 ────────────────────────────────────────────
    print("=" * 50)
    print("Version 1: NumPy softmax")
    print("=" * 50)

    print("\n--- Test 1: 小矩阵手动验证 ---")
    x1 = np.array([[1.0, 2.0, 3.0],
                    [1.0, 1.0, 1.0]])
    result1 = softmax_numpy(x1)
    print(f"Input:\n{x1}")
    print(f"Softmax:\n{result1}")
    print(f"Row sums: {result1.sum(axis=-1)}")

    row0_expected = np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum()
    row1_expected = np.array([1 / 3, 1 / 3, 1 / 3])
    err1 = np.max(np.abs(result1[0] - row0_expected))
    err2 = np.max(np.abs(result1[1] - row1_expected))
    print(f"Row 0 max error vs manual: {err1:.2e}")
    print(f"Row 1 max error vs manual: {err2:.2e}")

    print("\n--- Test 2: 与 scipy 对比 ---")
    try:
        from scipy.special import softmax as scipy_softmax
        x2 = np.random.randn(5, 8)
        ours = softmax_numpy(x2)
        ref = scipy_softmax(x2, axis=-1)
        max_err = np.max(np.abs(ours - ref))
        print(f"Shape: {x2.shape}, max error vs scipy: {max_err:.2e}")
        assert max_err < 1e-12, "scipy comparison FAILED"
        print("PASSED")
    except ImportError:
        print("scipy not available, skipping")

    print("\n--- Test 3: 大矩阵 + 数值稳定性 ---")
    x3 = np.random.randn(100, 50) * 100
    result3 = softmax_numpy(x3)
    row_sums = result3.sum(axis=-1)
    sum_err = np.max(np.abs(row_sums - 1.0))
    print(f"Shape: {x3.shape}, max row-sum deviation from 1: {sum_err:.2e}")
    assert sum_err < 1e-12, "Row sum check FAILED"
    assert np.all(result3 >= 0), "Negative values found"
    assert not np.any(np.isnan(result3)), "NaN values found"
    assert not np.any(np.isinf(result3)), "Inf values found"
    print("PASSED")

    print("\n--- Test 4: 含极端值（溢出边界）---")
    x4 = np.array([[1000.0, 1000.0, 1000.0],
                    [-1000.0, 0.0, 1000.0]])
    result4 = softmax_numpy(x4)
    print(f"Input:\n{x4}")
    print(f"Softmax:\n{result4}")
    assert not np.any(np.isnan(result4)), "NaN on extreme values"
    assert not np.any(np.isinf(result4)), "Inf on extreme values"
    row_sums4 = result4.sum(axis=-1)
    print(f"Row sums: {row_sums4}")
    assert np.max(np.abs(row_sums4 - 1.0)) < 1e-12
    print("PASSED")

    # ── 纯 Python 版本测试 ───────────────────────────────────────
    print("\n" + "=" * 50)
    print("Version 2: Pure Python softmax (no NumPy)")
    print("=" * 50)

    print("\n--- Test 5: 小矩阵 & 与 NumPy 版本交叉对比 ---")
    x5_list = [[1.0, 2.0, 3.0],
               [1.0, 1.0, 1.0]]
    result5 = softmax_pure(x5_list)
    print(f"Input:   {x5_list}")
    print(f"Softmax: {result5}")
    row_sums5 = list(map(sum, result5))
    print(f"Row sums: {row_sums5}")

    result5_np = np.array(result5)
    cross_err = np.max(np.abs(result5_np - result1))
    print(f"Max error vs NumPy version: {cross_err:.2e}")
    assert cross_err < 1e-15, "Cross-version comparison FAILED"
    print("PASSED")

    print("\n--- Test 6: 极端值 ---")
    x6_list = [[1000.0, 1000.0, 1000.0],
               [-1000.0, 0.0, 1000.0]]
    result6 = softmax_pure(x6_list)
    print(f"Input:   {x6_list}")
    print(f"Softmax: {result6}")
    assert all(not (math.isnan(v) or math.isinf(v))
               for row in result6 for v in row), "NaN/Inf found"
    sum_errs = list(map(lambda row: abs(sum(row) - 1.0), result6))
    print(f"Row sum errors: {sum_errs}")
    assert max(sum_errs) < 1e-12
    print("PASSED")

    print("\n--- Test 7: 随机矩阵交叉对比 ---")
    x7_np = np.random.randn(20, 15)
    x7_list = x7_np.tolist()
    res_np = softmax_numpy(x7_np)
    res_pure = np.array(softmax_pure(x7_list))
    cross_err7 = np.max(np.abs(res_np - res_pure))
    print(f"Shape: (20, 15), max error pure vs numpy: {cross_err7:.2e}")
    assert cross_err7 < 1e-14, "Random matrix cross-comparison FAILED"
    print("PASSED")

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
