import torch
from torch import nn

torch.manual_seed(42)

batch_size = 2
seq_len = 6
d_model = 16
num_q_heads = 4
num_kv_heads = 2
head_dim = d_model // num_q_heads


class GQAManual(nn.Module):
    """
    手写 GQA (Grouped-Query Attention)：
    - Q 使用更多头 (num_q_heads)
    - K/V 使用更少头 (num_kv_heads)
    - 每个 KV 头服务多个 Q 头
    """

    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        if d_model % num_q_heads != 0:
            raise ValueError("d_model 必须能被 num_q_heads 整除")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads 必须能被 num_kv_heads 整除")

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.q_per_kv = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        bsz, tgt_len, _ = x.shape

        q = self.q_proj(x).view(bsz, tgt_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, tgt_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, tgt_len, self.num_kv_heads, self.head_dim)

        # 转为 [B, H, T, Hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA 核心：把 KV 头重复到 Q 头数量
        # 例如 Q=4, KV=2，则每个 KV 头重复 2 次 -> [B, 4, T, Hd]
        k = k.repeat_interleave(self.q_per_kv, dim=1)
        v = v.repeat_interleave(self.q_per_kv, dim=1)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, Hq, T, T]
        scores = scores + self.causal_mask[:tgt_len, :tgt_len]
        attn = torch.softmax(scores, dim=-1)

        context = attn @ v  # [B, Hq, T, Hd]
        context = context.transpose(1, 2).contiguous().view(
            bsz, tgt_len, self.num_q_heads * self.head_dim
        )
        return self.out_proj(context)


if __name__ == "__main__":
    x = torch.randn(batch_size, seq_len, d_model)
    gqa = GQAManual(
        d_model=d_model,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )
    y = gqa(x)

    print("input shape :", x.shape)   # [B, T, D]
    print("output shape:", y.shape)   # [B, T, D]
    print("output sample:\n", y[0, :2, :4])
