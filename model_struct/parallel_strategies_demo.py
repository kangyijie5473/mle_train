import torch
from torch import nn


torch.manual_seed(0)


class TinyMLP(nn.Module):
    """最简单的两层前馈网络：Linear -> ReLU -> Linear"""

    def __init__(self, d_model: int = 8, d_hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


class TPVariant(nn.Module):
    """
    TP (Tensor Parallel) 变体（2-way）：
    - fc1 做列切分（输出 hidden 切两半）
    - fc2 做行切分（输入 hidden 切两半）后求和
    """

    def __init__(self, base: TinyMLP):
        super().__init__()
        d_model = base.fc1.in_features
        d_hidden = base.fc1.out_features
        if d_hidden % 2 != 0:
            raise ValueError("示例要求 d_hidden 可以被 2 整除")

        mid = d_hidden // 2
        self.w1_0 = nn.Parameter(base.fc1.weight[:mid, :].detach().clone())
        self.b1_0 = nn.Parameter(base.fc1.bias[:mid].detach().clone())
        self.w1_1 = nn.Parameter(base.fc1.weight[mid:, :].detach().clone())
        self.b1_1 = nn.Parameter(base.fc1.bias[mid:].detach().clone())

        # fc2: [D_out, D_hidden]，沿 D_hidden 切成两块
        self.w2_0 = nn.Parameter(base.fc2.weight[:, :mid].detach().clone())
        self.w2_1 = nn.Parameter(base.fc2.weight[:, mid:].detach().clone())
        self.b2 = nn.Parameter(base.fc2.bias.detach().clone())

        self.d_model = d_model
        self.d_hidden = d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h0 = torch.relu(torch.matmul(x, self.w1_0.t()) + self.b1_0)
        h1 = torch.relu(torch.matmul(x, self.w1_1.t()) + self.b1_1)

        # row parallel reduce-sum
        y0 = torch.matmul(h0, self.w2_0.t())
        y1 = torch.matmul(h1, self.w2_1.t())
        return y0 + y1 + self.b2


class SPVariant(nn.Module):
    """
    SP (Sequence Parallel) 变体：
    - 按序列维 T 切分，每段独立过同一个 MLP，再拼接
    """

    def __init__(self, base: TinyMLP, seq_chunks: int = 2):
        super().__init__()
        self.model = base
        self.seq_chunks = seq_chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        parts = torch.chunk(x, self.seq_chunks, dim=1)
        outs = [self.model(p) for p in parts]
        return torch.cat(outs, dim=1)


class PPVariant(nn.Module):
    """
    PP (Pipeline Parallel) 变体：
    - stage1: fc1 + relu
    - stage2: fc2
    - 通过 micro-batch 在 batch 维做流水
    """

    def __init__(self, base: TinyMLP, micro_batches: int = 2):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Linear(base.fc1.in_features, base.fc1.out_features),
            nn.ReLU(),
        )
        self.stage2 = nn.Linear(base.fc2.in_features, base.fc2.out_features)
        self.micro_batches = micro_batches

        # 复制权重，确保与基线可对齐
        with torch.no_grad():
            self.stage1[0].weight.copy_(base.fc1.weight)
            self.stage1[0].bias.copy_(base.fc1.bias)
            self.stage2.weight.copy_(base.fc2.weight)
            self.stage2.bias.copy_(base.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]，按 B 维切 micro-batch
        micro_inputs = torch.chunk(x, self.micro_batches, dim=0)
        s1_buffer = []
        outputs = []

        # 简化版 fill-drain 演示
        for micro in micro_inputs:
            h = self.stage1(micro)
            s1_buffer.append(h)
            out = self.stage2(s1_buffer.pop(0))
            outputs.append(out)

        return torch.cat(outputs, dim=0)


def main():
    batch, seq_len, d_model, d_hidden = 4, 6, 8, 16
    x = torch.randn(batch, seq_len, d_model)

    base = TinyMLP(d_model=d_model, d_hidden=d_hidden)
    tp = TPVariant(base)
    sp = SPVariant(base, seq_chunks=2)
    pp = PPVariant(base, micro_batches=2)

    y_base = base(x)
    y_tp = tp(x)
    y_sp = sp(x)
    y_pp = pp(x)

    print("base shape:", tuple(y_base.shape))
    print("TP   shape:", tuple(y_tp.shape))
    print("SP   shape:", tuple(y_sp.shape))
    print("PP   shape:", tuple(y_pp.shape))

    print("max|base-TP|:", (y_base - y_tp).abs().max().item())
    print("max|base-SP|:", (y_base - y_sp).abs().max().item())
    print("max|base-PP|:", (y_base - y_pp).abs().max().item())


if __name__ == "__main__":
    main()
