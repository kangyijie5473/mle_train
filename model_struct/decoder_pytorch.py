import torch
from torch import nn
vocab_size = 100
d_model = 8
seq_len = 4
d_ff = 4 * d_model  # 32

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__() 
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.line1 = nn.Linear(d_model,d_ff,bias=True)
        self.line2 = nn.Linear(d_ff,d_model,bias=True)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.register_buffer("causal_mask", torch.triu(torch.full((seq_len,seq_len), float('-inf')),diagonal=1))
    def attention(self,x):
        Q = self.q_proj(x)
        K = self.k_proj(x) 
        V = self.v_proj(x)
        d_k = Q.size(-1)  # 缩放因子的分母
        scores = Q @ K.transpose(-2,-1)
        scores_scaled =  scores / d_k ** 0.5
        attn_weights = torch.softmax(scores_scaled + self.causal_mask,dim=-1)  # shape: (4, 4)
        return attn_weights @ V
    def forward(self,x):
        attention_scores = self.attention(x)
        attention_scores =self.layernorm1(attention_scores + x)
        output = torch.relu(self.line1(attention_scores))
        output = self.line2(output)
        return self.layernorm2(output + attention_scores)
