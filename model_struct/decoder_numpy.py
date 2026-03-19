import numpy as np
np.random.seed(42) 
vocab_size = 100
d_model = 8
seq_len = 4
emb_table = np.array([    [i * 0.1 + j * 0.01 for j in range(d_model)]    for i in range(vocab_size)], dtype=float)
tokens = np.array([2,5,1,3])
tokens_emb = emb_table[tokens]
pe = np.zeros((seq_len, d_model))
for pos in range(seq_len):
    for i in range(0, d_model, 2):
        tmp = np.divide(pos,np.pow(10000, (2*i)/d_model))
        pe[pos, i] =  np.sin(tmp)
        pe[pos, i + 1] = np.cos(tmp)
x = tokens_emb + pe 

np.random.seed(0)
W_Q = np.array([    [(i+1) * 0.1 - j * 0.01 for j in range(d_model)]    for i in range(d_model)], dtype=float)
W_K = np.array([    [(i+1) * 0.05 + j * 0.01 for j in range(d_model)]    for i in range(d_model)], dtype=float)
W_V = np.array([    [(i+1) * 0.02 - j * 0.005 for j in range(d_model)]    for i in range(d_model)], dtype=float)
Q = x @ W_Q
K = x @ W_K 
V = x @ W_V

def softmax(x):
    result = np.zeros_like(x)
    for x_i in range(len(x)):
        exp_row = np.exp(x[x_i])
        result[x_i] = exp_row / np.sum(exp_row)
    return result

def self_attention(Q,K,V,d_model,mask):
    d_k = d_model  # 缩放因子的分母
    scores = Q @ K.T
    scores_scaled =  scores / np.sqrt(d_k)
    attn_weights = softmax(scores_scaled + mask)  # shape: (4, 4)
    return attn_weights @ V

mask = np.triu(np.full((seq_len,seq_len),-np.inf),k=1)
output = self_attention(Q,K,V,d_model,mask)
print("output shape:", output.shape)
print("output:\n", np.round(output, 4))

epsilon = 1e-6
gamma = np.ones(d_model)   # shape: (d_model,)
beta = np.zeros(d_model)   # shape: (d_model,)

def layer_norm(x, gamma, beta, epsilon):
    # x shape: (seq_len, d_model)
    # 对每一行求均值和标准差，注意 axis 和 keepdims
    mean = np.mean(x,axis=1,keepdims=1)   # shape: (seq_len, 1)
    std  = np.std(x,axis=1,keepdims=1)   # shape: (seq_len, 1)
    x_norm = np.divide(x-mean,(epsilon + std)) # 归一化
    return x_norm*gamma+beta   # 乘以 gamma 加 beta

# Add & Norm
x_residual = x + output                        # Add
x_norm = layer_norm(x_residual, gamma, beta, epsilon)  # Norm

print("x_norm shape:", x_norm.shape)
print("每行均值:", np.round(np.mean(x_norm, axis=1), 4))  # 应该接近全 0
print("每行标准差:", np.round(np.std(x_norm, axis=1), 4))  # 应该接近全 1

def relu(x):
     return np.maximum(np.zeros_like(x),x)  # 小于 0 的置 0，提示：np.maximum(0, x)

def ffn(x, W1, b1, W2, b2):
    hidden = x @ W1 + b1  # 第一层线性变换 + bias，shape: (4, 32)
    hidden = relu(hidden)  # ReLU 激活
    out    = hidden @ W2 + b2  # 第二层线性变换 + bias，shape: (4, 8)
    return out
d_ff = 4 * d_model  # 32

W1 = np.array([
    [(i+1)*0.01 + j*0.005 for j in range(d_ff)]
    for i in range(d_model)
], dtype=float)  # shape: (d_model, d_ff) = (8, 32)

b1 = np.zeros(d_ff)   # shape: (d_ff,)

W2 = np.array([
    [(i+1)*0.01 - j*0.005 for j in range(d_model)]
    for i in range(d_ff)
], dtype=float)  # shape: (d_ff, d_model) = (32, 8)

b2 = np.zeros(d_model)  # shape: (d_model,)
x_ffn = ffn(x_norm,W1,b1,W2,b2)
x_residual2 = x_ffn + x_norm
final_output = layer_norm(x_residual2,gamma,beta,epsilon)
print("final_output shape:", final_output.shape)
print("final_output:\n", np.round(final_output, 4))


