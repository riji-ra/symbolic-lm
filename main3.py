import torch
import torch.nn as nn
import tqdm
import numpy as np

class FourierScaledDotProductAttention(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.d_k = size
        self.q_imag = nn.Parameter(torch.zeros(size))
        self.q_real = nn.Parameter(torch.ones(size))
        self.k_imag = nn.Parameter(torch.zeros(size))
        self.k_real = nn.Parameter(torch.ones(size))
        self.v_imag = nn.Parameter(torch.zeros(size))
        self.v_real = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor):
        q_weight = self.q_imag*1j + self.q_real
        k_weight = self.k_imag*1j + self.k_real
        v_weight = self.v_imag*1j + self.v_real

        q = torch.fft.fft(x+0j)
        q = q * q_weight.unsqueeze(0).unsqueeze(0)
        q = torch.fft.ifft(q).real

        k = torch.fft.fft(x+0j)
        k = k * k_weight.unsqueeze(0).unsqueeze(0)
        k = torch.fft.ifft(k).real

        v = torch.fft.fft(x+0j)
        v = v * v_weight.unsqueeze(0).unsqueeze(0)
        v = torch.fft.ifft(v).real

        scalar = np.sqrt(self.d_k)
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar # 「Q * X^T / (D^0.5)」" を計算

        attention_weight = nn.functional.softmax(attention_weight, dim=2) # Attention weightを計算
        return torch.matmul(attention_weight, v) # (Attention weight) * X により重み付け.


class FourierForwardNetwork(torch.nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.a_imag = nn.Parameter(torch.zeros(size))
        self.a_real = nn.Parameter(torch.ones(size))
        self.b_imag = nn.Parameter(torch.zeros(size))
        self.b_real = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
    def forward(self, x: torch.Tensor):
        a = self.a_imag*1j + self.a_real
        b = self.b_imag*1j + self.b_real

        X = torch.fft.fft(x+0j)
        X = X * a.unsqueeze(0).unsqueeze(0)
        X = torch.fft.ifft(X).real

        X = X + self.bias.unsqueeze(0).unsqueeze(0)
        X = torch.nn.functional.gelu(X)
        ## X = torch.nn.SiLU(X)
        ## X = torch.sin(X)**2 + X

        X = torch.fft.fft(x+0j)
        X = X * b.unsqueeze(0).unsqueeze(0)
        X = torch.fft.ifft(X).real
        return X

class FourierFourierAttention(torch.nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.q_imag = nn.Parameter(torch.zeros(size))
        self.q_real = nn.Parameter(torch.ones(size))
        self.k_imag = nn.Parameter(torch.zeros(size))
        self.k_real = nn.Parameter(torch.ones(size))
        self.v_imag = nn.Parameter(torch.zeros(size))
        self.v_real = nn.Parameter(torch.ones(size))
        self.rt_imag = nn.Parameter(torch.zeros(size))
        self.rt_real = nn.Parameter(torch.ones(size))
    def forward(self, x: torch.Tensor):
        q_weight = self.q_imag*1j + self.q_real
        k_weight = self.k_imag*1j + self.k_real
        v_weight = self.v_imag*1j + self.v_real
        rt_weight = self.rt_imag*1j + self.rt_real

        q = torch.fft.fft(x+0j)
        q = q * q_weight.unsqueeze(0).unsqueeze(0)
        q = torch.fft.fft(q, dim=-2)

        k = torch.fft.fft(x+0j)
        k = k * k_weight.unsqueeze(0).unsqueeze(0)
        k = torch.fft.fft(k, dim=-2)

        v = torch.fft.fft(x+0j)
        v = v * v_weight.unsqueeze(0).unsqueeze(0)
        v = torch.fft.fft(v, dim=-2)

        rt = q / (k + 1e-12) * v
        rt = torch.fft.ifft(rt, dim=-2) * rt_weight.unsqueeze(0).unsqueeze(0)
        rt = torch.fft.ifft(rt)
        rt = rt.real
        return rt

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class FourierFFNBlock(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.ffn = FourierForwardNetwork(size)
        self.dyt = DyT(size)

    def forward(self, x):
        return self.ffn(self.dyt(x)) + x

class FourierScaledDotProductAttentionBlock(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.ffn = FourierScaledDotProductAttention(size)
        self.dyt = DyT(size)

    def forward(self, x):
        return self.ffn(self.dyt(x)) + x

class FourierFourierAttentionBlock(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.ffn = FourierFourierAttention(size)
        self.dyt = DyT(size)

    def forward(self, x):
        return self.ffn(self.dyt(x)) + x

class FourierTransformerLayer(nn.Module):
    def __init__(self, size=128, attn_type="scaled"):
        super().__init__()
        self.attn_a = FourierFourierAttentionBlock(size)
        self.attn_b = FourierScaledDotProductAttentionBlock(size)
        self.ffn = FourierFFNBlock(size)

    def forward(self, x):
        x = self.attn_a(x)
        x = self.attn_b(x)
        x = self.ffn(x)
        return x

class FourierTransformerEncoder(nn.Module):
    def __init__(self, size=128, depth=6,):
        super().__init__()
        self.layers = nn.ModuleList([
            FourierTransformerLayer(size)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FourierTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        size=128,
        depth=6
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, size)
        self.encoder = FourierTransformerEncoder(
            size=size,
            depth=depth,
        )
        self.output = nn.Linear(size, vocab_size)

    def forward(self, x):
        # x: (batch, seq)
        x = self.embedding(x)
        x = self.encoder(x)
        return self.output(x)

A = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"+":10,"-":11,"*":12,"/":13,"=":14, "_":15, "'":16}
def generatetekito():
    S = int(np.random.randint(1, 2 ** np.random.randint(4, 32)))
    S2 = int(np.random.randint(1, 2 ** np.random.randint(4, 32)))
    S3 = int(np.random.randint(1, 2 ** np.random.randint(4, 32)))
    enz = np.random.randint(0, 13)
    if enz == 0:  return f"{S}+{S2}={S+S2}"
    if enz == 1:  return f"{S}-{S2}={S-S2}"
    if enz == 2:  return f"{S}*{S2}={S*S2}"
    if enz == 3:  return f"{S}/{S2}={S//S2}"
    if enz == 4:  return f"{S}+{S2}+{S3}={S+S2+S3}"
    if enz == 5:  return f"{S}+{S2}-{S3}={S+S2-S3}"
    if enz == 6:  return f"{S}+{S2}*{S3}={S+S2*S3}"
    if enz == 7:  return f"{S}*{S2}+{S3}={S*S2+S3}"
    if enz == 8:  return f"{S}*{S2}-{S3}={S*S2-S3}"
    if enz == 9:  return f"{S}-{S2}*{S3}={S-S2*S3}"
    if enz == 10: return f"{S}*{S2}/{S3}={int(S*S2/S3)}"
    if enz == 11: return f"{S}-{S2}/{S3}={S - S2//S3}"
    return f"{S}/{S2}*{S3}={int(S/S2*S3)}"

def gendata(g, tt, lt):
    at = [A[_] for _ in g]
    at_ = [A[_] for _ in g]
    s = ((len(at) - tt) / len(at))
    g = np.random.choice(lt, lt, replace=False)
    for i in range(tt):
        at[g[i]] = 16
    return at, at_

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

VOCAB_SIZE = 17
SEQ_LEN = 64
BATCH = 128
EPOCHS = 50
LR = 3e-4
MAX_NOISE = 64

def make_batch(batch_size=BATCH, max_noise=MAX_NOISE):
    xs = []
    ys = []

    for _ in range(batch_size):
        g = generatetekito()
        g2 = g[:SEQ_LEN].ljust(SEQ_LEN, "_")  # padding
        noise = np.random.randint(1, len(g))

        x_noised, x_clean = gendata(g2, noise, len(g))
        x_noised[len(g):] = np.ones(SEQ_LEN-len(g)) * 15
        x_clean[len(g):] = np.ones(SEQ_LEN-len(g)) * 15

        xs.append(x_noised)
        ys.append(x_clean)

    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )

model = FourierTransformer(
    vocab_size=VOCAB_SIZE,
    size=256,
    depth=8
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

idhk = 0.0
model.train()
for step in range(300):
    x, y = make_batch()

    logits = model(x)  # (B, T, V)
    loss = criterion(
        logits.view(-1, VOCAB_SIZE),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(step == 0):
        idhk = float(loss)

    print(f"{step}, {loss:.4f}")

def decode(tokens):
    inv = {v:k for k,v in A.items()}
    return "".join(inv[int(t)] for t in tokens)

for j in range(128):
    model.eval()
    x, y = make_batch(1, max_noise=8)
    with torch.no_grad():
        out = model(x).argmax(dim=-1)

    print("input :", decode(x[0]))
    print("target:", decode(y[0]))
    print("pred  :", decode(out[0]))