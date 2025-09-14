# train_toy_gpt.py
# A super tiny GPT model training loop for fun and learning
# Adapted from Karpathy's minGPT and nanoGPT

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Load dataset
with open("C:/Users/c08/alderamin_v1-3.txt", encoding="utf-8") as f:
    text = f.read()

# #Model learn with char-level
# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# # mapping char <-> int
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }

# def encode(s): return [stoi[c] for c in s]
# def decode(l): return ''.join([itos[i] for i in l])

# Convert to word-level tokens
words = text.split()  # split by whitespace
vocab = sorted(set(words))
vocab_size = len(vocab)

# Dictionaries for encoding/decoding
stoi = {w:i for i,w in enumerate(vocab)}
itos = {i:w for w,i in stoi.items()}

def encode(s):
    return [stoi[w] for w in s.split() if w in stoi]

def decode(l):
    return " ".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# train/test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
block_size = 512     # can “see” longer context
batch_size = 32
n_embd = 512        # larger embeddings
n_head = 8
n_layer = 6
max_iters = 1000    # longer training, adjust if overfitting occur
eval_interval = 200
learning_rate = 3e-4
grad_clip = 1.0       # Gradient clipping to stabilize training
dropout = 0.1         # Optional: helps prevent overfitting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# 3. Data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 4. Model definition
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 5. Initialize
model = GPT()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# 6. Training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = []
        for split in ['train','val']:
            X,Y = get_batch(split)
            with torch.no_grad():
                logits, loss = m(X, Y)
            losses.append(loss.item())
        print(f"step {iter}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 7. Generate some text
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
