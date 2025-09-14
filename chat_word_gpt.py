# chat_word_gpt.py
import torch
import torch.nn.functional as F
import pickle

# 1. Load vocab
with open("C:/Users/c08/vocab_word.pkl", "rb") as f:
    vocab = pickle.load(f)

stoi = {w:i for i,w in enumerate(vocab)}
itos = {i:w for i,w in enumerate(vocab)}
vocab_size = len(vocab)

# 2. Hyperparameters (must match training)
block_size = 128
n_embd = 512
n_head = 8
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 3. Encode/decode functions
def encode(s):
    return [stoi[w] for w in s.split() if w in stoi]

def decode(l):
    return " ".join([itos[i] for i in l])

# 4. GPT model definition (same as in training)
import torch.nn as nn

class Head(nn.Module):
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
        self.proj = nn.Linear(num_heads*head_size, n_embd)
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
            nn.Linear(4*n_embd, n_embd)
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

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 5. Load trained model
model = GPT().to(device)
model.load_state_dict(torch.load("C:/Users/c08/model_word.pth", map_location=device))
model.eval()

print("Word-level GPT Chat ready! Type 'quit' to exit.")

# 6. Chat loop with context
history = []

while True:
    prompt = input("You: ")
    if prompt.lower() == 'quit':
        break

    # encode prompt safely
    tokens = [stoi[w] for w in prompt.split() if w in stoi]
    if len(tokens) == 0:
        print("GPT: Sorry, I don’t know any of these words.")
        continue

    # append to history
    history += tokens
    input_ids = torch.tensor([history[-block_size:]], device=device, dtype=torch.long)

    # generate response
    output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)

    # decode only the newly generated tokens
    response = decode(output_ids[0].tolist()[len(history):])

    print("GPT:", response)

    # append model's response to history
    history += encode(response)