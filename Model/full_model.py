# Integrated Pipeline for BPE Tokenizer and Transformer Training

import os
import re
import emoji
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

#########################
# DATA LOADING & PREPROCESSING
#########################

# Read raw text files for L1, L2, L3, and L4
with open("../Data/L1.txt", "r", encoding="utf-8") as f:
    L1 = f.read()
with open("../Data/L2.txt", "r", encoding="utf-8") as f:
    L2 = f.read()
with open("../Data/L3.txt", "r", encoding="utf-8") as f:
    L3 = f.read()
with open("../Data/L4.txt", "r", encoding="utf-8") as f:
    L4 = f.read()

# For L4, perform cleaning to keep only English, math symbols, special characters, and emojis
english_regex = r"[a-zA-Z0-9\s]"
math_symbols = r"[\+\-\*/=<>∑∫√πθΣ∂∞]"
special_chars = r"[\.,!?;:'\"()\[\]{}#@%^&*_~]"

def keep_emojis(text):
    return "".join(c for c in text if emoji.is_emoji(c))

L4_cleaned = "".join(
    c for c in L4 if re.match(english_regex, c) or 
                    re.match(math_symbols, c) or 
                    re.match(special_chars, c) or 
                    emoji.is_emoji(c)
)

# Combine texts from all levels into one unified text
full_text = L1 + L2 + L3 + L4_cleaned

#########################
# CHARACTER-LEVEL TOKENIZATION (for initial vocab)
#########################

chars_full = sorted(list(set(full_text)))
vocab_size_full = len(chars_full)
stoi_full = { ch: i for i, ch in enumerate(chars_full) }
itos_full = { i: ch for i, ch in enumerate(chars_full) }

def encode_full(s):
    return [stoi_full[c] for c in s]

def decode_full(l):
    return "".join([itos_full[i] for i in l])

#########################
# BPE TOKENIZER TRAINING
#########################

# Convert full_text to bytes (UTF-8 encoding)
token = full_text.encode("utf-8")
token_list = list(token)

def get_stats(ids):
    """Return frequency counts of adjacent token pairs."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """In the list of token IDs, replace all consecutive occurrences of pair with new token idx."""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# Set desired vocabulary size (e.g., final vocab size = 276)
base_vocab = 256  # starting with 256 byte tokens
final_vocab_size = 276
num_merges = final_vocab_size - base_vocab
ids = token_list[:]  # working copy of token_list
merges = {}  # Dictionary to hold merge rules

for i in range(num_merges):
    stats = get_stats(ids)
    if not stats:
        break
    pair = max(stats, key=stats.get)
    idx = base_vocab + i
    print(f"Merging {pair} into new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

# Rebuild vocabulary: For each new token, combine the corresponding byte sequences
vocab = {i: bytes([i]) for i in range(base_vocab)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode_ids(ids):
    """Decode list of token IDs back into a UTF-8 string."""
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors="replace")

def encode_text(text):
    """Encode a given string using the learned BPE merges."""
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # Choose the pair with the smallest merge index (following training order)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

#########################
# TRANSFORMER MODEL DEFINITION (GPT-style)
#########################

# Hyperparameters for the transformer
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Prepare data for transformer: encode full_text using BPE
encoded_data = encode_text(full_text)
data_tensor = torch.tensor(encoded_data, dtype=torch.long)

# Train / validation split
n_train = int(0.9 * len(data_tensor))
train_data = data_tensor[:n_train]
val_data = data_tensor[n_train:]

def get_batch_transformer(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss_transformer(model):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for split in ['train', 'val']:
        for k in range(eval_iters):
            X, Y = get_batch_transformer(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Define supporting modules for the transformer (Head, MultiHeadAttention, FeedForward, Block)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
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

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ----------- Transformer Training Loop ------------
def train_transformer():
    model = GPTLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss_transformer(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch_transformer('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# ----------- Usage Example ------------
# Train the transformer model
transformer_model = train_transformer()

# Generate sample text using the BPE decoder
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = transformer_model.generate(context, max_new_tokens=500)[0].tolist()
print(decode_ids(generated_ids))
