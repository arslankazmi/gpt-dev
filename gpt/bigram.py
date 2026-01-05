import math
import os
import torch

# bigram.py
# An original, minimal bigram language model implementation in PyTorch.

import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(1337)

# Device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Hyperparameters
batch_size = 64       # number of sequences per batch
block_size = 32       # sequence length for training (context size); bigram model won't use >1
max_iters = 5000      # training steps
eval_interval = 500   # print loss every this many steps
eval_iters = 200      # steps to average eval loss
learning_rate = 3e-4  # AdamW LR
generate_tokens = 1000

# Data loading: expects an input.txt in the same directory; falls back to a tiny snippet
def load_text(path="bionicletext_cleaned.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # fallback text (very small)
    return (
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
    )

text = load_text()

# Build character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(ix):
    return "".join(itos[i] for i in ix)

# Encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    source = train_data if split == "train" else val_data
    if len(source) <= block_size + 1:
        raise RuntimeError("Dataset too small for the configured block_size.")
    ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Each token directly predicts the logits for the next token
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T), output logits: (B, T, C)
        logits = self.token_embedding(idx)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        # idx: (B, T) of indices; generates by sampling autoregressively
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # Only the last time step is used to sample the next token
            next_logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

def main():
    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}, Data length: {len(data)} chars")

    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(1, max_iters + 1):
        if step == 1 or step % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start token index 0
    out_idx = model.generate(context, max_new_tokens=generate_tokens)[0].tolist()
    print("\n=== Sample ===\n")
    print(decode(out_idx))

if __name__ == "__main__":
    main()