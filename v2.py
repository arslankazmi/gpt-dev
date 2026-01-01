import math
import os
import torch

# v2 of the bogram language model

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Hyperparamters

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
generate_tokens = 1000
n_embed = 32

# load data

def load_text(path="bionicletext_cleaned.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
        

text = load_text()

# char vocab

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(int):
    return "".join(itos[i] for i in int)


# encode entire input text as torch tensor

data = torch.tensor(encode(text), dtype=torch.long)

# Splits

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    source = train_data if split == 'train' else val_data
    if len(source) <= block_size + 1:
        raise RuntimeError("Dataset too small for configured block size.")
    
    random_indexes = torch.randint(0, len(source) - block_size - 1, (batch_size, ))
    x = torch.stack( [ source[i:i + block_size ] for i in random_indexes ] )
    y = torch.stack( [ source[i + 1:i + block_size + 1] for i in random_indexes ] )
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

class Head(nn.Module): 
    """One Self-Attention Head"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute scores or affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B , T, T)

        # perform the weighted average aggregation of the values
        y = self.value(x) # (B, T, C)
        out = wei @ y # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple self-attention headds in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([ Head(head_size) for _ in range(num_heads) ])
        self.projection = nn.Linear(num_heads * head_size, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate across that least dimension, pretty standard
        out = self.projection(out) # projection is just a linear transformation of the output of the previous layer (the heads)
        return out
    
class FeedForward(nn.Module):
    """Simple linear layer followed by enforced non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # 4x optimization from paper "Attentionis all you need", where they make the inner layer dimensionality 4x the outer layers
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            # nn.Linear(4 * n_embed, n_embed),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Block of transformer which intersperses communication and computation, in that order"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.layer_norm1(x)) # residual connections
        x = x + self.feed_forward(self.layer_norm2(x)) # residual connection
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.sa_head = Head(n_embed) #- replaced with multiple heads
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),

        )
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B,T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embeddings + positional_embeddings # (B, T, C)
        # x = self.sa_head(x) # apply only one head of self-attention
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, stream=False):

        for _ in range(max_new_tokens):

            # crop idx to the last block_szie tokens for positonal embeddings
            idx_cond = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx_cond)

            # only last time step used
            next_logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, next_idx], dim=1)

            if stream:
                yield next_idx
        
        if not stream:
            return idx
    
def main():
    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}, Data Length: {len(data)} chars")

    model = BigramLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(1, max_iters + 1):
        if step ==1 or step % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Step {step:5d} | train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f}")

        
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    
    #out_idx = model.generate(context, max_new_tokens=generate_tokens)[0].tolist()

    print("\n===== Sample =====\n")
    #print(decode(out_idx))

    for idx in model.generate(context, max_new_tokens=generate_tokens, stream=True):
        print(decode(idx[0].tolist()), end='', flush=True)


if __name__ == "__main__":
    main()