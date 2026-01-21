import torch
import torch.nn as nn
from dataclasses import dataclass
import sys
sys.path.insert(0, '/Users/akazmi/Documents/arslan/gpt-dev/gpt2')

# Minimal setup to check keys
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.ln_2 = nn.LayerNorm(config.n_embed)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([ Block(config) for _ in range(config.n_layer) ]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

config = GPTConfig()
model = GPT(config)

print("Our model keys:")
for k in sorted(model.state_dict().keys())[:10]:
    print(f"  {k}")
