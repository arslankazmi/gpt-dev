import torch
from transformers import GPT2LMHeadModel
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

# This is a minimal version of the GPT model to check keys
class MiniGPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        from torch.nn import Embedding, Linear, ModuleList
        
        self.transformer = torch.nn.ModuleDict(dict(
            wte=Embedding(config.vocab_size, config.n_embed),
            wpe=Embedding(config.block_size, config.n_embed),
            h=ModuleList([torch.nn.Identity() for _ in range(config.n_layer)]),
            ln_f=torch.nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = Linear(config.n_embed, config.vocab_size, bias=False)

config = GPTConfig()

# Load HF model
model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
sd_hf = model_hf.state_dict()

sd_keys_hf = sd_hf.keys()
sd_keys_hf = [ k for k in sd_keys_hf if not k.endswith('attn.masked_bias') ]
sd_keys_hf = [ k for k in sd_keys_hf if not k.endswith('attn.bias') ]

print(f"Total HF keys (after filtering): {len(sd_keys_hf)}")
print("\nHF Keys:")
for k in sorted(sd_keys_hf)[:20]:
    print(f"  {k}")

# Show filtered keys
sd_keys_hf_filtered = []
for k in sd_keys_hf:
    if k.startswith('transformer.'):
        sd_keys_hf_filtered.append(k.replace('transformer.', '', 1))
    elif not k.startswith('lm_head.'):
        sd_keys_hf_filtered.append(k)

print(f"\nFiltered HF keys: {len(sd_keys_hf_filtered)}")
print("\nFiltered Keys:")
for k in sorted(sd_keys_hf_filtered)[:20]:
    print(f"  {k}")

# Check what's different
print("\nKeys only in HF (after filtering):")
for k in sorted(sd_keys_hf_filtered):
    if not any(k.startswith(prefix) for prefix in ['h.0', 'h.1', 'h.2']):
        print(f"  {k}")
