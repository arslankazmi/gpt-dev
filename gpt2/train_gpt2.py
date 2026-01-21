from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # not really a bias but more of a mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        

    def forward(self, x):

        B, T, C = x.size()

        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embed, dim=2) # each is (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = ( q @ k.transpose(-2, -1) ) * ( 1.0 / math.sqrt(k.size(-1)) ) 
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # avengers re-assemble

        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config) # analogous to FeedForward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # x + provides a residual connection stream
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # tokens = 50K BPE merges + 256 bytes tokens + 1 for end of text delineation
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # wte - weights of token embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            # wpe - weights of position embeddings
            wpe = nn.Embedding(config. block_size, config.n_embed),
            # h for hidden layers
            h = nn.ModuleList([ Block(config) for _ in range(config.n_layer) ]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx):
        # input shape is (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # token and position embeddings
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T,)
        position_embeddings = self.transformer.wpe(positions) # (T, n_embed)
        # we use idx here as the token values represent indices in the vocabulary array
        token_embeddings = self.transformer.wte(idx) # (B, T, n_embed

        x = token_embeddings + position_embeddings # (B, T, n_embed)
        for block in self.transformer.h:
            x = block(x)

        # final layernorm and lm head
        x = self.transformer.ln_f(x) # (B, T, n_embed)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Load a GPT model from the Hugging Face transformers library."""

        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], "Unsupported model type"
        from transformers import GPT2LMHeadModel

        print(f"Loading pretrained model {model_type} from Hugging Face...")
        
        config_args = {
            "gpt2" : dict(n_layer=12, n_head=12, n_embed=768),
            "gpt2-medium" : dict(n_layer=24, n_head=16, n_embed=1024),
            "gpt2-large" : dict(n_layer=36, n_head=20, n_embed=1280),
            "gpt2-xl" : dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()

        sd_keys = sd.keys()

        sd_keys = [ k for k in sd_keys if not k.endswith('attn.bias') ]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        sd_hf = model_hf.state_dict()

        # now copy over all tensors to our own state dict
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [ k for k in sd_keys_hf if not k.endswith('attn.masked_bias') ]
        sd_keys_hf = [ k for k in sd_keys_hf if not k.endswith('attn.bias') ]
        
        # Create mapping from our keys to HF keys
        # Both have same structure: transformer.h.*, transformer.wte, transformer.wpe, transformer.ln_f, lm_head.*
        hf_key_map = {}  # Map from our key names to HF key names
        
        for k in sd_keys_hf:
            hf_key_map[k] = k
        
        # these weights are stored transposed in HF transformers
        transposed = [ 'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"State dict keys length mismatch: {len(sd_keys)} vs {len(sd_keys_hf)}"

        for k_our in sd_keys:
            k_hf = hf_key_map.get(k_our)
            if k_hf is None:
                print(f"Warning: Key {k_our} not found in HF model")
                continue
                continue
            
            if any(k_our.endswith(w) for w in transposed):
                assert sd_hf[k_hf].shape[::-1] == sd[k_our].shape, f"Shape mismatch for {k_our}: {sd_hf[k_hf].shape[::-1]} vs {sd[k_our].shape}"
                with torch.no_grad():
                    sd[k_our].copy_(sd_hf[k_hf].t()) # transposition
            else:
                assert sd_hf[k_hf].shape == sd[k_our].shape, f"Shape mismatch for {k_our}: {sd_hf[k_hf].shape} vs {sd[k_our].shape}"
                with torch.no_grad():
                    sd[k_our].copy_(sd_hf[k_hf])

        return model
    

# -------------------
num_return_sequences = 3
max_new_tokens = 30

model = GPT.from_pretrained('gpt2')
print("Model loaded successfully.")

model.eval()
model.to('mps')

import tiktoken
enc = tiktoken.get_encoding("gpt2")

tokens = enc.encode("I am a language model, and ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (num_return_sequences, sequence_length)
x = tokens.to('mps')

# generation loop

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.mps.manual_seed(42)

while x.size(1) < max_new_tokens:
    logits = model(x) # (B, T, vocab_size)
    logits = logits[:, -1, :] # (B, vocab_size) taking logits at the lat position
    probs = F.softmax(logits, dim=-1)

    # huggingface does top-k sampling by default
    top_k = 50
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

    # select 1 token from top k

    ix = torch.multinomial(top_k_probs, 1)

    xcol = torch.gather(top_k_indices, -1, ix)

    # i choose to print as we go in a streaming fashion
    # so decode xcol to string and print
    decoded = enc.decode(xcol.squeeze().cpu().numpy().tolist())
    print(decoded, end='', flush=True)

    # append to sequence
    x = torch.cat((x, xcol), dim=1)

