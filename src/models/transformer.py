"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache


@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)
    


    def forward(self, sequences: torch.Tensor, past_keys_values= None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)

        for i, block in enumerate(self.blocks):
            x = block(x,return_past= False, past_keys_values= None if past_keys_values is None else past_keys_values[i])

        x = self.ln_f(x)
        return x
    

    def forward_with_past(self, sequences: torch.Tensor, past_keys_values= None) -> torch.Tensor:
        if past_keys_values is not None: 
            past_keys_values= torch.cat(past_keys_values, dim=-2) 
            print("size of last past key", past_keys_values[-1].size())
            assert len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        presents=[]
        for i, block in enumerate(self.blocks):
            x, present = block(x, return_past= True, past_keys_values= None if past_keys_values is None else past_keys_values[i])
            presents.append(present)

        x = self.ln_f(x)
        return x, torch.stack(presents)


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, return_past= False, past_keys_values= None) -> torch.Tensor:
        if return_past is False: 
            x_attn= self.attn(self.ln1(x),  return_past= False, past_keys_values= past_keys_values)
        else: 
            x_attn, present= self.attn(self.ln1(x), return_past= True, past_keys_values= past_keys_values)
            

        x = x + x_attn
        x = x + self.mlp(self.ln2(x))

        if return_past is False:
            return x 
        else: 
            return x, present


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        assert config.attention in ('causal', 'block_causal')
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        self.register_buffer('mask', causal_mask if config.attention == 'causal' else block_causal_mask)

    def forward(self, x: torch.Tensor, return_past= False, past_keys_values= None) -> torch.Tensor:
        B, T, C = x.size()
        if past_keys_values is not None:
            _, b, nh, L, c =past_keys_values.size()
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

        present=torch.stack((k,v))


        if past_keys_values is not None:
            past_key, past_value = past_keys_values
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if past_keys_values is None:
            att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf')) # check how masks look like 
        # att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')

        y = self.resid_drop(self.proj(y))

        if return_past is False: 
            return y
        else: 
            print("present", present.size())
            return y, present
