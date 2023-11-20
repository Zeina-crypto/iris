from dataclasses import dataclass
from typing import Any, Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    past_keys_values: torch.tensor

class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size = obs_vocab_size
        self.config = config
        self.transformer = Transformer(config)
        obs_tokens_pattern = torch.ones(config.tokens_per_block)

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        
        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )
        
        self.apply(init_weights)
           

    def __repr__(self) -> str:
        return "world_model"
            
    def forward(self, obs_tokens: torch.LongTensor, past_keys_values= None) -> WorldModelOutput:

        num_steps = obs_tokens.shape[1]  # (B, T)
        if past_keys_values is None: 
            past_shape = 0 
        else : 
            past_shape = len(past_keys_values)
            print(past_shape)
        sequences = self.embedder(obs_tokens, num_steps, past_shape) + self.pos_emb(torch.arange(past_shape, past_shape+num_steps, device=obs_tokens.device))
        #print("sequence ",sequences.size())

        x = self.transformer.forward(sequences, past_keys_values)
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=past_shape)
        
        return WorldModelOutput(x, logits_observations, past_keys_values)
    


    def forward_with_past(self, obs_tokens: torch.LongTensor, past_keys_values=None) -> WorldModelOutput:

        num_steps = obs_tokens.shape[1]  # (B, T)
        if past_keys_values is None: 
            past_shape = 0 
        else : 
            past_shape = len(past_keys_values)
            print(past_shape)
        sequences = self.embedder(obs_tokens, num_steps, past_shape) + self.pos_emb(torch.arange(past_shape, past_shape+num_steps, device=obs_tokens.device))
        print("sequence ",sequences.size())

        x, past_keys_values = self.transformer.forward_with_past(sequences, past_keys_values)
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=past_shape)
        
        return WorldModelOutput(x, logits_observations, past_keys_values)
    
    
    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        
        with torch.no_grad():
            observations= rearrange(batch, 'b t c h w  -> (b t) c h w')
            obs_tokens = tokenizer.encode(observations).tokens  # (BL, K)
            shape_obs = batch.size()
            shape_token= obs_tokens.size()

        b = shape_obs[0]
        l = shape_obs[1]
        k = shape_token[1]
        tokens = obs_tokens.view(b, l*k) # (B, L(K))
        outputs = self.forward(tokens)
    

        labels_observations = self.compute_labels_world_model(tokens)

        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        #print("Cross entropy Losses", loss_obs)
        return LossWithIntermediateLosses(loss_obs=loss_obs)
    
    def compute_labels_world_model(self, obs_tokens: torch.Tensor):
        labels_observations = obs_tokens[:, 1:] # obs tokens from t to t_end, remove only the first one 
        return labels_observations.reshape(-1)
