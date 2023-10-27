import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision


class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        obs_tokens=rearrange(obs_tokens,'B T H -> B (T H)')
        #obs_tokens=obs_tokens.rearrange('C T H -> C (T*H)')
        #_, num_observations_tokens = obs_tokens.shape
        # if self.num_observations_tokens is None:
            # self._num_observations_tokens = num_observations_tokens

        # _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

        return obs_tokens

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=16)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)
    
    @torch.no_grad()
    def step(self, observations: torch.FloatTensor, latent_dim:int, horizon:int, obs_time:int):
    
        num_passes = latent_dim*(horizon-obs_time)
        generated_sequence = observations
        for k in range(num_passes):
            
            outputs_wm = self.world_model(generated_sequence)
            logits=outputs_wm.logits_observations
            probabilities = torch.softmax(logits[:, -1], dim=-1) 
            next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1) 
            generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(1)], dim=1)
            #print(generated_sequence.size())
    
    
        
        return generated_sequence, logits 




    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self, tokens, rec_length:int ) -> List[Image.Image]:
        generated_sequence=tokens
        generated_sequence=generated_sequence.squeeze(0)
        embedded_tokens = self.tokenizer.embedding(generated_sequence)     # (B, K, E)
        z = rearrange(embedded_tokens, '(b h w) e -> b e h w', b=rec_length, e=1024, h=8, w=8).contiguous()
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return rec

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
