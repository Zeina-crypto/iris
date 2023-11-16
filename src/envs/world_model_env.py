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
        self.obs_tokens = obs_tokens

        return obs_tokens

    # @torch.no_grad()
    # def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
    #     n, num_observations_tokens = obs_tokens.shape
    #     assert num_observations_tokens == self.num_observations_tokens
    #     self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
    #     #outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
    #     return self.keys_values_wm   # (B, K, E)
    
    @torch.no_grad()
    def step(self , observations, num_steps) -> None:
        sample=observations
        cond_len = observations.shape[1]
        past = None
        x=sample
        
        

        for k in range(num_steps):
            outputs_wm = self.world_model.forward_with_past(x, past)


            if past is None:
                past = [outputs_wm.past_keys_values]
            else:
                past.append(outputs_wm.past_keys_values)
                print("past len", len(past))

            logits = outputs_wm.logits_observations
            logits=logits[:, -1, :]
            token = Categorical(logits=logits).sample()
            x = token.unsqueeze(1) 
            sample = torch.cat((sample, x), dim=1)
        

        sample = sample[:, :] 


        return sample 

    




    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self, obs_tokens) -> List[Image.Image]:
        generated_sequence=obs_tokens
        generated_sequence=generated_sequence.squeeze(0)
        embedded_tokens = self.tokenizer.embedding(generated_sequence)     # (B, K, E)
        z = rearrange(embedded_tokens, '(b h w) e -> b e h w', e=1024, h=8, w=8).contiguous()
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        rec= rec.unsqueeze(0)
        return rec

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
