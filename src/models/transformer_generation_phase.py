from dataclasses import dataclass
from typing import Any, Optional, Union
import sys
import os
import matplotlib.pyplot as plt
from pysteps.visualization import plot_precip_field
from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Batch
from envs.world_model_env import WorldModelEnv
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import compute_lambda_returns, LossWithIntermediateLosses


class GenerationPhase():
    def __init__(self):
        super().__init__()

        
                

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, latent_dim:int, horizon: int, obs_time: int, **kwargs: Any):

        observations = rearrange(batch, 'b t c h w  -> (b t) c h w')
        assert batch.ndim == 5 and batch.shape[2:] == (1, 128, 128)

        device = batch.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)
        sequence_length= horizon + obs_time
        
        ground_truth_observations, logits_obs, predicted_observations = self.generate(batch, tokenizer, world_model, sequence_length)

        ########### COMPUTE THE CROSS ENTROPY LOSS BETWEEN THE PREDICTED SEQUENCE AND THE ACTUAL SEQUENCE ##############
        logits_tokens = logits_obs.squeeze(0)
        logits_tokens = logits_tokens[((obs_time*latent_dim)-1): , :]
        ground_truth_tokens= ground_truth_observations.squeeze(0)
        ground_truth_tokens= ground_truth_tokens[(obs_time*latent_dim):]
        sequence_loss= F.cross_entropy(logits_tokens, ground_truth_tokens)

        ######### IF WE COULD CALCULATE THE LOSSES BETWEEN RECONSTRUCTION AFTER PREDICTION AND THE GROUND TRUTH RECONSTRUCTION ##########
        reconstructed_from_groundtruth_tokens= wm_env.decode_obs_tokens(ground_truth_observations[:, (obs_time*latent_dim):])
        reconstructed_from_groundtruth_tokens= reconstructed_from_groundtruth_tokens.squeeze(0)
        reconstructed_from_predicted_tokens= wm_env.decode_obs_tokens(predicted_observations[:, (obs_time*latent_dim):])
        reconstructed_from_predicted_tokens= reconstructed_from_predicted_tokens.squeeze(0)
        reconstruction_loss_real_tokens = torch.abs(observations[obs_time:sequence_length,:,:,:] - reconstructed_from_groundtruth_tokens).mean()
        reconstruction_loss_generated_tokens = torch.abs(observations[obs_time:sequence_length,:,:,:] - reconstructed_from_predicted_tokens).mean()



        return reconstructed_from_predicted_tokens, sequence_loss, reconstruction_loss_real_tokens, reconstruction_loss_generated_tokens

    def generate(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, latent_dim:int, horizon: int, obs_time: int):
        
        initial_observations = batch
        device = initial_observations.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)
        ground_truth_observations = wm_env.reset_from_initial_observations(initial_observations)
        obs= ground_truth_observations[:, :(obs_time*latent_dim)]

        for k in range(horizon): 
        
            predicted_obs, logits_obs = wm_env.step(obs, should_predict_next_obs=(k < horizon - 1))
            predicted_observations= predicted_obs[horizon-1]
            obs=predicted_observations
        
        return ground_truth_observations, logits_obs, predicted_observations
    


    def show_prediction(self, observations, predictions, save_dir, epoch):
        #observations_plot=  observations[7:16,:,:,:]
        #predictions=reconstructed_from_predicted_tokens[i,0,:,:] this is from the generate function 
        os.makedirs(save_dir, exist_ok=True)
        for i in range(9):
            ar_display = tensor_to_np_frames(predictions[i,0,:,:])
            a_display = tensor_to_np_frames(observations[i,0,:,:])
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plot_precip_field(a_display, title="Ground-Truth")
            plt.subplot(1, 2, 2)
            plot_precip_field(ar_display, title="Prediction")
            plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_t_{i:03d}.png'))
            plt.show()
            plt.close()

        def tensor_to_np_frames(inputs):
            return inputs.cpu().numpy()*40
        
            
    
