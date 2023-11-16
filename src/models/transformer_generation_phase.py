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
from pysteps.verification.detcatscores import det_cat_fct
from pysteps.verification.detcontscores import det_cont_fct
from pysteps.verification.spatialscores import intensity_scale


class GenerationPhase():
    def __init__(self):
        super().__init__()

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, latent_dim:int, horizon: int, obs_time: int, **kwargs: Any):
        observations = rearrange(batch, 'b t c h w  -> (b t) c h w')
        assert batch.ndim == 5 and batch.shape[2:] == (1, 128, 128)
        device = batch.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)
        sequence_length= horizon
        
        ground_truth, logits_obs, predicted_observations = self.generate(batch, tokenizer, world_model, sequence_length)


        ########### COMPUTE THE CROSS ENTROPY LOSS BETWEEN THE PREDICTED SEQUENCE AND THE ACTUAL SEQUENCE ##############
        logits_tokens = logits_obs.squeeze(0)
        logits_tokens = logits_tokens[((obs_time*latent_dim)-1): , :]
        ground_truth_tokens= ground_truth.squeeze(0)
        ground_truth_tokens= ground_truth_tokens[(obs_time*latent_dim):]
        sequence_loss= F.cross_entropy(logits_tokens, ground_truth_tokens)


        ######### IF WE COULD CALCULATE THE LOSSES BETWEEN RECONSTRUCTION AFTER PREDICTION AND THE GROUND TRUTH RECONSTRUCTION ##########
        reconstructed_from_groundtruth_tokens= wm_env.decode_obs_tokens(ground_truth[:, (obs_time*latent_dim):], rec_length=(horizon-obs_time))
        reconstructed_from_groundtruth_tokens= reconstructed_from_groundtruth_tokens.squeeze(0)
        reconstructed_from_predicted_tokens= wm_env.decode_obs_tokens(predicted_observations[:, (obs_time*latent_dim):], rec_length=(horizon-obs_time))
        reconstructed_from_predicted_tokens= reconstructed_from_predicted_tokens.squeeze(0)
        reconstruction_loss_real_tokens = torch.abs(observations[obs_time:sequence_length,:,:,:] - reconstructed_from_groundtruth_tokens).mean()
        reconstruction_loss_generated_tokens = torch.abs(observations[obs_time:sequence_length,:,:,:] - reconstructed_from_predicted_tokens).mean()
        return sequence_loss, reconstruction_loss_real_tokens, reconstruction_loss_generated_tokens
    




    def generate(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, latent_dim:int, horizon: int, obs_time: int):
        
        initial_observations = batch
        device = initial_observations.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)
        input_image = initial_observations[:,:obs_time,:,:,:].to(device=device)
        obs_tokens = wm_env.reset_from_initial_observations(input_image)
        generated_sequence = wm_env.step(obs_tokens, num_steps=horizon*latent_dim)#, top_k=False, sample=False)
        reconstructed_predicted_sequence= wm_env.decode_obs_tokens(generated_sequence)

        return reconstructed_predicted_sequence
    

    def tensor_to_np_frames(self, inputs):
        return inputs.cpu().numpy()*40
    
    

    def compute_metrics (self, batch, predicted_frames, obs_time):
        input_images = rearrange(batch, 'b t c h w  -> (b t) c h w')
        input_images = input_images.squeeze(1) 

        total_images = input_images.shape[0]
                                        
        avg_metrics = {
            'MSE:': 0, 'MAE:': 0, 'PCC:': 0, 'CSI(1mm):': 0, 'CSI(2mm):': 0, 
            'CSI(8mm):': 0, 'ACC(1mm):': 0, 'ACC(2mm):': 0, 'ACC(8mm):': 0, 
            'FSS(1km):': 0, 'FSS(10km):': 0, 'FSS(20km):': 0, 'FSS(30km):': 0
        }

        
        
        for i in range(total_images):
            input_images_npy = self.tensor_to_np_frames(input_images[i])
            predicted_frames= self.generate(batch=batch[i:i+1,:,:,:,:], tokenizer=Tokenizer, world_model=WorldModel, latent_dim= 64, horizon= 6, obs_time= 3)                                 
            prediction = predicted_frames.squeeze(1) 
            prediction_npy = self.tensor_to_np_frames(prediction[i])
            scores_cat1 = det_cat_fct(prediction_npy, input_images_npy, 1)
            scores_cat2 = det_cat_fct(prediction_npy, input_images_npy, 2)
            scores_cat8 = det_cat_fct(prediction_npy, input_images_npy, 8)
            scores_cont = det_cont_fct(prediction_npy, input_images_npy, scores = ["MSE", "MAE", "corr_p"], thr=0.1)
            scores_spatial = intensity_scale(prediction_npy, input_images_npy, 'FSS', 0.1, [1,10,20,30])
            
            metrics = {'MSE:': scores_cont['MSE'],
                    'MAE:': scores_cont['MAE'], 
                    'PCC:': scores_cont['corr_p'], 
                    'CSI(1mm):': scores_cat1['CSI'],
                    'CSI(2mm):': scores_cat2['CSI'],
                    'CSI(8mm):': scores_cat8['CSI'],
                    'ACC(1mm):': scores_cat1['ACC'],
                    'ACC(2mm):': scores_cat2['ACC'],
                    'ACC(8mm):': scores_cat8['ACC'],
                    'FSS(1km):': scores_spatial[0][0],
                    'FSS(10km):': scores_spatial[1][0],
                    'FSS(20km):': scores_spatial[2][0],
                    'FSS(30km):': scores_spatial[3][0]
            }
            
            # Update avg_metrics dictionary
            for key in avg_metrics:
                avg_metrics[key] += metrics[key]
            
        # Compute average for each metric
        for key in avg_metrics:
            avg_metrics[key] = np.around(avg_metrics[key] / total_images, 3)
        
        return avg_metrics
    

        
    @torch.no_grad()
    def show_prediction(self, batch, obs_time, epoch, tokenizer, save_dir):
        B, T, C, H, W = batch.size()  
        batch_tokenizer= batch[:, obs_time: , :, :, :]
        original_frames = rearrange(batch_tokenizer, 'b t c h w  -> (b t) c h w')
        batch_wm = batch

        rec_frames = self.generate_reconstructions_with_tokenizer(batch_tokenizer, tokenizer)
        
        prediction_frames= self.generate(batch= batch, tokenizer=Tokenizer, world_model=WorldModel, latent_dim=64, horizon=6, obs_time=3)
        prediction_frames= prediction_frames.squeeze(0)
        
        os.makedirs(save_dir, exist_ok=True)

        for t in range(T-obs_time):
            original_frame = original_frames[t,0,:,:]
            a_display = self.tensor_to_np_frames(original_frame)
            rec_frame = rec_frames[t,0,:,:]
            ar_display = self.tensor_to_np_frames(rec_frame)
            pred_frame = prediction_frames[t,0,:,:]
            ap_display= self.tensor_to_np_frames(pred_frame) 

            # Plot the precipitation fields using your plot_precip_field function
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plot_precip_field(a_display, title="t+{}".format((t+1)*30), axis='off')
            
            plt.subplot(1, 3, 2)
            plot_precip_field(ar_display, title="Reconstruction")

            plt.subplot(1, 3, 3)
            plot_precip_field(ap_display, title="Prediction")



            plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_t_{t:03d}.png'))

            # Optionally, display the figure if needed
            plt.show()

            # Close the figure to free up resources
            plt.close()


        return
        
    @torch.no_grad()
    def generate_reconstructions_with_tokenizer(self, batch, tokenizer):
        #check_batch(batch)
        inputs = rearrange(batch, 'b t c h w  -> (b t) c h w') 
        outputs = self.reconstruct_through_tokenizer(inputs, tokenizer)
        rec_frames = outputs
        return rec_frames


    @torch.no_grad()
    def reconstruct_through_tokenizer(self, inputs, tokenizer):
        reconstructions = tokenizer.encode_decode(inputs, should_preprocess=True, should_postprocess=True)
        return torch.clamp(reconstructions, 0, 1)
  
        

    
