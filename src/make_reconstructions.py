from einops import rearrange
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import json
from pysteps.verification.detcatscores import det_cat_fct
from pysteps.verification.detcontscores import det_cont_fct
from pysteps.verification.spatialscores import intensity_scale
from pysteps.visualization import plot_precip_field


@torch.no_grad()
def make_reconstructions_from_batch(batch, save_dir, epoch, tokenizer):
    #check_batch(batch)

    original_frames = rearrange(batch['observations'], 'c t b h w  -> (b t) c h w')
    batch_tokenizer = batch['observations']

    rec_frames = generate_reconstructions_with_tokenizer(batch_tokenizer, tokenizer)
    metrics = compute_metrics(batch_tokenizer, rec_frames)
    
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, f'epoch_{epoch:03d}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


    for i in range(5):
        original_frame = original_frames[i,0,:,:]
        a_display = tensor_to_np_frames(original_frame)
        rec_frame = rec_frames[i,0,:,:]
        ar_display = tensor_to_np_frames(rec_frame)

        # Plot the precipitation fields using your plot_precip_field function
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_precip_field(a_display, title="Input")
        
        plt.subplot(1, 2, 2)
        plot_precip_field(ar_display, title="Reconstruction")

        plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_t_{i:03d}.png'))

        # Optionally, display the figure if needed
        plt.show()

        # Close the figure to free up resources
        plt.close()


    return


def tensor_to_np_frames(inputs):
    #check_float_btw_0_1(inputs)
    return inputs.cpu().numpy()*40


# def check_float_btw_0_1(inputs):
    # assert inputs.is_floating_point() and (inputs >= 0).all() and (inputs <= 1).all()


@torch.no_grad()
def generate_reconstructions_with_tokenizer(batch, tokenizer):
    #check_batch(batch)
    inputs = rearrange(batch, 'c t b h w  -> (b t) c h w')
    outputs = reconstruct_through_tokenizer(inputs, tokenizer)
    b, t, _, _, _ = batch.size()
    # outputs = rearrange(outputs, '(b t) c h w -> b t h w c', b=b, t=t)
    rec_frames = outputs
    return rec_frames


@torch.no_grad()
def reconstruct_through_tokenizer(inputs, tokenizer):
    #check_float_btw_0_1(inputs)
    reconstructions = tokenizer.encode_decode(inputs, should_preprocess=True, should_postprocess=True)
    return torch.clamp(reconstructions, 0, 1)


def compute_metrics (batch, rec_frames):
    #threshold = 0.0
    input_images = rearrange(batch, 'c t b h w  -> (b t) c h w')
    input_images = input_images.squeeze(1)                                  #  (bt) c h w -> (bt) h w
    #reconstruction = rearrange(rec_frames, 'c t b h w  -> (b t) c h w')
    reconstruction = rec_frames.squeeze(1)                                  #  (bt) c h w -> (bt) h w
    metrics_pysteps = {}
    pcc_average = 0.0
    # input images and reconstruction/prediction images should be of the same shape to perform the below operation
    
    for i in range(input_images.shape[0]):   #input images of shape [128, 128] for pysteps evaluation
        if i >=5: break
        input_images_npy = tensor_to_np_frames(input_images[i])
        reconstruction_npy = tensor_to_np_frames(reconstruction[i])
        #ar_display[ar_display < threshold] = 0.0   
        
        scores_cat1 = det_cat_fct(reconstruction_npy, input_images_npy, 1)
        scores_cat2 = det_cat_fct(reconstruction_npy, input_images_npy, 2)
        scores_cat8 = det_cat_fct(reconstruction_npy, input_images_npy, 8)
        scores_cont = det_cont_fct(reconstruction_npy, input_images_npy, thr=0.1)
        
        scores_spatial = intensity_scale(reconstruction_npy, input_images_npy, 'FSS', 0.1, [1,10,20,30])
        pcc_average += float(np.around(scores_cont['corr_p'],3))
        
        metrics_pysteps = {'MSE:': np.around(scores_cont['MSE'],3), 
                    'MAE:': np.around(scores_cont['MAE'],3), 
                    'PCC:': np.around(scores_cont['corr_p'],3), 
                    'CSI(1mm):': np.around(scores_cat1['CSI'],3),   # CSI: TP/(TP+FP+FN)
                    'CSI(2mm):': np.around(scores_cat2['CSI'],3),
                    'CSI(8mm):': np.around(scores_cat8['CSI'],3),
                    'ACC(1mm):': np.around(scores_cat1['ACC'],3),   # ACC: (TP+TF)/(TP+TF+FP+FN)
                    'ACC(2mm):': np.around(scores_cat2['ACC'],3),
                    'ACC(8mm):': np.around(scores_cat8['ACC'],3),
                    'FSS(1km):': np.around(scores_spatial[0][0],3),
                    'FSS(10km):': np.around(scores_spatial[1][0],3),
                    'FSS(20km):': np.around(scores_spatial[2][0],3),
                    'FSS(30km):': np.around(scores_spatial[3][0],3),
                    'pcc_average': pcc_average/i,
            }
        
    return metrics_pysteps   