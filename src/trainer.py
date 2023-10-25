from collections import defaultdict
from pathlib import Path
import shutil
import sys
import time
import math
from typing import Any, Dict

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import wandb

from agent import Agent
from collector import Collector
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch, generate_reconstructions_with_tokenizer,compute_metrics
from models.transformer_generation_phase import GenerationPhase
from models.world_model import WorldModel
from utils import configure_optimizer, set_seed


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)
        self.batch_size=cfg.common.batch_size
        self.obs_time = cfg.common.obs_time 
        self.pred_time = cfg.common.pred_time 
        self.time_interval = cfg.common.time_interval

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'
        #self.generation_dir= self.media_dir / 'generation'

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)
        #################################################################



        if self.cfg.training.should:
            self.train_collector = Collector()

        if self.cfg.evaluation.should:
            self.test_collector = Collector()


        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, config=instantiate(cfg.world_model))

        self.agent = Agent(tokenizer, world_model).to(self.device)
        ## the world model should be set in cuda 1 and the agent should be in cuda 0 
        print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
    

        self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)

        if cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)

        if cfg.common.resume:
            self.load_checkpoint()

    def run(self) -> None:
        
        training_data_dataloader, length_train =self.train_collector.collect_training_data(self.batch_size)
        testing_data_dataloader, length_test= self.test_collector.collect_testing_data(batch_size=1) #

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                
                if epoch <= self.cfg.collection.train.stop_after_epochs:

                    to_log += self.train_agent(epoch, training_data_dataloader)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                to_log += self.eval_agent(epoch, testing_data_dataloader, length_test)

            if self.cfg.training.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

        self.finish()

    def train_agent(self, epoch: int, training_data_dataloader) -> None:
        self.agent.train()
        self.agent.zero_grad()
        

        metrics_tokenizer, metrics_world_model= {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model



        if epoch >= cfg_tokenizer.start_after_epochs:
            loss_total_epoch = 0.0
            intermediate_losses = defaultdict(float)
            for batch in training_data_dataloader:
                batch= batch.unsqueeze(2)
                metrics_tokenizer, loss, intermediate_los = self.train_component(self.agent.tokenizer, self.optimizer_tokenizer,batch, loss_total_epoch, intermediate_losses, sequence_length=self.cfg.common.sequence_length, **cfg_tokenizer) 
                loss_total_epoch = loss 
                intermediate_losses = intermediate_los
            print("tokenizer_loss_total_epoch", loss_total_epoch)
        self.agent.tokenizer.eval()

        if epoch >= cfg_world_model.start_after_epochs:
            loss_total_epoch = 0.0
            intermediate_losses = defaultdict(float)
            for batch in training_data_dataloader:
                batch= batch.unsqueeze(2)
                metrics_world_model, loss, intermediate_los = self.train_component(self.agent.world_model, self.optimizer_world_model,batch, loss_total_epoch, intermediate_losses, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer, **cfg_world_model)
                loss_total_epoch = loss 
                intermediate_losses = intermediate_los
            print("worldmodel_loss_total_epoch", loss_total_epoch)
        self.agent.world_model.eval()
        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, batch,  loss_total_epoch, intermediate_losses,  batch_num_samples: int, grad_acc_steps: int, sequence_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        mini_batch= math.floor(batch.size(0)/(batch_num_samples*grad_acc_steps))
        counter=0
        # optimizer.zero_grad()
        for _ in range(mini_batch):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps): 
                batch_training= batch[(counter*batch_num_samples):(counter+1)*(batch_num_samples),:,:,:,:] #[:,:,(counter*batch_num_samples):(counter+1)*(batch_num_samples),:,:]
                batch_training = self._to_device(batch_training)
                losses = component.compute_loss(batch_training, **kwargs_loss) / grad_acc_steps
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                loss_total_epoch += loss_total_step.item() 

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value
                
                counter= counter + 1
                
            print("loss_total_batch", loss_total_epoch)

            optimizer.step()

            
        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}


        

        #batch = self._out_device(batch)
        return metrics, loss_total_epoch, intermediate_losses
    
    @torch.no_grad()
    def eval_agent(self, epoch: int, testing_data_dataloader, length_test) -> None:
        self.agent.eval()

        
        metrics_tokenizer, metrics_world_model = {}, {}
        

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        
# 
        if epoch >= cfg_tokenizer.start_after_epochs:
            loss_total_test_epoch = 0.0
            intermediate_losses = defaultdict(float)
            self.accumulated_metrics = defaultdict(float)           
            for batch in testing_data_dataloader:
                batch= batch.unsqueeze(2)            
                metrics_tokenizer, loss_test, intermediate_los  = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples,batch, loss_total_test_epoch, intermediate_losses, sequence_length=self.cfg.common.sequence_length)
                loss_total_test_epoch = loss_test 
                intermediate_losses = intermediate_los
                print("evaluation total loss", loss_total_test_epoch)
                
            for metrics_name, metrics_value in metrics_tokenizer.items():
                metrics_tokenizer[metrics_name] = metrics_value / length_test

                
            

        if epoch >= cfg_world_model.start_after_epochs:
            loss_total_test_epoch = 0.0
            intermediate_losses = defaultdict(float)
                    
            for batch in testing_data_dataloader:
                batch= batch.unsqueeze(2)
                metrics_world_model, loss_test, intermediate_los = self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples, batch, loss_total_test_epoch, intermediate_losses, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer)
                loss_total_test_epoch = loss_test 
                intermediate_losses = intermediate_los
                print("evaluation total loss", loss_total_test_epoch)



        if cfg_tokenizer.save_reconstructions:
            for batch in testing_data_dataloader:
                reconstruct_batch= batch.unsqueeze(2)
                break
            reconstruct_batch = self._to_device(reconstruct_batch)
            make_reconstructions_from_batch(reconstruct_batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)
        
        # if cfg_world_model.save_generations: 
            # for batch in testing_data_dataloader: 
                # generate_batch=batch.unsqueeze(2)
                # break 
            # generate_batch = self._to_device(generate_batch)
            # self.start_generation(generate_batch, epoch=epoch)
            # 


        return [metrics_tokenizer, metrics_world_model]
    
    

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, batch, loss_total_test_epoch, intermediate_losses, sequence_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        pysteps_metrics = {}
        
        batch_testing = self._to_device(batch)          
        losses = component.compute_loss(batch_testing, **kwargs_loss)
        loss_total_test_epoch += (losses.loss_total.item())

        for loss_name, loss_value in losses.intermediate_losses.items():
            intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

        ######## Pysteps Metrics Calculation
        
        if str(component) =='tokenizer':
            rec_frames = generate_reconstructions_with_tokenizer(batch_testing, component)
            pysteps_metrics = compute_metrics(batch_testing, rec_frames)
        
            for metrics_name, metrics_value in pysteps_metrics.items():
                if math.isnan(metrics_value):
                    metrics_value = 0.0
                self.accumulated_metrics[metrics_name] += metrics_value
        

            intermediate_losses = {k: v  for k, v in intermediate_losses.items()}
            metrics = {f'{str(component)}/eval/total_loss': loss_total_test_epoch, **intermediate_losses, **self.accumulated_metrics}
        else: 
            intermediate_losses = {k: v  for k, v in intermediate_losses.items()}
            metrics = {f'{str(component)}/eval/total_loss': loss_total_test_epoch, **intermediate_losses}
        
        # print("evaluation total loss", loss_total_test_epoch)

        return metrics, loss_total_test_epoch, intermediate_losses
    

    # @torch.no_grad()
    # def start_generation(self, batch, epoch) -> None:
        # predictions, sequence_loss, reconstruction_loss_real_tokens, reconstruction_loss_generated_tokens = GenerationPhase.compute_loss(self, batch, tokenizer= self.agent.tokenizer,world_model= self.agent.world_model,latent_dim=16, horizon=9, obs_time=7)
        # observations= batch[7:16,:,:,:]
        # GenerationPhase.show_prediction(observations,predictions, save_dir=self.generation_dir, epoch=epoch)



    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({
                "optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                "optimizer_world_model": self.optimizer_world_model.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        print(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: torch.Tensor):
        return batch.to(self.device)

    def _out_device(self, batch: torch.Tensor):
        return batch.detach()
    
    def finish(self) -> None:
        wandb.finish()
