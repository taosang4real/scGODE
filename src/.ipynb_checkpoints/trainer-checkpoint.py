import os
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf 

from .utils import get_optimizer, get_scheduler, save_checkpoint, load_checkpoint
from .losses import CombinedLoss
from .models.combined_model import CombinedModel
from .models.graph_autoencoder import GraphAutoencoder


class Trainer:
    def __init__(self, config, model, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if config.training_params.ddp.use_ddp and torch.cuda.is_available() else config.training_params.device)
        
        self.model = model.to(self.device)
        if config.training_params.ddp.use_ddp and world_size > 1 and torch.cuda.is_available():
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True) 

        self.loss_fn = CombinedLoss(config.loss_weights, device=self.device)
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.config.training_params.get("use_amp", False))


    def _initialize_optimizer_scheduler(self, mode="gae_pretrain"):
        lr_key = f"learning_rate_{mode.replace('train_', '')}" if mode != "joint_train" else "learning_rate_joint"
        
        current_lr = self.config.training_params.get(lr_key, self.config.training_params.learning_rate_joint)

        temp_opt_config = type('OptConfig', (object,), {
            'optimizer': str(self.config.training_params.optimizer), 
            'learning_rate_joint': float(current_lr), 
            'weight_decay': float(self.config.training_params.weight_decay)
        })()
        
        target_params = self.model.parameters()
        actual_model_module = self.model.module if isinstance(self.model, DDP) else self.model

        if mode == "pretrain_gae":
            if hasattr(actual_model_module, 'graph_autoencoder'):
                 target_params = actual_model_module.graph_autoencoder.parameters()
            elif isinstance(actual_model_module, GraphAutoencoder):
                 target_params = actual_model_module.parameters()

        self.optimizer = get_optimizer(target_params, temp_opt_config)

        lr_scheduler_name = self.config.training_params.get('lr_scheduler', None)
        lr_scheduler_params_conf = self.config.training_params.get('lr_scheduler_params', {})
        
        temp_sched_config = type('SchedConfig', (object,), {
            'lr_scheduler': str(lr_scheduler_name) if lr_scheduler_name else None,
            'lr_scheduler_params': lr_scheduler_params_conf 
        })()
        self.scheduler = get_scheduler(self.optimizer, temp_sched_config)


    def _run_epoch(self, dataloader, mode, epoch):
        is_train = "train" in mode
        current_model_to_use = self.model.module if isinstance(self.model, DDP) else self.model

        if is_train:
            current_model_to_use.train()
            if mode == "pretrain_gae" and hasattr(current_model_to_use, 'graph_autoencoder'):
                current_model_to_use.graph_autoencoder.train()
        else: 
            current_model_to_use.eval()
            if hasattr(current_model_to_use, 'graph_autoencoder'):
                current_model_to_use.graph_autoencoder.eval()
            if hasattr(current_model_to_use, 'neural_ode'):
                 current_model_to_use.neural_ode.eval()

        total_epoch_loss = 0.0
        all_loss_dicts = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{mode}] Rank {self.rank}", disable=(self.rank != 0 or not dataloader))

        for batch_idx, data_batch in enumerate(progress_bar):
            if data_batch is None: continue 

            if mode == "pretrain_gae" or mode == "eval_gae":
                pyg_batch = data_batch['pyg_batch'].to(self.device)
                original_expressions = data_batch['original_expressions'].to(self.device) 
                
                effective_model = current_model_to_use
                if hasattr(current_model_to_use, 'graph_autoencoder'): 
                    effective_model = current_model_to_use.graph_autoencoder
                
                if not isinstance(effective_model, GraphAutoencoder):
                    raise TypeError("Model for GAE mode is not GraphAutoencoder.")

                with torch.set_grad_enabled(is_train), torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    reconstructed_output, _, _, _ = effective_model(pyg_batch, return_pooling_details=False)
                    
                    batch_size_gae = original_expressions.shape[0]
                    num_genes_cfg = int(self.config.model_params.num_genes) 
                    
                    reconstructed_expressions_reshaped = reconstructed_output
                    if reconstructed_output.shape[0] == batch_size_gae * num_genes_cfg and reconstructed_output.shape[1] == 1:
                        reconstructed_expressions_reshaped = reconstructed_output.view(batch_size_gae, num_genes_cfg)
                    elif reconstructed_output.numel() == original_expressions.numel(): 
                        try:
                            reconstructed_expressions_reshaped = reconstructed_output.view_as(original_expressions)
                        except RuntimeError:
                            print(f"Warning: Could not reshape GAE reconstruction output {reconstructed_output.shape} to {original_expressions.shape}. Using original output for loss.")
                            pass 
                    else:
                         print(f"Warning: Shape mismatch for GAE reconstruction. Output: {reconstructed_output.shape}, Target: {original_expressions.shape}. Using original output for loss.")


                    model_outputs_for_loss = {"reconstructed_expressions": reconstructed_expressions_reshaped}
                    original_data_for_loss = {"original_expressions": original_expressions}
                    loss, loss_dict = self.loss_fn(model_outputs_for_loss, original_data_for_loss, mode="pretrain_gae")
            
            elif "joint" in mode or "eval_joint" in mode:
                batch_t0 = data_batch['batch_t0'].to(self.device)
                batch_t1 = data_batch['batch_t1'].to(self.device)
                original_expressions_t0 = data_batch['original_expressions_t0'].to(self.device) 
                original_expressions_t1 = data_batch['original_expressions_t1'].to(self.device) 
                day_t0_scalar = data_batch['day_t0_scalar'].to(self.device)
                day_t1_scalar = data_batch['day_t1_scalar'].to(self.device)

                if not isinstance(current_model_to_use, CombinedModel):
                     raise TypeError("Model for joint training/eval must be CombinedModel.")

                with torch.set_grad_enabled(is_train), torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    model_outputs_raw = current_model_to_use(
                        data_t0=batch_t0,
                        t0_scalar=day_t0_scalar,
                        t1_scalar=day_t1_scalar,
                        data_t1=batch_t1,
                        return_latent_t1_real=True 
                    )
                    
                    num_genes_cfg = int(self.config.model_params.num_genes)
                    model_outputs_for_loss = model_outputs_raw.copy() 

                    x_t0_rec_raw = model_outputs_raw.get("x_t0_reconstructed")
                    if x_t0_rec_raw is not None:
                        batch_size_t0 = original_expressions_t0.shape[0]
                        x_t0_rec_reshaped = x_t0_rec_raw
                        if x_t0_rec_raw.shape[0] == batch_size_t0 * num_genes_cfg and x_t0_rec_raw.shape[1] == 1:
                            x_t0_rec_reshaped = x_t0_rec_raw.view(batch_size_t0, num_genes_cfg)
                        elif x_t0_rec_raw.numel() == original_expressions_t0.numel():
                            try:
                                x_t0_rec_reshaped = x_t0_rec_raw.view_as(original_expressions_t0)
                            except RuntimeError:
                                print(f"Warning: Could not reshape x_t0_reconstructed {x_t0_rec_raw.shape} to {original_expressions_t0.shape}. Using original for loss.")
                                pass
                        else:
                            print(f"Warning: Shape mismatch for x_t0_reconstructed. Output: {x_t0_rec_raw.shape}, Target: {original_expressions_t0.shape}. Using original for loss.")
                        model_outputs_for_loss["x_t0_reconstructed"] = x_t0_rec_reshaped

                    x_t1_pred_rec_raw = model_outputs_raw.get("x_t1_predicted_reconstructed")
                    if x_t1_pred_rec_raw is not None:
                        batch_size_t1 = original_expressions_t1.shape[0]
                        x_t1_pred_rec_reshaped = x_t1_pred_rec_raw
                        if x_t1_pred_rec_raw.shape[0] == batch_size_t1 * num_genes_cfg and x_t1_pred_rec_raw.shape[1] == 1:
                            x_t1_pred_rec_reshaped = x_t1_pred_rec_raw.view(batch_size_t1, num_genes_cfg)
                        elif x_t1_pred_rec_raw.numel() == original_expressions_t1.numel():
                            try:
                                x_t1_pred_rec_reshaped = x_t1_pred_rec_raw.view_as(original_expressions_t1)
                            except RuntimeError:
                                 print(f"Warning: Could not reshape x_t1_pred_reconstructed {x_t1_pred_rec_raw.shape} to {original_expressions_t1.shape}. Using original for loss.")
                                 pass
                        else:
                            print(f"Warning: Shape mismatch for x_t1_predicted_reconstructed. Output: {x_t1_pred_rec_raw.shape}, Target: {original_expressions_t1.shape}. Using original for loss.")
                        model_outputs_for_loss["x_t1_predicted_reconstructed"] = x_t1_pred_rec_reshaped

                    original_data_for_loss = {
                        "original_expressions_t0": original_expressions_t0,
                        "original_expressions_t1": original_expressions_t1
                    }
                    loss, loss_dict = self.loss_fn(model_outputs_for_loss, original_data_for_loss, mode="train_joint")
            else:
                raise ValueError(f"Unsupported mode in _run_epoch: {mode}")


            if is_train and loss is not None:
                grad_accum_steps = self.config.training_params.get("gradient_accumulation_steps", 1)
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps
                
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    grad_clip_norm = self.config.training_params.get("gradient_clipping_norm", None)
                    if grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(current_model_to_use.parameters(), grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)


            if loss is not None:
                grad_accum_steps = self.config.training_params.get("gradient_accumulation_steps", 1)
                current_batch_total_loss = loss.item() * grad_accum_steps if is_train and grad_accum_steps > 1 else loss.item()
                total_epoch_loss += current_batch_total_loss
                all_loss_dicts.append(loss_dict)
                
                if self.rank == 0 and dataloader: 
                    postfix_dict = {"batch_loss": f"{current_batch_total_loss:.4f}"}
                    # Add individual loss components to postfix, abbreviate keys if too long
                    for k, v in loss_dict.items():
                        if k != "total_loss": # total_loss is already covered by current_batch_total_loss or loss.item()
                             pk = "".join([s[0] for s in k.split("_")[:2]]).upper() # e.g., L_recon_t0 -> LRT0
                             if len(pk) < 2: pk = k[:4] # Fallback for short names
                             postfix_dict[pk] = f"{v:.3f}"
                    progress_bar.set_postfix(postfix_dict)
        
        avg_epoch_loss = total_epoch_loss / len(dataloader) if dataloader and len(dataloader) > 0 else 0
        
        aggregated_loss_dict = {}
        if all_loss_dicts:
            for key in all_loss_dicts[0].keys(): 
                valid_values = [d[key] for d in all_loss_dicts if key in d and d[key] is not None]
                if valid_values:
                    aggregated_loss_dict[key] = np.mean(valid_values)
        
        if self.rank == 0:
            print(f"Epoch {epoch+1} [{mode}] Rank {self.rank} Avg Loss: {avg_epoch_loss:.4f}")
            for k, v_avg in aggregated_loss_dict.items():
                print(f"  Avg {k}: {v_avg:.4f}")

        return avg_epoch_loss, aggregated_loss_dict


    def train_loop(self, train_dataloader, val_dataloader, num_epochs, mode, checkpoint_prefix):
        if self.optimizer is None or self.scheduler is None: 
            self._initialize_optimizer_scheduler(mode=mode)

        for epoch in range(self.current_epoch, num_epochs):
            if train_dataloader and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            train_loss, train_loss_dict = self._run_epoch(train_dataloader, mode=mode, epoch=epoch) 
            
            val_loss = self.best_val_loss 
            if val_dataloader:
                eval_mode_suffix = mode.split("train_")[-1] if "train_" in mode else mode
                val_loss, val_loss_dict = self._run_epoch(val_dataloader, mode=f"eval_{eval_mode_suffix}", epoch=epoch) 

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_dataloader else train_loss)
                else:
                    self.scheduler.step()
            
            if self.rank == 0:
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    print(f"New best validation loss: {self.best_val_loss:.4f}")
                else:
                    self.epochs_no_improve += 1
                
                save_freq = self.config.training_params.get("save_checkpoint_freq_epochs", 10)
                if epoch % save_freq == 0 or is_best:
                    self.save_model_checkpoint(epoch, checkpoint_prefix, is_best, self.best_val_loss)
                
                early_stop_patience = self.config.training_params.get("early_stopping_patience", 0)
                if early_stop_patience > 0 and self.epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
            
            if self.config.training_params.ddp.use_ddp and self.world_size > 1:
                dist.barrier() 

        self.current_epoch = num_epochs 
        return self.best_val_loss


    def pretrain_gae(self, train_dataloader, val_dataloader=None):
        if self.rank == 0: print(f"Starting GAE pretraining on rank {self.rank}...")
        
        current_model_is_ddp = isinstance(self.model, DDP)
        actual_model_module = self.model.module if current_model_is_ddp else self.model

        if hasattr(actual_model_module, 'graph_autoencoder'):
            for name, param in actual_model_module.named_parameters():
                if 'graph_autoencoder' in name:
                    param.requires_grad = True
                else: 
                    param.requires_grad = False
        elif not isinstance(actual_model_module, GraphAutoencoder): 
            raise TypeError("Model for GAE pretraining must be GraphAutoencoder or CombinedModel with 'graph_autoencoder' attribute.")

        self._initialize_optimizer_scheduler(mode="pretrain_gae") 
        
        num_epochs_gae = self.config.training_params.get("num_epochs_gae_pretrain", 50)
        self.train_loop(
            train_dataloader, val_dataloader,
            num_epochs_gae,
            mode="pretrain_gae", checkpoint_prefix="gae_pretrain"
        )
        if self.rank == 0: print(f"GAE pretraining finished on rank {self.rank}.")

        if isinstance(actual_model_module, CombinedModel):
            if self.config.training_params.get("freeze_gae_after_pretrain", False):
                actual_model_module.freeze_gae_parameters()
            else: 
                actual_model_module.unfreeze_gae_parameters()


    def train_joint(self, train_dataloader, val_dataloader=None):
        if self.rank == 0: print(f"Starting Joint training on rank {self.rank}...")
        actual_model_module = self.model.module if isinstance(self.model, DDP) else self.model
        if not isinstance(actual_model_module, CombinedModel):
            raise TypeError("Model for joint training must be CombinedModel.")

        if not self.config.training_params.get("freeze_gae_after_pretrain", False):
             actual_model_module.unfreeze_gae_parameters()
        
        if hasattr(actual_model_module, 'neural_ode'): 
            for param in actual_model_module.neural_ode.parameters():
                param.requires_grad = True

        self._initialize_optimizer_scheduler(mode="train_joint")
        num_epochs_joint = self.config.training_params.get("num_epochs_joint_train", 100)
        self.train_loop(
            train_dataloader, val_dataloader,
            num_epochs_joint,
            mode="train_joint", checkpoint_prefix="joint_train"
        )
        if self.rank == 0: print(f"Joint training finished on rank {self.rank}.")


    def save_model_checkpoint(self, epoch, prefix, is_best, best_metric_val):
        if self.rank != 0: return

        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        model_state = model_to_save.state_dict()
        
        opt_state = self.optimizer.state_dict() if self.optimizer else None
        sched_state = self.scheduler.state_dict() if self.scheduler else None

        state = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': opt_state,
            'scheduler_state_dict': sched_state,
            'best_metric': best_metric_val,
            'config_dict': OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False) 
        }
        
        checkpoint_dir = self.config.training_params.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        filename = f"{prefix}_epoch_{epoch+1}.pt"
        filepath = os.path.join(checkpoint_dir, filename)
        save_checkpoint(state, filepath)
        if self.rank == 0: print(f"Checkpoint saved to {filepath}")

        if is_best:
            best_filename = f"{prefix}_best.pt"
            best_filepath = os.path.join(checkpoint_dir, best_filename)
            save_checkpoint(state, best_filepath) 
            if self.rank == 0: print(f"Best checkpoint updated to {best_filepath}")

    def load_model_checkpoint(self, filepath):
        start_epoch, best_metric = load_checkpoint(
            filepath,
            self.model, 
            optimizer=self.optimizer, 
            scheduler=self.scheduler,
            device=self.device
        )
        self.current_epoch = start_epoch
        self.best_val_loss = best_metric if best_metric is not None else float('inf')
        
        if self.rank == 0:
            print(f"Model, optimizer, and scheduler states loaded from {filepath}. Resuming training from epoch {self.current_epoch}.")

