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
        self.device = torch.device(
            f"cuda:{rank}" if config.training_params.ddp.use_ddp and torch.cuda.is_available() else config.training_params.device)

        self.model = model.to(self.device)
        if config.training_params.ddp.use_ddp and world_size > 1 and torch.cuda.is_available():
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)

        self.loss_fn = CombinedLoss(config.loss_weights, config=self.config, device=self.device)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(device=self.device,
                                           enabled=self.config.training_params.get("use_amp", False))
        self.is_variational_encoder = bool(self.config.model_params.encoder.get("is_variational", False))
        self.node_type = str(self.config.model_params.node.get("type", "NeuralODE")).upper()

    def _setup_optimizer_and_scheduler(self):
        actual_model = self.model.module if isinstance(self.model, DDP) else self.model
        base_lr = float(self.config.training_params.get("learning_rate", 0.0005))

        params_for_optimizer = []

        param_groups_cfg = self.config.training_params.get("param_groups")
        if not param_groups_cfg:
            print(
                "Warning: `training_params.param_groups` not found. Optimizing all trainable parameters with base LR.")
            params_for_optimizer = filter(lambda p: p.requires_grad, actual_model.parameters())
        else:
            for group_config in param_groups_cfg:
                group_name = group_config.get("name")
                is_trainable = bool(group_config.get("trainable", False))
                lr_multiplier = float(group_config.get("lr_multiplier", 1.0))

                if not hasattr(actual_model, group_name):
                    if self.rank == 0: print(
                        f"Warning: Submodule '{group_name}' not found in model. Skipping param group.")
                    continue

                submodule = getattr(actual_model, group_name)

                for param in submodule.parameters():
                    param.requires_grad = is_trainable

                if is_trainable:
                    group_lr = base_lr * lr_multiplier
                    params_for_optimizer.append({
                        "params": submodule.parameters(),
                        "lr": group_lr
                    })
                    if self.rank == 0:
                        print(f"Parameter group '{group_name}' is TRAINABLE with LR = {group_lr:.6f}")
                else:
                    if self.rank == 0:
                        print(f"Parameter group '{group_name}' is FROZEN.")

        trainable_params_list = list(filter(lambda p: p.requires_grad, actual_model.parameters()))
        if not trainable_params_list:
            raise ValueError("No trainable parameters found. Check `param_groups` in config or model structure.")

        opt_name = str(self.config.training_params.optimizer).lower()
        weight_decay = float(self.config.training_params.get("weight_decay", 0.0))

        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(params_for_optimizer, lr=base_lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(params_for_optimizer, lr=base_lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(params_for_optimizer, lr=base_lr, weight_decay=weight_decay)

        self.scheduler = get_scheduler(self.optimizer, self.config.training_params)

    def _get_loss_weight(self, loss_name):
        return float(self.config.loss_weights.get(loss_name, 0.0))

    def _get_current_kl_weight(self, epoch):
        kl_anneal_cfg = self.config.training_params.get("kl_annealing", {"enabled": False})
        if not kl_anneal_cfg.get("enabled", False):
            return self._get_loss_weight("L_KL_latent")
        warmup_epochs = int(kl_anneal_cfg.get("warmup_epochs", 0))
        anneal_epochs = int(kl_anneal_cfg.get("anneal_epochs", 1))
        final_kl_weight = self._get_loss_weight("L_KL_latent")
        if epoch < warmup_epochs:
            return 0.0
        elif epoch < warmup_epochs + anneal_epochs:
            return final_kl_weight * ((epoch - warmup_epochs) / anneal_epochs)
        else:
            return final_kl_weight

    def _reshape_params_for_loss(self, params_raw, target_expressions_shape_b_ng, num_dist_params):
        if params_raw is None or params_raw.numel() == 0: return params_raw
        batch_size, num_genes = target_expressions_shape_b_ng

        params_reshaped = params_raw
        expected_flat_shape_dim0 = batch_size * num_genes

        if params_raw.dim() == 2 and params_raw.shape[0] == expected_flat_shape_dim0 and params_raw.shape[
            1] == num_dist_params:
            try:
                params_reshaped = params_raw.view(batch_size, num_genes, num_dist_params)
            except RuntimeError as e:
                print(f"Warning: Reshape failed in _reshape_params_for_loss. Error: {e}")
        elif not (params_raw.dim() == 3 and params_raw.shape == (batch_size, num_genes, num_dist_params)):
            print(
                f"Warning: Shape mismatch for _reshape_params_for_loss. Raw: {params_raw.shape}, Target: {target_expressions_shape_b_ng}, NumParams: {num_dist_params}.")

        return params_reshaped

    def _run_epoch(self, dataloader, mode, epoch):
        is_train = "train" in mode
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.train() if is_train else model.eval()
        kl_weight = self._get_current_kl_weight(epoch) if is_train else self._get_loss_weight("L_KL_latent")
        total_loss, all_losses = 0.0, []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [{mode}]", disable=(self.rank != 0))

        for i, batch in enumerate(progress_bar):
            if batch is None: continue
            with torch.set_grad_enabled(is_train), torch.amp.autocast(device_type=self.device.type,
                                                                      enabled=self.scaler.is_enabled()):
                if mode in ["pretrain_gae", "eval_gae"]:
                    outputs = model.autoencoder(batch['pyg_batch'].to(self.device))
                    originals = {"original_expressions": batch['original_expressions'].to(self.device)}
                    loss, losses = self.loss_fn(outputs, originals, "pretrain_gae", kl_weight)

                elif mode in ["train_joint", "eval_joint"]:
                    outputs = model(
                        data_t0=batch['data_t0'].to(self.device), t0_scalar=batch['day_t0_scalar'].to(self.device),
                        t1_scalar=batch['day_t1_scalar'].to(self.device), data_t1=batch['data_t1'].to(self.device),
                        return_latent_t1_real=True,
                        num_ot_reg_intervals=int(self.config.model_params.node.get("num_ot_reg_intervals", 0)),
                        joint_training_mode=self.config.training_params.get("joint_training_mode", "end_to_end")
                    )

                    if outputs.get("x_t0_reconstructed_params") is not None:
                        outputs["x_t0_reconstructed_params"] = self._reshape_params_for_loss(
                            outputs.get("x_t0_reconstructed_params"), batch['original_expressions_t0'].shape,
                            self.config.model_params.decoder.num_dist_params)
                    if outputs.get("x_t1_predicted_reconstructed_params") is not None:
                        outputs["x_t1_predicted_reconstructed_params"] = self._reshape_params_for_loss(
                            outputs.get("x_t1_predicted_reconstructed_params"), batch['original_expressions_t1'].shape,
                            self.config.model_params.decoder.num_dist_params)

                    originals = {"original_expressions_t0": batch['original_expressions_t0'].to(self.device),
                                 "original_expressions_t1": batch['original_expressions_t1'].to(self.device)}
                    loss, losses = self.loss_fn(outputs, originals, "train_joint", kl_weight)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

            if is_train and loss is not None:
                if not torch.isfinite(loss):
                    if self.rank == 0: print(f"!!! FATAL: Loss is {loss}. Skipping update. Components: {losses}")
                    continue
                grad_steps = self.config.training_params.get("gradient_accumulation_steps", 1)
                self.scaler.scale(loss / grad_steps if grad_steps > 1 else loss).backward()
                if (i + 1) % grad_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                                   self.config.training_params.get("gradient_clipping_norm", 1.0))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            if loss is not None:
                total_loss += loss.item()
                all_losses.append(losses)
                if self.rank == 0:
                    postfix = {k[:8]: f"{v:.3f}" for k, v in losses.items() if v is not None and k != "total_loss"}
                    postfix["batch_loss"] = f"{loss.item():.4f}";
                    postfix["kl_w"] = f"{kl_weight:.4f}"
                    progress_bar.set_postfix(postfix)

        avg_loss = np.nanmean([d.get('total_loss', np.nan) for d in all_losses]) if all_losses else 0.0
        agg_losses = {k: np.nanmean([d[k] for d in all_losses if d and k in d and d[k] is not None]) for k in
                      (all_losses[0] or {})}
        if self.rank == 0:
            print(f"\nEpoch {epoch + 1} [{mode}] Rank 0 Avg Loss: {avg_loss:.4f}")
            for k, v in agg_losses.items(): print(f"  Avg {k}: {v:.4f}")
        return avg_loss, agg_losses

    def train_loop(self, train_dl, val_dl, epochs, mode, prefix):
        self._setup_optimizer_and_scheduler()
        for epoch in range(self.current_epoch, epochs):
            if train_dl and isinstance(train_dl.sampler, DistributedSampler): train_dl.sampler.set_epoch(epoch)
            train_loss, _ = self._run_epoch(train_dl, mode, epoch)
            val_loss = self.best_val_loss
            if val_dl:
                val_loss, _ = self._run_epoch(val_dl, f"eval_{mode.split('train_')[-1]}", epoch)
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            if self.rank == 0:
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss, self.epochs_no_improve = val_loss, 0
                    print(f"New best validation loss: {self.best_val_loss:.4f}")
                else:
                    self.epochs_no_improve += 1
                if epoch % self.config.training_params.get("save_checkpoint_freq_epochs", 5) == 0 or is_best:
                    self.save_model_checkpoint(epoch, prefix, is_best, self.best_val_loss)
                patience = int(self.config.training_params.get("early_stopping_patience", 20) or 0)
                if patience > 0 and self.epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}.");
                    break
            if self.config.training_params.ddp.use_ddp and self.world_size > 1: dist.barrier()
        return self.best_val_loss

    def pretrain_gae(self, train_dataloader, val_dataloader=None):
        if self.rank == 0: print("Starting GAE pretraining (controlled by `param_groups` in config)...")
        num_epochs_gae = self.config.training_params.get("num_epochs_gae_pretrain", 50)
        self.train_loop(train_dataloader, val_dataloader, num_epochs_gae, "pretrain_gae", "gae_pretrain")
        if self.rank == 0: print("GAE pretraining finished.")

    def train_joint(self, train_dataloader, val_dataloader=None):
        if self.rank == 0: print("Starting Joint training (controlled by `param_groups` in config)...")
        if not isinstance(self.model.module if isinstance(self.model, DDP) else self.model, CombinedModel):
            raise TypeError("Model for joint training must be CombinedModel.")
        num_epochs_joint = self.config.training_params.get("num_epochs_joint_train", 100)
        self.train_loop(train_dataloader, val_dataloader, num_epochs_joint, "train_joint", "joint_train")
        if self.rank == 0: print("Joint training finished.")

    def save_model_checkpoint(self, epoch, prefix, is_best, best_metric):
        if self.rank != 0: return
        state = {'epoch': epoch + 1,
                 'model_state_dict': (self.model.module if isinstance(self.model, DDP) else self.model).state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                 'best_metric': best_metric, 'config_dict': OmegaConf.to_container(self.config, resolve=True)}
        checkpoint_dir = self.config.training_params.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_checkpoint(state, os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch + 1}.pt"))
        if is_best: save_checkpoint(state, os.path.join(checkpoint_dir, f"{prefix}_best.pt"))

    def load_model_checkpoint(self, filepath):
        self.current_epoch, self.best_val_loss = load_checkpoint(filepath, self.model, self.optimizer, self.scheduler,
                                                                 self.device)
        if self.rank == 0: print(f"Model state loaded from {filepath}. Resuming from epoch {self.current_epoch}.")
