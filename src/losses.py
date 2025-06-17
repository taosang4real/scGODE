import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from omegaconf import OmegaConf

try:
    import geomloss
except ImportError:
    print("geomloss not found. Please install it for OT loss: pip install geomloss")
    geomloss = None


class ReconstructionLoss(nn.Module):
    def __init__(self, distribution_type="gaussian", deterministic_criterion="mse"):
        super(ReconstructionLoss, self).__init__()
        self.distribution_type = distribution_type.lower()
        self.deterministic_criterion_str = deterministic_criterion.lower()

        if self.distribution_type == "deterministic":
            if self.deterministic_criterion_str == "mse":
                self.criterion = nn.MSELoss()
            elif self.deterministic_criterion_str == "l1":
                self.criterion = nn.L1Loss()
            else:
                raise ValueError(f"Unsupported criterion for deterministic reconstruction: {deterministic_criterion}")
        elif self.distribution_type == "gaussian":
            pass
        else:
            raise ValueError(f"Unsupported distribution type in ReconstructionLoss init: {self.distribution_type}")

    def forward(self, predicted_params, target_x):
        if predicted_params is None or target_x is None:
            ref_tensor = predicted_params if predicted_params is not None else target_x
            device = ref_tensor.device if ref_tensor is not None else 'cpu'
            return torch.tensor(0.0, device=device)

        if self.distribution_type == "gaussian":
            if predicted_params.shape[-1] != 2:
                raise ValueError(
                    f"Gaussian distribution expects 2 parameters, got {predicted_params.shape[-1]} from predicted_params shape {predicted_params.shape}")

            mu = predicted_params[..., 0]
            std = predicted_params[..., 1]

            if not (std > 1e-6).all():
                std = torch.clamp(std, min=1e-6)

            try:
                dist = torch.distributions.Normal(mu, std)
                nll = -dist.log_prob(target_x)
            except ValueError as e:
                print(f"Error creating Normal distribution or calculating log_prob: {e}")
                print(f"Mu shape: {mu.shape}, Std shape: {std.shape}, Target shape: {target_x.shape}")
                if mu.numel() > 0 and std.numel() > 0:
                    print(
                        f"Mu min/max: {mu.min().item()}/{mu.max().item()}, Std min/max: {std.min().item()}/{std.max().item()}")
                raise e
            return nll.sum() / target_x.numel() if target_x.numel() > 0 else torch.tensor(0.0, device=target_x.device)

        elif self.distribution_type == "deterministic":
            pred_x = predicted_params
            if pred_x.dim() == target_x.dim() + 1 and pred_x.shape[-1] == 1:
                pred_x = pred_x.squeeze(-1)

            if pred_x.shape != target_x.shape:
                if pred_x.numel() == target_x.numel():
                    try:
                        pred_x = pred_x.view_as(target_x)
                    except RuntimeError:
                        raise ValueError(
                            f"Shape mismatch for deterministic loss. Predicted: {pred_x.shape}, Target: {target_x.shape}")
                else:
                    raise ValueError(
                        f"Shape mismatch for deterministic loss. Predicted: {pred_x.shape}, Target: {target_x.shape}")
            return self.criterion(pred_x, target_x)
        else:
            raise ValueError(f"Forward pass for distribution type '{self.distribution_type}' not implemented.")


class OptimalTransportLoss(nn.Module):
    def __init__(self, cost_type="L2", blur=0.05, scaling=0.9, reach=None, backend="auto", p=2):
        super(OptimalTransportLoss, self).__init__()
        if geomloss is None:
            print("Warning: geomloss not found. OT Loss will be skipped if used.")
            self.ot_loss = None
            return

        p_val = 1 if cost_type.lower() == "l1" else 2 if cost_type.lower() == "l2" else p

        self.ot_loss = geomloss.SamplesLoss(
            loss="sinkhorn", p=p_val, blur=blur, scaling=scaling,
            reach=reach, backend=backend, debias=True
        )

    def forward(self, x_features, y_features, x_batch=None, y_batch=None):
        if self.ot_loss is None: return torch.tensor(0.0, device=x_features.device if isinstance(x_features,
                                                                                                 torch.Tensor) else 'cpu')
        if x_features.ndim == 1: x_features = x_features.unsqueeze(0)
        if y_features.ndim == 1: y_features = y_features.unsqueeze(0)
        if x_features.size(0) == 0 or y_features.size(0) == 0: return torch.tensor(0.0, device=x_features.device)
        if torch.isnan(x_features).any() or torch.isinf(x_features).any():
            print("Warning: NaNs or Infs found in x_features for OT loss. Returning 0 loss.")
            return torch.tensor(0.0, device=x_features.device, dtype=x_features.dtype)
        if torch.isnan(y_features).any() or torch.isinf(y_features).any():
            print("Warning: NaNs or Infs found in y_features for OT loss. Returning 0 loss.")
            return torch.tensor(0.0, device=y_features.device, dtype=y_features.dtype)
        return self.ot_loss(x_features, y_features)


class KLDivergenceLoss(nn.Module):
    def forward(self, mu, log_var):
        if mu is None or log_var is None or mu.numel() == 0 or log_var.numel() == 0:
            return torch.tensor(0.0, device=mu.device if mu is not None else 'cpu')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return kl_div.mean()


class CNFNLLLoss(nn.Module):
    def forward(self, initial_log_p_z0, delta_logp_t1):
        if initial_log_p_z0 is None or delta_logp_t1 is None:
            return torch.tensor(0.0, device=(initial_log_p_z0 or delta_logp_t1 or torch.tensor(0.0)).device)
        return -(initial_log_p_z0 + delta_logp_t1)


class CombinedLoss(nn.Module):
    def __init__(self, loss_weights_config, config, device='cpu'):
        super(CombinedLoss, self).__init__()
        self.weights = loss_weights_config
        self.config = config
        self.device = device
        self.is_variational_encoder = bool(self.config.model_params.encoder.get("is_variational", False))
        self.node_type = str(self.config.model_params.node.get("type", "NeuralODE")).upper()

        decoder_conf = self.config.model_params.decoder
        self.recon_loss_fn = ReconstructionLoss(distribution_type=decoder_conf.get("distribution", "gaussian"),
                                                deterministic_criterion=decoder_conf.get("deterministic_criterion",
                                                                                         "mse"))

        if self.is_variational_encoder: self.kl_loss_fn = KLDivergenceLoss()
        if self.node_type.startswith("CNF"): self.cnf_nll_loss_fn = CNFNLLLoss()

        ot_params_conf = self.config.model_params.get("ot_params", OmegaConf.create({}))
        ot_blur, ot_scaling, ot_reach = float(ot_params_conf.get("blur", 0.05)), float(
            ot_params_conf.get("scaling", 0.8)), ot_params_conf.get("reach", None)
        if ot_reach is not None: ot_reach = float(ot_reach)
        if geomloss is not None:
            self.ot_loss_latent_fn = self.ot_loss_expression_fn = self.ot_loss_reg_fn = OptimalTransportLoss(
                blur=ot_blur, scaling=ot_scaling, reach=ot_reach)
        else:
            self.ot_loss_latent_fn = self.ot_loss_expression_fn = self.ot_loss_reg_fn = None

    def _get_weight(self, loss_name):
        return float(self.weights.get(loss_name, 0.0))

    def _calculate_log_p_z_given_params(self, z, mu, log_var):
        if z is None or mu is None or log_var is None: return None
        if z.shape != mu.shape or z.shape != log_var.shape: return None
        try:
            return torch.distributions.Normal(mu, torch.exp(0.5 * log_var)).log_prob(z).sum(dim=-1).mean()
        except ValueError:
            return None

    def forward(self, model_outputs, original_data, mode="train_joint", current_kl_weight=None):
        total_loss, loss_dict = torch.tensor(0.0, device=self.device), {}
        joint_training_mode = self.config.training_params.get("joint_training_mode", "end_to_end")
        freeze_encoder = self.config.training_params.get("freeze_encoder_during_joint_train", False)

        # Determine the KL weight to use for this forward pass
        kl_weight = current_kl_weight if current_kl_weight is not None else self._get_weight("L_KL_latent")

        # Define the local helper function to add loss terms
        def add_loss(name, value, weight_override=None):
            nonlocal total_loss
            if value is not None and torch.isfinite(value):
                # Use override for weight if provided (for KL annealing), otherwise use from config
                weight = weight_override if weight_override is not None else self._get_weight(name)
                loss_dict[name] = value.item()
                total_loss += weight * value
            else:
                loss_dict[name] = float('nan') if value is not None else None

        # --- GAE Pretraining Mode ---
        if mode == "pretrain_gae":
            add_loss("L_recon_pretrain", self.recon_loss_fn(model_outputs.get("reconstructed_expressions_params"),
                                                            original_data.get("original_expressions")))
            if self.is_variational_encoder:
                add_loss("L_KL_pretrain",
                         self.kl_loss_fn(model_outputs.get("mu_nodes"), model_outputs.get("log_var_nodes")),
                         weight_override=kl_weight)

        # --- Joint Training / Evaluation Mode ---
        elif mode == "train_joint":
            if self.is_variational_encoder and not freeze_encoder:
                add_loss("L_KL_t0",
                         self.kl_loss_fn(model_outputs.get("mu_t0_nodes"), model_outputs.get("log_var_t0_nodes")),
                         weight_override=kl_weight)

            # For t1_real KL, just record it, don't add to loss
            l_kl_t1_real_val = self.kl_loss_fn(model_outputs.get("mu_t1_real_nodes"),
                                               model_outputs.get("log_var_t1_real_nodes"))
            if l_kl_t1_real_val is not None: loss_dict["L_KL_t1_real"] = l_kl_t1_real_val.item()

            if self.node_type.startswith("CNF"):
                initial_log_p = self._calculate_log_p_z_given_params(
                    model_outputs.get("z_t0_nodes_sampled_or_deterministic"), model_outputs.get("mu_t0_nodes"),
                    model_outputs.get("log_var_t0_nodes")) if self.is_variational_encoder else torch.tensor(0.0,
                                                                                                            device=self.device)
                add_loss("L_CNF_NLL", self.cnf_nll_loss_fn(initial_log_p, model_outputs.get("delta_logp_t1")))
                add_loss("L_kinetic_energy", model_outputs.get("kinetic_energy"))
                add_loss("L_jacobian_reg", model_outputs.get("jacobian_reg"))

            if self.ot_loss_latent_fn and model_outputs.get("z_t1_predicted_nodes") is not None:
                pred_z_global = global_mean_pool(model_outputs["z_t1_predicted_nodes"], model_outputs["z_t0_batch"]) if \
                model_outputs["z_t1_predicted_nodes"].numel() > 0 else torch.empty(0, device=self.device)
                real_z_global = global_mean_pool(model_outputs.get("z_t1_real_nodes_sampled_or_deterministic"),
                                                 model_outputs.get("z_t1_real_batch")) if model_outputs.get(
                    "z_t1_real_nodes_sampled_or_deterministic") is not None and model_outputs[
                                                                                              "z_t1_real_nodes_sampled_or_deterministic"].numel() > 0 else torch.empty(
                    0, device=self.device)
                if pred_z_global.numel() > 0 and real_z_global.numel() > 0:
                    add_loss("L_OT_latent", self.ot_loss_latent_fn(pred_z_global, real_z_global))

            if joint_training_mode == "end_to_end":
                add_loss("L_recon_t0", self.recon_loss_fn(model_outputs.get("x_t0_reconstructed_params"),
                                                          original_data.get("original_expressions_t0")))
                add_loss("L_recon_t1_predicted",
                         self.recon_loss_fn(model_outputs.get("x_t1_predicted_reconstructed_params"),
                                            original_data.get("original_expressions_t1")))
                if self.ot_loss_expression_fn:
                    pred_mean = _get_point_estimate_from_params(
                        model_outputs.get("x_t1_predicted_reconstructed_params"),
                        self.config.model_params.decoder.get("distribution"))
                    if pred_mean is not None and pred_mean.numel() > 0:
                        add_loss("L_OT_expression",
                                 self.ot_loss_expression_fn(pred_mean, original_data.get("original_expressions_t1")))

            if self.ot_loss_reg_fn and model_outputs.get(
                    "z_trajectory_solutions") is not None and self.node_type == "NEURALODE" and \
                    model_outputs["z_trajectory_solutions"].shape[0] > 1:
                traj, batch_vec = model_outputs["z_trajectory_solutions"], model_outputs["z_t0_batch"]
                if batch_vec is not None:
                    ot_reg_vals = [self.ot_loss_reg_fn(global_mean_pool(traj[i], batch_vec),
                                                       global_mean_pool(traj[i + 1], batch_vec)) for i in
                                   range(traj.shape[0] - 1)]
                    if ot_reg_vals: add_loss("L_OT_reg_trajectory", torch.stack(ot_reg_vals).mean())

        loss_dict["total_loss"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return total_loss, loss_dict


def _get_point_estimate_from_params(params_tensor, distribution_type, device='cpu'):
    if params_tensor is None or params_tensor.numel() == 0: return torch.empty(0, device=device)
    if distribution_type == "gaussian": return params_tensor[..., 0]
    if distribution_type == "deterministic": return params_tensor.squeeze(-1) if params_tensor.dim() == 3 and \
                                                                                 params_tensor.shape[
                                                                                     -1] == 1 else params_tensor
    if params_tensor.dim() > 1 and params_tensor.shape[-1] > 0: return params_tensor[..., 0]
    return params_tensor
