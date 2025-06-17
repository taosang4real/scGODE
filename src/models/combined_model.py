import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf

from .graph_autoencoder import GraphAutoencoder
from .neural_ode import NeuralODE
from .cnf_integrator import CNFIntegrator
from torch_geometric.nn import global_mean_pool


class CombinedModel(nn.Module):
    def __init__(self, config):
        super(CombinedModel, self).__init__()
        self.config = config
        self.model_params_conf = config.model_params
        self.latent_dim = int(self.model_params_conf.latent_dim)
        self.is_variational_encoder = bool(self.model_params_conf.encoder.get("is_variational", False))

        self.node_type = str(self.model_params_conf.node.get("type", "NeuralODE")).upper()

        self.autoencoder = GraphAutoencoder(self.model_params_conf)

        if self.node_type.startswith("CNF"):
            self.dynamics_model = CNFIntegrator(self.model_params_conf, cnf_node_config=self.model_params_conf.node)
        elif self.node_type == "NEURALODE":
            self.dynamics_model = NeuralODE(self.model_params_conf, node_specific_config=self.model_params_conf.node)
        else:
            raise ValueError(f"Unsupported node_type in config: {self.node_type}. Choose NeuralODE or CNFIntegrator.")

    @property
    def encoder(self):
        return self.autoencoder.encoder

    @property
    def decoder(self):
        return self.autoencoder.decoder

    def _run_dynamics(self, initial_z, t0, t1, edge_index, edge_weight, batch, num_reg_intervals):
        z_trajectory, delta_logp, kinetic_energy, jacobian_reg = None, None, None, None

        if initial_z.size(0) == 0:
            z_t1_pred = torch.empty((0, self.latent_dim), device=initial_z.device, dtype=initial_z.dtype)
            return z_t1_pred, None, None, None, None

        if self.node_type.startswith("CNF"):
            z_t1_pred, delta_logp, kinetic_energy, jacobian_reg = self.dynamics_model.solve_to_t1(
                initial_z, t0, t1, edge_index, edge_weight, batch
            )
        else:  # NeuralODE
            if num_reg_intervals > 0:
                _t0, _t1 = float(t0.item()), float(t1.item())
                time_points = torch.linspace(_t0, _t1, steps=num_reg_intervals + 2, device=initial_z.device)
                z_trajectory = self.dynamics_model(
                    initial_z, time_points, edge_index, edge_weight, batch
                )
                z_t1_pred = z_trajectory[-1]
            else:
                z_t1_pred = self.dynamics_model.solve_to_t1(
                    initial_z, t0, t1, edge_index, edge_weight, batch
                )

        return z_t1_pred, delta_logp, kinetic_energy, jacobian_reg, z_trajectory

    def forward(self, data_t0, t0_scalar, t1_scalar, data_t1=None,
                return_latent_t1_real=False, num_ot_reg_intervals=0,
                joint_training_mode="end_to_end"):

        z_t0_eff, mu_t0_nodes, log_var_t0_nodes, z_t0_batch, pooling_details_t0 = self.autoencoder.encode(
            data_t0, return_pooling_details=True
        )
        initial_z_for_dynamics = mu_t0_nodes if self.is_variational_encoder and mu_t0_nodes is not None else z_t0_eff

        outputs = {
            "z_t0_nodes_sampled_or_deterministic": z_t0_eff,
            "mu_t0_nodes": mu_t0_nodes,
            "log_var_t0_nodes": log_var_t0_nodes,
            "z_t0_batch": z_t0_batch,
            "pooling_details_t0": pooling_details_t0,
        }

        last_pool_detail = pooling_details_t0[-1] if pooling_details_t0 else {}
        edge_index_node = last_pool_detail.get("edge_index_after_pool")
        if edge_index_node is None and initial_z_for_dynamics is not None and initial_z_for_dynamics.numel() > 0:
            edge_index_node = data_t0.edge_index
        edge_weight_node = last_pool_detail.get("edge_attr_after_pool")

        z_t1_predicted_nodes, delta_logp_t1, kinetic_energy, jacobian_reg, z_trajectory_solutions = self._run_dynamics(
            initial_z_for_dynamics, t0_scalar, t1_scalar,
            edge_index_node, edge_weight_node, z_t0_batch, num_ot_reg_intervals
        )
        outputs["z_t1_predicted_nodes"] = z_t1_predicted_nodes
        outputs["delta_logp_t1"] = delta_logp_t1
        outputs["kinetic_energy"] = kinetic_energy
        outputs["jacobian_reg"] = jacobian_reg
        outputs["z_trajectory_solutions"] = z_trajectory_solutions

        if joint_training_mode == "end_to_end":
            x_t0_reconstructed_params, _ = self.autoencoder.decode(z_t0_eff, z_t0_batch, pooling_details_t0)
            x_t1_predicted_reconstructed_params, _ = self.autoencoder.decode(z_t1_predicted_nodes, z_t0_batch,
                                                                             pooling_details_t0)
            outputs["x_t0_reconstructed_params"] = x_t0_reconstructed_params
            outputs["x_t1_predicted_reconstructed_params"] = x_t1_predicted_reconstructed_params
        else:
            outputs["x_t0_reconstructed_params"] = None
            outputs["x_t1_predicted_reconstructed_params"] = None

        if return_latent_t1_real and data_t1 is not None:
            z_t1_real_nodes_eff, mu_t1_real_nodes, log_var_t1_real_nodes, z_t1_real_batch, _ = self.autoencoder.encode(
                data_t1, return_pooling_details=False
            )
            outputs["z_t1_real_nodes_sampled_or_deterministic"] = z_t1_real_nodes_eff
            outputs["mu_t1_real_nodes"] = mu_t1_real_nodes
            outputs["log_var_t1_real_nodes"] = log_var_t1_real_nodes
            outputs["z_t1_real_batch"] = z_t1_real_batch

        return outputs
