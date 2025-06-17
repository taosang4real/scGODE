import torch
import torch.nn as nn
# Corrected import path for helper functions
from .gnn_encoder import get_pooling_layer, get_activation_function


def hutchinson_trace_estimator(func_dz_dt_from_z, z, num_samples=1):
    divergence_accum = 0.0
    for i in range(num_samples):
        noise = torch.randn_like(z, device=z.device)

        current_z_requires_grad = z.requires_grad
        if not current_z_requires_grad: z.requires_grad_(True)

        f_output = func_dz_dt_from_z(z)

        jv_prod_or_vjp = torch.autograd.grad(
            outputs=f_output, inputs=z, grad_outputs=noise,
            create_graph=True, retain_graph=True
        )[0]

        if not current_z_requires_grad: z.requires_grad_(False)

        divergence_sample = (jv_prod_or_vjp * noise).sum(dim=-1)
        divergence_accum = divergence_accum + divergence_sample.sum()

    return divergence_accum / num_samples


class CNFDynamicsGNN(nn.Module):
    def __init__(self, latent_dim, hidden_dims, gnn_layer_type="GCNConv",
                 activation="relu", dropout_rate=0.1, time_dependent=False,
                 hutchinson_samples=1):
        super(CNFDynamicsGNN, self).__init__()

        self.latent_dim = int(latent_dim)
        self.time_dependent = time_dependent
        self.layer_type = str(gnn_layer_type).lower()

        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.activation_fn = get_activation_function(activation)

        _hidden_dims = [int(d) for d in hidden_dims]
        _dropout_rate = float(dropout_rate)

        current_f_input_dim = self.latent_dim
        if self.time_dependent:
            current_f_input_dim += 1

        for i, h_dim in enumerate(_hidden_dims):
            self.layers.append(get_processing_layer(self.layer_type, current_f_input_dim, h_dim))
            self.bn_layers.append(nn.BatchNorm1d(h_dim))
            self.dropout_layers.append(nn.Dropout(_dropout_rate))
            current_f_input_dim = h_dim

        self.final_layer = get_processing_layer(self.layer_type, current_f_input_dim, self.latent_dim)

    def get_dz_dt(self, t, z_nodes, edge_index, edge_weight=None, batch_vector=None):
        h = z_nodes
        if self.time_dependent and t is not None:
            if h.size(0) > 0:
                time_vec = torch.full((h.size(0), 1), float(t), device=h.device, dtype=h.dtype)
                h = torch.cat([h, time_vec], dim=1)

        for i in range(len(self.layers)):
            if h.size(0) == 0: break

            if self.layer_type == 'mlp':
                h = self.layers[i](h)
            else:
                h = self.layers[i](h, edge_index, edge_weight=edge_weight)

            if h.size(0) > 0:
                if self.bn_layers[i] is not None:
                    if h.size(0) > 1 or not self.bn_layers[i].track_running_stats:
                        h = self.bn_layers[i](h)
                h = self.activation_fn(h)
                h = self.dropout_layers[i](h)

        if h.size(0) > 0:
            if self.layer_type == 'mlp':
                dz_dt = self.final_layer(h)
            else:
                dz_dt = self.final_layer(h, edge_index, edge_weight=edge_weight)
        else:
            dz_dt = torch.zeros_like(z_nodes)
        return dz_dt
