import torch
import torch.nn as nn
# Import the helper functions from gnn_encoder, which is the correct source
from .gnn_encoder import get_gnn_layer, get_activation_function


class NODEDynamics(nn.Module):
    def __init__(self, latent_dim, hidden_dims, gnn_layer_type="GCNConv",
                 activation="relu", dropout_rate=0.1, time_dependent=False):
        super(NODEDynamics, self).__init__()

        self.latent_dim = int(latent_dim)
        self.time_dependent = time_dependent

        _hidden_dims = [int(d) for d in hidden_dims]
        _dropout_rate = float(dropout_rate)

        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.activation_fn = get_activation_function(activation)

        current_dim = self.latent_dim
        if self.time_dependent:
            current_dim += 1

        for i, h_dim in enumerate(_hidden_dims):
            # This now correctly uses get_gnn_layer
            self.gnn_layers.append(get_gnn_layer(str(gnn_layer_type), current_dim, h_dim))
            self.bn_layers.append(nn.BatchNorm1d(h_dim))
            self.dropout_layers.append(nn.Dropout(_dropout_rate))
            current_dim = h_dim

        self.final_layer = get_gnn_layer(str(gnn_layer_type), current_dim, self.latent_dim)

    def forward(self, t, x_nodes, edge_index, edge_weight=None, batch_vector=None):
        h = x_nodes

        if self.time_dependent and t is not None:
            if h.size(0) > 0:
                time_vec = torch.full((h.size(0), 1), float(t), device=h.device, dtype=h.dtype)
                h = torch.cat([h, time_vec], dim=1)

        for i in range(len(self.gnn_layers)):
            if h.size(0) == 0: break

            # The forward pass is now only for GNN layers
            h = self.gnn_layers[i](h, edge_index, edge_weight=edge_weight)

            if h.size(0) > 0:
                if self.bn_layers[i] is not None:
                    if h.size(0) > 1 or not self.bn_layers[i].track_running_stats:
                        h = self.bn_layers[i](h)
                h = self.activation_fn(h)
                h = self.dropout_layers[i](h)

        if h.size(0) > 0:
            dz_dt = self.final_layer(h, edge_index, edge_weight=edge_weight)
        else:
            dz_dt = torch.zeros_like(x_nodes)
        return dz_dt
