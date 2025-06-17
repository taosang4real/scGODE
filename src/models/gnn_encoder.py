import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TopKPooling, SAGPooling, global_mean_pool


def get_gnn_layer(layer_type, in_channels, out_channels, **kwargs):
    """Helper function to get a GNN layer from its name."""
    if str(layer_type).lower() == "gcnconv":
        return GCNConv(in_channels, out_channels, **kwargs)
    elif str(layer_type).lower() == "gatconv":
        return GATConv(in_channels, out_channels, **kwargs)
    elif str(layer_type).lower() == "sageconv":
        return SAGEConv(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unsupported GNN layer type: {layer_type}")


def get_pooling_layer(pool_type, ratio, in_channels, **kwargs):
    """Helper function to get a pooling layer from its name."""
    if pool_type.lower() == "topkpooling":
        return TopKPooling(in_channels, ratio=ratio, **kwargs)
    elif pool_type.lower() == "sagpooling":
        return SAGPooling(in_channels, ratio=ratio, **kwargs)
    else:
        raise ValueError(f"Unsupported pooling layer type: {pool_type}")


def get_activation_function(activation_name):
    """Helper function to get an activation function from its name."""
    if activation_name is None: return nn.Identity()
    name = str(activation_name).lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, gnn_hidden_dims, latent_dim,
                 pooling_ratios, gnn_layer_type="GCNConv", pooling_type="TopKPooling",
                 gnn_activation="relu", dropout_rate=0.1, final_mlp_layers=None,
                 is_variational=False, log_var_clamp_min=-10.0, log_var_clamp_max=10.0):
        super(GNNEncoder, self).__init__()

        self.is_variational = is_variational
        self.latent_dim = latent_dim
        self.log_var_clamp_min = log_var_clamp_min
        self.log_var_clamp_max = log_var_clamp_max

        self.gnn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.activation_fn = get_activation_function(gnn_activation)

        current_dim = input_dim
        for i, hidden_dim in enumerate(gnn_hidden_dims):
            self.gnn_layers.append(get_gnn_layer(gnn_layer_type, current_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout_rate))

            if i < len(pooling_ratios) and pooling_ratios[i] is not None and 0 < pooling_ratios[i] < 1:
                self.pool_layers.append(get_pooling_layer(pooling_type, pooling_ratios[i], hidden_dim))
            else:
                self.pool_layers.append(None)
            current_dim = hidden_dim

        final_projection_out_dim = 2 * self.latent_dim if self.is_variational else self.latent_dim
        self.final_projection = nn.Linear(current_dim, final_projection_out_dim)

    def reparameterize(self, mu, logvar):
        if not self.training: return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, return_pooling_details=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        if x.dim() == 1: x = x.unsqueeze(-1)

        pooling_details_list = []

        for i in range(len(self.gnn_layers)):
            num_nodes_before_pool = x.size(0)

            x = self.gnn_layers[i](x, edge_index, edge_weight=edge_weight)

            if x.size(0) > 0:
                if self.bn_layers[i] is not None and (x.size(0) > 1 or not self.bn_layers[i].track_running_stats):
                    x = self.bn_layers[i](x)
                x = self.activation_fn(x)
                x = self.dropout_layers[i](x)

            if i < len(self.pool_layers) and self.pool_layers[i] is not None and x.size(0) > 0:
                edge_index_before_pool_for_detail = edge_index.clone() if return_pooling_details else None
                batch_before_pool_for_detail = batch.clone() if return_pooling_details else None
                pool_out = self.pool_layers[i](x, edge_index, edge_attr=edge_weight, batch=batch)
                x, edge_index, edge_weight, batch, perm, score = pool_out

                if return_pooling_details:
                    pooling_details_list.append({
                        "num_nodes_before_pool": num_nodes_before_pool, "num_nodes_after_pool": x.size(0),
                        "perm": perm, "score": score,
                        "edge_index_before_pool": edge_index_before_pool_for_detail,
                        "edge_index_after_pool": edge_index.clone(),
                        "batch_before_pool": batch_before_pool_for_detail,
                        "batch_after_pool": batch.clone() if batch is not None else None,
                    })

        projected_output = self.final_projection(x) if x.size(0) > 0 else torch.empty(
            (0, self.final_projection.out_features), device=x.device, dtype=x.dtype)

        mu_nodes, log_var_nodes, sampled_z_nodes = None, None, None
        if self.is_variational:
            if projected_output.size(0) > 0:
                mu_nodes = projected_output[:, :self.latent_dim]
                log_var_nodes = torch.clamp(projected_output[:, self.latent_dim:], self.log_var_clamp_min,
                                            self.log_var_clamp_max)
                sampled_z_nodes = self.reparameterize(mu_nodes, log_var_nodes)
            else:
                empty_latent = torch.empty((0, self.latent_dim), device=projected_output.device,
                                           dtype=projected_output.dtype)
                mu_nodes, log_var_nodes, sampled_z_nodes = empty_latent, empty_latent, empty_latent
        else:
            sampled_z_nodes = projected_output

        return sampled_z_nodes, mu_nodes, log_var_nodes, batch, pooling_details_list if return_pooling_details else None
