import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import unbatch, unbatch_edge_index


def get_gnn_layer(layer_type, in_channels, out_channels, **kwargs):
    if layer_type.lower() == "gcnconv":
        return GCNConv(in_channels, out_channels, **kwargs)
    elif layer_type.lower() == "gatconv":
        return GATConv(in_channels, out_channels, **kwargs)
    elif layer_type.lower() == "sageconv":
        return SAGEConv(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unsupported GNN layer type: {layer_type}")


def get_activation_function(activation_name):
    if activation_name is None:
        return nn.Identity()
    if activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "elu":
        return nn.ELU()
    elif activation_name.lower() == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_name.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation_name.lower() == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, gnn_hidden_dims, output_params_per_gene,
                 gnn_layer_type="GCNConv", gnn_activation="relu",
                 dropout_rate=0.1, output_activation=None,
                 distribution_type="gaussian"):
        super(GNNDecoder, self).__init__()

        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.activation_fn = get_activation_function(gnn_activation)
        self.output_activation_fn = get_activation_function(output_activation)
        self.distribution_type = distribution_type
        self.output_params_per_gene = output_params_per_gene

        _gnn_hidden_dims = [int(d) for d in gnn_hidden_dims]
        _dropout_rate = float(dropout_rate)
        _latent_dim = int(latent_dim)

        current_dim = _latent_dim
        for i, hidden_dim in enumerate(_gnn_hidden_dims):
            self.gnn_layers.append(get_gnn_layer(str(gnn_layer_type), current_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(_dropout_rate))
            current_dim = hidden_dim

        self.final_projection = nn.Linear(current_dim, self.output_params_per_gene)

    def forward(self, x_latent, batch_latent, pooling_details_list):
        current_x = x_latent
        current_batch = batch_latent
        current_edge_index = None
        current_edge_weight = None

        if not pooling_details_list and (len(self.gnn_layers) > 0):
            if hasattr(x_latent, 'edge_index') and x_latent.edge_index is not None:
                current_edge_index = x_latent.edge_index
                current_edge_weight = x_latent.edge_attr if hasattr(x_latent, 'edge_attr') else None

        num_decoder_gnn_layers = len(self.gnn_layers)
        num_pool_stages_in_encoder = len(pooling_details_list) if pooling_details_list else 0

        for i in range(num_decoder_gnn_layers):
            pool_detail_idx = num_pool_stages_in_encoder - 1 - i

            if pooling_details_list and 0 <= pool_detail_idx < num_pool_stages_in_encoder:
                detail = pooling_details_list[pool_detail_idx]
                perm = detail.get("perm")

                if perm is not None:
                    num_nodes_target = detail["num_nodes_before_pool"]
                    edge_index_target = detail["edge_index_before_pool"]
                    batch_target = detail["batch_before_pool"]
                    edge_attr_target = detail.get("edge_attr_before_pool")

                    if current_x.size(0) == 0 and num_nodes_target > 0:
                        current_x = torch.zeros(num_nodes_target,
                                                current_x.size(1) if current_x.dim() > 1 and current_x.size(1) > 0 else
                                                self.gnn_layers[i].in_channels,
                                                device=x_latent.device, dtype=x_latent.dtype)
                    elif current_x.size(0) > 0:
                        unpooled_x = torch.zeros(num_nodes_target, current_x.size(1),
                                                 device=current_x.device, dtype=current_x.dtype)

                        if perm.max() >= unpooled_x.size(0):
                            raise ValueError(
                                f"perm.max() {perm.max().item()} is out of bounds for unpooled_x.size(0) {unpooled_x.size(0)}")

                        min_size = min(current_x.size(0), perm.size(0))
                        unpooled_x[perm[:min_size]] = current_x[:min_size]

                        current_x = unpooled_x

                    current_edge_index = edge_index_target
                    current_batch = batch_target
                    current_edge_weight = edge_attr_target

                elif detail:
                    current_edge_index = detail["edge_index_after_pool"]
                    current_batch = detail["batch_after_pool"]
                    current_edge_weight = detail.get("edge_attr_after_pool")

            if current_x.size(0) > 0:
                if current_edge_index is None:
                    if i == 0 and hasattr(x_latent, 'edge_index') and x_latent.edge_index is not None:
                        current_edge_index = x_latent.edge_index
                        current_edge_weight = x_latent.edge_attr if hasattr(x_latent, 'edge_attr') else None
                    else:
                        raise ValueError(f"Decoder GNN layer {i} has no edge_index.")

                if current_edge_index.numel() > 0 and current_x.size(0) <= current_edge_index.max():
                    raise RuntimeError(
                        f"Decoder GNN layer {i}: Max index in edge_index ({current_edge_index.max().item()}) "
                        f"is out of bounds for current_x node count ({current_x.size(0)}).")

                current_x = self.gnn_layers[i](current_x, current_edge_index, edge_weight=current_edge_weight)
                if current_x.size(0) > 0:
                    if self.bn_layers[i] is not None:
                        if current_x.size(0) > 1 or not self.bn_layers[i].track_running_stats:
                            current_x = self.bn_layers[i](current_x)
                    current_x = self.activation_fn(current_x)
                    current_x = self.dropout_layers[i](current_x)
            else:
                break

        if current_x.size(0) > 0:
            params_raw = self.final_projection(current_x)

            if self.distribution_type == "gaussian":
                if params_raw.shape[-1] != 2:
                    raise ValueError(
                        f"Expected {self.output_params_per_gene} parameters for Gaussian, got {params_raw.shape[-1]}")

                mu = params_raw[..., 0]
                raw_std_param = params_raw[..., 1]

                mu = self.output_activation_fn(mu)
                std = F.softplus(raw_std_param) + 1e-6

                reconstructed_params = torch.stack([mu, std], dim=-1)
            else:
                if self.output_params_per_gene == 1 and params_raw.shape[-1] == 1:
                    reconstructed_params = self.output_activation_fn(params_raw)
                elif self.output_params_per_gene == params_raw.shape[-1]:
                    reconstructed_params = self.output_activation_fn(params_raw)
                else:
                    raise ValueError(f"Output params per gene {self.output_params_per_gene} "
                                     f"mismatch with final_projection output {params_raw.shape[-1]} for dist {self.distribution_type}")

        else:
            reconstructed_params = torch.empty((0, self.output_params_per_gene),
                                               device=x_latent.device, dtype=x_latent.dtype)

        return reconstructed_params, current_batch
