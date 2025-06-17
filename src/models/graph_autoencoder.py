import torch
import torch.nn as nn
from .gnn_encoder import GNNEncoder
from .gnn_decoder import GNNDecoder


class GraphAutoencoder(nn.Module):
    def __init__(self, model_config_params):
        super(GraphAutoencoder, self).__init__()

        encoder_cfg = model_config_params.encoder
        decoder_cfg = model_config_params.decoder

        # Handle case where num_genes is populated later by data_loader
        _num_genes = int(model_config_params.num_genes) if model_config_params.num_genes is not None else -1
        if _num_genes == -1 and model_config_params.get("num_genes_from_data", None):
            _num_genes = int(model_config_params.num_genes_from_data)
        if _num_genes == -1:
            raise ValueError("num_genes is None and could not be inferred for GraphAutoencoder init.")

        self.latent_dim = int(model_config_params.latent_dim)

        self.is_variational_encoder = bool(encoder_cfg.get("is_variational", False))

        enc_gnn_hidden_dims = [int(d) for d in encoder_cfg.gnn_hidden_dims]
        enc_pooling_ratios_conf = encoder_cfg.pooling_ratios
        enc_pooling_ratios = [float(p) if p is not None else None for p in enc_pooling_ratios_conf] \
            if enc_pooling_ratios_conf is not None else [None] * len(enc_gnn_hidden_dims)

        final_mlp_layers_conf = encoder_cfg.get("final_mlp_layers", None)
        enc_final_mlp_layers = [int(d) for d in final_mlp_layers_conf] if final_mlp_layers_conf else None

        self.encoder = GNNEncoder(
            input_dim=1,
            gnn_hidden_dims=enc_gnn_hidden_dims,
            latent_dim=self.latent_dim,
            pooling_ratios=enc_pooling_ratios,
            gnn_layer_type=str(encoder_cfg.gnn_layer_type),
            pooling_type=str(encoder_cfg.pooling_type),
            gnn_activation=str(encoder_cfg.gnn_activation),
            dropout_rate=float(encoder_cfg.dropout_rate),
            final_mlp_layers=enc_final_mlp_layers,
            is_variational=self.is_variational_encoder,
            log_var_clamp_min=float(encoder_cfg.get("log_var_clamp_min", -10.0)),
            log_var_clamp_max=float(encoder_cfg.get("log_var_clamp_max", 10.0))
        )

        dec_gnn_hidden_dims = [int(d) for d in decoder_cfg.gnn_hidden_dims]
        output_activation_conf = decoder_cfg.get("output_activation", None)
        dec_output_activation = str(output_activation_conf) if output_activation_conf is not None else None

        dec_distribution_type = str(decoder_cfg.get("distribution", "gaussian"))
        dec_output_params_per_gene = int(
            decoder_cfg.get("num_dist_params", 2 if dec_distribution_type == "gaussian" else 1))

        self.decoder = GNNDecoder(
            latent_dim=self.latent_dim,
            gnn_hidden_dims=dec_gnn_hidden_dims,
            output_params_per_gene=dec_output_params_per_gene,
            gnn_layer_type=str(decoder_cfg.gnn_layer_type),
            gnn_activation=str(decoder_cfg.gnn_activation),
            dropout_rate=float(decoder_cfg.dropout_rate),
            output_activation=dec_output_activation,
            distribution_type=dec_distribution_type
        )

    def encode(self, data, return_pooling_details=False):
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        # GNNEncoder returns a tuple of 5, where mu and log_var are None if not variational
        sampled_z_nodes, mu_nodes, log_var_nodes, batch_vector, pooling_details = \
            self.encoder(data, return_pooling_details=return_pooling_details)

        return sampled_z_nodes, mu_nodes, log_var_nodes, batch_vector, pooling_details

    def decode(self, sampled_z_nodes, batch_vector, pooling_details_list):
        return self.decoder(sampled_z_nodes, batch_vector, pooling_details_list)

    def forward(self, data, return_pooling_details=False):
        if not hasattr(data, 'x') or data.x is None:
            raise ValueError("Input data object must have 'x' attribute (node features).")
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            raise ValueError("Input data object must have 'edge_index' attribute.")
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        sampled_z_nodes, mu_nodes, log_var_nodes, z_batch, pooling_details = self.encode(
            data, return_pooling_details=True
        )

        reconstructed_params, reconstructed_batch = self.decode(
            sampled_z_nodes, z_batch, pooling_details
        )

        outputs = {
            "reconstructed_params": reconstructed_params,
            "sampled_z_nodes": sampled_z_nodes,
            "mu_nodes": mu_nodes,
            "log_var_nodes": log_var_nodes,
            "z_batch": z_batch,
        }
        if return_pooling_details:
            outputs["pooling_details"] = pooling_details

        return outputs
