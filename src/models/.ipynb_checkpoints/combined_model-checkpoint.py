import torch
import torch.nn as nn
from .graph_autoencoder import GraphAutoencoder
from .neural_ode import NeuralODE
from torch_geometric.nn import global_mean_pool # Updated import


class CombinedModel(nn.Module):
    def __init__(self, model_config):
        super(CombinedModel, self).__init__()
        self.config = model_config
        self.latent_dim = model_config.model_params.latent_dim

        self.graph_autoencoder = GraphAutoencoder(model_config.model_params)
        
        self.neural_ode = NeuralODE(model_config.model_params, node_specific_config=model_config.model_params.node)

    def forward(self, data_t0, t0_scalar, t1_scalar, data_t1=None, return_latent_t1_real=False):
        z_t0_nodes, z_t0_batch, pooling_details_t0 = self.graph_autoencoder.encode(
            data_t0, return_pooling_details=True
        )

        x_t0_reconstructed, _ = self.graph_autoencoder.decode(
            z_t0_nodes, z_t0_batch, pooling_details_t0
        )

        if not pooling_details_t0:
            raise ValueError("Pooling details from t0 encoder are missing, cannot determine NODE graph structure.")
        
        edge_index_node = pooling_details_t0[-1]["edge_index_after_pool"]
        edge_weight_node = None 

        z_t1_predicted_nodes = self.neural_ode.solve_to_t1(
            z_t0_nodes, t0_scalar, t1_scalar,
            edge_index=edge_index_node,
            edge_weight=edge_weight_node,
            batch_vector=z_t0_batch
        )
        
        x_t1_predicted_reconstructed, _ = self.graph_autoencoder.decode(
            z_t1_predicted_nodes, z_t0_batch, pooling_details_t0 
        )

        outputs = {
            "x_t0_reconstructed": x_t0_reconstructed,
            "z_t0_nodes": z_t0_nodes,
            "z_t0_batch": z_t0_batch,
            "z_t1_predicted_nodes": z_t1_predicted_nodes,
            "x_t1_predicted_reconstructed": x_t1_predicted_reconstructed,
            "pooling_details_t0": pooling_details_t0
        }

        if return_latent_t1_real and data_t1 is not None:
            z_t1_real_nodes, z_t1_real_batch, _ = self.graph_autoencoder.encode(
                data_t1, return_pooling_details=False 
            )
            outputs["z_t1_real_nodes"] = z_t1_real_nodes
            outputs["z_t1_real_batch"] = z_t1_real_batch
            
        return outputs

    def freeze_gae_parameters(self):
        for param in self.graph_autoencoder.parameters():
            param.requires_grad = False
        print("GraphAutoencoder parameters frozen.")

    def unfreeze_gae_parameters(self):
        for param in self.graph_autoencoder.parameters():
            param.requires_grad = True
        print("GraphAutoencoder parameters unfrozen.")

