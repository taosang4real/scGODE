import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import TopKPooling, SAGPooling
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torchdiffeq import odeint_adjoint, odeint

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, encoder_hidden_dims, latent_dim, 
                 pooling_type="TopKPooling", pooling_ratios=None, 
                 dropout_rate=0.1, activation_fn_str="relu"):
        super(GNNEncoder, self).__init__()
        self.in_channels = in_channels
        self.encoder_hidden_dims = encoder_hidden_dims
        self.latent_dim = latent_dim # This is features on the final pooled graph nodes
        self.pooling_type = pooling_type
        self.pooling_ratios = pooling_ratios if pooling_ratios is not None else []
        self.dropout_rate = dropout_rate
        
        if activation_fn_str == "relu": self.activation_fn = F.relu
        elif activation_fn_str == "elu": self.activation_fn = F.elu
        elif activation_fn_str == "leaky_relu": self.activation_fn = F.leaky_relu
        else: self.activation_fn = F.relu
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        current_channels = in_channels
        for i, hidden_dim in enumerate(encoder_hidden_dims):
            self.convs.append(GCNConv(current_channels, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            if i < len(self.pooling_ratios) and 0.0 < self.pooling_ratios[i] < 1.0:
                if self.pooling_type == "TopKPooling":
                    self.pools.append(TopKPooling(hidden_dim, ratio=self.pooling_ratios[i]))
                elif self.pooling_type == "SAGPooling":
                    self.pools.append(SAGPooling(hidden_dim, ratio=self.pooling_ratios[i], GNN=GCNConv)) # SAGPool can take a GNN arg
                else:
                    self.pools.append(None)
            else:
                self.pools.append(None)
            current_channels = hidden_dim
        
        self.final_conv = GCNConv(current_channels, latent_dim)
        self.final_batch_norm = nn.BatchNorm1d(latent_dim)

    def forward(self, x, edge_index, edge_weight=None, batch=None, return_pooling_details=False):
        pooling_details_list = []
        current_x, current_edge_index, current_edge_weight, current_batch = x, edge_index, edge_weight, batch
        original_num_nodes_before_any_pooling = x.size(0)


        for i in range(len(self.convs)):
            original_x_shape_before_conv = current_x.shape
            original_edge_index_before_conv = current_edge_index
            original_batch_before_conv = current_batch

            current_x = self.convs[i](current_x, current_edge_index, current_edge_weight)
            current_x = self.batch_norms[i](current_x)
            current_x = self.activation_fn(current_x)
            current_x = F.dropout(current_x, p=self.dropout_rate, training=self.training)

            if self.pools[i] is not None:
                num_nodes_before_pool = current_x.size(0)
                pool_out = self.pools[i](current_x, current_edge_index, current_edge_weight, current_batch)
                
                pooled_x, pooled_edge_index, pooled_edge_weight, pooled_batch, perm, score = pool_out[0], pool_out[1], pool_out[2], pool_out[3], pool_out[4], pool_out[5] if len(pool_out) > 5 else None

                if return_pooling_details:
                    detail = {
                        'pooling_type': self.pooling_type,
                        'num_nodes_before_pool': num_nodes_before_pool,
                        'perm': perm, # Indices of kept nodes relative to input of this pooling layer
                        'original_x_shape_before_conv_at_this_level': original_x_shape_before_conv, # For unpooling feature dim
                        'original_edge_index_before_conv_at_this_level': original_edge_index_before_conv, # For GNN in unpool
                        'original_batch_before_conv_at_this_level': original_batch_before_conv, # For batching in unpool
                        'pooled_graph_size': pooled_x.size(0),
                        'score': score.clone().detach() if score is not None else None,
                        'pooled_edge_weight_for_unpool_gnn': pooled_edge_weight # Pass this to the GNN in unpool layer
                    }
                    pooling_details_list.append(detail)
                
                current_x, current_edge_index, current_edge_weight, current_batch = pooled_x, pooled_edge_index, pooled_edge_weight, pooled_batch
        
        z_nodes = self.final_conv(current_x, current_edge_index, current_edge_weight)
        z_nodes = self.final_batch_norm(z_nodes)
        z_nodes = self.activation_fn(z_nodes) 

        latent_graph_tuple = (z_nodes, current_edge_index, current_edge_weight, current_batch)
        
        if return_pooling_details:
            return latent_graph_tuple, pooling_details_list
        else:
            return latent_graph_tuple


class GNNUnpool(nn.Module):
    def __init__(self, method="scatter_zeros"):
        super(GNNUnpool, self).__init__()
        self.method = method

    def forward(self, x_pooled, perm, num_nodes_before_pool, batch_vector_before_pool):
        # x_pooled: features of the pooled graph [NumPooledNodes, Features]
        # perm: indices of kept nodes (from pooling layer) [NumPooledNodes]
        # num_nodes_before_pool: scalar, number of nodes before this pooling step happened
        # batch_vector_before_pool: batch vector for the graph before pooling
        
        num_features = x_pooled.size(1)
        device = x_pooled.device
        
        # Create a zero tensor for the unpooled features
        x_unpooled = torch.zeros(num_nodes_before_pool, num_features, device=device)
        x_unpooled[perm] = x_pooled
        
        # The batch vector for this unpooled graph is batch_vector_before_pool
        return x_unpooled, batch_vector_before_pool


class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, decoder_hidden_dims, out_channels, 
                 dropout_rate=0.1, activation_fn_str="relu"):
        super(GNNDecoder, self).__init__()
        self.latent_dim = latent_dim # num features on the minimal graph nodes
        self.decoder_hidden_dims = decoder_hidden_dims
        self.out_channels = out_channels # e.g., 1 for gene expression reconstruction per node
        self.dropout_rate = dropout_rate

        if activation_fn_str == "relu": self.activation_fn = F.relu
        elif activation_fn_str == "elu": self.activation_fn = F.elu
        elif activation_fn_str == "leaky_relu": self.activation_fn = F.leaky_relu
        else: self.activation_fn = F.relu

        self.unpools = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Decoder layers mirror encoder layers in reverse
        # The first conv layer in decoder takes latent_dim (from final pooled graph) + potentially skip connection
        
        # Build layers in reverse order of encoder's pooling/convs
        # The input to the first decoder GNN layer is self.latent_dim
        current_channels = self.latent_dim
        
        # The number of unpool/conv layers should match number of pooling/conv layers in encoder
        # The dimensions should be reversed from encoder_hidden_dims
        all_dims = [self.latent_dim] + self.decoder_hidden_dims # e.g., if encoder_dims=[256,128], latent=64 -> decoder_dims=[128,256], then all_dims=[64,128,256]
                                                                # or if decoder_hidden_dims=[128,256] then all_dims=[64,128,256]

        for i in range(len(self.decoder_hidden_dims)):
            # Unpooling happens conceptually *before* the GNN conv for that level
            self.unpools.append(GNNUnpool(method="scatter_zeros")) 
            # The GNN conv input dim is the current_channels (after unpooling)
            # The output dim is decoder_hidden_dims[i]
            self.convs.append(GCNConv(current_channels, self.decoder_hidden_dims[i]))
            self.batch_norms.append(nn.BatchNorm1d(self.decoder_hidden_dims[i]))
            current_channels = self.decoder_hidden_dims[i]
            
        # Final unpool if encoder started with pooling (i.e. len(pooling_details) == len(decoder_hidden_dims) + 1)
        # This logic assumes number of pool layers matches number of encoder_hidden_dims stages.
        # If there's a final_conv in encoder after last pool, corresponding initial_conv in decoder before first unpool.

        # After all unpooling and hidden GNN layers, a final GNN conv to get to out_channels
        self.final_unpool = GNNUnpool(method="scatter_zeros") # For the very first pooling in encoder
        self.final_conv = GCNConv(current_channels, self.out_channels)


    def forward(self, latent_graph_tuple, pooling_details_list):
        # latent_graph_tuple: (z_nodes, z_edge_index, z_edge_weight, z_batch) from NODE or Encoder
        # pooling_details_list: list of dicts from Encoder, in forward order. We need to use it in reverse.
        
        current_x, current_edge_index, current_edge_weight, current_batch = latent_graph_tuple
        
        # Iterate through pooling_details in reverse for unpooling
        # Number of unpooling/conv stages in decoder typically matches number of pooling/conv stages in encoder
        
        # The `pooling_details_list` corresponds to each pooling stage in the encoder.
        # The `decoder_hidden_dims` corresponds to GNN layers that should reverse encoder's GNN layers.
        # Number of pooling stages in encoder is len(pooling_details_list).
        # Number of conv/bn layers in decoder is len(self.convs).
        
        # This loop should iterate len(self.convs) times, using pooling_details in reverse.
        # pooling_details_list is ordered from first pool to last pool.
        # Decoder unpools from last pool to first pool.
        
        j = len(pooling_details_list) - 1 # Index for pooling_details_list (reverse)
        for i in range(len(self.convs)):
            if j >= 0 and self.unpools[i] is not None: # Check if there's a corresponding pooling detail
                pool_detail = pooling_details_list[j]
                current_x, current_batch = self.unpools[i](
                    current_x, 
                    pool_detail['perm'], 
                    pool_detail['num_nodes_before_pool'],
                    pool_detail['original_batch_before_conv_at_this_level']
                )
                # After unpooling, use the edge_index from *before* that pooling step in encoder
                current_edge_index = pool_detail['original_edge_index_before_conv_at_this_level']
                current_edge_weight = pool_detail.get('pooled_edge_weight_for_unpool_gnn', None) # This was edge_weight of the *pooled* graph. This is not right for unpooled.
                                                                                                  # Should use original edge_weight if available for original_edge_index.
                                                                                                  # For simplicity, assume GCNConv handles None edge_weight.
                current_edge_weight = None # Simplification: GCNConv will use unweighted if None.
                                           # Proper handling would be to also store original edge_weights in pooling_details.
                j -= 1

            current_x = self.convs[i](current_x, current_edge_index, current_edge_weight)
            current_x = self.batch_norms[i](current_x)
            current_x = self.activation_fn(current_x)
            current_x = F.dropout(current_x, p=self.dropout_rate, training=self.training)

        # Final unpool if necessary (to match initial graph structure if encoder's first conv was followed by pool)
        if j == 0: # If one more pooling detail remains, corresponding to the very first pool
            pool_detail = pooling_details_list[0]
            current_x, current_batch = self.final_unpool(
                current_x,
                pool_detail['perm'],
                pool_detail['num_nodes_before_pool'],
                pool_detail['original_batch_before_conv_at_this_level']
            )
            current_edge_index = pool_detail['original_edge_index_before_conv_at_this_level']
            current_edge_weight = None # Simplification

        x_reconstructed = self.final_conv(current_x, current_edge_index, current_edge_weight)
        return x_reconstructed, current_batch # Return batch for reshaping output


class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, encoder_hidden_dims, latent_dim, decoder_hidden_dims, 
                 pooling_type, pooling_ratios, dropout_rate, activation_fn_str="relu"):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GNNEncoder(in_channels, encoder_hidden_dims, latent_dim, 
                                  pooling_type, pooling_ratios, dropout_rate, activation_fn_str)
        
        self.decoder = GNNDecoder(latent_dim, decoder_hidden_dims, in_channels, # out_channels = in_channels
                                  dropout_rate, activation_fn_str)

    def forward(self, x, edge_index, edge_weight=None, batch=None, return_pooling_details=False):
        
        encoder_output_tuple, pooling_details_list = self.encoder(
            x, edge_index, edge_weight, batch, return_pooling_details=True # Always get details for decoder
        )
        
        # pooling_details_list is needed by the decoder to unpool correctly.
        reconstructed_x_nodes, reconstructed_batch = self.decoder(encoder_output_tuple, pooling_details_list)

        if return_pooling_details:
            return reconstructed_x_nodes, encoder_output_tuple, pooling_details_list, reconstructed_batch
        else:
            return reconstructed_x_nodes, encoder_output_tuple, reconstructed_batch


class NODEDynamics(nn.Module):
    def __init__(self, num_node_features_in_latent_graph, dynamics_hidden_dims, dropout_rate=0.1, time_dependent=False, activation_fn_str="relu"):
        super(NODEDynamics, self).__init__()
        self.time_dependent = time_dependent
        self.dropout_rate = dropout_rate
        
        if activation_fn_str == "relu": self.activation_fn = F.relu
        elif activation_fn_str == "elu": self.activation_fn = F.elu
        elif activation_fn_str == "leaky_relu": self.activation_fn = F.leaky_relu
        else: self.activation_fn = F.relu
            
        current_channels = num_node_features_in_latent_graph
        if self.time_dependent:
            current_channels += 1 
            
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for hidden_dim in dynamics_hidden_dims:
            self.convs.append(GCNConv(current_channels, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            current_channels = hidden_dim
        
        self.final_conv = GCNConv(current_channels, num_node_features_in_latent_graph) 

    def forward(self, t, z_tuple):
        node_features, edge_index, edge_weight, batch_vector = z_tuple
        
        h = node_features
        if self.time_dependent:
            time_feat = torch.ones_like(node_features[:, :1]) * t
            h = torch.cat([h, time_feat], dim=1)

        for i in range(len(self.convs)):
            h_conv = self.convs[i](h, edge_index, edge_weight)
            if h_conv.size(0) == batch_vector.size(0): # Ensure features and batch vector align
                 h = self.batch_norms[i](h_conv)
            else: # Should not happen if batch_vector is correct for h
                 h = h_conv
            h = self.activation_fn(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        dz_dt_node_features = self.final_conv(h, edge_index, edge_weight)
        return (dz_dt_node_features, edge_index, edge_weight, batch_vector)


class NeuralODE(nn.Module):
    def __init__(self, dynamics_fn, solver_method='dopri5', rtol=1e-5, atol=1e-7, use_adjoint=True, options=None):
        super(NeuralODE, self).__init__()
        self.dynamics_fn = dynamics_fn
        self.method = solver_method
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        self.options = options if options is not None else {}

    def forward(self, z0_tuple, t_eval):
        if self.use_adjoint:
            solution_tensors_over_time = odeint_adjoint(
                self.dynamics_fn, z0_tuple, t_eval, 
                method=self.method, rtol=self.rtol, atol=self.atol, options=self.options
            )
        else:
            solution_tensors_over_time = odeint(
                self.dynamics_fn, z0_tuple, t_eval, 
                method=self.method, rtol=self.rtol, atol=self.atol, options=self.options
            )
        
        final_node_features = solution_tensors_over_time[0][-1] 
        final_edge_index = z0_tuple[1]
        final_edge_weight = z0_tuple[2]
        final_batch_vector = z0_tuple[3]
        
        return (final_node_features, final_edge_index, final_edge_weight, final_batch_vector)

class CombinedModel(nn.Module):
    def __init__(self, num_genes, gae_config, node_config, node_solver_config, activation_fn_str="relu"):
        super(CombinedModel, self).__init__()
        
        self.gae = GraphAutoencoder(
            in_channels=1, # Gene expression is the first feature of each node
            encoder_hidden_dims=gae_config['encoder_hidden_dims'],
            latent_dim=gae_config['latent_dim'], 
            decoder_hidden_dims=gae_config['decoder_hidden_dims'],
            pooling_type=gae_config['pooling_type'],
            pooling_ratios=gae_config['pooling_ratios'],
            dropout_rate=gae_config['dropout_rate'],
            activation_fn_str=activation_fn_str
        )

        node_dynamics_fn = NODEDynamics(
            num_node_features_in_latent_graph=gae_config['latent_dim'],
            dynamics_hidden_dims=node_config['dynamics_hidden_dims'],
            dropout_rate=node_config['dropout_rate'],
            time_dependent=node_config['time_dependent'],
            activation_fn_str=activation_fn_str
        )
        
        self.neural_ode = NeuralODE(
            dynamics_fn=node_dynamics_fn,
            solver_method=node_solver_config['method'],
            rtol=node_solver_config['rtol'],
            atol=node_solver_config['atol'],
            use_adjoint=node_solver_config.get('use_adjoint', True),
            options=node_solver_config.get('options', {})
        )
        self.num_genes = num_genes


    def forward(self, batch_t0_pyg, batch_t1_pyg, day_t0_scalar, day_t1_scalar, return_pooling_details_for_t0_encode=False):
        
        outputs = {}

        x_t0, edge_index_t0, edge_weight_t0, batch_vec_t0 = batch_t0_pyg.x, batch_t0_pyg.edge_index, batch_t0_pyg.edge_attr, batch_t0_pyg.batch
        x_t1, edge_index_t1, edge_weight_t1, batch_vec_t1 = batch_t1_pyg.x, batch_t1_pyg.edge_index, batch_t1_pyg.edge_attr, batch_t1_pyg.batch

        # 1. Encode x_t0 to get latent state Z_t0 and pooling details
        z_t0_tuple, pooling_details_t0_list = self.gae.encoder(
            x_t0, edge_index_t0, edge_weight_t0, batch_vec_t0, return_pooling_details=True
        )
        outputs['z_t0_encoded_tuple'] = z_t0_tuple
        if return_pooling_details_for_t0_encode:
            outputs['pooling_details_t0_list'] = pooling_details_t0_list

        # 2. Encode x_t1 to get "true" latent state Z_t1_real_encoded
        z_t1_real_encoded_tuple, _ = self.gae.encoder( # No need for pooling details from this branch typically
            x_t1, edge_index_t1, edge_weight_t1, batch_vec_t1, return_pooling_details=False 
        )
        outputs['z_t1_real_encoded_tuple'] = z_t1_real_encoded_tuple
        
        # 3. Evolve Z_t0 through NeuralODE to get Z_t1_predicted_node
        time_points = torch.stack([day_t0_scalar.unique(), day_t1_scalar.unique()]).squeeze().to(z_t0_tuple[0].device)
        if time_points.ndim == 1 and time_points.size(0) == 2: # Ensure it's [t0, t1]
             pass
        elif time_points.ndim == 0 : # if t0=t1 or unique() resulted in scalar
             time_points = torch.tensor([time_points.item(), time_points.item()]).to(z_t0_tuple[0].device)

        z_t1_predicted_node_tuple = self.neural_ode(z_t0_tuple, time_points)
        outputs['z_t1_predicted_node_tuple'] = z_t1_predicted_node_tuple

        # 4. Decode Z_t0 to reconstruct X_t0_reconstructed (on full graph)
        x_t0_reconstructed_nodes, x_t0_reconstructed_batch_vec = self.gae.decoder(z_t0_tuple, pooling_details_t0_list)
        
        # Convert batched node features back to [NumCells, NumGenes]
        # Output of decoder is [TotalNodes, OutChannels=1]. We need to reshape to [NumCells, NumGenes]
        # where NumGenes is num_nodes for each graph in the batch.
        x_t0_reconstructed_dense, _ = to_dense_batch(x_t0_reconstructed_nodes, x_t0_reconstructed_batch_vec, fill_value=0, max_num_nodes=self.num_genes)
        outputs['x_t0_reconstructed'] = x_t0_reconstructed_dense.squeeze(-1) # Remove last dim if it's 1


        # 5. Decode Z_t1_predicted_node to reconstruct X_t1_predicted_reconstructed (on full graph)
        # Use pooling_details from t0 encoding, as NODE operates on that structure
        x_t1_predicted_reconstructed_nodes, x_t1_predicted_reconstructed_batch_vec = self.gae.decoder(z_t1_predicted_node_tuple, pooling_details_t0_list)
        
        x_t1_predicted_reconstructed_dense, _ = to_dense_batch(x_t1_predicted_reconstructed_nodes, x_t1_predicted_reconstructed_batch_vec, fill_value=0, max_num_nodes=self.num_genes)
        outputs['x_t1_predicted_reconstructed'] = x_t1_predicted_reconstructed_dense.squeeze(-1)

        return outputs

    def gae_pretrain_forward(self, pyg_batch):
        x, edge_index, edge_weight, batch_vec = pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr, pyg_batch.batch
        
        # GAE's forward returns (reconstructed_x_nodes, encoder_output_tuple, reconstructed_batch_vec)
        reconstructed_x_nodes, (z_nodes, _, _, _), reconstructed_batch_vec = self.gae(
            x, edge_index, edge_weight, batch_vec, 
            return_pooling_details=False # Not strictly needed for pretrain loss, but GAE forward now always gets them for decoder
        )
        
        # Reshape reconstructed_x_nodes back to [NumCells, NumGenes]
        reconstructed_x_dense, _ = to_dense_batch(reconstructed_x_nodes, reconstructed_batch_vec, fill_value=0, max_num_nodes=self.num_genes)
        reconstructed_x_dense = reconstructed_x_dense.squeeze(-1)

        return reconstructed_x_dense, z_nodes # z_nodes are from the latent graph