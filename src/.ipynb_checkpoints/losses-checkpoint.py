import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

try:
    import geomloss
except ImportError:
    geomloss = None

def compute_reconstruction_loss(predicted_expr, true_expr, loss_type="mse"):
    if loss_type == "mse":
        return F.mse_loss(predicted_expr, true_expr)
    elif loss_type == "l1":
        return F.l1_loss(predicted_expr, true_expr)
    else:
        raise ValueError(f"Unsupported reconstruction loss type: {loss_type}")

def compute_ot_loss(pred_samples, true_samples, ot_loss_type='sinkhorn', blur=0.05, scaling=0.8, p=2, backend='auto'):
    if geomloss is None:
        # Fallback or warning if geomloss is not installed
        # For now, let's use a simple MSE as a placeholder if geomloss is missing
        # print("Warning: geomloss library not found. Using MSE as a fallback for OT loss.")
        # return F.mse_loss(pred_samples, true_samples)
        raise ImportError("geomloss library is not installed. Please install it to use OT loss.")


    if pred_samples.ndim == 3 and pred_samples.shape[-1] == 1 and true_samples.ndim == 2:
        # This can happen if to_dense_batch output for single feature nodes is [Batch, MaxNodes, 1]
        # while true_samples is [Batch, MaxNodes]. Squeeze last dim of pred_samples.
        pred_samples = pred_samples.squeeze(-1)

    if pred_samples.shape != true_samples.shape:
        raise ValueError(f"Shape mismatch for OT loss: pred_samples {pred_samples.shape}, true_samples {true_samples.shape}")

    if pred_samples.size(0) == 0 or true_samples.size(0) == 0: # Handle empty batches if they occur
        return torch.tensor(0.0, device=pred_samples.device, requires_grad=True)


    loss_geom = geomloss.SamplesLoss(
        loss=ot_loss_type, 
        p=p, # L_p norm for cost function
        blur=blur, 
        scaling=scaling, 
        debias=True,
        backend=backend
    )
    return loss_geom(pred_samples, true_samples)


def calculate_total_loss(model_outputs, data_batch, loss_weights, device, config):
    total_loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    # 0. Get necessary data from data_batch (which now contains PyG Batch objects and original expressions)
    # original_expressions_t0 is [M_cells, N_genes]
    # original_expressions_t1 is [M_cells, N_genes]
    true_expr_t0 = data_batch['original_expressions_t0'].to(device)
    true_expr_t1 = data_batch['original_expressions_t1'].to(device)

    # 1. Reconstruction Loss for t0
    if loss_weights.get('recon_t0', 0) > 0 and 'x_t0_reconstructed' in model_outputs:
        pred_expr_t0 = model_outputs['x_t0_reconstructed'] # Should be [M_cells, N_genes]
        if pred_expr_t0.shape == true_expr_t0.shape:
            l_recon_t0 = compute_reconstruction_loss(pred_expr_t0, true_expr_t0)
            total_loss += loss_weights['recon_t0'] * l_recon_t0
            loss_dict['recon_t0'] = l_recon_t0.item()
        else:
            print(f"Warning: Shape mismatch for recon_t0. Predicted: {pred_expr_t0.shape}, True: {true_expr_t0.shape}. Skipping this loss.")
            loss_dict['recon_t0'] = 0.0


    # 2. Reconstruction Loss for t1 (from NODE prediction)
    if loss_weights.get('recon_t1_predicted', 0) > 0 and 'x_t1_predicted_reconstructed' in model_outputs:
        pred_expr_t1 = model_outputs['x_t1_predicted_reconstructed'] # Should be [M_cells, N_genes]
        if pred_expr_t1.shape == true_expr_t1.shape:
            l_recon_t1 = compute_reconstruction_loss(pred_expr_t1, true_expr_t1)
            total_loss += loss_weights['recon_t1_predicted'] * l_recon_t1
            loss_dict['recon_t1'] = l_recon_t1.item()
        else:
            print(f"Warning: Shape mismatch for recon_t1. Predicted: {pred_expr_t1.shape}, True: {true_expr_t1.shape}. Skipping this loss.")
            loss_dict['recon_t1'] = 0.0

    ot_blur = config.get('ot_loss_params', {}).get('blur', 0.05)
    ot_scaling = config.get('ot_loss_params', {}).get('scaling', 0.8)
    ot_p = config.get('ot_loss_params', {}).get('p', 2)


    # 3. OT Loss in Latent Space
    if loss_weights.get('ot_latent', 0) > 0 and \
       'z_t1_real_encoded_tuple' in model_outputs and \
       'z_t1_predicted_node_tuple' in model_outputs:
        
        z_t1_real_nodes, _, _, z_t1_real_batch_vec = model_outputs['z_t1_real_encoded_tuple']
        z_t1_pred_nodes, _, _, z_t1_pred_batch_vec = model_outputs['z_t1_predicted_node_tuple']

        # Pool node features to get per-cell/per-graph latent vectors
        # z_t1_real_nodes is [TotalNodes_t1_real, LatentDimOnNodes]
        # z_t1_real_batch_vec maps these nodes to graphs in the batch
        if z_t1_real_batch_vec is not None and z_t1_pred_batch_vec is not None :
            per_cell_latent_true = global_mean_pool(z_t1_real_nodes, z_t1_real_batch_vec)
            per_cell_latent_pred = global_mean_pool(z_t1_pred_nodes, z_t1_pred_batch_vec)
        
            if per_cell_latent_pred.shape == per_cell_latent_true.shape and per_cell_latent_pred.size(0) > 0:
                 l_ot_latent = compute_ot_loss(per_cell_latent_pred, per_cell_latent_true, blur=ot_blur, scaling=ot_scaling, p=ot_p)
                 total_loss += loss_weights['ot_latent'] * l_ot_latent
                 loss_dict['ot_latent'] = l_ot_latent.item()
            else:
                 print(f"Warning: Shape mismatch or empty tensor for ot_latent. Pred: {per_cell_latent_pred.shape}, True: {per_cell_latent_true.shape}. Skipping.")
                 loss_dict['ot_latent'] = 0.0
        else:
            print("Warning: Batch vectors for latent OT loss are None. Skipping ot_latent.")
            loss_dict['ot_latent'] = 0.0


    # 4. OT Loss in Expression Space
    if loss_weights.get('ot_expression', 0) > 0 and 'x_t1_predicted_reconstructed' in model_outputs:
        pred_expr_t1_for_ot = model_outputs['x_t1_predicted_reconstructed'] # Should be [M_cells, N_genes]
        
        if pred_expr_t1_for_ot.shape == true_expr_t1.shape and pred_expr_t1_for_ot.size(0) > 0:
            l_ot_expression = compute_ot_loss(pred_expr_t1_for_ot, true_expr_t1, blur=ot_blur, scaling=ot_scaling, p=ot_p)
            total_loss += loss_weights['ot_expression'] * l_ot_expression
            loss_dict['ot_expression'] = l_ot_expression.item()
        else:
            print(f"Warning: Shape mismatch or empty tensor for ot_expression. Pred: {pred_expr_t1_for_ot.shape}, True: {true_expr_t1.shape}. Skipping.")
            loss_dict['ot_expression'] = 0.0
            
    loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
    return total_loss, loss_dict


def calculate_gae_pretrain_loss(model_outputs, data_batch, device, config):
    # model_outputs for GAE pretrain is (reconstructed_x_dense, z_nodes)
    # data_batch for GAE pretrain is {'pyg_batch': pyg_batch, 'original_expressions': original_expressions_tensor}
    
    pred_expr = model_outputs[0] # reconstructed_x_dense, shape [BatchCells, NumGenes]
    true_expr = data_batch['original_expressions'].to(device) # shape [BatchCells, NumGenes]

    if pred_expr.shape != true_expr.shape:
        # This can happen if num_genes used in to_dense_batch (in model) is different from true_expr.shape[1]
        # Or if batch sizes somehow mismatch.
        print(f"Warning: Shape mismatch for GAE pretrain loss. Predicted: {pred_expr.shape}, True: {true_expr.shape}.")
        # Attempt to pad/truncate if only num_genes dimension mismatch slightly due to to_dense_batch max_num_nodes
        # This is a simplistic fix; underlying cause should be checked.
        min_genes = min(pred_expr.shape[1], true_expr.shape[1])
        pred_expr = pred_expr[:, :min_genes]
        true_expr = true_expr[:, :min_genes]

    recon_loss = compute_reconstruction_loss(pred_expr, true_expr)
    loss_dict = {'gae_recon_loss': recon_loss.item()}
    
    return recon_loss, loss_dict