import os
import torch
import numpy as np
import json
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool 

from .utils import plot_umap_comparison, load_checkpoint
from .losses import CombinedLoss 
from .models.combined_model import CombinedModel
from .models.graph_autoencoder import GraphAutoencoder

def evaluate_gae_reconstruction(model, dataloader, loss_fn_instance, device, config):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        eval_model = model.module.graph_autoencoder if hasattr(model.module, 'graph_autoencoder') else model.module
    else:
        eval_model = model.graph_autoencoder if hasattr(model, 'graph_autoencoder') else model
    
    if not isinstance(eval_model, GraphAutoencoder):
        raise TypeError("Model for GAE evaluation must be or contain GraphAutoencoder.")

    eval_model.eval()
    total_loss = 0.0
    all_reconstructed_expr_for_umap = [] 
    all_original_expr_for_umap = []    
    all_latent_z_nodes_for_umap = [] 

    loss_components = {}

    with torch.no_grad():
        for data_batch in tqdm(dataloader, desc="Evaluating GAE Reconstruction"):
            if data_batch is None: continue
            pyg_batch = data_batch['pyg_batch'].to(device)
            original_expressions = data_batch['original_expressions'].to(device) 

            reconstructed_output, z_nodes, z_batch_vec, _ = eval_model(pyg_batch, return_pooling_details=True) 
            
            batch_size_eval = original_expressions.shape[0]
            num_genes_eval = original_expressions.shape[1] 
            
            reconstructed_expressions_reshaped = reconstructed_output
            if reconstructed_output.shape[0] == batch_size_eval * num_genes_eval and reconstructed_output.shape[1] == 1:
                reconstructed_expressions_reshaped = reconstructed_output.view(batch_size_eval, num_genes_eval)
            elif reconstructed_output.numel() == original_expressions.numel():
                try:
                    reconstructed_expressions_reshaped = reconstructed_output.view_as(original_expressions)
                except RuntimeError:
                    print(f"Warning: Could not reshape GAE reconstruction output {reconstructed_output.shape} to {original_expressions.shape} during eval. Using original output for loss.")
                    pass
            else:
                print(f"Warning: Shape mismatch for GAE reconstruction during eval. Output: {reconstructed_output.shape}, Target: {original_expressions.shape}. Using original output for loss.")


            model_outputs_for_loss = {"reconstructed_expressions": reconstructed_expressions_reshaped}
            original_data_for_loss = {"original_expressions": original_expressions}
            
            loss, loss_dict_batch = loss_fn_instance(model_outputs_for_loss, original_data_for_loss, mode="pretrain_gae")
            total_loss += loss.item()
            
            for key, value in loss_dict_batch.items():
                loss_components[key] = loss_components.get(key, 0) + value

            if config.evaluation_params.generate_umap_plots:
                all_original_expr_for_umap.append(original_expressions.detach().cpu().numpy())
                all_reconstructed_expr_for_umap.append(reconstructed_expressions_reshaped.detach().cpu().numpy()) 
                if z_nodes is not None and z_batch_vec is not None:
                    z_global = global_mean_pool(z_nodes, z_batch_vec)
                    all_latent_z_nodes_for_umap.append(z_global.detach().cpu().numpy())


    avg_loss = total_loss / len(dataloader) if dataloader and len(dataloader) > 0 else 0
    for key in loss_components.keys():
        loss_components[key] /= len(dataloader) if dataloader and len(dataloader) > 0 else 1
    
    print(f"GAE Evaluation - Average Reconstruction Loss: {avg_loss:.4f}")
    for k,v in loss_components.items():
        print(f"  Avg {k}: {v:.4f}")

    results = {"avg_reconstruction_loss": avg_loss, **loss_components}

    if config.evaluation_params.generate_umap_plots and all_original_expr_for_umap and all_reconstructed_expr_for_umap:
        num_samples_umap = config.evaluation_params.get("num_samples_for_umap", 500)
        
        original_expr_np = np.concatenate(all_original_expr_for_umap, axis=0)[:num_samples_umap]
        reconstructed_expr_np = np.concatenate(all_reconstructed_expr_for_umap, axis=0)[:num_samples_umap]
        
        umap_data_expr = {
            "Original Expression": original_expr_np,
            "GAE Reconstructed Expression": reconstructed_expr_np
        }
        plot_umap_comparison(
            umap_data_expr,
            output_path=os.path.join(config.evaluation_params.get("output_dir", "results/eval_output"), "umap_gae_reconstruction_comparison.png"),
            title="UMAP: Original vs. GAE Reconstructed Gene Expression",
            n_neighbors=config.evaluation_params.umap_n_neighbors,
            min_dist=config.evaluation_params.umap_min_dist,
            metric=config.evaluation_params.umap_metric,
            random_state=config.seed
        )

        if all_latent_z_nodes_for_umap:
            latent_z_np = np.concatenate(all_latent_z_nodes_for_umap, axis=0)[:num_samples_umap]
            umap_data_latent = {"GAE Latent Space (Z)": latent_z_np}
            plot_umap_comparison(
                umap_data_latent,
                output_path=os.path.join(config.evaluation_params.get("output_dir", "results/eval_output"), "umap_gae_latent_space.png"),
                title="UMAP: GAE Latent Space (Z)",
                n_neighbors=config.evaluation_params.umap_n_neighbors,
                min_dist=config.evaluation_params.umap_min_dist,
                metric=config.evaluation_params.umap_metric,
                random_state=config.seed
            )
    return results


def evaluate_joint_prediction(model, dataloader, loss_fn_instance, device, config):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        eval_model = model.module
    else:
        eval_model = model
    
    if not isinstance(eval_model, CombinedModel):
        raise TypeError("Model for joint evaluation must be CombinedModel.")

    eval_model.eval()
    total_loss = 0.0
    all_metrics = {} 

    all_x_t0_real_for_umap, all_x_t1_real_for_umap, all_x_t1_pred_recon_for_umap = [], [], []
    all_z_t0_encoded_for_umap, all_z_t1_real_encoded_for_umap, all_z_t1_pred_node_for_umap = [], [], [] 

    with torch.no_grad():
        last_data_batch_for_title = None 
        for data_batch in tqdm(dataloader, desc="Evaluating Joint Model Prediction"):
            if data_batch is None: continue
            last_data_batch_for_title = data_batch 
            batch_t0 = data_batch['batch_t0'].to(device)
            batch_t1 = data_batch['batch_t1'].to(device)
            original_expressions_t0 = data_batch['original_expressions_t0'].to(device)
            original_expressions_t1 = data_batch['original_expressions_t1'].to(device)
            day_t0_scalar = data_batch['day_t0_scalar'].to(device)
            day_t1_scalar = data_batch['day_t1_scalar'].to(device)

            model_outputs_raw = eval_model(
                data_t0=batch_t0,
                t0_scalar=day_t0_scalar,
                t1_scalar=day_t1_scalar,
                data_t1=batch_t1,
                return_latent_t1_real=True
            )
            
            num_genes_cfg = int(config.model_params.num_genes) 
            model_outputs_for_loss = model_outputs_raw.copy()

            x_t0_rec_raw = model_outputs_raw.get("x_t0_reconstructed")
            if x_t0_rec_raw is not None:
                batch_size_t0 = original_expressions_t0.shape[0]
                x_t0_rec_reshaped = x_t0_rec_raw
                if x_t0_rec_raw.shape[0] == batch_size_t0 * num_genes_cfg and x_t0_rec_raw.shape[1] == 1:
                    x_t0_rec_reshaped = x_t0_rec_raw.view(batch_size_t0, num_genes_cfg)
                elif x_t0_rec_raw.numel() == original_expressions_t0.numel():
                    try:
                        x_t0_rec_reshaped = x_t0_rec_raw.view_as(original_expressions_t0)
                    except RuntimeError:
                        print(f"Warning: Could not reshape x_t0_reconstructed {x_t0_rec_raw.shape} to {original_expressions_t0.shape} during eval. Using original for loss.")
                        pass
                else:
                     print(f"Warning: Shape mismatch for x_t0_reconstructed during eval. Output: {x_t0_rec_raw.shape}, Target: {original_expressions_t0.shape}. Using original for loss.")
                model_outputs_for_loss["x_t0_reconstructed"] = x_t0_rec_reshaped
            
            x_t1_pred_rec_raw = model_outputs_raw.get("x_t1_predicted_reconstructed")
            if x_t1_pred_rec_raw is not None:
                batch_size_t1 = original_expressions_t1.shape[0]
                x_t1_pred_rec_reshaped = x_t1_pred_rec_raw
                if x_t1_pred_rec_raw.shape[0] == batch_size_t1 * num_genes_cfg and x_t1_pred_rec_raw.shape[1] == 1:
                    x_t1_pred_rec_reshaped = x_t1_pred_rec_raw.view(batch_size_t1, num_genes_cfg)
                elif x_t1_pred_rec_raw.numel() == original_expressions_t1.numel():
                    try:
                        x_t1_pred_rec_reshaped = x_t1_pred_rec_raw.view_as(original_expressions_t1)
                    except RuntimeError:
                        print(f"Warning: Could not reshape x_t1_pred_reconstructed {x_t1_pred_rec_raw.shape} to {original_expressions_t1.shape} during eval. Using original for loss.")
                        pass
                else:
                    print(f"Warning: Shape mismatch for x_t1_predicted_reconstructed during eval. Output: {x_t1_pred_rec_raw.shape}, Target: {original_expressions_t1.shape}. Using original for loss.")
                model_outputs_for_loss["x_t1_predicted_reconstructed"] = x_t1_pred_rec_reshaped


            original_data_for_loss = {
                "original_expressions_t0": original_expressions_t0,
                "original_expressions_t1": original_expressions_t1
            }
            loss, loss_dict_batch = loss_fn_instance(model_outputs_for_loss, original_data_for_loss, mode="train_joint")
            total_loss += loss.item()

            for key, value in loss_dict_batch.items():
                all_metrics[key] = all_metrics.get(key, 0) + value

            if config.evaluation_params.generate_umap_plots:
                all_x_t0_real_for_umap.append(original_expressions_t0.detach().cpu().numpy())
                all_x_t1_real_for_umap.append(original_expressions_t1.detach().cpu().numpy())
                all_x_t1_pred_recon_for_umap.append(model_outputs_for_loss["x_t1_predicted_reconstructed"].detach().cpu().numpy())
                
                if model_outputs_raw.get("z_t0_nodes") is not None and model_outputs_raw.get("z_t0_batch") is not None:
                    z_t0_glob = global_mean_pool(model_outputs_raw["z_t0_nodes"], model_outputs_raw["z_t0_batch"])
                    all_z_t0_encoded_for_umap.append(z_t0_glob.detach().cpu().numpy())

                if model_outputs_raw.get("z_t1_real_nodes") is not None and model_outputs_raw.get("z_t1_real_batch") is not None:
                    z_t1_real_glob = global_mean_pool(model_outputs_raw["z_t1_real_nodes"], model_outputs_raw["z_t1_real_batch"])
                    all_z_t1_real_encoded_for_umap.append(z_t1_real_glob.detach().cpu().numpy())
                
                if model_outputs_raw.get("z_t1_predicted_nodes") is not None and model_outputs_raw.get("z_t0_batch") is not None: 
                    z_t1_pred_glob = global_mean_pool(model_outputs_raw["z_t1_predicted_nodes"], model_outputs_raw["z_t0_batch"])
                    all_z_t1_pred_node_for_umap.append(z_t1_pred_glob.detach().cpu().numpy())


    avg_total_loss = total_loss / len(dataloader) if dataloader and len(dataloader) > 0 else 0
    for key in all_metrics.keys():
        all_metrics[key] /= len(dataloader) if dataloader and len(dataloader) > 0 else 1
    
    print(f"Joint Model Evaluation - Average Total Loss: {avg_total_loss:.4f}")
    for k, v in all_metrics.items():
        print(f"  Avg {k}: {v:.4f}")

    results = {"avg_total_loss": avg_total_loss, **all_metrics}

    if config.evaluation_params.generate_umap_plots:
        num_samples_umap = config.evaluation_params.get("num_samples_for_umap", 500)
        output_dir_eval = config.evaluation_params.get("output_dir", "results/eval_output")

        if all_x_t0_real_for_umap and all_x_t1_real_for_umap and all_x_t1_pred_recon_for_umap:
            x_t0_real_np = np.concatenate(all_x_t0_real_for_umap, axis=0)[:num_samples_umap]
            x_t1_real_np = np.concatenate(all_x_t1_real_for_umap, axis=0)[:num_samples_umap]
            x_t1_pred_np = np.concatenate(all_x_t1_pred_recon_for_umap, axis=0)[:num_samples_umap]
            
            day_t0_title = last_data_batch_for_title['day_t0_scalar'].item() if last_data_batch_for_title else 't0'
            day_t1_title = last_data_batch_for_title['day_t1_scalar'].item() if last_data_batch_for_title else 't1'

            umap_data_expr = {
                f"X_t0 Real (Day {day_t0_title:.1f})": x_t0_real_np,
                f"X_t1 Real (Day {day_t1_title:.1f})": x_t1_real_np,
                f"X_t1 Predicted (from t0 to t1)": x_t1_pred_np
            }
            plot_umap_comparison(
                umap_data_expr,
                output_path=os.path.join(output_dir_eval, "umap_joint_expression_prediction.png"),
                title="UMAP: Joint Model Gene Expression Prediction",
                n_neighbors=config.evaluation_params.umap_n_neighbors,
                min_dist=config.evaluation_params.umap_min_dist,
                metric=config.evaluation_params.umap_metric,
                random_state=config.seed
            )

        if all_z_t0_encoded_for_umap and all_z_t1_real_encoded_for_umap and all_z_t1_pred_node_for_umap:
            z_t0_enc_np = np.concatenate(all_z_t0_encoded_for_umap, axis=0)[:num_samples_umap]
            z_t1_real_enc_np = np.concatenate(all_z_t1_real_encoded_for_umap, axis=0)[:num_samples_umap]
            z_t1_pred_node_np = np.concatenate(all_z_t1_pred_node_for_umap, axis=0)[:num_samples_umap]

            day_t0_title = last_data_batch_for_title['day_t0_scalar'].item() if last_data_batch_for_title else 't0'
            day_t1_title = last_data_batch_for_title['day_t1_scalar'].item() if last_data_batch_for_title else 't1'

            umap_data_latent = {
                f"Z_t0 Encoded (Day {day_t0_title:.1f})": z_t0_enc_np,
                f"Z_t1 Real Encoded (Day {day_t1_title:.1f})": z_t1_real_enc_np,
                f"Z_t1 Predicted Node (from t0 to t1)": z_t1_pred_node_np
            }
            plot_umap_comparison(
                umap_data_latent,
                output_path=os.path.join(output_dir_eval, "umap_joint_latent_space_prediction.png"),
                title="UMAP: Joint Model Latent Space Prediction",
                n_neighbors=config.evaluation_params.umap_n_neighbors,
                min_dist=config.evaluation_params.umap_min_dist,
                metric=config.evaluation_params.umap_metric,
                random_state=config.seed
            )
    return results

def run_evaluation(config, model_checkpoint_path, eval_type="all", data_loaders=None):
    device = torch.device(config.training_params.device if torch.cuda.is_available() else "cpu")
    
    model = CombinedModel(config) 
    
    optimizer = None 
    start_epoch, best_metric = load_checkpoint(model_checkpoint_path, model, optimizer=None, scheduler=None, device=device)
    print(f"Loaded model from checkpoint: {model_checkpoint_path} (trained for {start_epoch-1} epochs, best metric: {best_metric})")
    
    model.to(device)
    model.eval()

    loss_fn_instance = CombinedLoss(config.loss_weights, device=device)
    all_results = {}

    if data_loaders is None:
        raise ValueError("data_loaders must be provided to run_evaluation.")

    if eval_type == "gae" or eval_type == "all":
        print("\n--- Evaluating GAE Reconstruction ---")
        gae_eval_loader = data_loaders.get("gae_eval_loader")
        if gae_eval_loader:
            gae_results = evaluate_gae_reconstruction(model, gae_eval_loader, loss_fn_instance, device, config)
            all_results["gae_reconstruction"] = gae_results
        else:
            print("GAE evaluation loader not provided, skipping GAE evaluation.")

    if eval_type == "joint" or eval_type == "all":
        print("\n--- Evaluating Joint Model Prediction ---")
        joint_eval_loader = data_loaders.get("joint_eval_loader")
        if joint_eval_loader:
            joint_results = evaluate_joint_prediction(model, joint_eval_loader, loss_fn_instance, device, config)
            all_results["joint_prediction"] = joint_results
        else:
            print("Joint evaluation loader not provided, skipping joint model evaluation.")
            
    output_dir_eval = config.evaluation_params.get("output_dir", "results/eval_output")
    os.makedirs(output_dir_eval, exist_ok=True)
    results_filepath = os.path.join(output_dir_eval, "evaluation_metrics.json")
    with open(results_filepath, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nEvaluation metrics saved to {results_filepath}")
    
    return all_results
