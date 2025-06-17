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


def _reshape_params_for_loss_or_eval(params_raw, target_expressions_shape_b_ng, num_dist_params):
    if params_raw is None or params_raw.numel() == 0: return params_raw
    batch_size, num_genes = target_expressions_shape_b_ng

    params_reshaped = params_raw
    expected_flat_shape_dim0 = batch_size * num_genes

    if params_raw.dim() == 2 and \
            params_raw.shape[0] == expected_flat_shape_dim0 and \
            params_raw.shape[1] == num_dist_params:
        try:
            params_reshaped = params_raw.view(batch_size, num_genes, num_dist_params)
        except RuntimeError as e:
            print(
                f"Warning: Reshape failed in _reshape_params_for_loss_or_eval. Raw: {params_raw.shape}, Target: [{batch_size},{num_genes},{num_dist_params}]. Error: {e}")
    elif not (params_raw.dim() == 3 and params_raw.shape[0] == batch_size and params_raw.shape[1] == num_genes and
              params_raw.shape[2] == num_dist_params):
        if not (params_raw.dim() == 2 and params_raw.shape[0] == batch_size and params_raw.shape[
            1] == num_genes and num_dist_params == 1):
            print(
                f"Warning: Shape mismatch for _reshape_params_for_loss_or_eval. Raw: {params_raw.shape}, Target (B,N_genes): {target_expressions_shape_b_ng}, NumParams: {num_dist_params}.")

    return params_reshaped


def _get_point_estimate_from_params(params_tensor, distribution_type, device='cpu'):
    if params_tensor is None or params_tensor.numel() == 0: return torch.empty(0, device=device)
    if distribution_type == "gaussian":
        if params_tensor.shape[-1] < 1 or params_tensor.dim() < 2:
            print(f"Warning: Gaussian params tensor {params_tensor.shape} has too few parameters/dims for mean.")
            return params_tensor
        return params_tensor[..., 0]
    elif distribution_type == "deterministic":
        if params_tensor.dim() == 3 and params_tensor.shape[-1] == 1:
            return params_tensor.squeeze(-1)
        return params_tensor
    else:
        print(
            f"Warning: Unknown distribution '{distribution_type}' in _get_point_estimate_from_params. Using first param set if available.")
        if params_tensor.dim() > 1 and params_tensor.shape[-1] > 0:
            return params_tensor[..., 0]
        else:
            return params_tensor


def evaluate_gae_reconstruction(model, dataloader, loss_fn_instance, device, config):
    actual_model_module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    if hasattr(actual_model_module, 'graph_autoencoder'):
        eval_model = actual_model_module.graph_autoencoder
    elif isinstance(actual_model_module, GraphAutoencoder):
        eval_model = actual_model_module
    else:
        raise TypeError(
            "Model for GAE evaluation must be GraphAutoencoder or CombinedModel with 'graph_autoencoder' attribute.")

    eval_model.eval()
    total_loss, all_loss_dicts = 0.0, []
    all_reconstructed_expr_for_umap, all_original_expr_for_umap, all_latent_z_for_umap = [], [], []

    num_dist_params = config.model_params.decoder.get("num_dist_params", 2)
    distribution_type = config.model_params.decoder.get("distribution", "gaussian")
    is_variational_encoder = bool(config.model_params.encoder.get("is_variational", False))

    with torch.no_grad():
        for data_batch in tqdm(dataloader, desc="Evaluating GAE Reconstruction"):
            if data_batch is None: continue
            pyg_batch = data_batch['pyg_batch'].to(device)
            original_expressions = data_batch['original_expressions'].to(device)

            gae_outputs = eval_model(pyg_batch, return_pooling_details=True)

            reconstructed_output_params = gae_outputs.get("reconstructed_params")
            z_batch_vec = gae_outputs.get("z_batch")

            latent_mu_nodes = gae_outputs.get("mu_nodes") if is_variational_encoder else None
            z_nodes_for_umap_calc = latent_mu_nodes if latent_mu_nodes is not None else gae_outputs.get(
                "sampled_z_nodes")

            if reconstructed_output_params is None: continue

            reconstructed_params_reshaped = _reshape_params_for_loss_or_eval(reconstructed_output_params,
                                                                             original_expressions.shape,
                                                                             num_dist_params)
            model_outputs_for_loss = {"reconstructed_expressions_params": reconstructed_params_reshaped,
                                      "mu_nodes": latent_mu_nodes, "log_var_nodes": gae_outputs.get("log_var_nodes")}
            original_data_for_loss = {"original_expressions": original_expressions}

            loss, loss_dict = loss_fn_instance(model_outputs_for_loss, original_data_for_loss, mode="pretrain_gae")
            if loss is not None: total_loss += loss.item()
            all_loss_dicts.append(loss_dict)

            if config.evaluation_params.generate_umap_plots:
                all_original_expr_for_umap.append(original_expressions.detach().cpu().numpy())
                point_estimate_reconstructed = _get_point_estimate_from_params(reconstructed_params_reshaped,
                                                                               distribution_type, device=device)
                all_reconstructed_expr_for_umap.append(point_estimate_reconstructed.detach().cpu().numpy())
                if z_nodes_for_umap_calc is not None and z_nodes_for_umap_calc.numel() > 0 and z_batch_vec is not None and z_batch_vec.numel() > 0:
                    z_global = global_mean_pool(z_nodes_for_umap_calc, z_batch_vec)
                    all_latent_z_for_umap.append(z_global.detach().cpu().numpy())

    avg_epoch_loss = np.nanmean([d.get('total_loss', np.nan) for d in all_loss_dicts]) if all_loss_dicts else 0.0
    aggregated_loss_dict = {}
    if all_loss_dicts and all_loss_dicts[0]:
        for key in all_loss_dicts[0].keys():
            valid_values = [d[key] for d in all_loss_dicts if
                            d and key in d and d[key] is not None and not np.isnan(d[key]) and not np.isinf(d[key])]
            if valid_values: aggregated_loss_dict[key] = np.mean(valid_values)

    print(f"GAE Evaluation - Average Total Loss: {avg_epoch_loss:.4f}")
    for k, v in aggregated_loss_dict.items(): print(f"  Avg {k}: {v:.4f}")
    results = {"avg_total_loss": avg_epoch_loss, **aggregated_loss_dict}

    if config.evaluation_params.generate_umap_plots and all_original_expr_for_umap and all_reconstructed_expr_for_umap:
        num_samples_umap = config.evaluation_params.get("num_samples_for_umap", 500)
        original_expr_np = np.concatenate(all_original_expr_for_umap, axis=0)[:num_samples_umap]
        reconstructed_expr_np = np.concatenate(all_reconstructed_expr_for_umap, axis=0)[:num_samples_umap]

        umap_data_expr_title = "GAE Reconstructed Expression" + (" (Mean)" if distribution_type == "gaussian" else "")
        umap_data_expr = {"Original Expression": original_expr_np, umap_data_expr_title: reconstructed_expr_np}
        plot_umap_comparison(umap_data_expr,
                             os.path.join(config.evaluation_params.get("output_dir", "results/eval_output"),
                                          "umap_gae_reconstruction_comparison.png"),
                             title=f"UMAP: Original vs. {umap_data_expr_title}", config=config)

        if all_latent_z_for_umap:
            latent_z_np = np.concatenate(all_latent_z_for_umap, axis=0)[:num_samples_umap]
            umap_latent_title = "GAE Latent Space (Z)" + (" (Mean)" if is_variational_encoder else "")
            umap_data_latent = {umap_latent_title: latent_z_np}
            plot_umap_comparison(umap_data_latent,
                                 os.path.join(config.evaluation_params.get("output_dir", "results/eval_output"),
                                              "umap_gae_latent_space.png"), title=umap_latent_title, config=config)

    return results


def evaluate_joint_prediction(model, dataloader, loss_fn_instance, device, config):
    eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if not isinstance(eval_model, CombinedModel): raise TypeError("Model for joint evaluation must be CombinedModel.")

    eval_model.eval()
    total_loss, all_metrics = 0.0, {}
    all_x_t0_real_for_umap, all_x_t1_real_for_umap, all_x_t1_pred_recon_for_umap = [], [], []
    all_z_t0_encoded_for_umap, all_z_t1_real_encoded_for_umap, all_z_t1_pred_node_for_umap = [], [], []

    num_dist_params = config.model_params.decoder.get("num_dist_params", 2)
    distribution_type = config.model_params.decoder.get("distribution", "gaussian")
    is_variational_encoder = bool(config.model_params.encoder.get("is_variational", False))

    with torch.no_grad():
        last_data_batch_for_title = None
        for data_batch in tqdm(dataloader, desc="Evaluating Joint Model Prediction"):
            if data_batch is None: continue
            last_data_batch_for_title = data_batch
            batch_t0 = data_batch['batch_t0'].to(device)
            batch_t1 = data_batch['batch_t1'].to(device)
            original_expressions_t0 = data_batch['original_expressions_t0'].to(device)
            original_expressions_t1 = data_batch['original_expressions_t1'].to(device)
            day_t0_scalar, day_t1_scalar = data_batch['day_t0_scalar'].to(device), data_batch['day_t1_scalar'].to(
                device)

            model_outputs_raw = eval_model(data_t0=batch_t0, t0_scalar=day_t0_scalar, t1_scalar=day_t1_scalar,
                                           data_t1=batch_t1, return_latent_t1_real=True)
            model_outputs_for_loss = model_outputs_raw.copy()

            if model_outputs_for_loss.get("x_t0_reconstructed_params"):
                model_outputs_for_loss["x_t0_reconstructed_params"] = _reshape_params_for_loss_or_eval(
                    model_outputs_raw.get("x_t0_reconstructed_params"), original_expressions_t0.shape, num_dist_params)
            if model_outputs_for_loss.get("x_t1_predicted_reconstructed_params"):
                model_outputs_for_loss["x_t1_predicted_reconstructed_params"] = _reshape_params_for_loss_or_eval(
                    model_outputs_raw.get("x_t1_predicted_reconstructed_params"), original_expressions_t1.shape,
                    num_dist_params)

            original_data_for_loss = {"original_expressions_t0": original_expressions_t0,
                                      "original_expressions_t1": original_expressions_t1}
            loss, loss_dict_batch = loss_fn_instance(model_outputs_for_loss, original_data_for_loss, mode="train_joint")
            if loss is not None: total_loss += loss.item()
            for key, value in loss_dict_batch.items():
                if value is not None: all_metrics[key] = all_metrics.get(key, 0) + value

            if config.evaluation_params.generate_umap_plots:
                # ... UMAP data collection logic as before ...
                pass

    avg_total_loss = total_loss / len(dataloader) if dataloader and len(dataloader) > 0 else 0
    for key in all_metrics.keys(): all_metrics[key] /= len(dataloader) if dataloader and len(dataloader) > 0 else 1

    print(f"Joint Model Evaluation - Average Total Loss: {avg_total_loss:.4f}")
    for k, v in all_metrics.items(): print(f"  Avg {k}: {v:.4f}")
    results = {"avg_total_loss": avg_total_loss, **all_metrics}

    if config.evaluation_params.generate_umap_plots:
        # ... UMAP plotting logic as before ...
        pass

    return results


def run_evaluation(config, model_checkpoint_path, eval_type="all", data_loaders=None):
    device = torch.device(config.training_params.device if torch.cuda.is_available() else "cpu")
    model = CombinedModel(config)
    start_epoch, best_metric = load_checkpoint(model_checkpoint_path, model, optimizer=None, scheduler=None,
                                               device=device, strict=True)

    model.to(device)
    model.eval()
    loss_fn_instance = CombinedLoss(config.loss_weights, config=config, device=device)
    all_results = {}

    if data_loaders is None: raise ValueError("data_loaders must be provided to run_evaluation.")

    if eval_type == "gae" or eval_type == "all":
        print("\n--- Evaluating GAE Reconstruction ---")
        if data_loaders.get("gae_eval_loader"):
            all_results["gae_reconstruction"] = evaluate_gae_reconstruction(model, data_loaders["gae_eval_loader"],
                                                                            loss_fn_instance, device, config)
        else:
            print("GAE evaluation loader not provided, skipping.")

    if eval_type == "joint" or eval_type == "all":
        print("\n--- Evaluating Joint Model Prediction ---")
        if data_loaders.get("joint_eval_loader"):
            all_results["joint_prediction"] = evaluate_joint_prediction(model, data_loaders["joint_eval_loader"],
                                                                        loss_fn_instance, device, config)
        else:
            print("Joint evaluation loader not provided, skipping.")

    output_dir_eval = config.evaluation_params.get("output_dir", "results/eval_output")
    os.makedirs(output_dir_eval, exist_ok=True)
    results_filepath = os.path.join(output_dir_eval, "evaluation_metrics.json")
    with open(results_filepath, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nEvaluation metrics saved to {results_filepath}")

    return all_results
