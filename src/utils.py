import os
import random
import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import umap


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config_device_str="cuda"):
    if config_device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_ddp(rank, world_size, backend='nccl', init_method='env://'):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    if init_method == 'env://':
        dist.init_process_group(backend=backend, init_method=init_method)
    else:
        dist.init_process_group(backend=backend, init_method=init_method,
                                rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(state, filepath, is_best=False, best_filepath=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        os.makedirs(os.path.dirname(best_filepath), exist_ok=True)
        torch.save(state, best_filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu', strict=True):  # Added strict param
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    # Always load to CPU first to avoid GPU memory issues if checkpoint was saved on different device setup
    checkpoint = torch.load(filepath, map_location='cpu')

    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # Handle potential state_dict key mismatches (e.g. 'module.' prefix)
    ckpt_state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    is_ddp_checkpoint = any(key.startswith('module.') for key in ckpt_state_dict.keys())

    for k, v in ckpt_state_dict.items():
        name = k
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) and not is_ddp_checkpoint:
            name = 'module.' + k  # Prepend 'module.' if current model is DDP and checkpoint is not
        elif not isinstance(model, torch.nn.parallel.DistributedDataParallel) and is_ddp_checkpoint:
            name = k[7:]  # Remove 'module.' if current model is not DDP and checkpoint is
        new_state_dict[name] = v

    # Load state_dict with specified strictness
    load_report = model_to_load.load_state_dict(new_state_dict, strict=strict)

    if not strict and (load_report.missing_keys or load_report.unexpected_keys):
        print("Checkpoint loaded with strict=False. Report:")
        if load_report.missing_keys:
            print(f"  Missing keys: {load_report.missing_keys}")
        if load_report.unexpected_keys:  # Should be empty if we adjusted for module. prefix
            print(f"  Unexpected keys in checkpoint: {load_report.unexpected_keys}")
    elif strict and (load_report.missing_keys or load_report.unexpected_keys):
        # This case should ideally not happen if strict=True unless model def truly changed
        print(
            f"Warning: Strict loading reported issues. Missing: {load_report.missing_keys}, Unexpected: {load_report.unexpected_keys}")

    model.to(device)  # Move model to target device *after* loading state dict

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer states to device
            for state in optimizer.state.values():
                for k, v_opt in state.items():
                    if isinstance(v_opt, torch.Tensor):
                        state[k] = v_opt.to(device)
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    if scheduler and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_metric = checkpoint.get('best_metric', float('inf'))

    print(
        f"Loaded checkpoint from {filepath}. Model configured for device {device}. Resuming from epoch {start_epoch}.")
    return start_epoch, best_metric


def plot_umap_comparison(
        data_dict,
        output_path,
        title="UMAP Comparison",
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        n_components=2,
        random_state=42,
        point_size=5,
        alpha=0.7,
        show_legend=True,
        colors=None
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        low_memory=True
    )

    embeddings = []
    labels = []
    current_label_val = 0

    all_data_list = []
    for key, data_points in data_dict.items():
        if isinstance(data_points, torch.Tensor):
            data_points = data_points.detach().cpu().numpy()
        if data_points.ndim > 2:
            data_points = data_points.squeeze()
        if data_points.ndim == 1:
            data_points = data_points.reshape(1, -1)
        if data_points.shape[0] == 0:
            continue
        all_data_list.append(data_points)

    if not all_data_list:
        print("No data to plot for UMAP.")
        return

    combined_data = np.concatenate(all_data_list, axis=0)

    try:
        embedding_combined = reducer.fit_transform(combined_data)
    except Exception as e:
        print(f"Error during UMAP fitting/transforming: {e}")
        try:
            print("Attempting UMAP with default parameters...")
            reducer_fallback = umap.UMAP(random_state=random_state, low_memory=True)
            embedding_combined = reducer_fallback.fit_transform(combined_data)
        except Exception as e_fallback:
            print(f"Fallback UMAP also failed: {e_fallback}. Skipping plot.")
            return

    plt.figure(figsize=(12, 10))

    start_idx = 0
    plot_labels = []
    plot_embeddings_x = []
    plot_embeddings_y = []

    for i, (key, data_points) in enumerate(data_dict.items()):
        if data_points.shape[0] == 0:
            continue
        num_points = data_points.shape[0]
        end_idx = start_idx + num_points

        current_embedding = embedding_combined[start_idx:end_idx]

        if colors and i < len(colors):
            scatter_color = colors[i]
        else:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            default_colors = prop_cycle.by_key()['color']
            scatter_color = default_colors[i % len(default_colors)]

        plt.scatter(
            current_embedding[:, 0],
            current_embedding[:, 1],
            label=key,
            s=point_size,
            alpha=alpha,
            color=scatter_color
        )
        start_idx = end_idx

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.gca().set_aspect('equal', 'datalim')
    if show_legend:
        plt.legend(markerscale=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"UMAP plot saved to {output_path}")


def get_optimizer(model_parameters, config):
    opt_name = config.optimizer.lower()
    lr = config.learning_rate_joint
    weight_decay = getattr(config, "weight_decay", 0.0)

    if opt_name == "adam":
        return torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def get_scheduler(optimizer, config):
    scheduler_name = getattr(config, "lr_scheduler", None)
    if scheduler_name is None:
        return None

    scheduler_name = scheduler_name.lower()
    params_container = getattr(config, "lr_scheduler_params", {})
    # Ensure config.lr_scheduler (which is a string like "ReduceLROnPlateau") is used to get the correct sub-dict
    # Also, use .get on the sub-dict for individual hyperparams for safety.
    scheduler_specific_key = config.lr_scheduler  # The actual name e.g. "ReduceLROnPlateau"

    # OmegaConf specific: if params_container is an OmegaConf DictConfig,
    # and scheduler_specific_key is a valid key, direct access is fine.
    # If it might not exist, .get() is safer.
    params = params_container.get(scheduler_specific_key, {}) if hasattr(params_container, 'get') else \
        getattr(params_container, scheduler_specific_key, {}) if hasattr(params_container,
                                                                         scheduler_specific_key) else {}

    if scheduler_name == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(params.get("step_size", 30)),
                                               gamma=float(params.get("gamma", 0.1)))
    elif scheduler_name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=float(params.get("factor", 0.1)),
                                                          patience=int(params.get("patience", 10)))
    elif scheduler_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(params.get("T_max", 100)),
                                                          eta_min=float(params.get("eta_min", 0)))
    else:
        print(f"Unsupported scheduler: {scheduler_name}. No scheduler will be used.")
        return None

