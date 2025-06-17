import torch
import numpy as np
import random
import os
import yaml
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(config_device_str):
    if config_device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

def save_checkpoint(state, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    return model, optimizer, scheduler, epoch, best_loss

def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    if train_losses:
        plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def log_metrics_to_console(epoch, num_epochs, metrics_dict, prefix=""):
    log_str = f"Epoch [{epoch+1}/{num_epochs}]"
    if prefix:
        log_str = f"{prefix} " + log_str
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            log_str += f", {key}: {value:.4f}"
        else:
            log_str += f", {key}: {value}"
    print(log_str)

def setup_ddp(rank, world_size, config):
    os.environ['MASTER_ADDR'] = config['ddp'].get('master_addr', 'localhost')
    os.environ['MASTER_PORT'] = config['ddp'].get('master_port', '12355')
    
    init_method = config['ddp'].get('init_method', 'env://')
    
    torch.distributed.init_process_group(
        backend=config['ddp']['backend'],
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    torch.distributed.destroy_process_group()