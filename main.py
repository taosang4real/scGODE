import os
import argparse
from omegaconf import OmegaConf, ListConfig
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from functools import partial

from src.utils import set_seed, get_device, setup_ddp, cleanup_ddp, load_checkpoint
from src.data_loader import (
    load_gene_graph_data,
    create_cells_by_day_mapping,
    get_gae_pretrain_dataloader,
    get_transition_dataloader,
    SingleCellTransitionDataset
)
from src.models.combined_model import CombinedModel
from src.trainer import Trainer
from src.evaluate import run_evaluation


def _run_process(rank, world_size, config, args):
    is_main_process = rank == 0
    set_seed(config.seed + rank)
    device = get_device(config.training_params.device)

    if is_main_process: print("Loading data...")
    X_all_np, shared_edge_index, shared_edge_weight, _, _, meta_df = \
        load_gene_graph_data(config.data_params.data_dir, config)
    cells_by_day_indices = create_cells_by_day_mapping(meta_df, meta_df.index.tolist())

    sampler_class = DistributedSampler if config.training_params.ddp.use_ddp and world_size > 1 else None

    gae_train_loader, gae_eval_loader = None, None
    if args.mode == "pretrain_gae" or (
            args.mode == "evaluate" and (args.eval_type == "gae" or args.eval_type == "all")):
        gae_train_loader = get_gae_pretrain_dataloader(config, X_all_np, shared_edge_index, shared_edge_weight, meta_df,
                                                       sampler_class, is_eval=False)
        gae_eval_loader = get_gae_pretrain_dataloader(config, X_all_np, shared_edge_index, shared_edge_weight, meta_df,
                                                      None, is_eval=True)

    transition_train_loader, joint_eval_loader = None, None
    if args.mode == "train_joint" or (
            args.mode == "evaluate" and (args.eval_type == "joint" or args.eval_type == "all")):
        transition_train_loader, _ = get_transition_dataloader(config, X_all_np, shared_edge_index, shared_edge_weight,
                                                               meta_df, cells_by_day_indices, sampler_class)
        joint_eval_loader, _ = get_transition_dataloader(config, X_all_np, shared_edge_index, shared_edge_weight,
                                                         meta_df, cells_by_day_indices, None,
                                                         cell_sample_size_override=config.evaluation_params.get(
                                                             'joint_eval_cell_sample_size', 64))

    if is_main_process: print(f"Initializing model for mode: {args.mode}")
    model = CombinedModel(config)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path) and is_main_process:
        print(f"Checkpoint path provided: {args.checkpoint_path}.")
    elif args.checkpoint_path and not os.path.exists(args.checkpoint_path) and is_main_process:
        print(f"Warning: Provided checkpoint_path does not exist: {args.checkpoint_path}")

    if args.mode in ["pretrain_gae", "train_joint"]:
        trainer = Trainer(config, model, rank, world_size)
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            try:
                trainer.load_model_checkpoint(args.checkpoint_path)
            except RuntimeError as e:
                if is_main_process: print(
                    f"Error loading checkpoint {args.checkpoint_path}: {e}\nProceeding without loading.")
                trainer.current_epoch = 0

        if args.mode == "pretrain_gae":
            trainer.pretrain_gae(gae_train_loader, val_dataloader=gae_eval_loader)
        elif args.mode == "train_joint":
            trainer.train_joint(transition_train_loader, val_dataloader=joint_eval_loader)

    elif args.mode == "evaluate":
        if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError("Checkpoint path must be provided for evaluation.")

        eval_dataloaders = {"gae_eval_loader": gae_eval_loader, "joint_eval_loader": joint_eval_loader}
        if is_main_process:
            run_evaluation(config, args.checkpoint_path, args.eval_type, eval_dataloaders)
        if config.training_params.ddp.use_ddp and world_size > 1: dist.barrier()
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if is_main_process: print("Process finished.")


def main_worker(rank, world_size, config, args):
    if rank == 0: print(f"Running DDP on rank {rank}.")
    os.environ['MASTER_ADDR'] = str(config.training_params.ddp.get('master_addr', 'localhost'))
    os.environ['MASTER_PORT'] = str(config.training_params.ddp.get('master_port', '29500'))
    dist.init_process_group(config.training_params.ddp.get('backend', 'nccl'), rank=rank, world_size=world_size)
    _run_process(rank, world_size, config, args)
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-cell Dynamics Prediction Model")
    parser.add_argument("--config", type=str, default="configs/main_config.yaml", help="Path to YAML config file.")
    parser.add_argument("--mode", type=str, choices=["pretrain_gae", "train_joint", "evaluate"], required=True,
                        help="Operating mode.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a model checkpoint.")
    parser.add_argument("--eval_type", type=str, choices=["gae", "joint", "all"], default="all",
                        help="Type of evaluation.")
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP.')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    experiment_name = config.get("experiment_name", "experiment_default")
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


    def resolve_path(path_template):
        if path_template is None: return None
        path_with_exp_name = path_template.replace("experiment_default", experiment_name)
        return os.path.join(PROJECT_ROOT_DIR, path_with_exp_name) if not os.path.isabs(
            path_with_exp_name) else path_with_exp_name


    config.training_params.checkpoint_dir = resolve_path(config.training_params.checkpoint_dir)
    config.logging.log_file = resolve_path(config.logging.log_file)
    config.logging.tensorboard_log_dir = resolve_path(config.logging.tensorboard_log_dir)
    if hasattr(config.evaluation_params, "output_dir"): config.evaluation_params.output_dir = resolve_path(
        config.evaluation_params.output_dir)
    if not os.path.isabs(config.data_params.data_dir): config.data_params.data_dir = os.path.join(PROJECT_ROOT_DIR,
                                                                                                  config.data_params.data_dir)

    use_ddp = config.training_params.ddp.use_ddp and torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(main_worker, args=(world_size, config, args), nprocs=world_size, join=True)
    else:
        if config.training_params.ddp.use_ddp:
            print("DDP requested but not enabled. Running on a single process.")
        else:
            print("Running on a single process.")
        _run_process(0, 1, config, args)