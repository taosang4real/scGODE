import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix, coo_matrix
from functools import partial
from omegaconf import OmegaConf, ListConfig

def load_gene_graph_data(data_dir, config):
    data_params = config.data_params
    model_params = config.model_params
    expr_filename = data_params.get('expr_filename', 'ALL_cells_HVG_expr.csv')
    meta_filename = data_params.get('meta_filename', 'ALL_cells_meta.csv')
    adj_filename = data_params.get('adj_filename', 'model_input_adj_mat.npy')
    expr_path = os.path.join(data_dir, expr_filename)
    meta_path = os.path.join(data_dir, meta_filename)
    adj_path = os.path.join(data_dir, adj_filename)
    if not all(os.path.exists(p) for p in [expr_path, meta_path, adj_path]):
        raise FileNotFoundError(f"One or more data files not found in {data_dir}")
    expr_df = pd.read_csv(expr_path, index_col=0).fillna(0).replace([np.inf, -np.inf], 0)
    meta_df = pd.read_csv(meta_path, index_col=0)
    common_cells = expr_df.index.intersection(meta_df.index)
    expr_df, meta_df = expr_df.loc[common_cells], meta_df.loc[common_cells]
    adj_mat_loaded = np.load(adj_path, allow_pickle=True)
    adj_mat_sparse = adj_mat_loaded.item() if adj_mat_loaded.ndim == 0 and hasattr(adj_mat_loaded.item(), 'shape') else csr_matrix(adj_mat_loaded)
    num_genes_from_expr = expr_df.shape[1]
    if model_params.get('num_genes') is None:
        model_params.num_genes = num_genes_from_expr
        if OmegaConf.is_config(config.model_params):
             OmegaConf.set_struct(config.model_params, False)
             config.model_params.num_genes_from_data = num_genes_from_expr
             OmegaConf.set_struct(config.model_params, True)
    elif model_params.get('num_genes') != num_genes_from_expr:
        raise ValueError("Configured num_genes mismatch.")
    X_all_np = expr_df.values
    day_values = pd.to_numeric(meta_df[data_params.day_column], errors='coerce')
    valid_mask = day_values.notna()
    if day_values.isnull().any():
        X_all_np, meta_df = X_all_np[valid_mask], meta_df[valid_mask]
        day_values = day_values[valid_mask]
    meta_df['day_numeric'] = day_values.astype(float)
    if adj_mat_sparse.shape[0] != num_genes_from_expr:
        raise ValueError("Adjacency matrix dimension mismatch.")
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_mat_sparse)
    return X_all_np, edge_index.long(), edge_weight.float() if edge_weight is not None else None, list(expr_df.columns), list(expr_df.index), meta_df

def create_cells_by_day_mapping(meta_df, cell_names):
    cells_by_day = {}
    barcode_to_idx = {barcode: i for i, barcode in enumerate(cell_names)}
    for day, group in meta_df.groupby('day_numeric'):
        if pd.notna(day):
            cells_by_day[float(day)] = [barcode_to_idx[b] for b in group.index if b in barcode_to_idx]
    return cells_by_day

def generate_training_transition_pairs(all_available_days, excluded_day_config=None, start_day_config=None):
    excluded_day = float(excluded_day_config) if excluded_day_config is not None else None
    valid_days = sorted([d for d in all_available_days if pd.notna(d) and d != excluded_day])
    days_to_start_from, raw_start_days = [], start_day_config.get('days') if start_day_config else None
    if raw_start_days is not None:
        processed_start_days = [float(raw_start_days)] if isinstance(raw_start_days, (int, float)) else [float(d) for d in raw_start_days] if isinstance(raw_start_days, (list, tuple, ListConfig)) and len(raw_start_days) > 0 else []
        days_to_start_from = [d for d in processed_start_days if d in valid_days]
    if not days_to_start_from and len(valid_days) > 1:
        days_to_start_from = valid_days[:-1]
    return list(set([(s, t) for s in days_to_start_from for t in valid_days if t > s]))

class SingleCellTransitionDataset(Dataset):
    def __init__(self, transition_pairs): self.transition_pairs = transition_pairs
    def __len__(self): return len(self.transition_pairs)
    def __getitem__(self, idx): return self.transition_pairs[idx]

def custom_collate_fn_transitions(batch, X_all, edge_index, edge_weight, cells_by_day, sample_size, num_genes):
    if not batch: return None
    day_from, day_to = batch[0]
    from_pool, to_pool = cells_by_day.get(day_from, []), cells_by_day.get(day_to, [])
    if not from_pool or not to_pool: return None
    size = min(len(from_pool), len(to_pool), int(sample_size))
    if size == 0: return None
    from_indices = np.random.choice(from_pool, size, replace=len(from_pool) < size)
    to_indices = np.random.choice(to_pool, size, replace=len(to_pool) < size)
    expressions_t0 = torch.from_numpy(X_all[from_indices]).float()
    expressions_t1 = torch.from_numpy(X_all[to_indices]).float()
    data_list_t0 = [Data(x=expr.unsqueeze(-1), edge_index=edge_index) for expr in expressions_t0]
    data_list_t1 = [Data(x=expr.unsqueeze(-1), edge_index=edge_index) for expr in expressions_t1]
    return {'data_t0': Batch.from_data_list(data_list_t0), 'data_t1': Batch.from_data_list(data_list_t1),
            'original_expressions_t0': expressions_t0, 'original_expressions_t1': expressions_t1,
            'day_t0_scalar': torch.tensor(day_from, dtype=torch.float32),
            'day_t1_scalar': torch.tensor(day_to, dtype=torch.float32)}

def get_transition_dataloader(config, X_all, edge_index, edge_weight, meta, cells_by_day, sampler_class=None, cell_sample_size_override=None):
    all_days = sorted(meta['day_numeric'].dropna().unique())
    transitions = generate_training_transition_pairs(all_days, config.data_params.get('excluded_day_for_eval'), config.data_params.get('start_day_logic_for_pairs'))
    if not transitions: return DataLoader(SingleCellTransitionDataset([]), batch_size=1, collate_fn=lambda x: None), None
    dataset = SingleCellTransitionDataset(transitions)
    batch_size = config.training_params.get('joint_train_collate_batch_size', 1)
    sample_size = cell_sample_size_override if cell_sample_size_override is not None else config.data_params.epoch_fixed_sample_size_per_day
    collate_fn = partial(custom_collate_fn_transitions, X_all=X_all, edge_index=edge_index, edge_weight=edge_weight,
                         cells_by_day=cells_by_day, sample_size=sample_size, num_genes=config.model_params.num_genes)
    sampler = sampler_class(dataset, shuffle=True, drop_last=True) if sampler_class and config.training_params.ddp.use_ddp else None
    shuffle = sampler is None and (cell_sample_size_override is None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config.get('num_workers', 0),
                      collate_fn=collate_fn, pin_memory=True, sampler=sampler), None

class AllCellsDataset(Dataset):
    def __init__(self, X, days): self.X, self.days = X, days
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], float(self.days[idx])

def collate_fn_gae_pretrain(batch, edge_index, edge_weight, num_genes):
    expressions, _ = zip(*batch)
    original_expressions = torch.tensor(np.array(expressions), dtype=torch.float32)
    data_list = [Data(x=expr.unsqueeze(-1), edge_index=edge_index) for expr in original_expressions]
    return {'pyg_batch': Batch.from_data_list(data_list), 'original_expressions': original_expressions}

def get_gae_pretrain_dataloader(config, X_all, edge_index, edge_weight, meta, sampler_class=None, is_eval=False):
    dataset = AllCellsDataset(X_all, meta['day_numeric'].values)
    batch_size = config.evaluation_params.get('eval_batch_size', 64) if is_eval else config.training_params.get('gae_pretrain_batch_size', 64)
    shuffle = not is_eval
    collate_fn = partial(collate_fn_gae_pretrain, edge_index=edge_index, edge_weight=edge_weight, num_genes=config.model_params.num_genes)
    sampler = sampler_class(dataset, shuffle=True, drop_last=True) if sampler_class and not is_eval and config.training_params.ddp.use_ddp else None
    if sampler: shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config.get('num_workers', 0),
                      collate_fn=collate_fn, pin_memory=True, sampler=sampler)
