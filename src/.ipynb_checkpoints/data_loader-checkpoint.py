import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix, coo_matrix
from functools import partial

def load_gene_graph_data(data_dir, config):
    expr_filename = config.get('expr_filename', 'ALL_cells_HVG_expr.csv')
    meta_filename = config.get('meta_filename', 'ALL_cells_meta.csv')
    adj_filename = config.get('adj_filename', 'model_input_adj_mat.npy')

    expr_path = os.path.join(data_dir, expr_filename)
    meta_path = os.path.join(data_dir, meta_filename)
    adj_path = os.path.join(data_dir, adj_filename)

    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"Expression file not found: {expr_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"Adjacency matrix file not found: {adj_path}")

    expr_df = pd.read_csv(expr_path, index_col=0)
    
    if expr_df.isnull().values.any():
        expr_df = expr_df.fillna(0) 
    if np.isinf(expr_df.values).any():
        expr_df = expr_df.replace([np.inf, -np.inf], 0)

    meta_df = pd.read_csv(meta_path, index_col=0)

    if not expr_df.index.equals(meta_df.index):
        common_cells = expr_df.index.intersection(meta_df.index)
        if len(common_cells) == 0:
            raise ValueError("No common cells found between expression and metadata. Please check cell IDs.")
        expr_df = expr_df.loc[common_cells]
        meta_df = meta_df.loc[common_cells]

    adj_mat_loaded = np.load(adj_path, allow_pickle=True)

    if isinstance(adj_mat_loaded, (coo_matrix, csr_matrix)):
        adj_mat_sparse = adj_mat_loaded
    elif isinstance(adj_mat_loaded, np.ndarray):
        if adj_mat_loaded.ndim == 0 and isinstance(adj_mat_loaded.item(), (coo_matrix, csr_matrix)):
            adj_mat_sparse = adj_mat_loaded.item()
        else:
            adj_mat_sparse = csr_matrix(adj_mat_loaded)
    else:
      raise TypeError(f"Unsupported adjacency matrix type: {type(adj_mat_loaded)}")

    gene_names = list(expr_df.columns)
    cell_names = list(expr_df.index)
    
    num_genes_from_expr = expr_df.shape[1]
    if config.get('model_params') is not None and config['model_params'].get('num_genes') is None:
        config['model_params']['num_genes'] = num_genes_from_expr
    elif config.get('model_params') is not None and config['model_params'].get('num_genes') != num_genes_from_expr:
        raise ValueError(f"Configured num_genes {config['model_params']['num_genes']} != actual num_genes {num_genes_from_expr}")
    
    X_all_np = expr_df.values

    if 'day' not in meta_df.columns:
        raise ValueError("Missing 'day' column in meta_df, which is crucial for temporal modeling.")
    
    day_values_series = pd.to_numeric(meta_df['day'], errors='coerce')
    
    valid_day_mask = day_values_series.notna().values
    if day_values_series.isnull().any():
        X_all_np = X_all_np[valid_day_mask]
        meta_df = meta_df[valid_day_mask]
        cell_names = list(meta_df.index) 
        day_values_series = day_values_series[valid_day_mask]

    meta_df['day_numeric'] = day_values_series
    
    if adj_mat_sparse.shape[0] != len(gene_names):
        raise ValueError(f"Adjacency matrix dimension ({adj_mat_sparse.shape[0]}) does not match gene count ({len(gene_names)}).")

    edge_index, edge_weight = from_scipy_sparse_matrix(adj_mat_sparse)
    shared_edge_index = edge_index.long()
    shared_edge_weight = edge_weight.float()
    
    return X_all_np, shared_edge_index, shared_edge_weight, gene_names, cell_names, meta_df


def create_cells_by_day_mapping(meta_df):
    cells_by_day_indices = {}
    if 'day_numeric' not in meta_df.columns:
        raise ValueError("Numeric 'day' column ('day_numeric') not found in meta_df.")
    
    for day_val, group in meta_df.groupby('day_numeric'):
        if pd.notna(day_val):
            indices_for_day = group.index.map(lambda x: meta_df.index.get_loc(x)).tolist()
            cells_by_day_indices[float(day_val)] = indices_for_day
    return cells_by_day_indices


def generate_training_transition_pairs(all_available_days, excluded_day=None, start_day_config=None):
    training_transitions = []
    valid_days = sorted([d for d in all_available_days if d != excluded_day and pd.notna(d)])
    
    days_to_start_from = []
    if start_day_config and 'days' in start_day_config and start_day_config['days'] is not None:
        specified_start_days = start_day_config['days']
        if not isinstance(specified_start_days, list):
            specified_start_days = [float(specified_start_days)]
        else:
            specified_start_days = [float(d) for d in specified_start_days]
        
        for start_day in specified_start_days:
            if start_day not in valid_days:
                print(f"Warning: Specified start_day {start_day} for pair generation is not in valid_days or is the excluded_day. Skipping.")
                continue
            days_to_start_from.append(start_day)
    else: 
        if len(valid_days) > 1:
            days_to_start_from = [float(d) for d in valid_days[:-1]]
        else:
            days_to_start_from = []
                
    for start_day in days_to_start_from:
        for target_day in valid_days:
            if target_day > start_day:
                training_transitions.append((float(start_day), float(target_day)))

    if not training_transitions:
        print("Warning: No training transition pairs were generated. Check day configurations and excluded_day.")

    return list(set(training_transitions))


class SingleCellTransitionDataset(Dataset):
    def __init__(self, transition_pairs):
        self.transition_pairs = transition_pairs

    def __len__(self):
        return len(self.transition_pairs)

    def __getitem__(self, idx):
        return self.transition_pairs[idx]


def custom_collate_fn_transitions(batch_of_transition_tuples, 
                                  X_all_expressions_np, 
                                  shared_edge_index, 
                                  shared_edge_weight, 
                                  cells_by_day_indices, 
                                  epoch_fixed_sample_size_per_day,
                                  num_genes):
    day_from, day_to = batch_of_transition_tuples[0] 

    from_indices_pool_full = cells_by_day_indices.get(day_from, [])
    to_indices_pool_full = cells_by_day_indices.get(day_to, [])

    if not from_indices_pool_full or not to_indices_pool_full:
        return None

    num_from_to_sample = min(len(from_indices_pool_full), epoch_fixed_sample_size_per_day)
    num_to_to_sample = min(len(to_indices_pool_full), epoch_fixed_sample_size_per_day)
    
    actual_sample_size = min(num_from_to_sample, num_to_to_sample)
    
    if actual_sample_size == 0 :
        return None
    
    replace_from = len(from_indices_pool_full) < actual_sample_size
    replace_to = len(to_indices_pool_full) < actual_sample_size

    selected_pos_indices_from = np.random.choice(np.array(from_indices_pool_full, dtype=np.int64), 
                                                 size=actual_sample_size, replace=replace_from)
    selected_pos_indices_to = np.random.choice(np.array(to_indices_pool_full, dtype=np.int64), 
                                               size=actual_sample_size, replace=replace_to)

    data_list_t0 = []
    original_expressions_t0 = []
    for i in range(actual_sample_size):
        cell_expr_np_t0 = X_all_expressions_np[selected_pos_indices_from[i]] 
        original_expressions_t0.append(torch.tensor(cell_expr_np_t0, dtype=torch.float32))
        node_features_t0 = torch.tensor(cell_expr_np_t0, dtype=torch.float32).unsqueeze(-1) 
        data_list_t0.append(Data(x=node_features_t0, edge_index=shared_edge_index, edge_attr=shared_edge_weight, day=torch.tensor(day_from, dtype=torch.float32), num_nodes=num_genes))
    
    data_list_t1 = []
    original_expressions_t1 = []
    for i in range(actual_sample_size):
        cell_expr_np_t1 = X_all_expressions_np[selected_pos_indices_to[i]]
        original_expressions_t1.append(torch.tensor(cell_expr_np_t1, dtype=torch.float32))
        node_features_t1 = torch.tensor(cell_expr_np_t1, dtype=torch.float32).unsqueeze(-1)
        data_list_t1.append(Data(x=node_features_t1, edge_index=shared_edge_index, edge_attr=shared_edge_weight, day=torch.tensor(day_to, dtype=torch.float32), num_nodes=num_genes))

    batch_t0 = Batch.from_data_list(data_list_t0)
    batch_t1 = Batch.from_data_list(data_list_t1)
    
    original_expressions_t0_tensor = torch.stack(original_expressions_t0)
    original_expressions_t1_tensor = torch.stack(original_expressions_t1)

    return {
        'batch_t0': batch_t0,    
        'batch_t1': batch_t1,
        'original_expressions_t0': original_expressions_t0_tensor, 
        'original_expressions_t1': original_expressions_t1_tensor, 
        'day_t0_scalar': torch.tensor(day_from, dtype=torch.float32), 
        'day_t1_scalar': torch.tensor(day_to, dtype=torch.float32),
        'num_samples_in_batch': actual_sample_size 
    }

def get_transition_dataloader(config, X_all_expressions_np, shared_edge_index, shared_edge_weight, meta_df, cells_by_day_indices, sampler_class=None): 
    all_days_unique = sorted(meta_df['day_numeric'].dropna().unique())
    num_genes = config['model_params']['num_genes']
    
    excluded_day_eval_config = config.get('excluded_day_for_eval', None)
    excluded_day_eval = float(excluded_day_eval_config) if excluded_day_eval_config is not None else None

    start_day_logic_config = config.get('start_day_logic_for_pairs', {'days': None})

    training_transitions = generate_training_transition_pairs(
        all_available_days=all_days_unique,
        excluded_day=excluded_day_eval,
        start_day_config=start_day_logic_config
    )

    if not training_transitions:
        print("Error: No training transitions could be generated. Training cannot proceed.")
        return None

    train_dataset = SingleCellTransitionDataset(training_transitions)
    
    dataloader_batch_size = 1 
    
    collate_fn_partial = partial(custom_collate_fn_transitions, 
                                 X_all_expressions_np=X_all_expressions_np,
                                 shared_edge_index=shared_edge_index,
                                 shared_edge_weight=shared_edge_weight,
                                 cells_by_day_indices=cells_by_day_indices,
                                 epoch_fixed_sample_size_per_day=config.get('epoch_fixed_sample_size_per_day', 200),
                                 num_genes=num_genes) 
    
    pin_memory_flag = True if config.get('device', 'cpu') == 'cuda' else False

    sampler = None
    shuffle = True
    if sampler_class is not None and config.get('ddp', {}).get('use_ddp', False):
        sampler = sampler_class(train_dataset, shuffle=True) 
        shuffle = False 

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_batch_size, 
        shuffle=shuffle, 
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn_partial,
        pin_memory=pin_memory_flag,
        drop_last=True if sampler is not None else False, 
        sampler=sampler
    )
    
    val_loader = None 
    
    return train_loader, val_loader


class AllCellsDataset(Dataset):
    def __init__(self, X_all_expressions_np, list_of_cell_days_np, num_genes):
        self.X_all_expressions_np = X_all_expressions_np
        self.list_of_cell_days_np = list_of_cell_days_np 
        self.num_genes = num_genes

    def __len__(self):
        return self.X_all_expressions_np.shape[0]

    def __getitem__(self, idx):
        cell_expr = self.X_all_expressions_np[idx]
        cell_day = self.list_of_cell_days_np[idx]
        return cell_expr, cell_day


def collate_fn_gae_pretrain(batch_of_cells_data, shared_edge_index, shared_edge_weight, num_genes):
    data_list = []
    original_expressions = []
    for cell_expr_np, cell_day_np in batch_of_cells_data:
        original_expressions.append(torch.tensor(cell_expr_np, dtype=torch.float32))
        node_features = torch.tensor(cell_expr_np, dtype=torch.float32).unsqueeze(-1) 
        data_list.append(Data(x=node_features, edge_index=shared_edge_index, edge_attr=shared_edge_weight, day=torch.tensor(cell_day_np, dtype=torch.float32), num_nodes=num_genes))
    
    pyg_batch = Batch.from_data_list(data_list)
    original_expressions_tensor = torch.stack(original_expressions) 

    return {
        'pyg_batch': pyg_batch,
        'original_expressions': original_expressions_tensor 
    }

def get_gae_pretrain_dataloader(config, X_all_expressions_np, shared_edge_index, shared_edge_weight, meta_df, sampler_class=None):
    num_genes = config['model_params']['num_genes']
    cell_days_np = meta_df['day_numeric'].values 

    dataset = AllCellsDataset(X_all_expressions_np, cell_days_np, num_genes) 
    gae_batch_size = config.get('gae_pretrain_batch_size', 64)
    pin_memory_flag = True if config.get('device', 'cpu') == 'cuda' else False
    
    collate_fn_partial_gae = partial(collate_fn_gae_pretrain,
                                     shared_edge_index=shared_edge_index,
                                     shared_edge_weight=shared_edge_weight,
                                     num_genes=num_genes)
    sampler = None
    shuffle = True
    if sampler_class is not None and config.get('ddp', {}).get('use_ddp', False):
        sampler = sampler_class(dataset, shuffle=True)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=gae_batch_size, 
        shuffle=shuffle,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn_partial_gae, 
        pin_memory=pin_memory_flag,
        drop_last=True if sampler is not None else False, 
        sampler=sampler
    )
    return dataloader