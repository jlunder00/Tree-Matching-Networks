#configs/default_tree_config.py 
from ...GMN.configure import get_default_config
import yaml
from pathlib import Path
import torch

def get_tree_config(task_type='entailment', base_config_path=None, override_path=None):
    """Get configuration for tree matching with better organization"""
    # Start with GMN base config
    config = get_default_config()
    
    # Load base task config
    if base_config_path:
        base_path = Path(base_config_path) 
    else:
        # Default to configs within package
        base_path = Path(__file__).parent / "experiment_configs"
        base_config_path = base_path / f"{task_type}_config.yaml"
        
    with open(base_config_path) as f:
        task_config = yaml.safe_load(f)
    config.update(task_config)
    
    # Load override config if provided
    if override_path:
        with open(override_path) as f:
            override_config = yaml.safe_load(f)
            config.update(override_config)
            
    # Add runtime configs
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config

# def get_tree_config(dataset_type='snli', task_type='entailment', config_path=None):
#     """Get configuration for tree matching"""
#     # Start with GMN base config
#     config = get_default_config()
#     
#     # Add tree-specific defaults
#     tree_config = {
#         'model': {
#             'task_type': task_type,
#             'loss_params': {
#                 'thresholds': [-0.3, 0.3] if task_type == 'entailment' else None,
#                 'margin': 0.1 if task_type == 'similarity' else None
#             },
#             'name': 'tree_matching',
#             'node_feature_dim': 804,
#             'edge_feature_dim': 22,
#             'node_hidden_dim': 256,
#             'edge_hidden_dim': 128,
#             'n_prop_layers': 5,
#             'dropout': 0.2,
#             'layer_norm': True,
#             'weight_norm': True
#         },
#         'data': {
#             'dataset_type': dataset_type,
#             'task_type': task_type,
#             'spacy_variant': 'trf',
#             'loading_pattern': 'sequential',
#             'batch_size': 128,
#             'max_nodes_per_batch': 1000,
#             'max_edges_per_batch': 2000,
#             'num_workers': 2,
#             'prefetch_factor': 2
#         },
#         'train': {
#             'learning_rate': 5e-4,
#             'min_learning_rate': 1e-6,
#             'weight_decay': 1e-5,
#             'scheduler': {
#                 'cosine': {
#                     'T_0': 5,
#                     'T_mult': 2,
#                     'eta_min': 1e-6
#                 },
#                 'plateau': {
#                     'factor': 0.5,
#                     'patience': 3,
#                     'min_lr': 1e-6
#                 }
#             },
#             'n_epochs': 100,
#             'patience': 10,
#             'warmup_steps': 1000,
#             'gradient_accumulation_steps': 4,
#             'clip_value': 1.0,
#             'cleanup_interval': 5
#         },
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#         'wandb': {
#             'project': 'tree-matching',
#             'tags': [dataset_type, task_type],
#             'log_interval': 100,
#             'memory_logging': True
#         }
#     }
#     config.update(tree_config)
#     
#     # Override with user config if provided
#     if config_path:
#         with open(config_path) as f:
#             user_config = yaml.safe_load(f)
#             config.update(user_config)
#     
#     return config
