from tree_matching_networks.GMN.configure import get_default_config
import yaml
from pathlib import Path
import torch

def get_tree_config(config_path=None):
    """Get configuration for tree matching"""
    # Start with GMN base config
    config = get_default_config()
    
    # Add tree-specific defaults
    tree_config = {
        'model': {
            'name': 'tree_matching',
            'node_feature_dim': 768,  # BERT embedding size
            'edge_feature_dim': 64,   # Dependency feature size
            'node_hidden_dim': 256,
            'edge_hidden_dim': 128,
            'n_prop_layers': 5,
            'dropout': 0.1,
        },
        'data': {
            'spacy_variant': 'trf',
            'loading_pattern': 'sequential',
            'batch_size': 32,
            'use_worker_sharding': False,
            'max_partitions_in_memory': 2
        },
        'train': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'n_epochs': 100,
            'patience': 10,
            'warmup_steps': 1000
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'wandb': {
            'project': 'tree-matching',
            'tags': ['linguistic-trees'],
            'log_interval': 100
        }
    }
    config.update(tree_config)
    
    # Override with user config if provided
    if config_path:
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)
    
    return config
