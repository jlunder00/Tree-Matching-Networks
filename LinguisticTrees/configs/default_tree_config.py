#configs/default_tree_config.py
from GMN.configure import get_default_config

def get_tree_config():
    """Base configuration for tree models"""
    config = get_default_config()
    
    # Add tree-specific configs
    config.update({
        'model': {
            'node_feature_dim': 768,  # BERT embedding dimension
            'edge_feature_dim': 64,   # Dependency feature dimension
            'node_hidden_dim': 256,
            'edge_hidden_dim': 128,
            'n_prop_layers': 5,
            'dropout': 0.1
        },
        'data': {
            'max_tree_size': 100,
            'batch_size': 32,
            'num_workers': 4
        },
        'training': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'n_epochs': 100,
            'patience': 10
        }
    })
    return config

