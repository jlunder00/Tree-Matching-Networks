# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

#configs/default_tree_config.py 
from ...GMN.configure import get_default_config
import yaml
from pathlib import Path
import torch

def get_tree_config(task_type='entailment', base_config=None, override_config=None):
    """Get configuration for tree matching with better organization"""
    # Start with GMN base config
    config = get_default_config()
    
    # Load base task config
    if base_config is None:
        # Default to configs within package
        base_path = Path(__file__).parent / "experiment_configs"
        base_config_path = base_path / f"{task_type}_config.yaml"
        
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
    config.update(base_config)
    
    # Load override config if provided
    if override_config is not None:
        config.update(override_config)
            
    # Add runtime configs
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config

