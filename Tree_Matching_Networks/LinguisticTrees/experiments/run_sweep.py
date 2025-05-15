# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#designed to work with non text level groups. needs updates to work with newest data format. used to do grid search on hyperparams during initial pretraining

import wandb
import torch
from pathlib import Path
import logging
import argparse
import os
import yaml
import sys
import json

try:
    from ..configs.default_tree_config import get_tree_config
    from ..configs.tree_data_config import TreeDataConfig
    from ..configs.sweep_config import get_sweep_config
    from ..models.tree_matching import TreeMatchingNet
    from ..models.tree_embedding import TreeEmbeddingNet
    from ..training.experiment import ExperimentManager
    from .train_contrastive import train_contrastive
except:
    from Tree_Matching_Networks.LinguisticTrees.configs.default_tree_config import get_tree_config
    from Tree_Matching_Networks.LinguisticTrees.configs.tree_data_config import TreeDataConfig
    from Tree_Matching_Networks.LinguisticTrees.configs.sweep_config import get_sweep_config
    from Tree_Matching_Networks.LinguisticTrees.models.tree_matching import TreeMatchingNet
    from Tree_Matching_Networks.LinguisticTrees.models.tree_embedding import TreeEmbeddingNet
    from Tree_Matching_Networks.LinguisticTrees.training.experiment import ExperimentManager
    from Tree_Matching_Networks.LinguisticTrees.experiments.train_contrastive import train_contrastive

logger = logging.getLogger(__name__)

def update_nested_dict(d, key_path, value):
    """Update a nested dictionary using a dot-separated key path"""
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return d

def apply_sweep_config(config, sweep_config):
    """Apply sweep parameters to the config dict"""
    for key, value in sweep_config.items():
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            # Parse list parameters
            try:
                parsed_value = json.loads(value.replace("'", '"'))
                update_nested_dict(config, key, parsed_value)
            except:
                logger.warning(f"Failed to parse list parameter {key}: {value}")
        else:
            update_nested_dict(config, key, value)
    return config

def train_sweep_agent():
    """Training function for the sweep agent"""
    # Initialize wandb run with sweep configuration
    run = wandb.init()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get base config path from sweep
    base_config_path = os.environ.get('BASE_CONFIG_PATH',
                                      'Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/contrastive_config.yaml')
    
    # Load base config
    config = get_tree_config(
        task_type='info_nce',
        base_config_path=base_config_path
    )
    
    # Apply sweep configuration
    config = apply_sweep_config(config, wandb.config)
    
    # Define a simple arguments object to pass to train_contrastive
    class Args:
        def __init__(self):
            self.resume = None
            self.config = None
            self.override = None
            self.debug = False
    
    args = Args()
    
    # Store the modified config for this run
    temp_config_path = f"sweep_config_{wandb.run.id}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    args.config = temp_config_path
    
    try:
        # Run training with the sweep configuration
        train_contrastive(args)
    except Exception as e:
        logger.exception(f"Sweep run failed: {e}")
        wandb.run.summary['status'] = 'failed'
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def run_sweep(args):
    """Initialize and start a wandb sweep"""
    # Get sweep configuration
    sweep_config = get_sweep_config()
    
    # Add environment variable for base config path
    os.environ['BASE_CONFIG_PATH'] = args.config or '/home/jlunder/research/Tree-Matching-Networks/Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/contrastive_config.yaml'
    
    # Initialize the sweep
    if args.create_only:
        # Just create the sweep without starting agents
        sweep_id = wandb.sweep(
            sweep_config, 
            project=args.project or "tree-embedding"
        )
        print(f"Created sweep with ID: {sweep_id}")
        print(f"To run agents, use: python -m Tree_Matching_Networks.LinguisticTrees.experiments.run_sweep --sweep_id {sweep_id} --agent")
    elif args.sweep_id:
        # Connect to existing sweep
        sweep_id = args.sweep_id
        # Start the sweep agent
        wandb.agent(sweep_id, function=train_sweep_agent, count=args.count)
    else:
        # Create and start the sweep in a single call
        sweep_id = wandb.sweep(
            sweep_config, 
            project=args.project or "tree-embedding"
        )
        print(f"Created sweep with ID: {sweep_id}")
        # Start the sweep agent
        wandb.agent(sweep_id, function=train_sweep_agent, count=args.count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                      help='Base config path')
    parser.add_argument('--project', type=str, default="tree-embedding",
                      help='WandB project name')
    parser.add_argument('--count', type=int, default=20,
                      help='Number of runs in the sweep')
    parser.add_argument('--sweep_id', type=str,
                      help='ID of existing sweep to connect to')
    parser.add_argument('--create_only', action='store_true',
                      help='Only create the sweep without starting agents')
    parser.add_argument('--agent', action='store_true',
                      help='Run as an agent for an existing sweep')
    
    args = parser.parse_args()
    
    run_sweep(args)
