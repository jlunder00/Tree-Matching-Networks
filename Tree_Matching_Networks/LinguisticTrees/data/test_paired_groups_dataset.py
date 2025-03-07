# test_paired_groups_dataset.py

import logging
import os
import sys
import json
from pathlib import Path
import torch

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.paired_groups_dataset import (
    create_paired_groups_dataset,
    get_paired_groups_dataloader
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_non_strict_matching_dataset():
    """Test loading the non-strict matching dataset with direct labeled pairs."""
    
    # Sample configuration
    config = {
        'data': {
            'batch_size': 16,
            'num_workers': 0,  # Using 0 for easier debugging
            'max_batches_per_epoch': 2,  # Just test a few batches
        },
        'model': {
            'model_type': 'matching',
        }
    }
    
    # Path to your data directory containing paired groups
    # Adjust this to your actual data path
    data_dir = "/home/jlunder/research/data/snli_1.0/dev"
    
    logger.info(f"Creating dataset from {data_dir}")
    
    # Create the dataset
    dataset = create_paired_groups_dataset(
        data_dir=data_dir,
        config=config,
        model_type='matching',
        strict_matching=False,
        contrastive_mode=False,  # Direct labeled mode
        batch_size=config['data']['batch_size'],
        shuffle_files=True,
        prefetch_factor=2,
        max_active_files=2,
        min_trees_per_group=1,
        # avg_trees_per_subgroup=5
    )
    
    logger.info(f"Created dataset: {type(dataset).__name__}")
    
    # Create dataloader
    dataloader = get_paired_groups_dataloader(
        dataset,
        num_workers=config['data']['num_workers'],
        pin_memory=False  # Set to False for easier debugging
    )
    
    logger.info("Starting iteration through dataloader")
    
    # Get the first batch
    for i, (graphs, batch_info) in enumerate(dataloader):
        logger.info(f"Batch {i+1}:")
        logger.info(f"  Number of groups: {len(batch_info.group_indices)}")
        logger.info(f"  Number of trees: {graphs.n_graphs}")
        
        # Print group information
        for g_idx in range(len(batch_info.group_indices)):
            group_id = batch_info.group_ids[g_idx]
            label = batch_info.group_labels[g_idx]
            trees_a = batch_info.trees_a_indices[g_idx]
            trees_b = batch_info.trees_b_indices[g_idx]
            
            logger.info(f"  Group {g_idx} (ID: {group_id}):")
            logger.info(f"    Label: {label}")
            logger.info(f"    Trees A: {len(trees_a)} indices")
            logger.info(f"    Trees B: {len(trees_b)} indices")
        
        # Print pair information
        logger.info(f"  Total pairs: {len(batch_info.pair_indices)}")
        pos_pairs = [p for p in batch_info.pair_indices if p[2] > 0.5]
        neg_pairs = [p for p in batch_info.pair_indices if p[2] <= 0.5]
        logger.info(f"    Positive pairs: {len(pos_pairs)}")
        logger.info(f"    Negative pairs: {len(neg_pairs)}")
        
        # Print first few pairs
        if batch_info.pair_indices:
            logger.info("  Sample pairs (a_idx, b_idx, label):")
            for p in batch_info.pair_indices[:5]:
                logger.info(f"    {p}")
        
        # Print graph properties
        logger.info(f"  Graph properties:")
        logger.info(f"    Nodes: {len(graphs.node_features)}")
        logger.info(f"    Edges: {len(graphs.from_idx)}")
        
        # Check if all trees are covered by at least one pair
        covered_a = set(a_idx for a_idx, _, _ in batch_info.pair_indices)
        covered_b = set(b_idx for _, b_idx, _ in batch_info.pair_indices)
        
        all_a = set()
        all_b = set()
        for g_idx in range(len(batch_info.group_indices)):
            all_a.update(batch_info.trees_a_indices[g_idx])
            all_b.update(batch_info.trees_b_indices[g_idx])
        
        uncovered_a = all_a - covered_a
        uncovered_b = all_b - covered_b
        
        if uncovered_a:
            logger.warning(f"  Uncovered A trees: {uncovered_a}")
        if uncovered_b:
            logger.warning(f"  Uncovered B trees: {uncovered_b}")
            
        logger.info("-" * 80)
        
        # Only test a few batches
        if i >= config['data']['max_batches_per_epoch'] - 1:
            break
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    test_non_strict_matching_dataset()
