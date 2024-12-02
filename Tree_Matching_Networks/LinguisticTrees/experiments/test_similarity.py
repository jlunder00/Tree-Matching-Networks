import torch
import argparse
import logging
from pathlib import Path
import wandb
from ..configs.default_tree_config import get_tree_config
from ..configs.tree_data_config import TreeDataConfig
from ..data.partition_datasets import MultiPartitionTreeDataset
from ..models.tree_matching import TreeMatchingNet, TreeMatchingNetlg
from ..training.metrics import TreeMatchingMetrics
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def test_similarity_model(args):
    """Test similarity model on test dataset"""
    # Load checkpoint and config
    checkpoint = torch.load(args.checkpoint)
    config = checkpoint['config']
    
    # Initialize wandb for logging
    wandb.init(
        project=config['wandb']['project'],
        name=f"similarity_test_{Path(args.checkpoint).stem}",
        config=config,
        tags=['test', 'similarity', *config['wandb'].get('tags', [])]
    )
    
    # Load test dataset
    data_config = TreeDataConfig(
        dataset_type=args.dataset,
        task_type='similarity',
        use_sharded_train=False,
        use_sharded_validate=False
    )
    
    test_dataset = MultiPartitionTreeDataset(
        data_config.test_path,
        config=config,
        num_workers=config['data']['num_workers'],
        prefetch_factor=config['data']['prefetch_factor']
    )
    
    # Initialize model
    model = TreeMatchingNetlg(config).to(config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test loop
    all_similarities = []
    all_labels = []
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, (graphs, labels) in enumerate(test_dataset.pairs(config['data']['batch_size'])):
            # Move to device
            graphs = graphs.to(config['device'])
            labels = labels.to(config['device'])
            
            # Forward pass
            graph_vectors = model(
                graphs.node_features,
                graphs.edge_features,
                graphs.from_idx,
                graphs.to_idx,
                graphs.graph_idx,
                graphs.n_graphs
            )
            
            # Split vectors for similarity computation
            x, y = graph_vectors[::2], graph_vectors[1::2]
            similarities = torch.nn.functional.cosine_similarity(x, y)
            
            all_similarities.append(similarities.cpu())
            all_labels.append(labels.cpu())
            
            # Log progress
            if batch_idx % config['wandb']['log_interval'] == 0:
                logger.info(f"Processed {batch_idx} batches")
                MemoryMonitor.log_memory()
    
    # Compute final metrics
    similarities = torch.cat(all_similarities)
    labels = torch.cat(all_labels)
    
    metrics = TreeMatchingMetrics.compute_similarity_metrics(similarities, labels)
    
    # Log results
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
        wandb.run.summary[f"test_{name}"] = value
    
    wandb.finish()
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='para50m',
                       choices=['semeval', 'para50m'],
                       help='Test dataset to use')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
                       
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_similarity_model(args)
