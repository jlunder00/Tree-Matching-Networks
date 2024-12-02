# experiments/similarity.py
import torch.nn.functional as F
from ..data.data_utils import GraphData
import wandb
import torch
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from ..configs.default_tree_config import get_tree_config
from ..configs.tree_data_config import TreeDataConfig
from ..data.partition_datasets import MultiPartitionTreeDataset
from ..models.tree_matching import TreeMatchingNet, TreeMatchingNetlg, TreeMatchingNetSimilarity
from ..training.experiment import ExperimentManager
from ..training.train import train_epoch
from ..training.validation import validate_epoch
from ..utils.memory_utils import MemoryMonitor
import json

logger = logging.getLogger(__name__)

def similarity(args):
    # Initialize experiment/load checkpoint
    checkpoint = {}
    if args.resume or args.test:
        checkpoint_path = args.resume if args.resume else args.test_checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint, experiment = ExperimentManager.load_checkpoint(checkpoint_path)
        config = checkpoint['config']
        start_epoch = checkpoint['epoch'] + 1 if args.resume else 0
    else:
        config = get_tree_config(
            task_type='similarity',
            base_config_path=args.config,
            override_path=args.override
        )
        experiment = ExperimentManager('similarity', config)
        start_epoch = 0
    
    # Data config
    data_config = TreeDataConfig(
        dataset_type=args.dataset,
        task_type='similarity',
        # use_sharded_train=True,
        use_sharded_train=False,
        use_sharded_validate=False
    )
    
    # Initialize wandb
    run_name = f"similarity_{experiment.timestamp}"
    if args.test:
        run_name = f"test_{run_name}"
    elif args.resume:
        run_name = f"resume_{run_name}"
    wandb.init(
        project=config['wandb']['project'],
        name=run_name,
        config=config,
        tags=['test' if args.test else 'train', 'similarity', args.dataset]
    )
    
    logger.info("Creating datasets...")
    # Create datasets
    if args.test:
        dataset = MultiPartitionTreeDataset(
            data_config.test_path,
            config=config,
            num_workers=config['data']['num_workers'],
            prefetch_factor=config['data']['prefetch_factor'],
            max_active_files=4
        )
    else:
        train_dataset = MultiPartitionTreeDataset(
            data_config.train_path,
            config=config, 
            num_workers=config['data']['num_workers'],
            prefetch_factor=config['data']['prefetch_factor'],
            max_active_files=4
        )
        
        val_dataset = MultiPartitionTreeDataset(
            data_config.dev_path,
            config=config,
            num_workers=max(1, config['data']['num_workers'] // 2),
            prefetch_factor=config['data']['prefetch_factor'],
            max_active_files=4
        )
    
    model = TreeMatchingNetlg(config).to(config['device'])
    if args.test:
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loop(model, dataset, config, experiment)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config['train']['learning_rate']),
            weight_decay=float(config['train']['weight_decay'])
        )
        # Initialize schedulers
        schedulers = {
            'cosine': CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config['train']['scheduler']['cosine']['T_0'],
                T_mult=config['train']['scheduler']['cosine']['T_mult'],
                eta_min=config['train']['scheduler']['cosine']['eta_min']
            ),
            'plateau': ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config['train']['scheduler']['plateau']['factor'],
                patience=config['train']['scheduler']['plateau']['patience'],
                min_lr=config['train']['scheduler']['plateau']['min_lr']
            )
        }
        if args.resume:
            model.load_state_dict(checkpoint['model_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_states' in checkpoint:
                for name, state in checkpoint['scheduler_states'].items():
                    schedulers[name].load_state_dict(state)
            logger.info(f"Resumed from epoch {start_epoch}")
        
            # Log best metrics from checkpoint
            if 'best_metrics' in checkpoint:
                for k, v in checkpoint['best_metrics'].items():
                    wandb.run.summary[f"best_{k}"] = v
        best_val_loss = checkpoint.get('bess_val_loss', -1.0) if args.resume else -1.0


        train_loop(model, train_dataset, val_dataset, config, experiment, optimizer, schedulers, best_val_loss, start_epoch)



# def test_loop(model, dataset, config, experiment):
#     """Test loop for similarity model"""
#     model.eval()
#     metrics = validate_epoch(model, dataset, config, -1)  # -1 for test epoch
#     
#     # Log results
#     logger.info("Test Results:")
#     for name, value in metrics.items():
#         logger.info(f"{name}: {value:.4f}")
#         wandb.run.summary[f"test_{name}"] = value
#     
#     # Save results
#     results_path = experiment.experiment_dir / "test_results.json"
#     with open(results_path, 'w') as f:
#         json.dump(metrics, f, indent=2)

def test_loop(model, dataset, config, experiment):
    """Test loop for similarity model with pure evaluation metrics"""
    model.eval()
    all_similarities = []
    all_labels = []
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, (graphs, labels) in enumerate(dataset.pairs(config['data']['batch_size'])):
            # Move to device
            device = config['device']
            graphs = GraphData(
                node_features=graphs.node_features.to(device, non_blocking=True),
                edge_features=graphs.edge_features.to(device, non_blocking=True),
                from_idx=graphs.from_idx.to(device, non_blocking=True),
                to_idx=graphs.to_idx.to(device, non_blocking=True),
                graph_idx=graphs.graph_idx.to(device, non_blocking=True),
                n_graphs=graphs.n_graphs
            )
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            graph_vectors = model(
                graphs.node_features,
                graphs.edge_features,
                graphs.from_idx,
                graphs.to_idx,
                graphs.graph_idx,
                graphs.n_graphs
            )
            
            # Split vectors and compute cosine similarity
            x, y = graph_vectors[::2], graph_vectors[1::2]
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            similarities = torch.sum(x_norm * y_norm, dim=1)
            
            all_similarities.append(similarities.cpu())
            all_labels.append(labels.cpu())
            
            # Log progress periodically
            if batch_idx % config['wandb']['log_interval'] == 0:
                logger.info(f"Processed {batch_idx} batches")
                MemoryMonitor.log_memory()
    
    # Concatenate all results
    similarities = torch.cat(all_similarities).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # Compute metrics
    from scipy.stats import pearsonr, spearmanr
    pearson_corr, _ = pearsonr(similarities, labels)
    spearman_corr, _ = spearmanr(similarities, labels)
    mse = np.mean((similarities - labels) ** 2)
    mae = np.mean(np.abs(similarities - labels))
    
    # For binned accuracy metrics (optional)
    binned_acc = compute_binned_accuracy(similarities, labels)
    
    metrics = {
        'pearson_correlation': float(pearson_corr),
        'spearman_correlation': float(spearman_corr),
        'mse': float(mse),
        'mae': float(mae),
        'binned_accuracy': binned_acc
    }
    
    # Log results
    logger.info("\nTest Results:")
    for name, value in metrics.items():
        if isinstance(value, dict):
            logger.info(f"\n{name}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
                wandb.run.summary[f"test_{name}/{k}"] = v
        else:
            logger.info(f"{name}: {value:.4f}")
            wandb.run.summary[f"test_{name}"] = value
    
    # Save detailed results
    results = {
        'metrics': metrics,
        'raw_predictions': similarities.tolist(),
        'raw_labels': labels.tolist()
    }
    
    results_path = experiment.experiment_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    wandb.finish()
    return metrics

def compute_binned_accuracy(similarities, labels, n_bins=10):
    """Compute accuracy metrics for binned similarity scores"""
    # Create bins for similarity scores
    bins = np.linspace(-1, 1, n_bins + 1)
    digitized = np.digitize(labels, bins) - 1  # -1 to get bin indices starting at 0
    
    accuracies = {}
    for i in range(n_bins):
        bin_mask = digitized == i
        if not np.any(bin_mask):
            continue
            
        bin_similarities = similarities[bin_mask]
        bin_labels = labels[bin_mask]
        
        # Compute accuracy for this bin
        bin_mse = np.mean((bin_similarities - bin_labels) ** 2)
        bin_mae = np.mean(np.abs(bin_similarities - bin_labels))
        
        bin_range = f"{bins[i]:.2f} to {bins[i+1]:.2f}"
        accuracies[bin_range] = {
            'mse': float(bin_mse),
            'mae': float(bin_mae),
            'count': int(np.sum(bin_mask))
        }
    
    return accuracies

def train_loop(model, train_dataset, val_dataset, config, experiment, optimizer, schedulers={}, best_val_loss=0.0, start_epoch=0):
    """Full training loop for similarity task"""

    # Training loop
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['train']['n_epochs']):
        # Train
        train_metrics = train_epoch(model, train_dataset, optimizer, config, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_dataset, config, epoch)
        
        # Update schedulers
        schedulers['cosine'].step()
        schedulers['plateau'].step(val_metrics['correlation'])
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)
        
        # Save checkpoint
        scheduler_states = {
            name: scheduler.state_dict() 
            for name, scheduler in schedulers.items()
        }
        experiment.save_checkpoint(
            model, optimizer, epoch,
            {**metrics, 'scheduler_states': scheduler_states}
        )
        
        # Early stopping based on correlation
        if val_metrics['loss'] > best_val_loss:
            bess_val_loss = val_metrics['loss']
            experiment.save_best_model(
                model, optimizer, epoch,
                {**metrics, 'bess_val_loss': bess_val_loss}
            )
            patience_counter = 0
            logger.info(f"New best validation loss: {bess_val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['train']['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
    logger.info(f"Training completed! Best validation loss: {bess_val_loss:.4f}")
    wandb.finish()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='semeval',
#                       choices=['semeval', 'para50m'],
#                       help='Dataset to use for similarity task')
#     parser.add_argument('--config', type=str,
#                       help='Base config path')
#     parser.add_argument('--override', type=str,
#                       help='Override config path')
#     parser.add_argument('--resume', type=str,
#                       help='Path to checkpoint to resume from')
#     parser.add_argument('--debug', action='store_true',
#                       help='Enable debug mode')
#     
#     args = parser.parse_args()
#     
#     # Setup logging
#     logging.basicConfig(
#         level=logging.DEBUG if args.debug else logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#     
#     try:
#         train_similarity(args)
#     except Exception as e:
#         logger.exception("Training failed with error:")
#         raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='semeval',
                      choices=['semeval', 'para50m'],
                      help='Dataset to use for similarity task')
    parser.add_argument('--config', type=str,
                      help='Base config path')
    parser.add_argument('--override', type=str,
                      help='Override config path')
    parser.add_argument('--resume', type=str,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                      help='Run in test mode')
    parser.add_argument('--test_checkpoint', type=str,
                      help='Model checkpoint for testing')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
                      
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        similarity(args)
    except Exception as e:
        logger.exception("Training failed with error:")
        raise

