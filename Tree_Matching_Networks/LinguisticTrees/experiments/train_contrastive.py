# experiments/train_contrastive.py
import torch.multiprocessing as mp
import wandb
import torch
from pathlib import Path
import logging
import argparse
from datetime import datetime
import sys
try:
    from ..configs.default_tree_config import get_tree_config
    from ..configs.tree_data_config import TreeDataConfig
    from ..data import GroupedTreeDataset, DynamicCalculatedContrastiveDataset, get_dynamic_calculated_dataloader
    from ..models.tree_matching import TreeMatchingNet
    from ..models.tree_embedding import TreeEmbeddingNet
    from ..training.experiment import ExperimentManager
    from ..training.train import train_epoch
    from ..training.validation import validate_epoch
    from ..utils.memory_utils import MemoryMonitor
except:
    from Tree_Matching_Networks.LinguisticTrees.configs.default_tree_config import get_tree_config
    from Tree_Matching_Networks.LinguisticTrees.configs.tree_data_config import TreeDataConfig
    from Tree_Matching_Networks.LinguisticTrees.data import GroupedTreeDataset, DynamicCalculatedContrastiveDataset, get_dynamic_calculated_dataloader
    from Tree_Matching_Networks.LinguisticTrees.models.tree_matching import TreeMatchingNet
    from Tree_Matching_Networks.LinguisticTrees.models.tree_embedding import TreeEmbeddingNet
    from Tree_Matching_Networks.LinguisticTrees.training.experiment import ExperimentManager
    from Tree_Matching_Networks.LinguisticTrees.training.train import train_epoch
    from Tree_Matching_Networks.LinguisticTrees.training.validation import validate_epoch
    from Tree_Matching_Networks.LinguisticTrees.utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def train_contrastive(args):
    """Full training loop for contrastive learning"""
    # Initialize experiment
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        logger.info(f"override config passed is: {args.config}")
        if args.config:
            config = get_tree_config(
                task_type='info_nce',
                base_config_path=args.config,
                override_path=args.override
            ) 
        else:
            config = None

        checkpoint, experiment, config = ExperimentManager.load_checkpoint(args.resume, config)
        start_epoch = checkpoint['epoch'] + 1
    else:
        # Load fresh config
        config = get_tree_config(
            task_type='info_nce',  # New task type
            base_config_path=args.config if args.config else '/home/jlunder/research/Tree-Matching-Networks/Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/contrastive_config.yaml',
            override_path=args.override
        )
        experiment = ExperimentManager('contrastive', config)
        start_epoch = 0

    # Data config
    data_config = TreeDataConfig(
        dataset_specs=config.get('data', {}).get('dataset_specs', 
                                               [config.get('data', {}).get('dataset_type', 'wikiqs')]),
        task_type='info_nce',
        use_sharded_train=True,
        use_sharded_validate=True,
        allow_cross_dataset_negatives=config.get('data', {}).get('allow_cross_dataset_negatives', True)
    )
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        name=f"contrastive_{experiment.timestamp}",
        config=config,
        tags=['contrastive', *config['wandb'].get('tags', [])]
    )
    
    logger.info("Creating datasets...")
    # Adjust paths for your environment
    # train_dataset = GroupedTreeDataset(
    #     data_path=data_config.train_path / "shard_000000.json",  # Adjust path as needed
    #     config=config
    # )
    # 
    # val_dataset = GroupedTreeDataset(
    #     data_path=data_config.dev_path / "shard_000002.json",  # Adjust path as needed
    #     config=config
    # )
    train_dataset = DynamicCalculatedContrastiveDataset(
        data_dir=[str(path) for path in data_config.train_paths],
        config=config,
        batch_size=config['data']['batch_size'],  # if a matching model, batch size is defined in terms of pairs. if an embedding model, batch size determined in terms of embeddings
        anchors_per_group=config['data'].get('anchors_per_group', 1),
        pos_pairs_per_anchor=config['data'].get('pos_pairs_per_anchor', 1),
        # neg_pairs_per_anchor=config['data'].get('neg_pairs_per_anchor', 10),
        # min_groups_per_batch=config['data'].get('min_groups_per_batch', 8),
        shuffle_files=True,
        prefetch_factor=config['data'].get('prefetch_factor', 2),
        max_active_files=4,
        allow_cross_dataset_negatives=data_config.allow_cross_dataset_negatives,
        recycle_leftovers=True,
        model_type=config['model'].get('model_type', 'matching'),
        strict_matching=config['data'].get('strict_matching', False)
    )
    
    val_dataset = DynamicCalculatedContrastiveDataset(
        data_dir=[str(path) for path in data_config.dev_paths],
        config=config,
        batch_size=config['data']['batch_size'],  # if a matching model, batch size is defined in terms of pairs. if an embedding model, batch size determined in terms of embeddings
        anchors_per_group=config['data'].get('anchors_per_group', 1),
        pos_pairs_per_anchor=config['data'].get('pos_pairs_per_anchor', 1),
        # neg_pairs_per_anchor=config['data'].get('neg_pairs_per_anchor', 10),
        # min_groups_per_batch=config['data'].get('min_groups_per_batch', 8),
        shuffle_files=True,
        prefetch_factor=config['data'].get('prefetch_factor', 2),
        max_active_files=4,
        allow_cross_dataset_negatives=data_config.allow_cross_dataset_negatives,
        recycle_leftovers=True,
        model_type=config['model'].get('model_type', 'matching'),
        strict_matching=config['data'].get('strict_matching', False)
    )
    
    # Initialize model and optimizer
    logger.info("Initializing model...")
    model_type = config['model'].get('model_type', 'matching')
    if model_type == 'embedding':
        model = TreeEmbeddingNet(config).to(config['device'])
    else:
        model = TreeMatchingNet(config).to(config['device'])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )

    # Load checkpoint if resuming
    if args.resume:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Resumed from epoch {start_epoch}")
        
        # Log best metrics from checkpoint
        if 'best_metrics' in checkpoint:
            for k, v in checkpoint['best_metrics'].items():
                wandb.run.summary[f"best_{k}"] = v

    # Training loop
    best_val_loss = float('inf') if not args.resume else checkpoint.get('best_val_loss', float('inf'))
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['train']['n_epochs']):
        # Train
        train_metrics = train_epoch(model, train_dataset, optimizer, config, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_dataset, config, epoch)
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)
        
        # Save checkpoint
        experiment.save_checkpoint(
            model, optimizer, epoch, metrics
        )
        
        # Early stopping based on validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            experiment.save_best_model(
                model, optimizer, epoch,
                {**metrics, 'best_val_loss': best_val_loss}
            )
            patience_counter = 0
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['train']['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                      help='Base config path')
    parser.add_argument('--override', type=str,
                      help='Override config path')
    parser.add_argument('--resume', type=str,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        train_contrastive(args)
    except Exception as e:
        logger.exception("Training failed with error:")
        raise
