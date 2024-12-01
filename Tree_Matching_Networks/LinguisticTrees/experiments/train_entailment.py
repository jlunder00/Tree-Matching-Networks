# experiments/train_entailment.py
import wandb
import torch
from pathlib import Path
import logging
import argparse
from datetime import datetime
from ..configs.default_tree_config import get_tree_config
from ..configs.tree_data_config import TreeDataConfig
from ..data.partition_datasets import MultiPartitionTreeDataset
from ..models.tree_matching import TreeMatchingNet
from ..training.experiment import ExperimentManager
from ..training.train import train_epoch
from ..training.validation import validate_epoch
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def train_entailment(args):
    """Full training loop for entailment task"""
    # Initialize experiment
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint, experiment = ExperimentManager.load_checkpoint(args.resume)
        config = checkpoint['config']
        start_epoch = checkpoint['epoch'] + 1
    else:
        # Load fresh config
        config = get_tree_config(
            task_type='entailment',
            base_config_path=args.config,
            override_path=args.override
        )
        experiment = ExperimentManager('entailment', config)
        start_epoch = 0

    # Data config
    data_config = TreeDataConfig(
        dataset_type='snli',
        task_type='entailment',
        use_sharded_train=True,
        use_sharded_validate=False
    )
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        name=f"entailment_{experiment.timestamp}",
        config=config,
        tags=['entailment', *config['wandb'].get('tags', [])]
    )
    
    logger.info("Creating datasets...")
    train_dataset = MultiPartitionTreeDataset(
        data_config.train_path,
        config=config,
        # num_workers=0
        num_workers=config['data']['num_workers'],
        prefetch_factor=config['data']['prefetch_factor'],
        max_active_files=16
    )
    
    val_dataset = MultiPartitionTreeDataset(
        data_config.dev_path,
        config=config,
        num_workers=max(1, config['data']['num_workers'] // 2),
        prefetch_factor=config['data']['prefetch_factor'],
        max_active_files=16
        # num_workers=0
    )
    
    # Initialize model and optimizer
    logger.info("Initializing model...")
    model = TreeMatchingNet(config).to(config['device'])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )

    # Load checkpoint state if resuming
    if args.resume:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Resumed from epoch {start_epoch}")
        
        # Log best metrics from checkpoint
        if 'best_metrics' in checkpoint:
            for k, v in checkpoint['best_metrics'].items():
                wandb.run.summary[f"best_{k}"] = v

    # Training loop
    best_val_acc = checkpoint.get('best_val_acc', 0.0) if args.resume else 0.0
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
        
        # Early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            experiment.save_best_model(
                model, optimizer, epoch,
                {**metrics, 'best_val_acc': best_val_acc}
            )
            patience_counter = 0
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['train']['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
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
        train_entailment(args)
    except Exception as e:
        logger.exception("Training failed with error:")
        raise

