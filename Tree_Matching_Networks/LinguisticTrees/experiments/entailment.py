# experiments/entailment.py
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
from ..models.tree_matching import TreeMatchingNetEntailment, TreeEntailmentNet
from ..training.experiment import ExperimentManager
from ..training.train import train_epoch
from ..training.validation import validate_epoch
from ..utils.memory_utils import MemoryMonitor
import json

logger = logging.getLogger(__name__)

def entailment(args):
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
    run_name = f"entailment_{experiment.timestamp}"
    if args.test:
        run_name = f"test_{run_name}"
    elif args.resume:
        run_name = f"resume_{run_name}"
    wandb.init(
        project=config['wandb']['project'],
        name=run_name,
        config=config,
        tags=['test' if args.test else 'train', 'entailment']
    )
    
    logger.info("Creating datasets...")
    # Create datasets
    if args.test:
        dataset = MultiPartitionTreeDataset(
            data_config.test_path,
            config=config,
            num_workers=config['data']['num_workers'],
            prefetch_factor=config['data']['prefetch_factor'],
            max_active_files=16
        )
    else:
        train_dataset = MultiPartitionTreeDataset(
            data_config.train_path,
            config=config,
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
        )
    
    model = TreeEntailmentNet(config).to(config['device'])
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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_states' in checkpoint:
                for name, state in checkpoint['scheduler_states'].items():
                    schedulers[name].load_state_dict(state)
            logger.info(f"Resumed from epoch {start_epoch}")
            
            if 'best_metrics' in checkpoint:
                for k, v in checkpoint['best_metrics'].items():
                    wandb.run.summary[f"best_{k}"] = v
        
        best_val_acc = checkpoint.get('best_val_acc', 0.0) if args.resume else 0.0
        train_loop(model, train_dataset, val_dataset, config, experiment, 
                  optimizer, schedulers, best_val_acc, start_epoch)

def test_loop(model, dataset, config, experiment):
    """Test loop for entailment model"""
    model.eval()
    metrics = validate_epoch(model, dataset, config, -1)
    
    logger.info("Test Results:")
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
        wandb.run.summary[f"test_{name}"] = value
        
    results_path = experiment.experiment_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    wandb.finish()

def train_loop(model, train_dataset, val_dataset, config, experiment, 
               optimizer, schedulers, best_val_acc=0.0, start_epoch=0):
    """Training loop for entailment model"""
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['train']['n_epochs']):
        train_metrics = train_epoch(model, train_dataset, optimizer, config, epoch)
        val_metrics = validate_epoch(model, val_dataset, config, epoch)
        
        # Update schedulers using accuracy instead of correlation
        schedulers['cosine'].step()
        schedulers['plateau'].step(val_metrics['accuracy'])
        
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
        
        # Early stopping based on accuracy
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
        entailment(args)
    except Exception as e:
        logger.exception("Training failed with error:")
        raise
