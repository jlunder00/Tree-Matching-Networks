# experiments/train_similarity.py
import wandb
import torch
from pathlib import Path
import logging
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from ..configs.default_tree_config import get_tree_config
from ..configs.tree_data_config import TreeDataConfig
from ..data.partition_datasets import MultiPartitionTreeDataset
from ..models.tree_matching import TreeMatchingNet
from ..training.experiment import ExperimentManager
from ..training.train import train_epoch
from ..training.validation import validate_epoch
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def train_similarity(args):
    """Full training loop for similarity task"""
    # Initialize experiment
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint, experiment = ExperimentManager.load_checkpoint(args.resume)
        config = checkpoint['config']
        start_epoch = checkpoint['epoch'] + 1
    else:
        # Load fresh config
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
        task_type='similarity'
    )
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        name=f"similarity_{experiment.timestamp}",
        config=config,
        tags=['similarity', args.dataset, *config['wandb'].get('tags', [])]
    )
    
    logger.info("Creating datasets...")
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
    
    # Initialize model and optimizer
    logger.info("Initializing model...")
    model = TreeMatchingNet(config).to(config['device'])
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

    # Load checkpoint state if resuming
    if args.resume:
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

    # Training loop
    best_val_corr = checkpoint.get('best_val_corr', -1.0) if args.resume else -1.0
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
        if val_metrics['correlation'] > best_val_corr:
            best_val_corr = val_metrics['correlation']
            experiment.save_best_model(
                model, optimizer, epoch,
                {**metrics, 'best_val_corr': best_val_corr}
            )
            patience_counter = 0
            logger.info(f"New best validation correlation: {best_val_corr:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['train']['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
    logger.info(f"Training completed! Best validation correlation: {best_val_corr:.4f}")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='semeval',
                      choices=['semeval'],
                      help='Dataset to use for similarity task')
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
        train_similarity(args)
    except Exception as e:
        logger.exception("Training failed with error:")
        raise


# #experiments/train_similarity.py
# import wandb
# import json
# import torch
# from pathlib import Path
# from datetime import datetime
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
# import logging
# from tqdm import tqdm
# from ..configs.default_tree_config import get_tree_config
# from ..configs.tree_data_config import TreeDataConfig
# from ..data.partition_datasets import MultiPartitionTreeDataset
# from ..models.tree_matching import TreeMatchingNet
# from ..training.train import train_epoch
# from ..training.validation import validate_epoch
# from ..training.metrics import TreeMatchingMetrics
# from ..utils.memory_utils import MemoryMonitor

# logger = logging.getLogger(__name__)

# def train_similarity(dataset_type='semeval'):
#     """Full training loop for similarity task"""
#     # Load config
#     config = get_tree_config(dataset_type=dataset_type, task_type='similarity')
#     data_config = TreeDataConfig(
#         dataset_type=dataset_type,
#         task_type='similarity'
#     )
#     
#     # Setup logging
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_name = f"similarity_{dataset_type}_{timestamp}"
#     
#     # Initialize wandb
#     wandb.init(
#         project=config['wandb']['project'],
#         config=config,
#         name=run_name,
#         tags=[*config['wandb']['tags'], 'similarity']
#     )
#     
#     logger.info("Creating datasets...")
#     train_dataset = MultiPartitionTreeDataset(
#         data_config.train_path,
#         config=config,
#         num_workers=8
#     )
#     
#     val_dataset = MultiPartitionTreeDataset(
#         data_config.dev_path,
#         config=config,
#         num_workers=4
#     )
#     
#     logger.info("Initializing model...")
#     model = TreeMatchingNet(config).to(config['device'])
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=config['train'].get('learning_rate', 5e-4),
#         weight_decay=config['train'].get('weight_decay', 1e-5)
#     )

#     # Add two schedulers:
#     # 1. Cosine annealing for cyclic learning rate
#     cosine_scheduler = CosineAnnealingWarmRestarts(
#         optimizer,
#         T_0=5,  # Restart every 5 epochs
#         T_mult=2,  # Double period after each restart
#         eta_min=1e-6  # Min learning rate
#     )
#     
#     # 2. ReduceLROnPlateau to reduce LR when stuck
#     plateau_scheduler = ReduceLROnPlateau(
#         optimizer,
#         mode='max',  # Looking at correlation 
#         factor=0.5,  # Reduce LR by half
#         patience=3,  # Wait 3 epochs before reducing
#         min_lr=1e-6
#     )
#     
#     # Create checkpoint directory
#     checkpoint_dir = Path(config['train'].get('checkpoint_dir', 'checkpoints'))
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     
#     # Training loop
#     best_val_corr = -1  # Using correlation instead of accuracy for similarity
#     patience_counter = 0
#     
#     logger.info("Starting training...")
#     epoch_pbar = tqdm(range(config['train']['n_epochs']), desc="Training", position=0)
#     
#     for epoch in epoch_pbar:
#         # Monitor memory before epoch
#         MemoryMonitor.log_memory(step=epoch, prefix=f'Epoch {epoch} start: ')
#         
#         # Train
#         logger.info(f"training epoch: {epoch}")
#         train_metrics = train_epoch(model, train_dataset, optimizer, config, epoch)
#         
#         # Validate
#         logger.info(f"validating epoch: {epoch}")
#         val_metrics = validate_epoch(model, val_dataset, config, epoch)
#         
#         logger.info(f"cosine schedule stepping epoch: {epoch}")
#         cosine_scheduler.step()
#         logger.info(f"plateau scheduler stepping epoch: {epoch}")
#         plateau_scheduler.step(val_metrics['correlation'])
#         # Update progress bar with correlation instead of accuracy
#         logger.info("setting postfix")
#         epoch_pbar.set_postfix({
#             'train_loss': f"{train_metrics['loss']:.4f}",
#             'train_corr': f"{train_metrics['correlation']:.4f}",
#             'val_loss': f"{val_metrics['loss']:.4f}",
#             'val_corr': f"{val_metrics['correlation']:.4f}"
#         })
#         
#         # Log metrics
#         wandb.log({
#             'epoch': epoch,
#             'train/loss': train_metrics['loss'],
#             'train/correlation': train_metrics['correlation'],
#             'train/mse': train_metrics['mse'],
#             'train/batch_time': train_metrics['batch_time'],
#             # 'train/data_time': train_metrics['data_time'],
#             'val/loss': val_metrics['loss'],
#             'val/correlation': val_metrics['correlation'],
#             'val/mse': val_metrics['mse'],
#             'learning_rate': optimizer.param_groups[0]['lr']
#         })
#         
#         # Save checkpoint
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'train_metrics': train_metrics,
#             'val_metrics': val_metrics,
#             'config': config
#         }
#         
#         checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
#         torch.save(checkpoint, checkpoint_path)
#         
#         # Early stopping based on correlation
#         if val_metrics['correlation'] > best_val_corr:
#             best_val_corr = val_metrics['correlation']
#             best_checkpoint_path = checkpoint_dir / 'best_model.pt'
#             torch.save(checkpoint, best_checkpoint_path)
#             patience_counter = 0
#             logger.info(f"New best validation correlation: {best_val_corr:.4f}")
#         else:
#             patience_counter += 1
#             
#         if patience_counter >= config['train']['patience']:
#             logger.info(f"Early stopping triggered after {epoch + 1} epochs")
#             break
#         
#         # Monitor memory after epoch
#         MemoryMonitor.log_memory(step=epoch, prefix=f'Epoch {epoch} end: ')
#     
#         # Optional cleanup between epochs
#         torch.cuda.empty_cache()
#     
#     logger.info(f"Training completed! Best validation correlation: {best_val_corr:.4f}")
#     
#     # Save final results summary
#     results = {
#         'best_val_correlation': best_val_corr,
#         'epochs_trained': epoch + 1,
#         'early_stopped': patience_counter >= config['train']['patience']
#     }
#     
#     results_file = checkpoint_dir / 'training_results.json'
#     with open(results_file, 'w') as f:
#         json.dump(results, f, indent=2)
#     
#     wandb.save(str(results_file))
#     wandb.finish()

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='Train tree matching model for similarity task')
#     parser.add_argument('--dataset', type=str, default='semeval',
#                       choices=['semeval'],
#                       help='Dataset to use for similarity task')
#     parser.add_argument('--config', type=str,
#                       help='Optional path to config override file')
#     parser.add_argument('--debug', action='store_true',
#                       help='Enable debug mode with limited batches')
#     args = parser.parse_args()
#     
#     # Setup logging
#     logging.basicConfig(
#         level=logging.DEBUG if args.debug else logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#     
#     try:
#         train_similarity(
#             dataset_type=args.dataset
#         )
#     except Exception as e:
#         logger.exception("Training failed with error:")
#         raise
