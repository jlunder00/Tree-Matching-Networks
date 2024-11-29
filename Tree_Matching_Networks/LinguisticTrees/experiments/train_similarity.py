#experiments/train_similarity.py
import wandb
import torch
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from ..configs.default_tree_config import get_tree_config
from ..configs.tree_data_config import TreeDataConfig
from ..data.partition_datasets import MultiPartitionTreeDataset
from ..models.tree_matching import TreeMatchingNet
from ..training.train import train_epoch
from ..training.validation import validate_epoch
from ..training.metrics import TreeMatchingMetrics
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def train_similarity(dataset_type='semeval'):
    """Full training loop for similarity task"""
    # Load config
    config = get_tree_config(dataset_type=dataset_type, task_type='similarity')
    data_config = TreeDataConfig(
        dataset_type=dataset_type,
        task_type='similarity'
    )
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"similarity_{dataset_type}_{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        config=config,
        name=run_name,
        tags=[*config['wandb']['tags'], 'similarity']
    )
    
    logger.info("Creating datasets...")
    train_dataset = MultiPartitionTreeDataset(
        data_config.train_path,
        config=config,
        num_workers=8
    )
    
    val_dataset = MultiPartitionTreeDataset(
        data_config.dev_path,
        config=config,
        num_workers=4
    )
    
    logger.info("Initializing model...")
    model = TreeMatchingNet(config).to(config['device'])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['train'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_corr = -1  # Using correlation instead of accuracy for similarity
    patience_counter = 0
    
    logger.info("Starting training...")
    epoch_pbar = tqdm(range(config['train']['n_epochs']), desc="Training", position=0)
    
    for epoch in epoch_pbar:
        # Monitor memory before epoch
        MemoryMonitor.log_memory(step=epoch, prefix=f'Epoch {epoch} start: ')
        
        # Train
        train_metrics = train_epoch(model, train_dataset, optimizer, config, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_dataset, config, epoch)
        
        # Update progress bar with correlation instead of accuracy
        epoch_pbar.set_postfix({
            'train_loss': f"{train_metrics['loss']:.4f}",
            'train_corr': f"{train_metrics['correlation']:.4f}",
            'val_loss': f"{val_metrics['loss']:.4f}",
            'val_corr': f"{val_metrics['correlation']:.4f}"
        })
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/correlation': train_metrics['correlation'],
            'train/mse': train_metrics['mse'],
            'train/batch_time': train_metrics['batch_time'],
            'train/data_time': train_metrics['data_time'],
            'val/loss': val_metrics['loss'],
            'val/correlation': val_metrics['correlation'],
            'val/mse': val_metrics['mse']
        })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Early stopping based on correlation
        if val_metrics['correlation'] > best_val_corr:
            best_val_corr = val_metrics['correlation']
            best_checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_checkpoint_path)
            patience_counter = 0
            logger.info(f"New best validation correlation: {best_val_corr:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['train']['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Monitor memory after epoch
        MemoryMonitor.log_memory(step=epoch, prefix=f'Epoch {epoch} end: ')
    
        # Optional cleanup between epochs
        torch.cuda.empty_cache()
    
    logger.info(f"Training completed! Best validation correlation: {best_val_corr:.4f}")
    
    # Save final results summary
    results = {
        'best_val_correlation': best_val_corr,
        'epochs_trained': epoch + 1,
        'early_stopped': patience_counter >= config['train']['patience']
    }
    
    results_file = checkpoint_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    wandb.save(str(results_file))
    wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train tree matching model for similarity task')
    parser.add_argument('--dataset', type=str, default='semeval',
                      choices=['semeval'],
                      help='Dataset to use for similarity task')
    parser.add_argument('--config', type=str,
                      help='Optional path to config override file')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with limited batches')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        train_similarity(
            dataset_type=args.dataset
        )
    except Exception as e:
        logger.exception("Training failed with error:")
        raise