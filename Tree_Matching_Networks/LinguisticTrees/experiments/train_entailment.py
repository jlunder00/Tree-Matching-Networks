#experiments/train_entailment.py
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

def train_model():
    """Full training loop for entailment task"""
    # Load config
    config = get_tree_config()
    data_config = TreeDataConfig()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"entailment_{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        config=config,
        name=run_name,
        tags=[*config['wandb']['tags'], 'entailment']
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
    best_val_acc = 0
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
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'train_loss': f"{train_metrics['loss']:.4f}",
            'train_acc': f"{train_metrics['accuracy']:.4f}",
            'val_loss': f"{val_metrics['loss']:.4f}",
            'val_acc': f"{val_metrics['accuracy']:.4f}"
        })
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/accuracy': train_metrics['accuracy'],
            'train/batch_time': train_metrics['batch_time'],
            'train/data_time': train_metrics['data_time'],
            'val/loss': val_metrics['loss'],
            'val/accuracy': val_metrics['accuracy'],
            **{f"val/{k}": v for k, v in val_metrics.items() if k not in ['loss', 'accuracy']}
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
        
        # Early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_checkpoint_path)
            patience_counter = 0
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['train']['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Monitor memory after epoch
        MemoryMonitor.log_memory(step=epoch, prefix=f'Epoch {epoch} end: ')
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    wandb.finish()

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    train_model()



