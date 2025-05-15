# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#training/logger.py
import wandb
from typing import Dict, Any
import torch
from pathlib import Path

class WandbLogger:
    """WandB logger wrapper"""
    
    def __init__(self, config):
        self.config = config
        self.run = wandb.init(
            project=config.project_name,
            config=config,
            name=config.run_name
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics to WandB"""
        wandb.log(metrics, step=step)
        
    def log_batch(self, batch_metrics: Dict[str, torch.Tensor], step: int):
        """Log batch-level metrics"""
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in batch_metrics.items()
        }
        self.log_metrics(metrics, step)
        
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        # Log to WandB
        wandb.save(str(path))
        
    def finish(self):
        """Clean up logging"""
        wandb.finish()
