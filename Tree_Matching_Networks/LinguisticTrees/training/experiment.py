# training/experiment.py
from pathlib import Path
import yaml
import torch
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ExperimentManager:
    """Manages experiment checkpoints and configs"""
    
    def __init__(self, task_type, config, timestamp=None):
        self.task_type = task_type
        self.config = config
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories
        self.base_dir = Path("experiments")
        self.experiment_dir = self.base_dir / f"{task_type}_{self.timestamp}"
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.config_dir = self.experiment_dir / "config"
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
        # Save config
        self._save_config()
        
    def _save_config(self):
        """Save experiment config"""
        with open(self.config_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
            
    def get_checkpoint_path(self, epoch=None):
        """Get path for checkpoint"""
        if epoch is not None:
            return self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        return self.checkpoint_dir / "best_model.pt"
        
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        path = self.get_checkpoint_path(epoch)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def save_best_model(self, model, optimizer, epoch, metrics):
        """Save best model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        path = self.get_checkpoint_path()
        torch.save(checkpoint, path)
        logger.info(f"Saved best model to {path}")
        
    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        # Extract timestamp from path if possible
        path = Path(checkpoint_path)
        if path.parent.parent.name.startswith(('entailment_', 'similarity_')):
            timestamp = path.parent.parent.name.split('_', 1)[1]
        else:
            timestamp = None
            
        # Create experiment manager
        manager = cls(
            task_type=checkpoint['config']['model']['task_type'],
            config=checkpoint['config'],
            timestamp=timestamp
        )
        
        return checkpoint, manager
