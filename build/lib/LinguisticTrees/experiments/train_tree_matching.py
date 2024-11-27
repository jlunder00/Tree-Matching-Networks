#experiments/train_tree_matching.py
import wandb
from ..configs.default_tree_config import get_tree_config
from ..data.tree_dataset import TreeMatchingDataset
from ..models.tree_matching import TreeMatchingNet
from ..training.train import train_epoch

def main():
    # Initialize config
    config = get_tree_config()
    
    # Initialize WandB
    wandb.init(
        project="tree-matching",
        config=config
    )
    
    # Create datasets
    train_dataset = TreeMatchingDataset(
        config.data.train_path,
        config
    )
    val_dataset = TreeMatchingDataset(
        config.data.val_path,
        config
    )
    
    # Create model
    model = TreeMatchingNet(config)
    
    # Training loop
    for epoch in range(config.training.n_epochs):
        train_loss = train_epoch(
            model, 
            train_dataset,
            optimizer,
            config
        )
        
        val_metrics = evaluate_model(
            model,
            val_dataset,
            config
        )
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })

if __name__ == '__main__':
    main()
