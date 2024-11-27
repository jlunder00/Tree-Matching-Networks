import wandb
import torch
from pathlib import Path
from ..configs.default_tree_config import get_tree_config
from ..data.partition_datasets import MultiPartitionTreeDataset
from ..models.tree_matching import TreeMatchingNet
from ..training.train import train_epoch
from ..training.metrics import TreeMatchingMetrics

def test_training():
    """Test training loop with dev dataset"""
    # Load config
    config = get_tree_config()
    config['data']['train_path'] = 'data/processed_data/dev/final_dataset.json'
    config['train']['n_epochs'] = 2  # Short test run
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        config=config,
        tags=[*config['wandb']['tags'], 'dev-test']
    )
    
    # Create dataset
    dataset = MultiPartitionTreeDataset(config['data']['train_path'], config, num_workers=12)
    
    # Create model and optimizer
    model = TreeMatchingNet(config).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    # Test training loop
    print("Starting test training...")
    for epoch in range(config['train']['n_epochs']):
        train_loss = train_epoch(model, dataset, optimizer, config)
        print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss
        })
    
    print("Test training completed!")
    wandb.finish()

if __name__ == '__main__':
    test_training()
