#training/train.py
import wandb
from ...GMN.evaluation import compute_similarity, auc
import torch

def train_epoch(model, train_loader, optimizer, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (graphs, labels) in enumerate(train_loader):
        # Move to device
        graphs = graphs.to(config.device)
        labels = labels.to(config.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
        
        # Compute loss
        loss = compute_tree_matching_loss(outputs, labels, config)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log to WandB
        if batch_idx % config.log_interval == 0:
            wandb.log({
                'train_loss': loss.item(),
                'step': batch_idx
            })
    
    return total_loss / len(train_loader)
