#training/validation.py
import torch
from .loss import TreeMatchingLoss
import wandb
from tqdm import tqdm
from ..utils.memory_utils import MemoryMonitor

@torch.no_grad()
def validate_epoch(model, dataset, config, epoch):
    """Run validation epoch"""
    model.eval()
    device = torch.device(config['device'])
    
    # Initialize metrics
    metrics = {
        'loss': 0.0,
        'accuracy': 0.0,
        'batch_time': 0.0,
        'data_time': 0.0
    }
    
    # Create loss function
    loss_fn = TreeMatchingLoss(
        task_type=config['model']['task_type'],
        **config['model'].get('loss_params', {})
    ).to(device)
    
    # Get total batches for progress bar
    n_samples = len(dataset) if hasattr(dataset, '__len__') else None
    n_batches = n_samples // config['data']['batch_size'] if n_samples else None
    
    # Create progress bar
    pbar = tqdm(
        enumerate(dataset.pairs(config['data']['batch_size'])),
        total=n_batches,
        desc=f'Validation Epoch {epoch}',
        leave=False
    )
    
    all_predictions = []
    all_labels = []
    
    # Monitor memory at start
    MemoryMonitor.log_memory(prefix='Validation start: ')
    
    for batch_idx, (graphs, labels) in pbar:
        # Move to device
        labels = labels.to(device)
        
        # Forward pass
        graph_vectors = model(
            graphs.node_features.to(device),
            graphs.edge_features.to(device),
            graphs.from_idx.to(device),
            graphs.to_idx.to(device),
            graphs.graph_idx.to(device),
            graphs.n_graphs
        )
        
        # Split into pairs
        x, y = graph_vectors[::2], graph_vectors[1::2]
        
        # Compute loss, predictions and accuracy
        loss, predictions, accuracy = loss_fn(x, y, labels)
        
        # Update metrics
        metrics['loss'] += loss.item()
        metrics['accuracy'] += accuracy.item()
        
        # Store predictions for confusion matrix
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy.item():.4f}"
        })
        
        # Optional early break for testing
        if config.get('debug_max_batches') and batch_idx >= config['debug_max_batches']:
            break
    
    # Compute final metrics
    n_batches = batch_idx + 1
    metrics = {k: v / n_batches for k, v in metrics.items()}
    
    # For entailment task, add confusion matrix
    if config['model']['task_type'] == 'entailment' and wandb.run is not None:
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        wandb.log({
            'val_confusion_matrix': wandb.plot.confusion_matrix(
                preds=all_predictions.numpy(),
                y_true=all_labels.numpy(),
                class_names=['Contradiction', 'Neutral', 'Entailment']
            )
        })
    
    # Monitor memory at end
    MemoryMonitor.log_memory(prefix='Validation end: ')
    
    return metrics
