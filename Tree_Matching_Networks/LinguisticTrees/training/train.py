#training/train.py
from ..data.data_utils import GraphData
from tqdm import tqdm
import wandb
from .loss import TreeMatchingLoss
import torch
import logging
import time
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def train_epoch(model, dataset, optimizer, config, epoch):
    """Train for one epoch with progress tracking"""
    model.train()
    device = torch.device(config['device'])
    
    # Initialize metrics
    metrics = {
        'loss': 0.0,
        'accuracy': 0.0,
        'batch_time': 0.0,
        'data_time': 0.0
    }
    
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
        desc=f'Training Epoch {epoch}',
        leave=False
    )
    
    # Monitor memory at start
    MemoryMonitor.log_memory(prefix='Training start: ')
    
    start_time = time.time()
    data_time = 0
    
    all_predictions = []
    all_labels = []
    
    for batch_idx, (graphs, labels) in pbar:
        data_time = time.time() - start_time
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Move data to device
        graphs = GraphData(
            node_features=graphs.node_features.to(device),
            edge_features=graphs.edge_features.to(device),
            from_idx=graphs.from_idx.to(device),
            to_idx=graphs.to_idx.to(device),
            graph_idx=graphs.graph_idx.to(device),
            n_graphs=graphs.n_graphs
        )
        labels = labels.to(device)
        
        # Forward pass
        graph_vectors = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
        
        # Split vectors and compute loss
        x, y = graph_vectors[::2], graph_vectors[1::2]
        loss, predictions, accuracy = loss_fn(x, y, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if 'clip_value' in config['train']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['train']['clip_value']
            )
            
        optimizer.step()
        
        # Store predictions for confusion matrix
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
        
        # Update metrics
        batch_time = time.time() - start_time
        metrics['loss'] += loss.item()
        metrics['accuracy'] += accuracy.item()
        metrics['batch_time'] += batch_time
        metrics['data_time'] += data_time
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy.item():.4f}",
            'batch_time': f"{batch_time:.3f}s",
            'data_time': f"{data_time:.3f}s"
        })
        
        # Log to wandb periodically
        if batch_idx % config['wandb']['log_interval'] == 0:
            wandb.log({
                'batch/loss': loss.item(),
                'batch/accuracy': accuracy.item(),
                'batch/learning_rate': optimizer.param_groups[0]['lr'],
                'batch/data_time': data_time,
                'batch/batch_time': batch_time,
                'batch': batch_idx + epoch * n_batches if n_batches else batch_idx,
            })
        
        start_time = time.time()
    
    # Compute epoch metrics
    n_batches = batch_idx + 1
    metrics = {k: v / n_batches for k, v in metrics.items()}
    
    # For entailment task, add confusion matrix
    if config['model']['task_type'] == 'entailment' and wandb.run is not None:
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        wandb.log({
            'train_confusion_matrix': wandb.plot.confusion_matrix(
                preds=all_predictions.numpy(),
                y_true=all_labels.numpy(),
                class_names=['Contradiction', 'Neutral', 'Entailment']
            )
        })
    
    # Monitor memory at end
    MemoryMonitor.log_memory(prefix='Training end: ')
    
    return metrics
