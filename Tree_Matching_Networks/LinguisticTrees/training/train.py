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

def train_step(model, graphs, labels, optimizer, loss_fn, config):
    """Single training step with memory management"""
    device = config['device']
    
    try:
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
        
        # Scale loss for gradient accumulation
        loss = loss / config['train']['gradient_accumulation_steps']
        
        # Backward pass
        loss.backward()
        
        # Cleanup
        del graphs
        del graph_vectors
        del x
        del y
        torch.cuda.empty_cache()
        
        return loss.item() * config['train']['gradient_accumulation_steps'], predictions, accuracy
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("OOM during training step")
            torch.cuda.empty_cache()
            raise
        else:
            raise

def train_epoch(model, dataset, optimizer, config, epoch):
    """Train for one epoch with memory management"""
    model.train()
    device = config['device']
    
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
        desc=f'Training Epoch {epoch}'
    )
    
    # Monitor memory at start
    MemoryMonitor.log_memory(prefix='Training start: ')
    
    start_time = time.time()
    optimizer.zero_grad()
    
    all_predictions = []
    all_labels = []
    
    for batch_idx, (graphs, labels) in pbar:
        try:
            # Training step
            loss, predictions, accuracy = train_step(
                model, graphs, labels, optimizer, loss_fn, config
            )
            
            # Step optimizer on schedule
            if (batch_idx + 1) % config['train']['gradient_accumulation_steps'] == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['train']['clip_value']
                )
                optimizer.step()
                optimizer.zero_grad()
            
            # Store predictions
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            # Update metrics
            batch_time = time.time() - start_time
            metrics['loss'] += loss
            metrics['accuracy'] += accuracy.item()
            metrics['batch_time'] += batch_time
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss:.4f}",
                'acc': f"{accuracy.item():.4f}",
                'time': f"{batch_time:.3f}s"
            })
            
            # Periodic cleanup
            if batch_idx % config['train']['cleanup_interval'] == 0:
                MemoryMonitor.clear_memory()
                
            # Log to wandb
            if batch_idx % config['wandb']['log_interval'] == 0:
                mem_stats = MemoryMonitor.get_memory_usage()
                wandb.log({
                    'batch/loss': loss,
                    'batch/accuracy': accuracy.item(),
                    'batch/learning_rate': optimizer.param_groups[0]['lr'],
                    'batch/time': batch_time,
                    'memory/ram_used_gb': mem_stats['ram_used_gb'],
                    'memory/gpu_used_gb': mem_stats['gpu_used_gb'],
                    'batch': batch_idx + epoch * n_batches if n_batches else batch_idx
                })
                
            start_time = time.time()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM in batch {batch_idx}, clearing memory...")
                torch.cuda.empty_cache()
                MemoryMonitor.clear_memory()
                optimizer.zero_grad()
                continue
            else:
                raise e
    
    # Final optimizer step if needed
    if (batch_idx + 1) % config['train']['gradient_accumulation_steps'] != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Compute epoch metrics
    n_batches = batch_idx + 1
    metrics = {k: v / n_batches for k, v in metrics.items()}
    
    # Final memory check
    MemoryMonitor.log_memory(prefix='Training end: ')
    
    return metrics

