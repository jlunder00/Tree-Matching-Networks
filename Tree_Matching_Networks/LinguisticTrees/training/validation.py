# training/validation.py
import torch
import logging
from .loss import SimilarityLoss, EntailmentLoss 
import wandb
from tqdm import tqdm
import time
from ..utils.memory_utils import MemoryMonitor
from ..data.data_utils import GraphData

logger = logging.getLogger(__name__)

def validate_step(model, graphs, labels, loss_fn, device, config):
    """Single validation step with memory management"""
    try:
        # Move data to device
        graphs = GraphData(
            node_features=graphs.node_features.to(device, non_blocking=True),
            edge_features=graphs.edge_features.to(device, non_blocking=True),
            from_idx=graphs.from_idx.to(device, non_blocking=True),
            to_idx=graphs.to_idx.to(device, non_blocking=True),
            graph_idx=graphs.graph_idx.to(device, non_blocking=True),
            n_graphs=graphs.n_graphs
        )
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
        
        # Compute metrics based on task
        if config['model']['task_type'] == 'similarity':
            x, y = outputs[::2], outputs[1::2]
            loss, predictions, metrics = loss_fn(x, y, labels)
            del x, y
        else:  # entailment
            loss, predictions, metrics = loss_fn(outputs, labels)
        
        # Cleanup
        del graphs
        del outputs
        torch.cuda.empty_cache()
        
        return loss.item(), predictions, metrics
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("OOM during validation step")
            torch.cuda.empty_cache()
            raise
        else:
            raise

# def validate_step(model, graphs, labels, loss_fn, device, config):
#     """Single validation step with memory management"""
#     try:
#         # Move data to device
#         graphs = GraphData(
#             node_features=graphs.node_features.to(device, non_blocking=True),
#             edge_features=graphs.edge_features.to(device, non_blocking=True),
#             from_idx=graphs.from_idx.to(device, non_blocking=True),
#             to_idx=graphs.to_idx.to(device, non_blocking=True),
#             graph_idx=graphs.graph_idx.to(device, non_blocking=True),
#             n_graphs=graphs.n_graphs
#         )
#         labels = labels.to(device, non_blocking=True)
#         
#         # Forward pass
#         graph_vectors = model(
#             graphs.node_features,
#             graphs.edge_features,
#             graphs.from_idx,
#             graphs.to_idx,
#             graphs.graph_idx,
#             graphs.n_graphs
#         )
#         
#         # Split vectors and compute metrics
#         # x, y = graph_vectors[::2], graph_vectors[1::2]
#         # loss, predictions, metrics = loss_fn(x, y, labels)
#         if config['model']['task_type'] == 'similarity':
#             x, y = graph_vectors[::2], graph_vectors[1::2]
#             loss, predictions, metrics = loss_fn(x, y, labels)
#             del x
#             del y
#         else:  # entailment
#             loss, predictions, metrics = loss_fn(graph_vectors, labels)
#         
#         # Cleanup
#         del graphs
#         del graph_vectors
#         torch.cuda.empty_cache()
#         
#         return loss.item(), predictions, metrics
#         
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             logger.error("OOM during validation step")
#             torch.cuda.empty_cache()
#             raise
#         else:
#             raise

@torch.no_grad()
def validate_epoch(model, dataset, config, epoch):
    """Run validation epoch with memory management"""
    model.eval()
    device = config['device']
    task_type = config['model']['task_type']
    
    # Initialize metrics based on task
    if task_type == 'similarity':
        metrics = {
            'loss': 0.0,
            'correlation': 0.0,
            'spearman': 0.0,
            'mse': 0.0,
            'batch_time': 0.0
        }
    else:
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'batch_time': 0.0
        }
    
    # Create loss function
    # loss_fn = TreeMatchingLoss(
    #     device=device,
    #     task_type=config['model']['task_type'],
    #     **config['model'].get('loss_params', {})
    # ).to(device, non_blocking=True)
    if task_type == 'similarity':
        loss_fn = SimilarityLoss(
            device = device
            # **config['mode'].get('loss_params', {})
        ).to(device, non_blocking=True)
    else:
        loss_fn = EntailmentLoss(device).to(device=device, non_blocking=True)
    
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
    start_time = time.time()
    
    # Monitor memory at start
    MemoryMonitor.log_memory(prefix='Validation start: ')
    
    for batch_idx, (graphs, labels) in pbar:
        try:
            # Validation step
            loss, predictions, batch_metrics = validate_step(
                model, graphs, labels, loss_fn, device, config
            )
            
            # Store predictions
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            # Update metrics
            batch_time = time.time() - start_time
            metrics['loss'] += loss
            metrics['batch_time'] += batch_time
            for k, v in batch_metrics.items():
                if k in metrics:
                    metrics[k] += v
            
            # Update progress bar based on task
            progress_metrics = {
                'loss': f"{loss:.4f}",
                'time': f"{batch_time:.3f}s"
            }
            if task_type == 'similarity':
                progress_metrics['corr'] = f"{batch_metrics['correlation']:.4f}"
                progress_metrics['mse'] = f"{batch_metrics['mse']:.4f}"
            else:
                progress_metrics['acc'] = f"{batch_metrics['accuracy']:.4f}"
                if 'f1' in batch_metrics:
                    progress_metrics['f1'] = f"{batch_metrics['f1']:.4f}"
                    
            pbar.set_postfix(progress_metrics)
            
            # Periodic cleanup
            if batch_idx % config['train']['cleanup_interval'] == 0:
                MemoryMonitor.clear_memory()
                
            # Log to wandb
            if batch_idx % config['wandb']['log_interval'] == 0:
                mem_stats = MemoryMonitor.get_memory_usage()
                wandb_metrics = {
                    'val_memory/ram_used_gb': mem_stats['ram_used_gb'],
                    'val_memory/gpu_used_gb': mem_stats['gpu_used_gb'],
                    'val_batch/time': batch_time,
                    'val_batch/loss': loss,
                    'val_batch': batch_idx
                }
                
                # Add batch metrics
                for k, v in batch_metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    wandb_metrics[f'val_batch/{k}'] = v
                    
                wandb.log(wandb_metrics)
            
            start_time = time.time()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM in validation batch {batch_idx}, clearing memory...")
                torch.cuda.empty_cache()
                MemoryMonitor.clear_memory()
                continue
            else:
                raise e
    
    # Compute final metrics
    n_batches = batch_idx + 1
    metrics = {k: v / n_batches for k, v in metrics.items()}
    
    # Final cleanup and memory check
    MemoryMonitor.clear_memory()
    MemoryMonitor.log_memory(prefix='Validation end: ')
    
    return metrics



