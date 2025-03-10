# training/validation.py
import torch
import torch.nn.functional as F
import logging
from .loss import SimilarityLoss, EntailmentLoss, InfoNCELoss 
import wandb
from tqdm import tqdm
import time
from ..utils.memory_utils import MemoryMonitor
from ..data.data_utils import GraphData
from ..data.dynamic_calculated_contrastive_dataset import get_dynamic_calculated_dataloader
from ..data.batch_utils import BatchInfo

logger = logging.getLogger(__name__)

@torch.no_grad()
def validate_step_contrastive(model, graphs, batch_info, loss_fn, device, config):
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
        
        # Forward pass
        embeddings = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
        
        loss, _, base_metrics = loss_fn(embeddings, batch_info)
        # Calculate similarity matrix for all embeddings
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1),  # [n_embeddings, 1, hidden_dim]
            embeddings.unsqueeze(0),  # [1, n_embeddings, hidden_dim]
            dim=2
        )
        
        # Compute accuracy metrics for contrastive learning
        accuracy_metrics = {}
        
        # Accuracy@1: Whether the most similar item is the correct positive
        top1_accuracy = 0
        total_anchors = 0
        
        for anchor_idx in batch_info.anchor_indices:
            # Get positive indices for this anchor
            positive_indices = [pos_idx for a_idx, pos_idx in batch_info.positive_pairs 
                             if a_idx == anchor_idx]
            
            if not positive_indices:
                continue
                
            # Get similarities for this anchor with all other samples
            anchor_similarities = similarity_matrix[anchor_idx]
            
            # Exclude self-similarity
            anchor_similarities[anchor_idx] = -float('inf')
            
            # Get the index of the highest similarity
            top_idx = torch.argmax(anchor_similarities).item()
            
            # Check if the top similarity is with a positive example
            is_correct = top_idx in positive_indices
            top1_accuracy += int(is_correct)
            total_anchors += 1
            
        if total_anchors > 0:
            accuracy_metrics['top1_accuracy'] = top1_accuracy / total_anchors
        
        # Recall@k (k=5): Percentage of positives in top k most similar items
        k = min(5, similarity_matrix.size(0) - 1)  # Ensure k is valid
        recall_at_k = 0
        
        for anchor_idx in batch_info.anchor_indices:
            positives = [pos_idx for a_idx, pos_idx in batch_info.positive_pairs 
                       if a_idx == anchor_idx]
            
            if not positives:
                continue
                
            # Get similarities excluding self
            sims = similarity_matrix[anchor_idx].clone()
            sims[anchor_idx] = -float('inf')
            
            # Get top k indices
            _, top_k_indices = torch.topk(sims, k)
            
            # Count positives in top k
            correct = sum(1 for idx in top_k_indices if idx.item() in positives)
            recall_at_k += correct / len(positives)
            
        if total_anchors > 0:
            accuracy_metrics[f'recall@{k}'] = recall_at_k / total_anchors
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for anchor_idx in batch_info.anchor_indices:
            positives = [pos_idx for a_idx, pos_idx in batch_info.positive_pairs 
                       if a_idx == anchor_idx]
            
            if not positives:
                continue
                
            # Get similarities excluding self
            sims = similarity_matrix[anchor_idx].clone()
            sims[anchor_idx] = -float('inf')
            
            # Get ranks of all items
            _, indices = torch.sort(sims, descending=True)
            
            # Find rank of first positive
            for rank, idx in enumerate(indices):
                if idx.item() in positives:
                    mrr += 1.0 / (rank + 1)  # +1 because ranks are 0-indexed
                    break
        
        if total_anchors > 0:
            accuracy_metrics['mrr'] = mrr / total_anchors
            
        # Combine with base metrics
        combined_metrics = {**base_metrics, **accuracy_metrics}
        
        return loss.item(), combined_metrics
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("OOM during validation step")
            torch.cuda.empty_cache()
            raise
        else:
            raise

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
    task_loader_type = config['model']['task_loader_type']    
    
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
    elif task_type == 'info_nce':
        loss_fn = InfoNCELoss(
            device=device,
            temperature=config['model'].get('temperature', 0.07)
        )
    else:
        loss_fn = EntailmentLoss(device).to(device=device, non_blocking=True)
    
    if task_loader_type == 'contrastive':
        return validate_epoch_contrastive(model, dataset, loss_fn, config, epoch)
    
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


@torch.no_grad()
def validate_epoch_contrastive(model, dataset, loss_fn, config, epoch):
    """Validation epoch routine for contrastive learning"""
    model.eval()
    device = config['device']
    
    # Get contrastive dataloader
    val_loader = get_dynamic_calculated_dataloader(
        dataset,
        num_workers=config['data'].get('num_workers', 1),
        pin_memory=True
    )

    n_batches = len(val_loader) if hasattr(val_loader, '__len__') else None
    
    pbar = tqdm(
        enumerate(val_loader),
        total=n_batches,
        desc=f'Validation Epoch {epoch}',
        leave=False
    )
    
    # Initialize metrics
    metrics = {
        'loss': 0.0,
        'batch_time': 0.0,
        'data_time': 0.0,
        'pos_similarity': 0.0, 
        'neg_similarity': 0.0,
        'raw_pos_sim': 0.0,
        'raw_neg_sim': 0.0,
        'top1_accuracy': 0.0,
        'recall@5': 0.0,
        'mrr': 0.0
    }
    
    # Initial memory check
    MemoryMonitor.log_memory(prefix='Validation start: ')
    
    start_time = time.time()
    data_start = time.time()
    
    for batch_idx, (graphs, batch_info) in pbar:
        data_time = time.time() - data_start
        metrics['data_time'] += data_time
        
        try:
            # Validation step
            loss, batch_metrics = validate_step_contrastive(
                model, graphs, batch_info, loss_fn, device, config
            )
                
            # Update metrics
            batch_time = time.time() - start_time
            metrics['batch_time'] += batch_time
            metrics['loss'] += loss
            for k, v in batch_metrics.items():
                if k in metrics:
                    metrics[k] += v
                    
            # Progress bar updates
            progress = {
                'loss': f'{loss:.4f}',
                'time': f'{batch_time:.3f}s',
                'pos_sim': f"{batch_metrics.get('pos_similarity', 0):.4f}",
                'neg_sim': f"{batch_metrics.get('neg_similarity', 0):.4f}",
            }
            
            # Add accuracy metrics if available
            if 'top1_accuracy' in batch_metrics:
                progress['top1_acc'] = f"{batch_metrics['top1_accuracy']:.4f}"
            if 'mrr' in batch_metrics:
                progress['mrr'] = f"{batch_metrics['mrr']:.4f}"
                
            pbar.set_postfix(progress)
            
            # Periodic cleanup
            if batch_idx % config['train']['cleanup_interval'] == 0:
                MemoryMonitor.clear_memory()
            
            # WandB logging
            if batch_idx % config['wandb']['log_interval'] == 0:
                mem_stats = MemoryMonitor.get_memory_usage()
                wandb_metrics = {
                    'val_memory/ram_used_gb': mem_stats['ram_used_gb'],
                    'val_memory/gpu_used_gb': mem_stats['gpu_used_gb'],
                    'val_batch/time': batch_time,
                    'val_batch/data_time': data_time,
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
            data_start = time.time()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM in validation batch {batch_idx}, clearing memory...")
                torch.cuda.empty_cache()
                MemoryMonitor.clear_memory()
                continue
            else:
                raise e
                
    # Compute epoch metrics
    if batch_idx >= 0:  # Ensure we had at least one batch
        n_batches = batch_idx + 1
        metrics = {k: v / n_batches for k, v in metrics.items()}
    
    # Final cleanup and memory check
    MemoryMonitor.clear_memory()
    MemoryMonitor.log_memory(prefix='Validation end: ')
    
    return metrics
