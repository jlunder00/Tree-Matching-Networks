# training/validation.py
import torch
import torch.nn.functional as F
import logging
from .loss import SimilarityLoss, EntailmentLoss, InfoNCELoss, TextLevelBinaryLoss, TextLevelSimilarityLoss, TextLevelEntailmentLoss, TextLevelContrastiveLoss 
import wandb
from tqdm import tqdm
import time
from ..data.paired_groups_dataset import get_paired_groups_dataloader
from ..utils.memory_utils import MemoryMonitor
from ..data.data_utils import GraphData
from ..data.dynamic_calculated_contrastive_dataset import get_dynamic_calculated_dataloader
from ..data.batch_utils import BatchInfo
from .loss_handlers import LOSS_HANDLERS as loss_handlers

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
        
        loss, predictions, base_metrics = loss_fn(embeddings, batch_info)

        if 'aggregative' in config['model']['task_loader_type']:
            accuracy_metrics = {}
            
            # Accuracy@1: Whether the most similar item is the correct positive
            top1_accuracy_overall = 0
            top1_accuracy_sim = 0
            top1_accuracy_dist = 0
            top1_accuracy_mid = 0
            sim_anchors = 0
            dist_anchors = 0
            mid_anchors = 0

            n_groups = len(batch_info.group_indices)
            anchor_indices = list(range(0, n_groups*2, 2))
            pos_indices = list(range(1, n_groups*2, 2))
            sim_matrix, dist_matrix, mid_matrix = predictions[0], predictions[1], predictions[2]
            
            for i, anchor_idx in enumerate(anchor_indices):
                # Get positive indices for this anchor
                pos_idx = pos_indices[i]
                
                positive_test = batch_info.group_labels[i] > 0
                negative_test = batch_info.group_labels[i] < 0 
                mid_test = batch_info.group_labels[i] < 0 
                if positive_test:
                    # Get similarities for this anchor with all other samples
                    anchor_out = sim_matrix[anchor_idx]
                elif negative_test: 
                    anchor_out = dist_matrix[anchor_idx]
                elif mid_test:
                    anchor_out = mid_matrix[anchor_idx]
                else:
                    continue
                
                # Exclude self-similarity
                # anchor_similarities[anchor_idx] = -float('inf')
                
                # Get the index of the highest similarity
                top_idx = torch.argmax(anchor_out).item()
                
                # Check if the top similarity is with a positive example
                is_correct = top_idx == pos_idx
                if positive_test: 
                    top1_accuracy_sim += int(is_correct)
                    sim_anchors += 1
                elif negative_test:
                    top1_accuracy_dist += int(is_correct)
                    dist_anchors += 1
                elif mid_test:
                    top1_accuracy_mid += int(is_correct)
                    mid_anchors += 1
                top1_accuracy_overall += int(is_correct)
                
            total_anchors = sim_anchors + dist_anchors + mid_anchors
            if total_anchors > 0:
                accuracy_metrics['top1_accuracy'] = top1_accuracy_overall / total_anchors
            if sim_anchors > 0:
                accuracy_metrics['top1_accuracy_sim'] = top1_accuracy_sim / sim_anchors
            if dist_anchors > 0:
                accuracy_metrics['top1_accuracy_dist'] = top1_accuracy_dist / dist_anchors
            if mid_anchors > 0:
                accuracy_metrics['top1_accuracy_mid'] = top1_accuracy_mid / mid_anchors
            
            # Recall@k (k=5): Percentage of positives in top k most similar items
            k = min(5, predictions[0].size(0) - 1)  # Ensure k is valid
            recall_at_k_overall = 0
            recall_at_k_sim = 0
            recall_at_k_dist = 0
            recall_at_k_mid = 0
            # Mean Reciprocal Rank (MRR)
            mrr = 0
            mrr_sim = 0
            mrr_dist = 0
            mrr_mid = 0
            
            for i, anchor_idx in enumerate(anchor_indices):
                pos_idx = pos_indices[i]
                
                positive_test = batch_info.group_labels[i] > 0 
                negative_test = batch_info.group_labels[i] < 0
                mid_test = batch_info.group_labels[i] < 0 
                if positive_test:
                    # Get similarities for this anchor with all other samples
                    anchor_out = sim_matrix[anchor_idx].clone()
                elif negative_test: 
                    anchor_out = dist_matrix[anchor_idx].clone()
                elif mid_test:
                    anchor_out = mid_matrix[anchor_idx].clone()
                else:
                    continue
                
                # Get top k indices
                _, top_k_indices = torch.topk(anchor_out, k)
                # Get ranks of all items
                _, all_indices = torch.sort(anchor_out, descending=True)
                
                # Count positives in top k
                correct = sum(1 for idx in top_k_indices if idx.item() == pos_idx)
                if positive_test:
                    recall_at_k_sim += correct
                elif negative_test:
                    recall_at_k_dist += correct
                elif mid_test:
                    recall_at_k_mid += correct
                recall_at_k_overall += correct 

                if positive_test:
                    for rank, idx in enumerate(all_indices):
                        if idx.item() == pos_idx:
                            mrr_sim += 1.0 / (rank + 1)
                            mrr += 1.0 / (rank + 1)
                            break
                elif negative_test:
                    for rank, idx in enumerate(all_indices):
                        if idx.item() == pos_idx:
                            mrr_dist += 1.0 / (rank + 1)
                            mrr += 1.0 / (rank + 1)
                            break
                elif mid_test:
                    for rank, idx in enumerate(all_indices):
                        if idx.item() == pos_idx:
                            mrr_mid += 1.0 / (rank + 1)
                            mrr += 1.0 / (rank + 1)
                            break

                
            if total_anchors > 0:
                accuracy_metrics[f'recall@{k}'] = recall_at_k_overall / total_anchors
                accuracy_metrics['mrr'] = mrr / total_anchors
            if sim_anchors > 0:
                accuracy_metrics[f'recall@{k}_sim'] = recall_at_k_sim / sim_anchors
                accuracy_metrics['mrr_sim'] = mrr_sim / total_anchors
            if dist_anchors > 0:
                accuracy_metrics[f'recall@{k}_dist'] = recall_at_k_dist / dist_anchors
                accuracy_metrics['mrr_dist'] = mrr_dist / total_anchors
            if mid_anchors > 0:
                accuracy_metrics[f'recall@{k}_mid'] = recall_at_k_mid / mid_anchors
                accuracy_metrics['mrr_mid'] = mrr_mid / total_anchors
            
                
            
        else:

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

def validate_step(model, graphs, batch_info, loss_fn, device, config):
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
        
        # Compute metrics based on task
        # if config['model']['task_type'] == 'similarity':
        #     x, y = outputs[::2], outputs[1::2]
        #     loss, predictions, metrics = loss_fn(x, y, batch_info)
        #     del x, y
        # else:  # entailment
        #     loss, predictions, metrics = loss_fn(outputs, batch_info)
        loss, predictions, metrics = loss_fn(embeddings, batch_info)
        
        # Cleanup
        del graphs
        del embeddings
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
    dataset.reset_epoch()
    device = config['device']
    task_type = config['model']['task_type']
    task_loader_type = config['model']['task_loader_type']    
    contrastive_types = ['infonce']
    
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
    loss_loader = 'other' if task_loader_type != 'aggregative' else task_loader_type
    loss_fn = loss_handlers[task_type][loss_loader](
        device = device,
        temperature = config['model'].get('temperature', 0.07),
        aggregation = config['model'].get('aggregation', 'attention'),
        threshold = config['model'].get("threshold", 0.5),
        num_classes = config['model'].get("num_classes", 3),
        classifier_input_dim = config['model'].get("graph_rep_dim", 1792)*2,
        classifier_hidden_dims = config['model'].get("classifier_hidden_dims", [512]),
        positive_infonce_weight = config['model'].get("positive_infonce_weight", 1.0),
        inverse_infonce_weight = config['model'].get("inverse_infonce_weight", 0.25),
        midpoint_infonce_weight = config['model'].get("midpoint_infonce_weight", 0.25)
    )
    if task_type in contrastive_types:
        return validate_epoch_contrastive(model, dataset, loss_fn, config, epoch)
    
    # Get total batches for progress bar
    
    if 'aggregative' in task_loader_type:
        data_loader = get_paired_groups_dataloader(dataset, config['data']['num_workers_val'], persistent_workers=False) 
    else:
        data_loader = dataset.pairs(config['data']['batch_size'])
    n_batches = len(data_loader) if hasattr(data_loader, '__len__') else None
    # Create progress bar
    pbar = tqdm(
        enumerate(data_loader),
        total=n_batches,
        desc=f'Validation Epoch {epoch}',
        leave=False
    )
    
    all_predictions = []
    start_time = time.time()
    
    # Monitor memory at start
    MemoryMonitor.log_memory(prefix='Validation start: ')
    
    for batch_idx, (graphs, batch_info) in pbar:
        try:
            # Validation step
            loss, predictions, batch_metrics = validate_step(
                model, graphs, batch_info, loss_fn, device, config
            )
            
            # Store predictions
            all_predictions.append(predictions.cpu())
            
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
            if task_type == 'similarity' or task_type == 'similarity_aggregative':
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
        'pos_distance': 0.0,
        'neg_distance': 0.0,
        'pos_midpoint': 0.0,
        'neg_midpoint': 0.0,
        'top1_accuracy': 0.0,
        'recall@5': 0.0,
        'mrr': 0.0,
        'top1_accuracy_dist': 0.0,
        'recall@5_dist': 0.0,
        'mrr_dist': 0.0,
        'top1_accuracy_mid': 0.0,
        'recall@5_mid': 0.0,
        'mrr_mid': 0.0
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
                'pos_dist': f"{batch_metrics.get('pos_distance', 0):.4f}",
                'neg_dist': f"{batch_metrics.get('neg_distance', 0):.4f}",
                'pos_mid': f"{batch_metrics.get('pos_midpoint', 0):.4f}",
                'neg_mid': f"{batch_metrics.get('neg_midpoint', 0):.4f}",
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
