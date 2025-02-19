# training/train.py
from ..data.data_utils import GraphData
from ..data.batch_utils import BatchInfo
from tqdm import tqdm
import wandb
from .loss import SimilarityLoss, EntailmentLoss, InfoNCELoss 
import torch
import logging
import time
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

# def train_step(model, graphs: GraphData, labels: torch.Tensor, 
#                optimizer, loss_fn, config: dict):
#     """Single training step with memory optimization"""
#     device = config['device']
#     
#     try:
#         # Data should already be on device and pinned since we're using DataLoader properly
#         if not graphs.node_features.is_cuda:
#             graphs = GraphData(
#                 node_features=graphs.node_features.to(device, non_blocking=True),
#                 edge_features=graphs.edge_features.to(device, non_blocking=True),
#                 from_idx=graphs.from_idx.to(device, non_blocking=True),
#                 to_idx=graphs.to_idx.to(device, non_blocking=True),
#                 graph_idx=graphs.graph_idx.to(device, non_blocking=True),
#                 n_graphs=graphs.n_graphs
#             )
#             labels = labels.to(device, non_blocking=True)
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
#         # Split vectors and compute loss
#         x, y = graph_vectors[::2], graph_vectors[1::2]
#         loss, predictions, metrics = loss_fn(x, y, labels)
#         
#         # Scale loss for gradient accumulation
#         loss = loss / config['train']['gradient_accumulation_steps']
#         loss.backward()
#         
#         # Move predictions to CPU for metrics
#         predictions = predictions.cpu()
#         
#         # Cleanup GPU tensors
#         del graphs
#         del graph_vectors
#         del x
#         del y
#         
#         return loss.item() * config['train']['gradient_accumulation_steps'], predictions, metrics
#         
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             logger.error("OOM during training step")
#             torch.cuda.empty_cache()
#             MemoryMonitor.clear_memory()
#             raise
#         else:
#             raise

def train_step_infonce(model, graphs: GraphData, batch_info: BatchInfo,
                       optimizer, loss_fn, config: dict):
    """Single training step with InfoNCE loss"""
    device = config['device']
    
    try:
        # Move data to device if needed 
        if not graphs.node_features.is_cuda:
            graphs = GraphData(
                node_features=graphs.node_features.to(device, non_blocking=True),
                edge_features=graphs.edge_features.to(device, non_blocking=True),
                from_idx=graphs.from_idx.to(device, non_blocking=True),
                to_idx=graphs.to_idx.to(device, non_blocking=True),
                graph_idx=graphs.graph_idx.to(device, non_blocking=True),
                n_graphs=graphs.n_graphs
            )
            
        # Forward pass to get embeddings for all graphs
        embeddings = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
        
        # Compute loss
        loss, _, metrics = loss_fn(embeddings, batch_info)
        
        # Scale loss for accumulation
        loss = loss / config['train']['gradient_accumulation_steps']
        loss.backward()


        return loss.item() * config['train']['gradient_accumulation_steps'], metrics
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("OOM during training step")
            torch.cuda.empty_cache()
            MemoryMonitor.clear_memory()
            raise
        else:
            raise

def train_step(model, graphs: GraphData, labels: torch.Tensor, 
               optimizer, loss_fn, config: dict):
    """Single training step with full gradient optimization"""
    device = config['device']
    scaler = torch.GradScaler(device)
    
    try:
        # Move data to device if needed
        if not graphs.node_features.is_cuda:
            graphs = GraphData(
                node_features=graphs.node_features.to(device, non_blocking=True),
                edge_features=graphs.edge_features.to(device, non_blocking=True),
                from_idx=graphs.from_idx.to(device, non_blocking=True),
                to_idx=graphs.to_idx.to(device, non_blocking=True),
                graph_idx=graphs.graph_idx.to(device, non_blocking=True),
                n_graphs=graphs.n_graphs
            )
            labels = labels.to(device, non_blocking=True)
            
        # Enable gradient checkpointing if configured
        # if config['model'].get('gradient_checkpointing', False):
        #     model.enable_checkpointing()
            
        # Mixed precision forward pass
        # with torch.autocast(device):
        # Forward pass
        outputs = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
        
        # Compute loss based on model type
        if config['model']['task_type'] == 'similarity':
            x, y = outputs[::2], outputs[1::2]
            loss, predictions, metrics = loss_fn(x, y, labels)
        else:  # entailment
            loss, predictions, metrics = loss_fn(outputs, labels)
        
        # Scale loss for accumulation
        loss = loss / config['train']['gradient_accumulation_steps']
            
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            config['train']['clip_value']
        )
        
        # Step optimizer and update scaler
        scaler.step(optimizer)
        scaler.update()
        
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Move predictions to CPU and clear cache
        predictions = predictions.cpu()
        if config['train'].get('aggressive_cleanup', False):
            torch.cuda.empty_cache()
        
        return loss.item() * config['train']['gradient_accumulation_steps'], predictions, metrics
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("OOM during training step")
            torch.cuda.empty_cache()
            MemoryMonitor.clear_memory()
            raise
        else:
            raise

def train_epoch(model, dataset, optimizer, config, epoch):
    """Train for one epoch with optimized data loading"""
    model.train()
    device = config['device']
    task_type = config['model']['task_type']
    task_loader_type = config['model']['task_loader_type']    
    
    # Initialize loss function
    # loss_fn = TreeMatchingLoss(
    #     device=device,
    #     task_type=task_type,
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
        if not hasattr(dataset, 'get_dataloader'):
            raise ValueError("Dataset must support contrastive loading")
        return train_epoch_contrastive(model, dataset, optimizer, loss_fn, config, epoch)


    # Initialize metrics
    metrics = {
        'loss': 0.0,
        'batch_time': 0.0,
        'data_time': 0.0
    }
    
    if task_type == 'similarity':
        metrics.update({
            'correlation': 0.0,
            'spearman': 0.0,
            'mse': 0.0,
        })
    else:
        metrics.update({
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            # 'f1': 0.0,
        })
    # Create data iterator with progress bar
    data_loader = dataset.pairs(config['data']['batch_size'])
    n_batches = len(data_loader) if hasattr(data_loader, '__len__') else None
    
    pbar = tqdm(
        enumerate(data_loader),
        total=n_batches,
        desc=f'Training Epoch {epoch}'
    )
    
    # Initial memory check
    MemoryMonitor.log_memory(prefix='Training start: ')
    
    # Training loop
    optimizer.zero_grad()
    start_time = time.time()
    data_start = time.time()
    
    for batch_idx, (graphs, labels) in pbar:
        # Measure data loading time
        data_time = time.time() - data_start
        metrics['data_time'] += data_time
        
        try:
            # Training step
            loss, predictions, batch_metrics = train_step(
                model, graphs, labels, optimizer, loss_fn, config
            )
            
            # Optimize on schedule
            if (batch_idx + 1) % config['train']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['train']['clip_value']
                )
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            batch_time = time.time() - start_time
            metrics['batch_time'] += batch_time
            metrics['loss'] += loss
            for k, v in batch_metrics.items():
                if k in metrics:
                    metrics[k] += v
            
            # Update progress bar
            progress = {
                'loss': f'{loss:.4f}',
                'time': f'{batch_time:.3f}s'
            }
            if task_type == 'similarity':
                progress.update({
                    'corr': f"{batch_metrics['correlation']:.4f}",
                    'mse': f"{batch_metrics['mse']:.4f}"
                })
            else:
                progress.update({
                    'acc': f"{batch_metrics['accuracy']:.4f}",
                    # 'f1': f"{batch_metrics.get('f1', 0):.4f}"
                })
            pbar.set_postfix(progress)
            
            # Periodic cleanup
            if batch_idx % config['train']['cleanup_interval'] == 0:
                MemoryMonitor.clear_memory()
            
            # WandB logging
            if batch_idx % config['wandb']['log_interval'] == 0:
                mem_stats = MemoryMonitor.get_memory_usage()
                wandb_metrics = {
                    'memory/ram_used_gb': mem_stats['ram_used_gb'],
                    'memory/gpu_used_gb': mem_stats['gpu_used_gb'],
                    'batch/time': batch_time,
                    'batch/data_time': data_time,
                    'batch/loss': loss,
                    'batch/learning_rate': optimizer.param_groups[0]['lr'],
                    'batch': batch_idx + epoch * n_batches if n_batches else batch_idx
                }
                # Add batch metrics
                for k, v in batch_metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    wandb_metrics[f'batch/{k}'] = v
                wandb.log(wandb_metrics)
            
            start_time = time.time()
            data_start = time.time()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM in batch {batch_idx}, clearing memory...")
                torch.cuda.empty_cache()
                MemoryMonitor.clear_memory()
                optimizer.zero_grad()
                continue
            else:
                raise e

    # Final optimization step if needed
    if (batch_idx + 1) % config['train']['gradient_accumulation_steps'] != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Compute epoch metrics
    n_batches = batch_idx + 1
    metrics = {k: v / n_batches for k, v in metrics.items()}
    
    # Final memory check
    MemoryMonitor.log_memory(prefix='Training end: ')
    
    return metrics

def train_epoch_contrastive(model, dataset, optimizer, loss_fn, config, epoch):
    """Training epoch routine for contrastive learning"""
    # Get contrastive dataloader
    train_loader = dataset.get_dataloader(
        batch_size=config['data']['batch_size'],
        pos_pairs_per_anchor=config['data'].get('pos_pairs_per_anchor', 2),
        neg_pairs_per_anchor=config['data'].get('neg_pairs_per_anchor', 4),
        min_groups_per_batch=config['data'].get('min_groups_per_batch', 4),
        anchors_per_group=config['data'].get('anchors_per_group', 2),
        num_workers=config['data'].get('num_workers', 1),
        pin_memory=True
    )

    n_batches = len(train_loader) if hasattr(train_loader, '__len__') else None
    
    pbar = tqdm(
        enumerate(train_loader),
        total=n_batches,
        desc=f'Training Epoch {epoch}'
    )
    
    metrics = {
        'loss': 0.0,
        'batch_time': 0.0,
        'data_time': 0.0,
        'pos_similarity': 0.0, 
        'neg_similarity': 0.0
    }
    
    # Initial memory check
    MemoryMonitor.log_memory(prefix='Training start: ')
    
    optimizer.zero_grad()
    start_time = time.time()
    data_start = time.time()
    
    for batch_idx, (graphs, batch_info) in pbar:
        data_time = time.time() - data_start
        metrics['data_time'] += data_time
        
        try:
            # Training step
            loss, batch_metrics = train_step_infonce(
                model, graphs, batch_info, optimizer, loss_fn, config
            )
            
            # Optimize on schedule
            if (batch_idx + 1) % config['train']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['train']['clip_value']
                )
                optimizer.step()
                optimizer.zero_grad()
                
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
                'pos_sim': f"{batch_metrics['pos_similarity']:.4f}",
                'neg_sim': f"{batch_metrics['neg_similarity']:.4f}"
            }
            pbar.set_postfix(progress)
            
            # Periodic cleanup
            if batch_idx % config['train']['cleanup_interval'] == 0:
                MemoryMonitor.clear_memory()
            
            # WandB logging
            if batch_idx % config['wandb']['log_interval'] == 0:
                mem_stats = MemoryMonitor.get_memory_usage()
                wandb_metrics = {
                    'memory/ram_used_gb': mem_stats['ram_used_gb'],
                    'memory/gpu_used_gb': mem_stats['gpu_used_gb'],
                    'batch/time': batch_time,
                    'batch/data_time': data_time,
                    'batch/loss': loss,
                    'batch/learning_rate': optimizer.param_groups[0]['lr'],
                    'batch': batch_idx + epoch * n_batches if n_batches else batch_idx
                }
                # Add batch metrics
                for k, v in batch_metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    wandb_metrics[f'batch/{k}'] = v
                wandb.log(wandb_metrics)
            
            start_time = time.time()
            data_start = time.time()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM in batch {batch_idx}, clearing memory...")
                torch.cuda.empty_cache()
                MemoryMonitor.clear_memory()
                optimizer.zero_grad()
                continue
            else:
                raise e
                
    # Final optimization step if needed
    if (batch_idx + 1) % config['train']['gradient_accumulation_steps'] != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Compute epoch metrics
    n_batches = batch_idx + 1
    metrics = {k: v / n_batches for k, v in metrics.items()}
    
    # Final memory check
    MemoryMonitor.log_memory(prefix='Training end: ')
    
    return metrics


