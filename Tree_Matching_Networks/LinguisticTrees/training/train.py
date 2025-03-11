# training/train.py
from ..data.data_utils import GraphData
from ..data.batch_utils import BatchInfo
from ..data.paired_groups_dataset import PairedGroupBatchInfo
from ..data.dynamic_calculated_contrastive_dataset import get_dynamic_calculated_dataloader
from ..data.paired_groups_dataset import get_paired_groups_dataloader
from tqdm import tqdm
import wandb
from .loss import SimilarityLoss, EntailmentLoss, InfoNCELoss, TextLevelSimilarityLoss, TextLevelBinaryLoss, TextLevelEntailmentLoss, TextLevelContrastiveLoss 
import torch
import logging
import time
from ..utils.memory_utils import MemoryMonitor
from .loss_handlers import LOSS_HANDLERS as loss_handlers

logger = logging.getLogger(__name__)


def train_step_infonce(model, graphs: GraphData, batch_info: BatchInfo,
                       optimizer, loss_fn, config: dict):
    """Single training step with InfoNCE loss"""
    device = config['device']
    
    try:
        # Move data to device if needed 
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

def train_step(model, graphs: GraphData, batch_info: PairedGroupBatchInfo, 
               optimizer, loss_fn, config: dict):
    """Single training step with full gradient optimization"""
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
            
        # Enable gradient checkpointing if configured
        # if config['model'].get('gradient_checkpointing', False):
        #     model.enable_checkpointing()
            
        # Mixed precision forward pass
        # with torch.autocast(device):
        # Forward pass
        embeddings = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
        # Compute loss
        loss, predictions, metrics = loss_fn(embeddings, batch_info)
        
        # Scale loss for accumulation
        loss = loss / config['train']['gradient_accumulation_steps']
        loss.backward()

        
        # # Compute loss based on model type
        # if config['model']['task_type'] == 'similarity':
        #     x, y = outputs[::2], outputs[1::2]
        #     loss, predictions, metrics = loss_fn(x, y, labels)
        # else:  # entailment
        #     loss, predictions, metrics = loss_fn(outputs, labels)
        # 
        # # Scale loss for accumulation
        # loss = loss / config['train']['gradient_accumulation_steps']
        #     
        # # Scaled backward pass
        # scaler.scale(loss).backward()
        # 
        # # Unscale gradients for clipping
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(
        #     model.parameters(), 
        #     config['train']['clip_value']
        # )
        # 
        # # Step optimizer and update scaler
        # scaler.step(optimizer)
        # scaler.update()
        # 
        # # Clear gradients
        # optimizer.zero_grad(set_to_none=True)
        # 
        # # Move predictions to CPU and clear cache
        # predictions = predictions.cpu()
        # if config['train'].get('aggressive_cleanup', False):
        #     torch.cuda.empty_cache()
        # 
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
    dataset.reset_epoch()
    device = config['device']
    task_type = config['model']['task_type']
    task_loader_type = config['model']['task_loader_type']    
    contrastive_types = ['infonce']
    
    # Initialize loss function
    # loss_fn = TreeMatchingLoss(
    #     device=device,
    #     task_type=task_type,
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
        classifier_hidden_dims = config['model'].get("classifier_hidden_dims", [512])
    )

    # if task_type == 'similarity':
    #     loss_fn = SimilarityLoss(
    #         device = device
    #         # **config['mode'].get('loss_params', {})
    #     ).to(device, non_blocking=True)
    # elif task_type == 'info_nce':
    #     loss_fn = InfoNCELoss(
    #         device=device,
    #         temperature=config['model'].get('temperature', 0.07)
    #     )
    # elif task_type == 'similarity_aggregative':
    #     loss_fn = TextLevelSimilarityLoss(
    #         device = device,
    #         aggregation = config['model']['aggregation']
    #     )
    # elif task_type == 'contrastive_aggregative':
    #     loss_fn = TextLevelContrastiveLoss(
    #         device = device,
    #         aggregation = config['model']['aggregation']
    #     )
    # else:
    #     loss_fn = EntailmentLoss(device).to(device=device, non_blocking=True)
    

    if task_type in contrastive_types:
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
    if task_loader_type == 'aggregative':
        data_loader = get_paired_groups_dataloader(dataset, config['data']['num_workers']) 
    else:
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
    
    for batch_idx, (graphs, batch_info) in pbar:
        # Measure data loading time
        data_time = time.time() - data_start
        metrics['data_time'] += data_time
        
        try:
            # Training step
            loss, predictions, batch_metrics = train_step(
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
            
            # Update progress bar
            progress = {
                'loss': f'{loss:.4f}',
                'time': f'{batch_time:.3f}s'
            }
            if task_type == 'similarity' or task_type == 'similarity_aggregative':
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
    train_loader = get_dynamic_calculated_dataloader(
        dataset,
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
        'neg_similarity': 0.0,
        'raw_pos_sim': 0.0,
        'raw_neg_sim': 0.0
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
                'neg_sim': f"{batch_metrics['neg_similarity']:.4f}",
                # 'raw_pos_sim': f"{batch_metrics['raw_pos_sim']:.4f}",
                # 'raw_neg_sim': f"{batch_metrics['raw_neg_sim']:.4f}"
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


