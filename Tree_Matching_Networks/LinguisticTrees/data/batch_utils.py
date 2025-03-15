# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

#data/batch_utils.py
import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, NamedTuple
from torch.utils.data.dataloader import default_collate
import random
from collections import defaultdict
import queue
import threading
try:
    from .data_utils import convert_tree_to_graph_data, GraphData
except:
    from data_utils import convert_tree_to_graph_data, GraphData

logger = logging.getLogger(__name__)

@dataclass
class BatchInfo():
    """Track batch sample info for coverage tracking"""
    group_indices: List[int] # the groups in the batch
    group_ids: List[str] # The ids of each group
    anchor_indices: List[int]  # Trees used from each group
    positive_pairs: List[Tuple[int, int]]  # Positive pair indices
    negative_pairs: List[Tuple[int, int]]  # Negative pair indices

def pad_sequences(sequences: List[torch.Tensor], 
                 max_len: int = None, 
                 padding_value: float = 0.0) -> torch.Tensor:
    """Pad sequence of tensors to same length"""
    if not sequences:
        return torch.tensor([])
        
    max_len = max_len or max(len(s) for s in sequences)
    
    # Pre-allocate output tensor
    first = sequences[0]
    out_shape = [len(sequences), max_len] + list(first.shape[1:])
    out_tensor = torch.full(out_shape, padding_value, dtype=first.dtype)
    
    # Fill tensor
    for i, seq in enumerate(sequences):
        length = len(seq)
        out_tensor[i, :length] = seq
            
    return out_tensor

def create_attention_mask(lengths: List[int], max_len: int = None) -> torch.Tensor:
    """Create attention mask for variable length sequences"""
    max_len = max_len or max(lengths)
    batch_size = len(lengths)
    
    # Pre-allocate mask
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0
    
    return mask

def check_batch_limits(trees: List[Dict], config: Dict) -> bool:
    """Check if batch exceeds memory limits"""
    total_nodes = sum(len(tree['node_features']) for tree in trees)
    total_edges = sum(len(tree['from_idx']) for tree in trees)
    
    if total_nodes > config['data']['max_nodes_per_batch']:
        logger.warning(
            f"Batch with {total_nodes} nodes exceeds limit "
            f"({config['data']['max_nodes_per_batch']})"
        )
        return False
        
    if total_edges > config['data']['max_edges_per_batch']:
        logger.warning(
            f"Batch with {total_edges} edges exceeds limit "
            f"({config['data']['max_edges_per_batch']})"
        )
        return False
        
    return True

def batch_trees(trees: List[Dict], config: Dict) -> Dict:
    """Batch multiple trees together with memory limits"""
    if not check_batch_limits(trees, config):
        raise ValueError("Batch exceeds memory limits")
    
    batch_size = len(trees)
    node_counts = [len(tree['node_features']) for tree in trees]
    max_nodes = max(node_counts)
    
    try:
        # Pre-allocate tensors
        node_features = torch.zeros(
            (batch_size, max_nodes, len(trees[0]['node_features'][0])),
            dtype=torch.float32
        )
        masks = torch.zeros(batch_size, max_nodes, dtype=torch.float32)
        
        from_idx = []
        to_idx = []
        edge_features = []
        graph_idx = []
        
        node_offset = 0
        for i, (tree, n_nodes) in enumerate(zip(trees, node_counts)):
            # Node features
            node_features[i, :n_nodes] = torch.tensor(
                tree['node_features'],
                dtype=torch.float32
            )
            
            # Edge indices with offset
            from_idx.extend([x + node_offset for x in tree['from_idx']])
            to_idx.extend([x + node_offset for x in tree['to_idx']])
            
            # Edge features
            edge_features.extend(tree['edge_features'])
            
            # Graph index
            graph_idx.extend([i] * n_nodes)
            
            # Mask
            masks[i, :n_nodes] = 1
            
            node_offset += max_nodes
            
        return {
            'node_features': node_features,
            'edge_features': torch.tensor(edge_features, dtype=torch.float32),
            'from_idx': torch.tensor(from_idx, dtype=torch.long),
            'to_idx': torch.tensor(to_idx, dtype=torch.long),
            'graph_idx': torch.tensor(graph_idx, dtype=torch.long),
            'masks': masks,
            'n_graphs': batch_size
        }
        
    except RuntimeError as e:
        logger.error(f"Failed to batch trees: {str(e)}")
        logger.error(f"Batch stats: size={batch_size}, max_nodes={max_nodes}, "
                    f"total_edges={sum(len(t['from_idx']) for t in trees)}")
        raise


