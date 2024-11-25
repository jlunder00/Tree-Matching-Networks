#data/batch_utils.py
import torch
from typing import List, Dict
import numpy as np

def pad_sequences(sequences: List[torch.Tensor], 
                 max_len: int = None, 
                 padding_value: float = 0.0) -> torch.Tensor:
    """Pad sequence of tensors to same length"""
    max_len = max_len or max(len(s) for s in sequences)
    padded = []
    
    for seq in sequences:
        pad_size = max_len - len(seq)
        if pad_size > 0:
            padded.append(
                torch.nn.functional.pad(
                    seq, 
                    (0, 0, 0, pad_size),
                    value=padding_value
                )
            )
        else:
            padded.append(seq)
            
    return torch.stack(padded)

def create_attention_mask(lengths: List[int], max_len: int = None) -> torch.Tensor:
    """Create attention mask for variable length sequences"""
    max_len = max_len or max(lengths)
    batch_size = len(lengths)
    
    mask = torch.zeros((batch_size, max_len))
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    
    return mask

def batch_trees(trees: List[Dict], config) -> Dict:
    """Batch multiple trees together"""
    batch_size = len(trees)
    max_nodes = max(len(tree['node_features']) for tree in trees)
    
    # Initialize tensors
    node_features = []
    edge_features = []
    from_idx = []
    to_idx = []
    graph_idx = []
    masks = []
    
    node_offset = 0
    for i, tree in enumerate(trees):
        n_nodes = len(tree['node_features'])
        
        # Node features
        node_features.append(
            pad_sequences([torch.tensor(tree['node_features'])], max_nodes)[0]
        )
        
        # Edge indices
        from_idx.extend([x + node_offset for x in tree['from_idx']])
        to_idx.extend([x + node_offset for x in tree['to_idx']])
        
        # Edge features
        edge_features.extend(tree['edge_features'])
        
        # Graph index
        graph_idx.extend([i] * n_nodes)
        
        # Mask
        mask = torch.zeros(max_nodes)
        mask[:n_nodes] = 1
        masks.append(mask)
        
        node_offset += max_nodes
        
    return {
        'node_features': torch.stack(node_features),
        'edge_features': torch.tensor(edge_features),
        'from_idx': torch.tensor(from_idx),
        'to_idx': torch.tensor(to_idx),
        'graph_idx': torch.tensor(graph_idx),
        'masks': torch.stack(masks),
        'n_graphs': batch_size
    }
