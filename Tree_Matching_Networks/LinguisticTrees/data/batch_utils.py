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

class ContrastiveBatchCollator:
    """Collates trees into batches for contrastive learning"""
    
    def __init__(self,
                 pos_pairs_per_anchor: int = 2,
                 neg_pairs_per_anchor: int = 4,
                 min_groups_per_batch: int = 4,
                 anchors_per_group: int = 2):
        self.pos_per_anchor = pos_pairs_per_anchor
        self.neg_per_anchor = neg_pairs_per_anchor
        self.min_groups = min_groups_per_batch
        self.anchors_per_group = anchors_per_group
        
    def __call__(self, batch: List[Dict]) -> Tuple[Dict, BatchInfo]:
        """Create batch with positive and negative pairs
        
        Args:
            batch: List of dicts from dataset __getitem__
            
        Returns:
            graphs: GraphData containing all trees
            batch_info: Info about batch composition
        """
        # Group trees by group_idx
        groups = defaultdict(list)
        group_ids = {}  # Map group_idx to group_id
        for item in batch:
            groups[item['group_idx']].append(item)
            group_ids[item['group_idx']] = item['group_id']
            
        # Ensure minimum number of groups
        if len(groups) < self.min_groups:
            raise ValueError(f"Batch must contain at least {self.min_groups} groups")
            
        # Collect trees and track pairs
        all_trees = []
        tree_map = {}  # Map (group_idx, tree_idx) to position in all_trees
        positive_pairs = []
        negative_pairs = []
        anchor_indices = []
        
        # Process each group
        for group_idx, items in groups.items():
            # Sample anchor trees from group
            n_anchors = min(len(items), self.anchors_per_group)
            anchor_items = random.sample(items, n_anchors)
            
            # Add anchors
            for anchor_item in anchor_items:
                anchor_pos = len(all_trees)
                all_trees.append(anchor_item['tree'])
                tree_map[(group_idx, anchor_item['tree_idx'])] = anchor_pos
                anchor_indices.append(anchor_pos)
                
                # Sample positive pairs
                available_pos = [i for i in items if i['tree']['text'] != anchor_item['tree']['text']]
                if available_pos:
                    pos_items = random.sample(
                        available_pos,
                        min(len(available_pos), self.pos_per_anchor)
                    )
                    
                    # Add positive pairs
                    for pos_item in pos_items:
                        pos_pos = len(all_trees)
                        all_trees.append(pos_item['tree'])
                        tree_map[(group_idx, pos_item['tree_idx'])] = pos_pos
                        positive_pairs.append((anchor_pos, pos_pos))
                        
        # Create negative pairs across groups
        for anchor_pos in anchor_indices:
            anchor_group = None
            # Find anchor's group
            for (g_idx, t_idx), pos in tree_map.items():
                if pos == anchor_pos:
                    anchor_group = g_idx
                    break
                    
            # Get trees from other groups
            other_trees = []
            for (g_idx, t_idx), pos in tree_map.items():
                if g_idx != anchor_group:
                    other_trees.append(pos)
                    
            # Sample negative pairs
            if other_trees:
                neg_indices = random.sample(
                    other_trees,
                    min(len(other_trees), self.neg_per_anchor)
                )
                for neg_idx in neg_indices:
                    negative_pairs.append((anchor_pos, neg_idx))
                    
        # Convert trees to graph format
        graphs = convert_tree_to_graph_data(all_trees)
        
        # Create batch info
        batch_info = BatchInfo(
            group_indices=list(groups.keys()),
            group_ids=[group_ids[idx] for idx in groups.keys()],
            anchor_indices=anchor_indices,
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs
        )
        
        return graphs, batch_info

# class BatchCollator:
#     """Collate batches for dataloader"""
#     
#     def __init__(self, dataset):
#         self.dataset = dataset
#         
#     def __call__(self, batch: List[TreeGroup]) -> Tuple[GraphData, torch.Tensor]:
#         """Collate a batch of tree groups
#         
#         Returns:
#             graphs: GraphData containing all trees
#             labels: Tensor of labels:
#                    1 for pairs from same group
#                    0 for pairs from different groups
#         """
#         all_pairs = []
#         all_labels = []
#         
#         # Create positive pairs within groups
#         for group in batch:
#             n_trees = len(group.trees1)
#             for i in range(n_trees):
#                 for j in range(i + 1, n_trees):
#                     if group.trees1[i]['text'] != group.trees2[j]['text']:
#                         all_pairs.append((group.trees1[i], group.trees2[j]))
#                         all_labels.append(1)
#                         
#         # Create negative pairs across groups
#         for i in range(len(batch)):
#             for j in range(i + 1, len(batch)):
#                 group1, group2 = batch[i], batch[j]
#                 
#                 # Sample a few trees from each group
#                 trees1 = random.sample(group1.trees1, min(2, len(group1.trees1)))
#                 trees2 = random.sample(group2.trees2, min(2, len(group2.trees2)))
#                 
#                 for tree1 in trees1:
#                     for tree2 in trees2:
#                         all_pairs.append((tree1, tree2))
#                         all_labels.append(0)
#                         
#         # Convert to tensor format
#         graphs = convert_tree_to_graph_data([p[0] for p in all_pairs] + [p[1] for p in all_pairs])
#         labels = torch.tensor(all_labels, dtype=torch.float32)
#         
#         return graphs, labels

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


