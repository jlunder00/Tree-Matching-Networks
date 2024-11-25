# LinguisticTrees/tree_matching/collate.py
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

def collate_tree_pairs(batch: List[Dict]):
    """Custom collation for tree pairs"""
    graphs1, graphs2 = [], []
    labels = []
    
    for graph_pair, label in batch:
        graphs1.append(graph_pair[0])
        graphs2.append(graph_pair[1]) 
        labels.append(label)

    # Pad node features
    max_nodes1 = max(g['node_features'].shape[0] for g in graphs1)
    max_nodes2 = max(g['node_features'].shape[0] for g in graphs2)
    
    def pad_graph_batch(graphs, max_nodes):
        node_features = []
        edge_features = []
        from_idx = []
        to_idx = []
        graph_idx = []
        masks = []
        
        for i, g in enumerate(graphs):
            n_nodes = g['node_features'].shape[0]
            pad_size = max_nodes - n_nodes
            
            # Pad node features
            node_feat_padded = torch.nn.functional.pad(
                g['node_features'], (0, 0, 0, pad_size))
            node_features.append(node_feat_padded)
            
            # Adjust indices
            from_idx.append(g['from_idx'] + i * max_nodes)
            to_idx.append(g['to_idx'] + i * max_nodes)
            
            # Create mask
            mask = torch.zeros(max_nodes)
            mask[:n_nodes] = 1
            masks.append(mask)
            
            # Edge features and graph indices
            edge_features.append(g['edge_features'])
            graph_idx.append(torch.full((n_nodes,), i))
            
        return {
            'node_features': torch.stack(node_features),
            'edge_features': torch.cat(edge_features),
            'from_idx': torch.cat(from_idx),
            'to_idx': torch.cat(to_idx),
            'graph_idx': torch.cat(graph_idx),
            'masks': torch.stack(masks)
        }
    
    batch1 = pad_graph_batch(graphs1, max_nodes1)
    batch2 = pad_graph_batch(graphs2, max_nodes2)
    labels = torch.tensor(labels)
    
    return batch1, batch2, labels
