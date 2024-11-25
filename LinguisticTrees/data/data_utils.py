#data/data_utils.py
import torch
from collections import namedtuple
from typing import List, Dict

GraphData = namedtuple('GraphData', [
    'from_idx', 'to_idx', 'node_features', 'edge_features',
    'graph_idx', 'n_graphs'
])

def convert_tree_to_graph_data(tree_pairs: List[Dict], config) -> GraphData:
    """Convert TMN_DataGen tree pairs to GraphData format"""
    batch_size = len(tree_pairs)
    
    # Initialize lists for batch
    all_node_features = []
    all_edge_features = []
    all_from_idx = []
    all_to_idx = []
    all_graph_idx = []
    
    node_offset = 0
    for batch_idx, (tree1, tree2) in enumerate(tree_pairs):
        # Process first tree
        n_nodes1 = len(tree1['node_features'])
        all_node_features.extend(tree1['node_features'])
        all_edge_features.extend(tree1['edge_features'])
        
        # Adjust indices for batch
        all_from_idx.extend(
            [x + node_offset for x in tree1['from_idx']]
        )
        all_to_idx.extend(
            [x + node_offset for x in tree1['to_idx']]
        )
        all_graph_idx.extend([batch_idx * 2] * n_nodes1)
        
        # Process second tree
        node_offset += n_nodes1
        n_nodes2 = len(tree2['node_features'])
        all_node_features.extend(tree2['node_features'])
        all_edge_features.extend(tree2['edge_features'])
        
        all_from_idx.extend(
            [x + node_offset for x in tree2['from_idx']]
        )
        all_to_idx.extend(
            [x + node_offset for x in tree2['to_idx']]
        )
        all_graph_idx.extend([batch_idx * 2 + 1] * n_nodes2)
        
        node_offset += n_nodes2
    
    return GraphData(
        from_idx=torch.tensor(all_from_idx),
        to_idx=torch.tensor(all_to_idx),
        node_features=torch.tensor(all_node_features),
        edge_features=torch.tensor(all_edge_features),
        graph_idx=torch.tensor(all_graph_idx),
        n_graphs=batch_size * 2
    )
