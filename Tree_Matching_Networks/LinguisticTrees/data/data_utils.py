# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#data/data_utils.py
import torch
from collections import namedtuple
import math
from typing import List, Dict

GraphData = namedtuple('GraphData', [
    'from_idx', 'to_idx', 'node_features', 'edge_features',
    'graph_idx', 'n_graphs'
])

def convert_tree_to_graph_data(trees: List[Dict]) -> GraphData:
    """Convert trees to GraphData format"""
    all_node_features = []
    all_edge_features = []
    all_from_idx = []
    all_to_idx = []
    all_graph_idx = []
    
    node_offset = 0
    for graph_idx, tree in enumerate(trees):
        n_nodes = len(tree['node_features'])
        
        # Features are already tensors
        all_node_features.append(tree['node_features'])
        all_edge_features.extend(tree['edge_features'])
        
        # Adjust indices
        all_from_idx.extend([x + node_offset for x in tree['from_idx']])
        all_to_idx.extend([x + node_offset for x in tree['to_idx']])
        all_graph_idx.extend([graph_idx] * n_nodes)
        
        node_offset += n_nodes
        
    return GraphData(
        from_idx=torch.tensor(all_from_idx),
        to_idx=torch.tensor(all_to_idx),
        node_features=torch.cat(all_node_features),
        edge_features=torch.tensor(all_edge_features),
        graph_idx=torch.tensor(all_graph_idx),
        n_graphs=len(trees)
    )

def get_min_groups_trees_per_group(anchors_per_group, positive_pair_trees_per_group, batch_size, ceil=False):
    '''
    Given the number of anchors per group, and the number of non anchor items taken from each group,
    find the required number of groups to satisfy batch_size pairs.

    quadratic equation: a * (a+b) * g^2 - a*g - batch_size = 0
    '''

    a, b = anchors_per_group, positive_pair_trees_per_group
    
    discriminant = a**2 + 4 * a * (a + b) * batch_size

    #take only the positive solution
    g = (a + math.sqrt(discriminant))/(2 * a * (a + b))

    g = math.floor(g) if not ceil else math.ceil(g)
    total_pairs = a * (a+b) * g^2 - a*g
    return g, total_pairs

def get_min_groups_pairs_per_anchor(anchors_per_group, positive_pairs_per_anchor, batch_size, ceil=False):
    '''
    Given the number of anchors per group, the number of positive pairs allowed per group,
    find the required number of groups to satisfy batch_size pairs

    a(a+b)g^2 - a^2 * g - batch_size = 0
    or rather
    T(g) = total pairs in terms of groups = a(a+b)g^2 - a^2 * g
    '''
    
    a, b = anchors_per_group, positive_pairs_per_anchor

    discriminant = a**4 + 4 * a * (a + b) * batch_size

    g = (a**2 + math.sqrt(discriminant)) / (2 * a * (a + b))

    g = math.floor(g) if not ceil else math.ceil(g) #set ceil to true to include potentially more pairs than original batch size
    total_pairs = a*(a+b)*(g**2) - (a**2)*g #now find the adjusted total pairs generated from this many groups
    return g, total_pairs

def normalize(x, old_min, old_max, new_min, new_max):
    return new_min + ((x - old_min) / (old_max - old_min)) * (new_max - new_min)



