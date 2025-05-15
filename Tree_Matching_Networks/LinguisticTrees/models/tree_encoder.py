# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#models/tree_encoder.py
import torch
import torch.nn as nn
from ...GMN.graphembeddingnetwork import GraphEncoder


#models/tree_encoder.py
class TreeEncoder(GraphEncoder):
    def __init__(self, node_feature_dim, edge_feature_dim, node_state_dim, edge_state_dim, dropout=0.1):
        super().__init__(
            node_feature_dim=node_feature_dim,  
            edge_feature_dim=edge_feature_dim,  
            node_hidden_sizes=[node_state_dim],
            edge_hidden_sizes=[edge_state_dim]
        )
        
        print(f"TreeEncoder initialized with:")
        print(f"Node feature dim: {node_feature_dim}")
        print(f"Edge feature dim: {edge_feature_dim}")
        print(f"Node state dim: {node_state_dim}")
        print(f"Edge state dim: {edge_state_dim}")
        
    def forward(self, node_features, edge_features=None):
        node_outputs, edge_outputs = super().forward(node_features, edge_features)
        
        return node_outputs, edge_outputs
