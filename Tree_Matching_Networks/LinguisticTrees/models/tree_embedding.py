#models/tree_embedding.py
import torch
import torch.nn as nn
from ...GMN.graphembeddingnetwork import GraphEncoder

class TreeEncoder(GraphEncoder):
    """Tree-specific encoder for linguistic features"""
    
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim):
        super().__init__(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            node_hidden_sizes=[hidden_dim],
            edge_hidden_sizes=[hidden_dim]
        )
        
        # Additional tree-specific layers
        self.node_transform = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.edge_transform = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, node_features, edge_features=None):
        """Enhanced encoding for linguistic features"""
        # Initial encoding from base class
        node_outputs, edge_outputs = super().forward(node_features, edge_features)
        
        # Additional tree-specific transformations
        node_outputs = self.node_transform(node_outputs)
        if edge_outputs is not None:
            edge_outputs = self.edge_transform(edge_outputs)
            
        return node_outputs, edge_outputs
