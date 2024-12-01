#models/tree_encoder.py
import torch
import torch.nn as nn
from ...GMN.graphembeddingnetwork import GraphEncoder


#models/tree_encoder.py
class TreeEncoderlg(GraphEncoder):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, dropout=0.1):
        # Call parent's __init__ first
        super().__init__(
            node_feature_dim=hidden_dim,  # Using transformed dimension
            edge_feature_dim=hidden_dim,  # Using transformed dimension
            node_hidden_sizes=[hidden_dim],
            edge_hidden_sizes=[hidden_dim]
        )
        
        print(f"TreeEncoder initialized with:")
        print(f"Node feature dim: {node_feature_dim}")
        print(f"Edge feature dim: {edge_feature_dim}")
        print(f"Hidden dim: {hidden_dim}")
        
        # Now create our transforms
        self.node_transform = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # Add extra layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_transform = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # Add extra layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, node_features, edge_features=None):
        # Print input shapes
        # print(f"Input node features shape: {node_features.shape}")
        # if edge_features is not None:
        #     print(f"Input edge features shape: {edge_features.shape}")
            
        # First transform raw features to hidden dimension
        node_features = self.node_transform(node_features)
        if edge_features is not None:
            edge_features = self.edge_transform(edge_features)
            
        # Then apply base encoder operations
        node_outputs, edge_outputs = super().forward(node_features, edge_features)
        
        return node_outputs, edge_outputs

#models/tree_encoder.py
class TreeEncoder(GraphEncoder):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, dropout=0.1):
        # Call parent's __init__ first
        super().__init__(
            node_feature_dim=hidden_dim,  # Using transformed dimension
            edge_feature_dim=hidden_dim,  # Using transformed dimension
            node_hidden_sizes=[hidden_dim],
            edge_hidden_sizes=[hidden_dim]
        )
        
        print(f"TreeEncoder initialized with:")
        print(f"Node feature dim: {node_feature_dim}")
        print(f"Edge feature dim: {edge_feature_dim}")
        print(f"Hidden dim: {hidden_dim}")
        
        # Now create our transforms
        self.node_transform = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),  # Add extra layer
            # nn.LayerNorm(hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        
        self.edge_transform = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),  # Add extra layer
            # nn.LayerNorm(hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )

    def forward(self, node_features, edge_features=None):
        # Print input shapes
        # print(f"Input node features shape: {node_features.shape}")
        # if edge_features is not None:
        #     print(f"Input edge features shape: {edge_features.shape}")
            
        # First transform raw features to hidden dimension
        node_features = self.node_transform(node_features)
        if edge_features is not None:
            edge_features = self.edge_transform(edge_features)
            
        # Then apply base encoder operations
        node_outputs, edge_outputs = super().forward(node_features, edge_features)
        
        return node_outputs, edge_outputs
