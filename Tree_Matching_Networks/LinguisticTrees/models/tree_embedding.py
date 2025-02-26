#models/tree_embedding.py
import torch
import torch.nn as nn
from ...GMN.graphembeddingnetwork import GraphEncoder, GraphEmbeddingNet, GraphAggregator

from .tree_encoder import TreeEncoder, TreeEncoderlg

class TreeEmbeddingNet(GraphEmbeddingNet):
    def __init__(self, config):
        encoder = TreeEncoder(
            node_feature_dim=config['model']['node_feature_dim'],  # 804
            edge_feature_dim=config['model']['edge_feature_dim'],  # 22 
            hidden_dim=config['model']['node_hidden_dim'],        # 256
            dropout=config['model'].get('dropout', 0.1)
        )
        
        # Create aggregator
        aggregator = GraphAggregator(
            node_hidden_sizes=[config['model']['node_hidden_dim']],
            graph_transform_sizes=[config['model']['node_hidden_dim']],
            input_size=[config['model']['node_hidden_dim']],
            gated=True
        )

        super().__init__(
            encoder = encoder,
            aggregator = aggregator,
            node_state_dim=config['model']['node_hidden_dim'],    # 256
            edge_state_dim=config['model']['node_hidden_dim'],    # Should match node_hidden_dim
            edge_hidden_sizes=[config['model']['node_hidden_dim'] // 2,  # Reduced intermediate sizes
                             config['model']['node_hidden_dim']],  # Final size matches node_dim
            node_hidden_sizes=[config['model']['node_hidden_dim']],
            n_prop_layers=config['model']['n_prop_layers'],
            share_prop_params=True  # Add parameter sharing to reduce memory
        )
