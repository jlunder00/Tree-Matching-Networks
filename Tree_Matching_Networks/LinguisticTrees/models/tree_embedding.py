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
            node_state_dim=config['model']['node_state_dim'], # 256
            edge_state_dim=config['model']['edge_state_dim'],
            dropout=config['model'].get('dropout', 0.1)
        )
        
        # Create aggregator
        aggregator = GraphAggregator(
            node_hidden_sizes=[config['model']['graph_rep_dim']],
            graph_transform_sizes=[config['model']['graph_rep_dim']],
            input_size=[config['model']['node_state_dim']],
            gated=True
        )
        print(f"aggregator node hidden sizes: {[config['model']['graph_rep_dim']]}")
        print(f"aggregator graph transform sizes: {[config['model']['graph_rep_dim']]}")
        print(f"aggregator input size: {[config['model']['node_state_dim']]}")

        print(f"model edge hidden sizes: {[config['model']['edge_state_dim'], config['model']['edge_state_dim']*2]}")
        print(f"model node hidden sizes: {[config['model']['node_state_dim']*2+config['model']['edge_state_dim']]}")

        super().__init__(
            encoder = encoder,
            aggregator = aggregator,
            node_state_dim=config['model']['node_state_dim'],    # 256
            edge_state_dim=config['model']['edge_state_dim'],    # Should match node_hidden_dim
            # edge_state_dim = 0,    # Should match node_hidden_dim
            edge_hidden_sizes=[config['model']['edge_state_dim'], config['model']['edge_state_dim']*2,  # Half size first
                             config['model']['node_state_dim']*2],
            node_hidden_sizes=[config['model']['node_state_dim'], config['model']['node_state_dim']*2],
            n_prop_layers=config['model']['n_prop_layers'],
            share_prop_params=True,  # Add parameter sharing to reduce memory
            use_reverse_direction=False,
            reverse_dir_param_different=True,
            node_update_type='gru',
            edge_net_init_scale=0.1,
            prop_type='embedding'
        )
