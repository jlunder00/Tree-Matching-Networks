# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

#models/tree_embedding.py
import torch
import torch.nn as nn
from ...GMN.graphembeddingnetwork import GraphEncoder, GraphEmbeddingNet, GraphAggregator

from .tree_encoder import TreeEncoder

class TreeEmbeddingNet(GraphEmbeddingNet):
    def __init__(self, config):
        node_feature_dim = config['model']['node_feature_dim']
        edge_feature_dim = config['model']['edge_feature_dim']
        node_state_dim = config['model']['node_state_dim']
        edge_state_dim = config['model']['edge_state_dim']
        node_hidden_sizes = config['model']['node_hidden_sizes']
        node_hidden_sizes.append(node_state_dim*2)
        edge_hidden_sizes = config['model']['edge_hidden_sizes']
        edge_hidden_sizes.append(node_state_dim*2)
        graph_rep_dim = config['model']['graph_rep_dim']
        graph_transform_sizes = config['model']['graph_transform_sizes']
        edge_net_init_scale = config['model']['edge_net_init_scale']
        n_prop_layers = config['model']['n_prop_layers']
        share_prop_params = config['model']['share_prop_params']  # Add parameter sharing to reduce memory
        use_reverse_direction = config['model']['use_reverse_direction']
        reverse_dir_param_different= config['model']['reverse_dir_param_different']
        if graph_transform_sizes is None:
            graph_transform_sizes = []
        graph_transform_sizes.append(graph_rep_dim)
        encoder = TreeEncoder(
            node_feature_dim=node_feature_dim,  # 804
            edge_feature_dim=edge_feature_dim,  # 22 
            node_state_dim=node_state_dim, # 256
            edge_state_dim=edge_state_dim,
            # dropout=config['model'].get('dropout', 0.1)
        )
        
        # Create aggregator
        aggregator = GraphAggregator(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=graph_transform_sizes,
            input_size=[node_state_dim],
            gated=True
        )
        
        print(f"aggregator node hidden sizes: {[graph_rep_dim]}")
        print(f"aggregator graph transform sizes: {graph_transform_sizes}")
        print(f"aggregator input size: {[node_state_dim]}")

        print(f"model edge hidden sizes: {edge_hidden_sizes}")
        print(f"model node hidden sizes: {node_hidden_sizes}")

        super().__init__(
            encoder = encoder,
            aggregator = aggregator,
            node_state_dim=node_state_dim,    # 256
            edge_state_dim=edge_state_dim,    # Should match node_hidden_dim
            edge_hidden_sizes=edge_hidden_sizes,  # Half size first
            node_hidden_sizes=node_hidden_sizes,
            n_prop_layers=n_prop_layers,
            share_prop_params=share_prop_params,  # Add parameter sharing to reduce memory
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            # node_update_type='gru',
            edge_net_init_scale=edge_net_init_scale,
            prop_type='embedding'
        )
