# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#models/tree_matching.py
import math
from ...GMN.graphmatchingnetwork import GraphMatchingNet
from ...GMN.graphembeddingnetwork import GraphEncoder, GraphAggregator
from .tree_encoder import TreeEncoder
from torch.nn.utils.parametrizations import weight_norm
from torch import nn
import torch

class TreeMatchingNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        node_feature_dim = config['model']['graph']['node_feature_dim']
        edge_feature_dim = config['model']['graph']['edge_feature_dim']
        node_state_dim = config['model']['graph']['node_state_dim']
        edge_state_dim = config['model']['graph']['edge_state_dim']
        node_hidden_sizes = config['model']['graph']['node_hidden_sizes']
        node_hidden_sizes.append(node_state_dim*2)
        edge_hidden_sizes = config['model']['graph']['edge_hidden_sizes']
        edge_hidden_sizes.append(node_state_dim*2)
        graph_rep_dim = config['model']['graph']['graph_rep_dim']
        graph_transform_sizes = config['model']['graph']['graph_transform_sizes']
        edge_net_init_scale = config['model']['graph']['edge_net_init_scale']
        n_prop_layers = config['model']['graph']['n_prop_layers']
        share_prop_params = config['model']['graph']['share_prop_params']  # Add parameter sharing to reduce memory
        use_reverse_direction = config['model']['graph']['use_reverse_direction']
        reverse_dir_param_different= config['model']['graph']['reverse_dir_param_different']
        if graph_transform_sizes is None:
            graph_transform_sizes = []
        graph_transform_sizes.append(graph_rep_dim)
        
        attention_config = config['model']['graph'].get('attention', {})
        use_message_attention = attention_config.get('use_message_attention', False)
        use_aggregation_attention = attention_config.get('use_aggregation_attention', False)
        use_node_update_attention = attention_config.get('use_node_update_attention', False)
        use_graph_attention = attention_config.get('use_graph_attention', False)
        attention_heads = attention_config.get('attention_heads', 4)

        encoder = TreeEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            node_state_dim=node_state_dim, # 256
            edge_state_dim=edge_state_dim
            # dropout=config['model'].get('dropout', 0.1)
        )
        
        aggregator = GraphAggregator(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=graph_transform_sizes,
            input_size=[node_state_dim],
            gated=True
        )
        if any([use_message_attention, use_aggregation_attention, 
                use_node_update_attention, use_graph_attention]):
            from ...GMN.graphmatchingnetwork import AttentionGraphMatchingNet
            self.gmn = AttentionGraphMatchingNet(
                encoder=encoder, aggregator=aggregator,
                node_state_dim=node_state_dim, edge_state_dim=edge_state_dim,
                edge_hidden_sizes=edge_hidden_sizes, node_hidden_sizes=node_hidden_sizes,
                n_prop_layers=n_prop_layers, share_prop_params=share_prop_params,
                edge_net_init_scale=edge_net_init_scale, use_reverse_direction=use_reverse_direction,
                reverse_dir_param_different=reverse_dir_param_different, prop_type='matching',
                # Attention parameters
                use_message_attention=use_message_attention,
                use_aggregation_attention=use_aggregation_attention,
                use_node_update_attention=use_node_update_attention,
                use_graph_attention=use_graph_attention,
                attention_heads=attention_heads
            )
        else:
            from ...GMN.graphmatchingnetwork import GraphMatchingNet
            self.gmn = GraphMatchingNet(
                encoder=encoder, aggregator=aggregator,
                node_state_dim=node_state_dim, edge_state_dim=edge_state_dim,
                edge_hidden_sizes=edge_hidden_sizes, node_hidden_sizes=node_hidden_sizes,
                n_prop_layers=n_prop_layers, share_prop_params=share_prop_params,
                edge_net_init_scale=edge_net_init_scale, use_reverse_direction=use_reverse_direction,
                reverse_dir_param_different=reverse_dir_param_different, prop_type='matching'
            )
    
    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
        return self.gmn(node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs)
