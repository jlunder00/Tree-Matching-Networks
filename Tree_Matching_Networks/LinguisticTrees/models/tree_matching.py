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

        # Copy lists from config (don't mutate the config dictionary)
        # Then append the final expansion layer: node_state_dim * 2
        # This creates the compress → expand → compress architecture pattern
        node_hidden_sizes = list(config['model']['graph']['node_hidden_sizes'])
        node_hidden_sizes.append(node_state_dim * 2)
        edge_hidden_sizes = list(config['model']['graph']['edge_hidden_sizes'])
        edge_hidden_sizes.append(node_state_dim * 2)
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

        # Create aggregator (either pooling or transformer-based)
        transformer_config = config['model']['graph'].get('transformer', {})
        use_transformer = transformer_config.get('use_transformer_aggregation', False)

        if use_transformer:
            # Use transformer-based aggregation with shape-aware positional encoding
            from ...GMN.transformer_tree_aggregator import TransformerTreeAggregator

            # Get internal_dim for dimension expansion (None = use node_state_dim)
            internal_dim = transformer_config.get('internal_dim', None)

            aggregator = TransformerTreeAggregator(
                node_state_dim=node_state_dim,
                graph_rep_dim=graph_rep_dim,
                max_nodes=transformer_config.get('max_nodes', 64),
                num_heads=transformer_config.get('num_heads', 8),
                num_layers=transformer_config.get('num_layers', 2),
                dropout=transformer_config.get('dropout', 0.1),
                positional_features=transformer_config.get('positional_features', None),
                positional_max_values=transformer_config.get('positional_max_values', None),
                internal_dim=internal_dim
            )

            print(f"Using TransformerTreeAggregator:")
            print(f"  node_state_dim: {node_state_dim}")
            print(f"  internal_dim: {internal_dim if internal_dim else node_state_dim} (no expansion)" if not internal_dim else f"  internal_dim: {internal_dim} (dimension expansion)")
            print(f"  max_nodes: {transformer_config.get('max_nodes', 64)}")
            print(f"  num_heads: {transformer_config.get('num_heads', 8)}")
            print(f"  num_layers: {transformer_config.get('num_layers', 2)}")
            print(f"  positional_features: {transformer_config.get('positional_features', 'all')}")
        else:
            # Use standard pooling-based aggregation (default)
            aggregator = GraphAggregator(
                node_hidden_sizes=[graph_rep_dim],
                graph_transform_sizes=graph_transform_sizes,
                input_size=[node_state_dim],
                gated=True
            )

            print(f"Using GraphAggregator (pooling):")
            print(f"  node hidden sizes: {[graph_rep_dim]}")
            print(f"  graph transform sizes: {graph_transform_sizes}")
            print(f"  input size: {[node_state_dim]}")
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
