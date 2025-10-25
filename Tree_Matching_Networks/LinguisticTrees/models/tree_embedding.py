# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#models/tree_embedding.py
import torch
import torch.nn as nn
from ...GMN.graphembeddingnetwork import GraphEncoder, GraphEmbeddingNet, GraphAggregator

from .tree_encoder import TreeEncoder

class TreeEmbeddingNet(GraphEmbeddingNet):
    def __init__(self, config):
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
        encoder = TreeEncoder(
            node_feature_dim=node_feature_dim,  # 804
            edge_feature_dim=edge_feature_dim,  # 22 
            node_state_dim=node_state_dim, # 256
            edge_state_dim=edge_state_dim,
            # dropout=config['model'].get('dropout', 0.1)
        )
        
        # Create aggregator (either pooling or transformer-based)
        transformer_config = config['model']['graph'].get('transformer', {})
        use_transformer = transformer_config.get('use_transformer_aggregation', False)

        if use_transformer:
            # Use transformer-based aggregation with shape-aware positional encoding
            from ...GMN.transformer_tree_aggregator import TransformerTreeAggregator

            aggregator = TransformerTreeAggregator(
                node_state_dim=node_state_dim,
                graph_rep_dim=graph_rep_dim,
                max_nodes=transformer_config.get('max_nodes', 64),
                num_heads=transformer_config.get('num_heads', 8),
                num_layers=transformer_config.get('num_layers', 2),
                dropout=transformer_config.get('dropout', 0.1),
                positional_features=transformer_config.get('positional_features', None),
                positional_max_values=transformer_config.get('positional_max_values', None)
            )

            print(f"Using TransformerTreeAggregator:")
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
