#models/tree_matching.py
from ...GMN.graphmatchingnetwork import GraphMatchingNet
from ...GMN.graphembeddingnetwork import GraphEncoder, GraphAggregator
from .tree_encoder import TreeEncoder, TreeEncoderlg
from torch.nn.utils.parametrizations import weight_norm
from torch import nn

#models/tree_matching.py
class TreeMatchingNet(GraphMatchingNet):
    def __init__(self, config):
        # Create encoder with consistent dimensions
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

        # When initializing base class, ensure all dimensions match hidden_dim
        super().__init__(
            encoder=encoder,
            aggregator=aggregator,
            node_state_dim=config['model']['node_hidden_dim'],    # 256
            edge_state_dim=config['model']['node_hidden_dim'],    # Should match node_hidden_dim
            edge_hidden_sizes=[config['model']['node_hidden_dim'] // 2,  # Reduced intermediate sizes
                             config['model']['node_hidden_dim']],  # Final size matches node_dim
            node_hidden_sizes=[config['model']['node_hidden_dim']],
            n_prop_layers=config['model']['n_prop_layers'],
            share_prop_params=True  # Add parameter sharing to reduce memory
        )

        if config['model']['task_type'] == 'similarity':
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    weight_norm(module)

    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
        # Ensure input tensors require gradients
        if self.training:
            node_features = node_features.detach().requires_grad_(True)
            edge_features = edge_features.detach().requires_grad_(True)
            
        return super().forward(
            node_features,
            edge_features,
            from_idx,
            to_idx,
            graph_idx,
            n_graphs
        )

    #models/tree_matching.py
class TreeMatchingNetlg(GraphMatchingNet):
    def __init__(self, config):
        # Create encoder with consistent dimensions
        encoder = TreeEncoderlg(
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

        # When initializing base class, ensure all dimensions match hidden_dim
        super().__init__(
            encoder=encoder,
            aggregator=aggregator,
            node_state_dim=config['model']['node_hidden_dim'],    # 256
            edge_state_dim=config['model']['node_hidden_dim'],    # Should match node_hidden_dim
            edge_hidden_sizes=[config['model']['node_hidden_dim'] // 2,  # Reduced intermediate sizes
                             config['model']['node_hidden_dim']],  # Final size matches node_dim
            node_hidden_sizes=[config['model']['node_hidden_dim']],
            n_prop_layers=config['model']['n_prop_layers'],
            share_prop_params=True  # Add parameter sharing to reduce memory
        )

        if config['model']['task_type'] == 'similarity':
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    weight_norm(module)

    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
        # Ensure input tensors require gradients
        if self.training:
            node_features = node_features.detach().requires_grad_(True)
            edge_features = edge_features.detach().requires_grad_(True)
            
        return super().forward(
            node_features,
            edge_features,
            from_idx,
            to_idx,
            graph_idx,
            n_graphs
        )

# class TreeMatchingNet(GraphMatchingNet):
#     """Tree-specific matching network"""
#     
#     def __init__(self, config):
#         # Create encoder
#         encoder = TreeEncoder(
#             node_feature_dim=config['model']['node_feature_dim'],
#             edge_feature_dim=config['model']['edge_feature_dim'],


# class TreeMatchingNet(GraphMatchingNet):
#     """Tree-specific matching network"""
#     
#     def __init__(self, config):
#         # Create encoder
#         encoder = TreeEncoder(
#             node_feature_dim=config['model']['node_feature_dim'],
#             edge_feature_dim=config['model']['edge_feature_dim'],
#             hidden_dim=config['model']['node_hidden_dim']
#         )
#         
#         # Create aggregator with properly specified input size
#         aggregator = GraphAggregator(
#             node_hidden_sizes=[config['model']['node_hidden_dim']],
#             graph_transform_sizes=[config['model']['node_hidden_dim']],
#             input_size=[config['model']['node_hidden_dim']],  # Add this line
#             gated=True
#         )
#         
#         # Initialize base class
#         super().__init__(
#             encoder=encoder,
#             aggregator=aggregator,
#             node_state_dim=config['model']['node_hidden_dim'],
#             edge_state_dim=config['model']['edge_hidden_dim'],
#             edge_hidden_sizes=[config['model']['edge_hidden_dim']],  # Add explicit edge sizes
#             node_hidden_sizes=[config['model']['node_hidden_dim']],  # Add explicit node sizes 
#             n_prop_layers=config['model']['n_prop_layers']
#         )

    def _build_aggregator(self, config):
        """Build tree-aware aggregator"""
        return GraphAggregator(
            node_hidden_sizes=[config['model']['node_hidden_dim']],
            graph_transform_sizes=[config['model']['node_hidden_dim']],
            input_size=[config['model']['node_hidden_dim']],  # Add this line 
            gated=True
        )
