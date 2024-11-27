#models/tree_matching.py
from ...GMN.graphmatchingnetwork import GraphMatchingNet
from ...GMN.graphembeddingnetwork import GraphEncoder, GraphAggregator
from .tree_encoder import TreeEncoder

class TreeMatchingNet(GraphMatchingNet):
    """Tree-specific matching network"""
    
    def __init__(self, config):
        encoder = TreeEncoder(
            node_feature_dim=config.model.node_feature_dim,
            edge_feature_dim=config.model.edge_feature_dim,
            hidden_dim=config.model.node_hidden_dim
        )
        
        aggregator = GraphAggregator(
            node_hidden_sizes=[config.model.node_hidden_dim],
            graph_transform_sizes=[config.model.node_hidden_dim],
            gated=True
        )
        
        super().__init__(
            encoder=encoder,
            aggregator=aggregator,
            node_state_dim=config.model.node_hidden_dim,
            edge_state_dim=config.model.edge_hidden_dim,
            n_prop_layers=config.model.n_prop_layers
        )

    def _build_aggregator(self, config):
        """Build tree-aware aggregator"""
        return GraphAggregator(
            node_hidden_sizes=[config.model.node_hidden_dim],
            graph_transform_sizes=[config.model.node_hidden_dim],
            gated=True
        )
