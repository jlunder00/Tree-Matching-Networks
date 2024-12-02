#models/tree_matching.py
import math
from ...GMN.graphmatchingnetwork import GraphMatchingNet
from ...GMN.graphembeddingnetwork import GraphEncoder, GraphAggregator
from .tree_encoder import TreeEncoder, TreeEncoderlg
from torch.nn.utils.parametrizations import weight_norm
from torch import nn
import torch

class BaseTreeMatchingNet(GraphMatchingNet):
    """Base class with shared functionality"""
    def __init__(self, config):
        # Create encoder with consistent dimensions
        encoder = TreeEncoderlg(
            node_feature_dim=config['model']['node_feature_dim'],
            edge_feature_dim=config['model']['edge_feature_dim'],
            hidden_dim=config['model']['node_hidden_dim'],
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
            encoder=encoder,
            aggregator=aggregator,
            node_state_dim=config['model']['node_hidden_dim'],
            edge_state_dim=config['model']['node_hidden_dim'],
            edge_hidden_sizes=[
                config['model']['node_hidden_dim'] // 2,
                config['model']['node_hidden_dim']
            ],
            node_hidden_sizes=[config['model']['node_hidden_dim']],
            n_prop_layers=config['model']['n_prop_layers'],
            share_prop_params=True
        )
        
        # Gradient stabilization components
        self.layer_norm = nn.LayerNorm(config['model']['node_hidden_dim'])
        self.input_norm = nn.LayerNorm(config['model']['node_feature_dim'])
        self.output_norm = nn.LayerNorm(config['model']['node_hidden_dim'])
        self.dropout = nn.Dropout(config['model'].get('dropout', 0.1))
        
        # Training settings
        self.use_residuals = config['model'].get('use_residuals', True)
        self.checkpointing = True
        
        # self._init_weights(config['model'].get('init_scale', 0.01))
        
    # def _init_weights(self, scale=0.1):
    #     """Initialize weights with scaled Xavier/Glorot initialization"""
    #     for name, param in self.named_parameters():
    #         if 'weight' in name:
    #             if 'norm' not in name:  # Skip LayerNorm weights
    #                 nn.init.xavier_uniform_(param, gain=scale)
    #         elif 'bias' in name:
    #             nn.init.zeros_(param)
    
    def _init_weights(self, scale=0.1):
        """Initialize weights with scaled Xavier/Glorot initialization, properly handling all parameter types"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'norm' not in name:  # Skip LayerNorm weights
                    if len(param.shape) >= 2:
                        # For 2D+ tensors, use xavier
                        nn.init.xavier_uniform_(param, gain=scale)
                    else:
                        # For 1D tensors, use uniform initialization
                        bound = 1 / math.sqrt(param.shape[0])
                        nn.init.uniform_(param, -bound * scale, bound * scale)
            elif 'bias' in name:
                nn.init.zeros_(param)


    def enable_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        self.checkpointing = True
        
    def disable_checkpointing(self):
        """Disable gradient checkpointing"""
        self.checkpointing = False

class TreeMatchingNetSimilarity(BaseTreeMatchingNet):
    """Tree matching network for similarity task"""
    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config['model']['node_hidden_dim']
        self.input_dim = config['model']['node_feature_dim']
        
    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
        # Input normalization - [n_nodes, input_dim]
        node_features = self.input_norm(node_features)
        node_features = self.dropout(node_features)
        
        # Get graph vectors through base model - [n_nodes, hidden_dim]
        if self.checkpointing and self.training:
            graph_vectors = torch.utils.checkpoint.checkpoint(
                super().forward,
                node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs,
                use_reentrant=False
            )
        else:
            graph_vectors = super().forward(
                node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs
            )
        
        # Split vectors - each [n_graphs/2, hidden_dim]
        x, y = graph_vectors[::2], graph_vectors[1::2]
        
        if self.use_residuals:
            # Get node means using graph indices
            node_means = []
            for i in range(n_graphs):
                mask = graph_idx == i
                graph_nodes = node_features[mask]
                node_means.append(graph_nodes.mean(dim=0))
            node_means = torch.stack(node_means)  # [n_graphs, input_dim]
            
            # Split means for each graph pair
            node_means_x = node_means[::2]  # [batch_size, input_dim]
            node_means_y = node_means[1::2]  # [batch_size, input_dim]
            
            # Transform means to hidden dimension
            transformed_mean_x = self._encoder.node_transform(node_means_x)
            transformed_mean_y = self._encoder.node_transform(node_means_y)
            
            # Add residual connections with layer norm
            x = x + 0.1 * self.layer_norm(transformed_mean_x)
            y = y + 0.1 * self.layer_norm(transformed_mean_y)
        
        # Final normalization
        x = self.output_norm(x)
        y = self.output_norm(y)
        
        return torch.cat([x, y], dim=0)


class TreeMatchingNetEntailment(BaseTreeMatchingNet):
    """Tree matching network for entailment classification"""
    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config['model']['node_hidden_dim']
        self.input_dim = config['model']['node_feature_dim']
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model'].get('dropout', 0.1)),
            nn.Linear(self.hidden_dim, 3)
        )
        
    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
        # Input normalization - [n_nodes, input_dim]
        node_features = self.input_norm(node_features)
        node_features = self.dropout(node_features)
        
        # Get graph vectors - [n_nodes, hidden_dim]
        if self.checkpointing and self.training:
            graph_vectors = torch.utils.checkpoint.checkpoint(
                super().forward,
                node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs,
                use_reentrant=False
            )
        else:
            graph_vectors = super().forward(
                node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs
            )
            
        # Split vectors - each [n_graphs/2, hidden_dim]
        x, y = graph_vectors[::2], graph_vectors[1::2]
        
        if self.use_residuals:
            # Get node means using graph indices
            node_means = []
            for i in range(n_graphs):
                mask = graph_idx == i
                graph_nodes = node_features[mask]
                node_means.append(graph_nodes.mean(dim=0))
            node_means = torch.stack(node_means)  # [n_graphs, input_dim]
            
            # Split means for each graph pair
            node_means_x = node_means[::2]  # [batch_size, input_dim]
            node_means_y = node_means[1::2]  # [batch_size, input_dim]
            
            # Transform means to hidden dimension
            transformed_mean_x = self._encoder.node_transform(node_means_x)
            transformed_mean_y = self._encoder.node_transform(node_means_y)
            
            # Add residual connections with layer norm
            x = x + 0.1 * self.layer_norm(transformed_mean_x)
            y = y + 0.1 * self.layer_norm(transformed_mean_y)
            
        # Layer norm before combining
        x = self.output_norm(x)
        y = self.output_norm(y)
        
        # Combine embeddings for classification
        combined = torch.cat([x, y], dim=1)
        logits = self.classifier(combined)
        
        return logits

# class TreeMatchingNetSimilarity(BaseTreeMatchingNet):
#     """Tree matching network for similarity task"""
#     def __init__(self, config):
#         super().__init__(config)
#         # Store dimensions for easier access
#         self.hidden_dim = config['model']['node_hidden_dim']
#         self.input_dim = config['model']['node_feature_dim']
#         
#     def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
#         # Input normalization - [n_nodes, input_dim]
#         node_features = self.input_norm(node_features)
#         node_features = self.dropout(node_features)
#         
#         # Get graph vectors through base model - [n_nodes, hidden_dim]
#         if self.checkpointing and self.training:
#             graph_vectors = torch.utils.checkpoint.checkpoint(
#                 super().forward,
#                 node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs
#             )
#         else:
#             graph_vectors = super().forward(
#                 node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs
#             )
#         
#         # Split vectors - each [n_graphs/2, hidden_dim]
#         x, y = graph_vectors[::2], graph_vectors[1::2]
#         
#         if self.use_residuals:
#             batch_size = x.shape[0]
#             
#             # Split and reshape node features for mean
#             node_features_x = node_features[::2].view(batch_size, -1, self.input_dim)
#             node_features_y = node_features[1::2].view(batch_size, -1, self.input_dim)
#             
#             # Compute means - [batch_size, input_dim]
#             mean_x = node_features_x.mean(dim=1)
#             mean_y = node_features_y.mean(dim=1)
#             
#             # Transform means to hidden dimension
#             transformed_mean_x = self._encoder.node_transform(mean_x)
#             transformed_mean_y = self._encoder.node_transform(mean_y)
#             
#             # Add residual connections
#             x = x + 0.1 * self.layer_norm(transformed_mean_x)
#             y = y + 0.1 * self.layer_norm(transformed_mean_y)
#         
#         # Final normalization
#         x = self.output_norm(x)
#         y = self.output_norm(y)
#         
#         return torch.cat([x, y], dim=0)

# class TreeMatchingNetEntailment(BaseTreeMatchingNet):
#     """Tree matching network for entailment classification"""
#     def __init__(self, config):
#         super().__init__(config)
#         self.hidden_dim = config['model']['node_hidden_dim']
#         self.input_dim = config['model']['node_feature_dim']
#         
#         # Add classification head
#         self.classifier = nn.Sequential(
#             nn.Linear(self.hidden_dim * 2, self.hidden_dim),
#             nn.LayerNorm(self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(config['model'].get('dropout', 0.1)),
#             nn.Linear(self.hidden_dim, 3)
#         )
#         
#     def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
#         # Input normalization - [n_nodes, input_dim]
#         node_features = self.input_norm(node_features)
#         node_features = self.dropout(node_features)
#         
#         # Get graph vectors - [n_nodes, hidden_dim]
#         if self.checkpointing and self.training:
#             graph_vectors = torch.utils.checkpoint.checkpoint(
#                 super().forward,
#                 node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs
#             )
#         else:
#             graph_vectors = super().forward(
#                 node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs
#             )
#             
#         # Split vectors - each [n_graphs/2, hidden_dim]
#         x, y = graph_vectors[::2], graph_vectors[1::2]
#         
#         if self.use_residuals:
#             batch_size = x.shape[0]
#             
#             # Split and reshape node features for mean
#             node_features_x = node_features[::2].view(batch_size, -1, self.input_dim)
#             node_features_y = node_features[1::2].view(batch_size, -1, self.input_dim)
#             
#             # Compute means - [batch_size, input_dim]
#             mean_x = node_features_x.mean(dim=1)
#             mean_y = node_features_y.mean(dim=1)
#             
#             # Transform means to hidden dimension
#             transformed_mean_x = self._encoder.node_transform(mean_x)
#             transformed_mean_y = self._encoder.node_transform(mean_y)
#             
#             # Add residual connections
#             x = x + 0.1 * self.layer_norm(transformed_mean_x)
#             y = y + 0.1 * self.layer_norm(transformed_mean_y)
#         
#         # Layer norm before combining
#         x = self.output_norm(x)
#         y = self.output_norm(y)
#         
#         # Combine for classification
#         combined = torch.cat([x, y], dim=1)
#         logits = self.classifier(combined)
#         
#         return logits

# class TreeMatchingNetSimilarity(BaseTreeMatchingNet):
#     """Tree matching network for similarity task"""
#     def __init__(self, config):
#         super().__init__(config)
#         
#     def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
#         # Input normalization
#         node_features = self.input_norm(node_features)
#         node_features = self.dropout(node_features)
#         
#         if self.checkpointing and self.training:
#             graph_vectors = torch.utils.checkpoint.checkpoint(
#                 super().forward,
#                 node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs
#             )
#         else:
#             graph_vectors = super().forward(
#                 node_features,
#                 edge_features,
#                 from_idx,
#                 to_idx,
#                 graph_idx,
#                 n_graphs
#             )
#         
#         # Split vectors for residual connection
#         x, y = graph_vectors[::2], graph_vectors[1::2]
#         
#         if self.use_residuals:
#             # Add scaled residual connections
#             x = x + 0.1 * self.layer_norm(node_features[::2].mean(dim=1))
#             y = y + 0.1 * self.layer_norm(node_features[1::2].mean(dim=1))
#             
#         # Final normalization
#         x = self.output_norm(x)
#         y = self.output_norm(y)
#         
#         return torch.cat([x, y], dim=0)

# class TreeMatchingNetEntailment(BaseTreeMatchingNet):
#     """Tree matching network for entailment classification"""
#     def __init__(self, config):
#         super().__init__(config)
#         
#         # Add classification head
#         self.classifier = nn.Sequential(
#             nn.Linear(config['model']['node_hidden_dim'] * 2, 
#                      config['model']['node_hidden_dim']),
#             nn.LayerNorm(config['model']['node_hidden_dim']),
#             nn.ReLU(),
#             nn.Dropout(config['model'].get('dropout', 0.1)),
#             nn.Linear(config['model']['node_hidden_dim'], 3)
#         )
#         
#     def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
#         # Input normalization
#         node_features = self.input_norm(node_features)
#         node_features = self.dropout(node_features)
#         
#         if self.checkpointing and self.training:
#             graph_vectors = torch.utils.checkpoint.checkpoint(
#                 super().forward,
#                 node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs
#             )
#         else:
#             graph_vectors = super().forward(
#                 node_features,
#                 edge_features,
#                 from_idx,
#                 to_idx,
#                 graph_idx,
#                 n_graphs
#             )
#             
#         # Split vectors
#         x, y = graph_vectors[::2], graph_vectors[1::2]
#         
#         if self.use_residuals:
#             # Add scaled residual connections
#             x = x + 0.1 * self.layer_norm(node_features[::2].mean(dim=1))
#             y = y + 0.1 * self.layer_norm(node_features[1::2].mean(dim=1))
#             
#         # Layer norm before combining
#         x = self.output_norm(x)
#         y = self.output_norm(y)
#         
#         # Combine embeddings for classification
#         combined = torch.cat([x, y], dim=1)
#         logits = self.classifier(combined)
#         
#         return logits

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

#models/tree_matching.py
class TreeEntailmentNet(GraphMatchingNet):
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

        # Initialize base class
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

        # Add classification layers
        hidden_dim = config['model']['node_hidden_dim']
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because we concatenate both graph vectors
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['model'].get('dropout', 0.1)),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config['model'].get('dropout', 0.1)),
            nn.Linear(hidden_dim // 2, 3)  # 3 classes: contradiction, neutral, entailment
        )


    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
        # Ensure input tensors require gradients
        if self.training:
            node_features = node_features.detach().requires_grad_(True)
            edge_features = edge_features.detach().requires_grad_(True)
            
        # Get graph vectors from parent class
        graph_vectors = super().forward(
            node_features,
            edge_features,
            from_idx,
            to_idx,
            graph_idx,
            n_graphs
        )
        
        # Split vectors for each graph in the pairs
        vec1, vec2 = graph_vectors[::2], graph_vectors[1::2]
        
        # Concatenate vectors from both graphs
        paired = torch.cat([vec1, vec2], dim=1)
        
        # Pass through classification layers
        logits = self.classifier(paired)
        
        return logits

