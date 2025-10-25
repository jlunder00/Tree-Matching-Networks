# GMN/transformer_tree_aggregator.py
"""
Transformer-Based Tree Aggregation

Replaces simple pooling aggregation with multi-headed self-attention over nodes,
using tree shape-based positional encoding.
"""

import torch
import torch.nn as nn
from .tree_shape_positional_encoder import TreeShapePositionalEncoder


class TransformerTreeAggregator(nn.Module):
    """
    Transformer-based tree aggregation with shape-aware positional encoding.

    Replaces GraphAggregator's simple pooling with multi-headed self-attention.
    Maintains same interface for drop-in compatibility.
    """

    def __init__(self,
                 node_state_dim,
                 graph_rep_dim,
                 max_nodes=64,
                 num_heads=8,
                 num_layers=2,
                 dropout=0.1,
                 positional_features=None,
                 positional_max_values=None):
        """
        Args:
            node_state_dim: Dimension of node states from graph propagation
            graph_rep_dim: Output dimension for graph representation
            max_nodes: Maximum nodes per tree (for padding/memory allocation)
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            dropout: Dropout probability
            positional_features: List of features for positional encoding
                                (None = use all available)
            positional_max_values: Dict of max values for positional features
        """
        super().__init__()

        self.node_state_dim = node_state_dim
        self.graph_rep_dim = graph_rep_dim
        self.max_nodes = max_nodes

        # Positional encoder with LEARNED embeddings
        # Initialized with sinusoidal patterns, then made learnable (like BERT)
        self.pos_encoder = TreeShapePositionalEncoder(
            embed_dim=node_state_dim,
            max_values=positional_max_values,
            features=positional_features,
            learned=True  # Use learned positional embeddings
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_state_dim,
            nhead=num_heads,
            dim_feedforward=node_state_dim * 4,  # Standard: 4x expansion
            dropout=dropout,
            batch_first=True,  # Input shape: [batch, seq, features]
            norm_first=False,  # Post-norm (standard Transformer)
            activation='relu'
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection to graph representation dimension
        # (if node_state_dim != graph_rep_dim)
        if node_state_dim != graph_rep_dim:
            self.output_projection = nn.Linear(node_state_dim, graph_rep_dim)
        else:
            self.output_projection = nn.Identity()

        # Layer norm before output
        self.output_norm = nn.LayerNorm(node_state_dim)

    def forward(self, node_states, graph_idx, n_graphs, from_idx=None, to_idx=None):
        """
        Aggregate node states to graph representations using transformer.

        Args:
            node_states: [n_total_nodes, node_state_dim] from graph propagation
            graph_idx: [n_total_nodes] which graph each node belongs to
            n_graphs: int, number of graphs in batch
            from_idx: [n_edges] edge sources (for positional encoding)
            to_idx: [n_edges] edge targets (for positional encoding)

        Returns:
            graph_embeddings: [n_graphs, graph_rep_dim]
        """
        # Compute positional encodings
        if from_idx is not None and to_idx is not None:
            pos_encodings = self.pos_encoder(from_idx, to_idx, graph_idx, n_graphs)
        else:
            # Fallback: zero positional encoding (shouldn't happen in normal use)
            pos_encodings = torch.zeros_like(node_states)

        # Add positional encoding to node states
        node_states_with_pos = node_states + pos_encodings

        # Group nodes by graph and pad to max_nodes
        batched_nodes, padding_mask = self._group_and_pad(
            node_states_with_pos, graph_idx, n_graphs
        )
        # batched_nodes: [n_graphs, max_nodes, node_state_dim]
        # padding_mask: [n_graphs, max_nodes] (True = padding position)

        # Apply transformer encoder
        # PyTorch TransformerEncoder expects padding_mask where True = ignore
        attended_nodes = self.transformer(
            batched_nodes,
            src_key_padding_mask=padding_mask
        )
        # attended_nodes: [n_graphs, max_nodes, node_state_dim]

        # Aggregate to graph representation (mean pooling over non-padded nodes)
        graph_embeddings = self._aggregate_nodes(attended_nodes, padding_mask)
        # graph_embeddings: [n_graphs, node_state_dim]

        # Normalize and project to final dimension
        graph_embeddings = self.output_norm(graph_embeddings)
        graph_embeddings = self.output_projection(graph_embeddings)
        # graph_embeddings: [n_graphs, graph_rep_dim]

        return graph_embeddings

    def _group_and_pad(self, node_states, graph_idx, n_graphs):
        """
        Group nodes by graph and pad to fixed max_nodes size.

        Returns:
            batched_nodes: [n_graphs, max_nodes, node_state_dim]
            padding_mask: [n_graphs, max_nodes] (True for padding positions)
        """
        device = node_states.device

        # Pre-allocate batched tensor
        batched_nodes = torch.zeros(
            n_graphs, self.max_nodes, self.node_state_dim,
            dtype=node_states.dtype, device=device
        )

        # Pre-allocate padding mask (True = padding)
        padding_mask = torch.ones(
            n_graphs, self.max_nodes,
            dtype=torch.bool, device=device
        )

        # Fill in actual nodes for each graph
        for g in range(n_graphs):
            # Get nodes for this graph
            mask = (graph_idx == g)
            nodes = node_states[mask]
            n_nodes = min(nodes.size(0), self.max_nodes)

            # Place nodes in batch
            batched_nodes[g, :n_nodes] = nodes[:n_nodes]

            # Mark non-padding positions as False
            padding_mask[g, :n_nodes] = False

        return batched_nodes, padding_mask

    def _aggregate_nodes(self, attended_nodes, padding_mask):
        """
        Aggregate attended nodes to single graph embedding.

        Uses mean pooling over non-padded nodes.

        Args:
            attended_nodes: [n_graphs, max_nodes, node_state_dim]
            padding_mask: [n_graphs, max_nodes] (True = padding)

        Returns:
            graph_embeddings: [n_graphs, node_state_dim]
        """
        # Mask out padding positions (set to zero)
        attended_nodes_masked = attended_nodes.masked_fill(
            padding_mask.unsqueeze(-1),  # [n_graphs, max_nodes, 1]
            0.0
        )

        # Compute mean over non-padding nodes
        # Count non-padding nodes per graph
        lengths = (~padding_mask).sum(dim=1, keepdim=True).float()  # [n_graphs, 1]

        # Sum over sequence dimension and divide by length
        graph_embeddings = attended_nodes_masked.sum(dim=1) / lengths.clamp(min=1.0)
        # graph_embeddings: [n_graphs, node_state_dim]

        return graph_embeddings

    def get_attention_weights(self, node_states, graph_idx, n_graphs,
                             from_idx=None, to_idx=None):
        """
        Get attention weights for analysis/visualization.

        Returns attention weights from the last transformer layer.
        Useful for debugging and understanding what the model learns.

        Returns:
            attention_weights: List of [n_graphs, num_heads, max_nodes, max_nodes]
                              one per transformer layer
        """
        # Compute positional encodings
        if from_idx is not None and to_idx is not None:
            pos_encodings = self.pos_encoder(from_idx, to_idx, graph_idx, n_graphs)
        else:
            pos_encodings = torch.zeros_like(node_states)

        node_states_with_pos = node_states + pos_encodings
        batched_nodes, padding_mask = self._group_and_pad(
            node_states_with_pos, graph_idx, n_graphs
        )

        # Store attention weights by registering hooks
        attention_weights = []

        def hook_fn(module, input, output):
            # TransformerEncoderLayer returns (output, None) or just output
            # We need to access attention weights from MultiheadAttention
            # This is a simplified version - full implementation would need
            # to properly extract attention weights
            pass

        # For now, return placeholder
        # Full implementation would register hooks on attention layers
        return attention_weights
