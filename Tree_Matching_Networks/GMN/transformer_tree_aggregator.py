# GMN/transformer_tree_aggregator.py
"""
Transformer-Based Tree Aggregation

Replaces simple pooling aggregation with multi-headed self-attention over nodes,
using tree shape-based positional encoding.

Supports dimension expansion: node states from graph propagation can be projected
to a larger internal dimension before the transformer, allowing more parameters
in the aggregation stage without increasing graph propagation cost.

Supports CLS token aggregation: optionally uses a learned CLS token (prepended to
each sequence) whose output becomes the graph embedding, similar to BERT-style
sentence classification. When disabled, uses mean pooling over attended nodes.
"""

import torch
import torch.nn as nn
from .tree_shape_positional_encoder import TreeShapePositionalEncoder


class TransformerTreeAggregator(nn.Module):
    """
    Transformer-based tree aggregation with shape-aware positional encoding.

    Replaces GraphAggregator's simple pooling with multi-headed self-attention.
    Maintains same interface for drop-in compatibility.

    Supports dimension expansion via internal_dim parameter:
    - If internal_dim > node_state_dim: projects up before transformer
    - If internal_dim == node_state_dim or None: no projection (original behavior)

    Supports CLS token aggregation via use_cls_token and cls_token_type parameters:
    - use_cls_token=False: mean pooling over attended nodes (default)
    - use_cls_token=True, cls_token_type="virtual": learnable CLS token prepended to sequence
    - use_cls_token=True, cls_token_type="root": use actual tree root node as CLS
    """

    def __init__(self,
                 node_state_dim,
                 graph_rep_dim,
                 max_nodes=64,
                 num_heads=8,
                 num_layers=2,
                 dropout=0.1,
                 positional_features=None,
                 positional_max_values=None,
                 internal_dim=None,
                 use_cls_token=False,
                 cls_token_type="virtual"):
        """
        Args:
            node_state_dim: Dimension of node states from graph propagation
            graph_rep_dim: Output dimension for graph representation
            max_nodes: Maximum nodes per tree (for padding/memory allocation)
            num_heads: Number of attention heads (must divide internal_dim)
            num_layers: Number of transformer encoder layers
            dropout: Dropout probability
            positional_features: List of features for positional encoding
                                (None = use all available)
            positional_max_values: Dict of max values for positional features
            internal_dim: Internal dimension for transformer processing.
                         If None or equal to node_state_dim, no projection is used.
                         If larger, adds a linear projection for dimension expansion.
            use_cls_token: If True, use CLS token for aggregation instead of mean pooling.
            cls_token_type: Type of CLS token when use_cls_token=True:
                           - "virtual": learnable CLS token prepended to sequence with
                                       special positional encoding (depth=0, subtree_size=1,
                                       distance_to_leaf=0, etc.)
                           - "root": use actual tree root node as CLS (no virtual token,
                                    root identified by having no incoming edges)
        """
        super().__init__()

        self.node_state_dim = node_state_dim
        self.graph_rep_dim = graph_rep_dim
        self.max_nodes = max_nodes
        self.use_cls_token = use_cls_token
        self.cls_token_type = cls_token_type if use_cls_token else None

        # Validate cls_token_type
        if use_cls_token and cls_token_type not in ("virtual", "root"):
            raise ValueError(f"cls_token_type must be 'virtual' or 'root', got '{cls_token_type}'")

        # Determine internal dimension (dimension used inside transformer)
        if internal_dim is None or internal_dim == node_state_dim:
            self.internal_dim = node_state_dim
            self.use_input_projection = False
        else:
            self.internal_dim = internal_dim
            self.use_input_projection = True

        # Validate that internal_dim is divisible by num_heads
        if self.internal_dim % num_heads != 0:
            raise ValueError(
                f"internal_dim ({self.internal_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Input projection: node_state_dim -> internal_dim (single linear layer)
        # Applied per-node to project to transformer's working dimension
        if self.use_input_projection:
            self.input_projection = nn.Linear(node_state_dim, self.internal_dim)
            print(f"TransformerTreeAggregator: Using dimension expansion {node_state_dim} -> {self.internal_dim}")
        else:
            self.input_projection = nn.Identity()

        # CLS token components (only for virtual CLS)
        if self.use_cls_token and self.cls_token_type == "virtual":
            # Learnable CLS embedding prepended to each sequence
            self.cls_embedding = nn.Parameter(torch.randn(1, 1, self.internal_dim))
            nn.init.normal_(self.cls_embedding, mean=0.0, std=0.02)
            print(f"TransformerTreeAggregator: Using virtual CLS token aggregation")
        elif self.use_cls_token and self.cls_token_type == "root":
            print(f"TransformerTreeAggregator: Using root-as-CLS aggregation")

        # Positional encoder with LEARNED embeddings
        # Uses internal_dim since it's applied after input projection
        self.pos_encoder = TreeShapePositionalEncoder(
            embed_dim=self.internal_dim,
            max_values=positional_max_values,
            features=positional_features,
            learned=True  # Use learned positional embeddings
        )

        # Transformer encoder layers (operates at internal_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.internal_dim,
            nhead=num_heads,
            dim_feedforward=self.internal_dim * 4,  # Standard: 4x expansion
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
        if self.internal_dim != graph_rep_dim:
            self.output_projection = nn.Linear(self.internal_dim, graph_rep_dim)
        else:
            self.output_projection = nn.Identity()

        # Layer norm before output (at internal_dim)
        self.output_norm = nn.LayerNorm(self.internal_dim)

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
        # Project node states to internal dimension (if using dimension expansion)
        # node_states: [n_total_nodes, node_state_dim] -> [n_total_nodes, internal_dim]
        projected_states = self.input_projection(node_states)

        # Compute positional encodings (at internal_dim)
        if from_idx is not None and to_idx is not None:
            pos_encodings = self.pos_encoder(from_idx, to_idx, graph_idx, n_graphs)
        else:
            # Fallback: zero positional encoding (shouldn't happen in normal use)
            pos_encodings = torch.zeros_like(projected_states)

        # Add positional encoding to projected node states
        node_states_with_pos = projected_states + pos_encodings

        # For root-as-CLS, get root indices before grouping
        root_indices = None
        if self.use_cls_token and self.cls_token_type == "root":
            if from_idx is not None and to_idx is not None:
                root_indices = self.pos_encoder.get_root_indices(
                    from_idx, to_idx, graph_idx, n_graphs
                )

        # Group nodes by graph and pad to max_nodes
        # For virtual CLS, we prepend CLS token in _group_and_pad_with_cls
        if self.use_cls_token and self.cls_token_type == "virtual":
            batched_nodes, padding_mask = self._group_and_pad_with_cls(
                node_states_with_pos, graph_idx, n_graphs
            )
        else:
            batched_nodes, padding_mask = self._group_and_pad(
                node_states_with_pos, graph_idx, n_graphs
            )
        # batched_nodes: [n_graphs, max_nodes(+1 for virtual CLS), internal_dim]
        # padding_mask: [n_graphs, max_nodes(+1)] (True = padding position)

        # Apply transformer encoder
        # PyTorch TransformerEncoder expects padding_mask where True = ignore
        attended_nodes = self.transformer(
            batched_nodes,
            src_key_padding_mask=padding_mask
        )
        # attended_nodes: [n_graphs, max_nodes(+1), internal_dim]

        # Aggregate to graph representation
        if self.use_cls_token and self.cls_token_type == "virtual":
            # Virtual CLS: take position 0 output (the CLS token)
            graph_embeddings = attended_nodes[:, 0, :]  # [n_graphs, internal_dim]
        elif self.use_cls_token and self.cls_token_type == "root":
            # Root-as-CLS: extract root node outputs
            graph_embeddings = self._extract_root_embeddings(
                attended_nodes, padding_mask, graph_idx, n_graphs, root_indices
            )
        else:
            # Mean pooling over non-padded nodes
            graph_embeddings = self._aggregate_nodes(attended_nodes, padding_mask)
        # graph_embeddings: [n_graphs, internal_dim]

        # Normalize and project to final dimension
        graph_embeddings = self.output_norm(graph_embeddings)
        graph_embeddings = self.output_projection(graph_embeddings)
        # graph_embeddings: [n_graphs, graph_rep_dim]

        return graph_embeddings

    def _group_and_pad(self, node_states, graph_idx, n_graphs):
        """
        Group nodes by graph and pad to fixed max_nodes size.

        Args:
            node_states: [n_total_nodes, internal_dim] (already projected if using expansion)

        Returns:
            batched_nodes: [n_graphs, max_nodes, internal_dim]
            padding_mask: [n_graphs, max_nodes] (True for padding positions)
        """
        device = node_states.device
        feature_dim = node_states.size(-1)  # internal_dim after projection

        # Pre-allocate batched tensor
        batched_nodes = torch.zeros(
            n_graphs, self.max_nodes, feature_dim,
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

    def _group_and_pad_with_cls(self, node_states, graph_idx, n_graphs):
        """
        Group nodes by graph, prepend virtual CLS token, and pad to fixed size.

        The CLS token is prepended at position 0 for each graph. It receives a
        special positional encoding representing a "virtual root" of its own
        trivial tree: depth=0, subtree_size=1, distance_to_leaf=0, etc.

        Args:
            node_states: [n_total_nodes, internal_dim] (with positional encoding added)

        Returns:
            batched_nodes: [n_graphs, max_nodes + 1, internal_dim] (CLS at position 0)
            padding_mask: [n_graphs, max_nodes + 1] (True for padding positions)
        """
        device = node_states.device
        feature_dim = node_states.size(-1)

        # Sequence length is max_nodes + 1 for CLS token
        seq_len = self.max_nodes + 1

        # Pre-allocate batched tensor
        batched_nodes = torch.zeros(
            n_graphs, seq_len, feature_dim,
            dtype=node_states.dtype, device=device
        )

        # Pre-allocate padding mask (True = padding)
        padding_mask = torch.ones(
            n_graphs, seq_len,
            dtype=torch.bool, device=device
        )

        # Compute CLS positional encoding (special "virtual root" encoding)
        # This is computed once and applied to all graphs in the batch
        cls_pos_encoding = self._compute_cls_positional_encoding(device)

        # Place CLS token at position 0 for all graphs
        # CLS embedding + CLS positional encoding
        cls_with_pos = self.cls_embedding + cls_pos_encoding  # [1, 1, internal_dim]
        batched_nodes[:, 0, :] = cls_with_pos.squeeze(0).squeeze(0)  # [n_graphs, internal_dim]
        padding_mask[:, 0] = False  # CLS is never padding

        # Fill in actual nodes for each graph (starting at position 1)
        for g in range(n_graphs):
            mask = (graph_idx == g)
            nodes = node_states[mask]
            n_nodes = min(nodes.size(0), self.max_nodes)

            # Place nodes after CLS token (positions 1 to n_nodes)
            batched_nodes[g, 1:n_nodes + 1] = nodes[:n_nodes]
            padding_mask[g, 1:n_nodes + 1] = False

        return batched_nodes, padding_mask

    def _compute_cls_positional_encoding(self, device):
        """
        Compute positional encoding for virtual CLS token.

        The CLS token gets a special "virtual root" encoding:
        - depth = 0 (root-like)
        - num_siblings = 0 (no siblings)
        - num_children = 0 (no children)
        - num_grandparent_children = 0
        - subtree_size = 1 (just itself, distinguishes from real root)
        - parent_num_children = 0 (no parent)
        - distance_to_leaf = 0 (is its own leaf, distinguishes from real root)
        - nodes_at_level = 1 (only CLS at its level)

        Returns:
            cls_pos_encoding: [1, 1, internal_dim]
        """
        # Create feature values for CLS (all zeros except subtree_size=1)
        feature_values = {
            'depth': torch.tensor([0], dtype=torch.long, device=device),
            'num_siblings': torch.tensor([0], dtype=torch.long, device=device),
            'num_children': torch.tensor([0], dtype=torch.long, device=device),
            'num_grandparent_children': torch.tensor([0], dtype=torch.long, device=device),
            'subtree_size': torch.tensor([1], dtype=torch.long, device=device),
            'parent_num_children': torch.tensor([0], dtype=torch.long, device=device),
            'distance_to_leaf': torch.tensor([0], dtype=torch.long, device=device),
            'nodes_at_level': torch.tensor([1], dtype=torch.long, device=device),
        }

        # Build positional encoding using the same logic as TreeShapePositionalEncoder
        pos_encoding = torch.zeros(1, self.internal_dim, device=device)

        for feature_name in self.pos_encoder.features:
            start_dim, end_dim = self.pos_encoder.feature_dims[feature_name]
            values = feature_values.get(feature_name, torch.tensor([0], device=device))

            if self.pos_encoder.learned:
                # Use learned embedding lookup
                clamped_values = values.clamp(0, self.pos_encoder.max_values[feature_name])
                feature_enc = self.pos_encoder.feature_embeddings[feature_name](clamped_values)
            else:
                # Compute sinusoidal encoding
                feature_dim = end_dim - start_dim
                feature_enc = self.pos_encoder._sinusoidal_encoding(
                    values, feature_dim, self.pos_encoder.max_values[feature_name]
                )

            pos_encoding[:, start_dim:end_dim] = feature_enc

        return pos_encoding.unsqueeze(0)  # [1, 1, internal_dim]

    def _extract_root_embeddings(self, attended_nodes, padding_mask, graph_idx, n_graphs, root_indices):
        """
        Extract root node embeddings from attended nodes (for root-as-CLS).

        Args:
            attended_nodes: [n_graphs, max_nodes, internal_dim]
            padding_mask: [n_graphs, max_nodes] (True = padding)
            graph_idx: [n_total_nodes] which graph each node belongs to
            n_graphs: int, number of graphs
            root_indices: [n_graphs] global indices of root nodes

        Returns:
            graph_embeddings: [n_graphs, internal_dim]
        """
        device = attended_nodes.device
        graph_embeddings = torch.zeros(
            n_graphs, attended_nodes.size(-1),
            dtype=attended_nodes.dtype, device=device
        )

        # For each graph, find root's local position and extract its embedding
        for g in range(n_graphs):
            if root_indices is not None:
                # Get global root index
                global_root_idx = root_indices[g].item()

                # Find local position of root within this graph's nodes
                graph_mask = (graph_idx == g)
                graph_node_indices = torch.where(graph_mask)[0]

                # Find where global_root_idx appears in this graph's nodes
                local_positions = (graph_node_indices == global_root_idx).nonzero(as_tuple=True)[0]

                if len(local_positions) > 0:
                    local_root_pos = local_positions[0].item()
                    # Clamp to max_nodes in case of truncation
                    if local_root_pos < self.max_nodes:
                        graph_embeddings[g] = attended_nodes[g, local_root_pos]
                    else:
                        # Root was truncated, fall back to mean pooling for this graph
                        graph_embeddings[g] = self._aggregate_single_graph(
                            attended_nodes[g], padding_mask[g]
                        )
                else:
                    # Root not found, fall back to mean pooling
                    graph_embeddings[g] = self._aggregate_single_graph(
                        attended_nodes[g], padding_mask[g]
                    )
            else:
                # No root indices provided, fall back to mean pooling
                graph_embeddings[g] = self._aggregate_single_graph(
                    attended_nodes[g], padding_mask[g]
                )

        return graph_embeddings

    def _aggregate_single_graph(self, nodes, mask):
        """
        Mean pooling for a single graph (fallback for root-as-CLS edge cases).

        Args:
            nodes: [max_nodes, internal_dim]
            mask: [max_nodes] (True = padding)

        Returns:
            embedding: [internal_dim]
        """
        nodes_masked = nodes.masked_fill(mask.unsqueeze(-1), 0.0)
        length = (~mask).sum().float().clamp(min=1.0)
        return nodes_masked.sum(dim=0) / length

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
