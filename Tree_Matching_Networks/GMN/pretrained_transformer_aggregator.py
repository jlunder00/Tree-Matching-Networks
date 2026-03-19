# Tree_Matching_Networks/GMN/pretrained_transformer_aggregator.py

import torch
import torch.nn as nn
from transformers import AutoModel
from .tree_shape_positional_encoder import TreeShapePositionalEncoder


def extract_encoder_from_hf_model(hf_model):
    """Extract transformer encoder layers from a HuggingFace model.

    Different model families store the encoder differently:
    - BERT/RoBERTa/MiniLM: model.encoder
    - DistilBERT: model.transformer
    """
    if hasattr(hf_model, 'encoder'):
        return hf_model.encoder
    elif hasattr(hf_model, 'transformer'):
        return hf_model.transformer
    else:
        raise ValueError(
            f"Cannot extract encoder from {type(hf_model).__name__}. "
            f"Expected 'encoder' or 'transformer' attribute."
        )


class PretrainedTransformerAggregator(nn.Module):
    """Drop-in replacement for TransformerTreeAggregator using pre-trained HF weights.

    Same forward interface:
        forward(node_states, graph_idx, n_graphs, from_idx=None, to_idx=None)
        → [n_graphs, graph_rep_dim]

    Architecture:
        1. input_projection: node_state_dim → hf_hidden_dim
        2. TreeShapePositionalEncoder at hf_hidden_dim
        3. Group/pad nodes per graph (+ optional CLS token)
        4. HF encoder layers (pre-trained)
        5. Aggregation (mean pooling or CLS extraction)
        6. output_projection: hf_hidden_dim → graph_rep_dim
    """

    def __init__(self,
                 node_state_dim,
                 graph_rep_dim,
                 hf_model_name,
                 max_nodes=64,
                 positional_features=None,
                 positional_max_values=None,
                 use_cls_token=False,
                 cls_token_type="virtual",
                 freeze_transformer=False):
        super().__init__()

        self.max_nodes = max_nodes
        self.use_cls_token = use_cls_token
        self.cls_token_type = cls_token_type
        self._frozen = False
        self.hf_model_name = hf_model_name

        # Load HF model and extract encoder
        hf_model = AutoModel.from_pretrained(hf_model_name)
        self.hf_hidden_dim = hf_model.config.hidden_size
        self.encoder = extract_encoder_from_hf_model(hf_model)

        # Check if this is a DistilBERT-style model (different forward interface)
        self._is_distilbert = hasattr(hf_model, 'transformer') and not hasattr(hf_model, 'encoder')

        # Clean up — we only keep the encoder
        del hf_model

        # Input projection: node_state_dim → hf_hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(node_state_dim, self.hf_hidden_dim),
            nn.LayerNorm(self.hf_hidden_dim)
        )

        # Tree positional encoding (always newly initialized, always trainable)
        self.pos_encoder = TreeShapePositionalEncoder(
            embed_dim=self.hf_hidden_dim,
            max_values=positional_max_values,
            features=positional_features,
            learned=True
        )

        # CLS token (virtual)
        if use_cls_token and cls_token_type == "virtual":
            self.cls_embedding = nn.Parameter(torch.randn(1, 1, self.hf_hidden_dim))
            nn.init.normal_(self.cls_embedding, mean=0.0, std=0.02)

        # Output projection: hf_hidden_dim → graph_rep_dim
        self.output_norm = nn.LayerNorm(self.hf_hidden_dim)
        self.output_projection = nn.Linear(self.hf_hidden_dim, graph_rep_dim)

        if freeze_transformer:
            self.freeze_transformer()

    def freeze_transformer(self):
        """Freeze pre-trained encoder parameters only."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze_transformer(self):
        """Unfreeze pre-trained encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._frozen = False

    def get_parameter_groups(self, base_lr, pretrained_lr_scale=0.1):
        """Return parameter groups with differential learning rates."""
        pretrained_params = []
        new_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('encoder.'):
                pretrained_params.append(param)
            else:
                new_params.append(param)

        groups = []
        if new_params:
            groups.append({'params': new_params, 'lr': base_lr})
        if pretrained_params:
            groups.append({'params': pretrained_params, 'lr': base_lr * pretrained_lr_scale})
        return groups

    def _group_and_pad(self, node_states, graph_idx, n_graphs):
        """Group nodes by graph and pad to max_nodes.

        Replicates TransformerTreeAggregator._group_and_pad logic.

        Returns:
            batched: [n_graphs, max_nodes_in_batch, dim]
            padding_mask: [n_graphs, max_nodes_in_batch] (True = padding)
        """
        dim = node_states.size(-1)
        device = node_states.device

        # Find actual max nodes in this batch
        counts = torch.zeros(n_graphs, dtype=torch.long, device=device)
        counts.scatter_add_(0, graph_idx, torch.ones_like(graph_idx, dtype=torch.long))
        max_nodes_in_batch = min(int(counts.max().item()), self.max_nodes)

        batched = torch.zeros(n_graphs, max_nodes_in_batch, dim, device=device)
        padding_mask = torch.ones(n_graphs, max_nodes_in_batch, dtype=torch.bool, device=device)

        for i in range(n_graphs):
            mask = (graph_idx == i)
            nodes = node_states[mask]
            n = min(nodes.size(0), max_nodes_in_batch)
            batched[i, :n] = nodes[:n]
            padding_mask[i, :n] = False

        return batched, padding_mask

    def _group_and_pad_with_cls(self, node_states, graph_idx, n_graphs):
        """Group nodes and prepend virtual CLS token at position 0."""
        dim = node_states.size(-1)
        device = node_states.device

        counts = torch.zeros(n_graphs, dtype=torch.long, device=device)
        counts.scatter_add_(0, graph_idx, torch.ones_like(graph_idx, dtype=torch.long))
        max_nodes_in_batch = min(int(counts.max().item()), self.max_nodes)

        seq_len = max_nodes_in_batch + 1  # +1 for CLS
        batched = torch.zeros(n_graphs, seq_len, dim, device=device)
        padding_mask = torch.ones(n_graphs, seq_len, dtype=torch.bool, device=device)

        # CLS token at position 0
        cls_expanded = self.cls_embedding.expand(n_graphs, -1, -1)
        batched[:, 0:1, :] = cls_expanded
        padding_mask[:, 0] = False

        # Add CLS positional encoding
        cls_pos = self._compute_cls_positional_encoding(device)
        batched[:, 0:1, :] = batched[:, 0:1, :] + cls_pos.unsqueeze(0)

        for i in range(n_graphs):
            mask = (graph_idx == i)
            nodes = node_states[mask]
            n = min(nodes.size(0), max_nodes_in_batch)
            batched[i, 1:n+1] = nodes[:n]  # Offset by 1 for CLS
            padding_mask[i, 1:n+1] = False

        return batched, padding_mask

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
            cls_pos_encoding: [1, 1, hf_hidden_data]
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
        pos_encoding = torch.zeros(1, self.hf_hidden_data, device=device)

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

        return pos_encoding.unsqueeze(0)  # [1, 1, hidden_dim]
        # """Compute positional encoding for virtual CLS token (virtual root)."""
        # return self.pos_encoder.compute_single_node_encoding(
        #     depth=0, num_siblings=0, num_children=1,
        #     num_grandparent_children=0, subtree_size=1,
        #     parent_num_children=0, distance_to_leaf=0,
        #     nodes_at_level=1, device=device
        # )

    def _aggregate_nodes(self, attended_nodes, padding_mask):
        """Mean pooling over non-padded positions."""
        attended_nodes = attended_nodes.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        lengths = (~padding_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        return attended_nodes.sum(dim=1) / lengths

    def _extract_root_embeddings(self, attended_nodes, padding_mask, graph_idx, n_graphs, root_indices):
        """Extract root node embeddings for root-as-CLS mode."""
        if root_indices is None:
            return self._aggregate_nodes(attended_nodes, padding_mask)

        graph_embeddings = torch.zeros(n_graphs, attended_nodes.size(-1), device=attended_nodes.device)

        for i in range(n_graphs):
            if i < len(root_indices) and root_indices[i] >= 0:
                # Convert global root index to local position within padded sequence
                mask = (graph_idx == i)
                graph_nodes = mask.nonzero(as_tuple=True)[0]
                root_global = root_indices[i]
                root_local = (graph_nodes == root_global).nonzero(as_tuple=True)[0]
                if len(root_local) > 0 and root_local[0] < attended_nodes.size(1):
                    graph_embeddings[i] = attended_nodes[i, root_local[0]]
                else:
                    graph_embeddings[i] = self._aggregate_nodes(
                        attended_nodes[i:i+1], padding_mask[i:i+1]
                    ).squeeze(0)
            else:
                graph_embeddings[i] = self._aggregate_nodes(
                    attended_nodes[i:i+1], padding_mask[i:i+1]
                ).squeeze(0)

        return graph_embeddings

    def forward(self, node_states, graph_idx, n_graphs, from_idx=None, to_idx=None):
        """Same interface as TransformerTreeAggregator.forward()."""

        # 1. Project node states to HF hidden dim
        projected = self.input_projection(node_states)

        # 2. Add tree positional encoding
        if from_idx is not None and to_idx is not None:
            pos_enc = self.pos_encoder(from_idx, to_idx, graph_idx, n_graphs)
            projected = projected + pos_enc

        # 3. Get root indices for root-as-CLS
        root_indices = None
        if self.use_cls_token and self.cls_token_type == "root":
            if from_idx is not None and to_idx is not None:
                root_indices = self.pos_encoder.get_root_indices(from_idx, to_idx, graph_idx, n_graphs)

        # 4. Group and pad
        if self.use_cls_token and self.cls_token_type == "virtual":
            batched, padding_mask = self._group_and_pad_with_cls(projected, graph_idx, n_graphs)
        else:
            batched, padding_mask = self._group_and_pad(projected, graph_idx, n_graphs)

        # 5. Run through HF encoder
        # Convert padding mask to HF attention mask format
        hf_attention_mask = (~padding_mask).float()  # 1=attend, 0=ignore

        if self._is_distilbert:
            # DistilBERT forward: (x, attn_mask) → (hidden_states,)
            encoder_output = self.encoder(batched, attn_mask=hf_attention_mask)
            attended = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output.last_hidden_state
        else:
            # BERT-style forward: needs extended attention mask [batch, 1, 1, seq_len]
            extended_mask = hf_attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * torch.finfo(batched.dtype).min
            encoder_output = self.encoder(batched, attention_mask=extended_mask)
            attended = encoder_output.last_hidden_state if hasattr(encoder_output, 'last_hidden_state') else encoder_output[0]

        # 6. Aggregate
        if self.use_cls_token and self.cls_token_type == "virtual":
            graph_embeddings = attended[:, 0, :]
        elif self.use_cls_token and self.cls_token_type == "root":
            graph_embeddings = self._extract_root_embeddings(attended, padding_mask, graph_idx, n_graphs, root_indices)
        else:
            graph_embeddings = self._aggregate_nodes(attended, padding_mask)

        # 7. Normalize and project to graph_rep_dim
        graph_embeddings = self.output_norm(graph_embeddings)
        return self.output_projection(graph_embeddings)

