"""
Unit tests for Pre-trained HuggingFace Transformer Aggregation components.

Tests PretrainedTransformerAggregator and all pretrained model variants.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Tree_Matching_Networks.GMN.pretrained_transformer_aggregator import (
    PretrainedTransformerAggregator,
    extract_encoder_from_hf_model,
)

# Default HF model for tests — small and fast to load
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_HIDDEN_DIM = 384  # all-MiniLM-L6-v2 hidden size


# ---------------------------------------------------------------------------
# Fixtures: valid tree graph data
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_tree():
    """Single graph: root with 2 children.

        0 (root)
       / \\
      1   2
    """
    return {
        "node_states_dim": 3,  # n_nodes
        "from_idx": torch.tensor([0, 0]),
        "to_idx": torch.tensor([1, 2]),
        "graph_idx": torch.tensor([0, 0, 0]),
        "n_graphs": 1,
    }


@pytest.fixture
def binary_trees_batch():
    """Batch of 3 graphs with valid binary tree edges.

    Graph 0: 10 nodes (indices 0-9), root=0
    Graph 1: 15 nodes (indices 10-24), root=10
    Graph 2: 8 nodes (indices 25-32), root=25
    """
    from_idx = []
    to_idx = []

    # Graph 0: binary tree rooted at 0
    for i in range(1, 10):
        parent = (i - 1) // 2
        from_idx.append(parent)
        to_idx.append(i)

    # Graph 1: binary tree rooted at 10
    for i in range(11, 25):
        parent = 10 + (i - 11) // 2
        from_idx.append(parent)
        to_idx.append(i)

    # Graph 2: binary tree rooted at 25
    for i in range(26, 33):
        parent = 25 + (i - 26) // 2
        from_idx.append(parent)
        to_idx.append(i)

    n_nodes = 33
    graph_idx = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(15, dtype=torch.long),
        torch.full((8,), 2, dtype=torch.long),
    ])

    return {
        "n_nodes": n_nodes,
        "from_idx": torch.tensor(from_idx),
        "to_idx": torch.tensor(to_idx),
        "graph_idx": graph_idx,
        "n_graphs": 3,
    }


@pytest.fixture
def prop_heavy_config():
    """Config dict matching prop_heavy parameter distribution."""
    return {
        "model": {
            "graph": {
                "node_feature_dim": 804,
                "edge_feature_dim": 70,
                "node_state_dim": 1280,
                "edge_state_dim": 512,
                "node_hidden_sizes": [1280, 1280],
                "edge_hidden_sizes": [512, 1024],
                "graph_rep_dim": 2048,
                "graph_transform_sizes": [1280, 2048],
                "edge_net_init_scale": 0.1,
                "n_prop_layers": 5,
                "share_prop_params": True,
                "use_reverse_direction": True,
                "reverse_dir_param_different": False,
                "pretrained_transformer": {
                    "model_name": HF_MODEL_NAME,
                    "max_nodes": 64,
                },
            },
        },
    }


@pytest.fixture
def text_config():
    """Config dict for Condition A (text mode)."""
    return {
        "model": {
            "pretrained": {
                "model_name": HF_MODEL_NAME,
            },
            "graph": {
                "graph_rep_dim": 2048,
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests: PretrainedTransformerAggregator
# ---------------------------------------------------------------------------

class TestPretrainedTransformerAggregator:
    """Tests for the core pretrained aggregator component (Issue #3)."""

    # TODO: test output shape with single graph
    def testShapeSingle(self, simple_tree):
        agg = PretrainedTransformerAggregator(
            node_state_dim=1280, graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64
        )
        tree = simple_tree
        out = agg(tree['node_states_dim'], tree['graph_idx'], 1, from_idx = tree['from_idx'], to_idx = tree['to_idx'])
        print(f'Output shape: {out.shape}')
        assert out.shape == (2, 2048)

    # TODO: test output shape with batched graphs
    # TODO: test freeze_transformer freezes encoder params only
    # TODO: test unfreeze_transformer restores requires_grad
    # TODO: test input_projection and pos_encoder stay trainable when frozen
    # TODO: test get_parameter_groups returns correct grouping
    # TODO: test get_parameter_groups excludes frozen params
    # TODO: test with CLS token (virtual)
    # TODO: test with CLS token (root)
    # TODO: test mean pooling aggregation (no CLS)
    pass


# ---------------------------------------------------------------------------
# Tests: PretrainedNoPropEmbeddingNet (Condition B)
# ---------------------------------------------------------------------------

class TestPretrainedNoPropEmbedding:
    """Tests for Condition B embedding model (Issue #4)."""

    # TODO: test forward produces correct output shape
    # TODO: test freeze/unfreeze delegation to aggregator
    pass


# ---------------------------------------------------------------------------
# Tests: PretrainedTreeEmbeddingNet (Conditions D/E/F)
# ---------------------------------------------------------------------------

class TestPretrainedTreeEmbedding:
    """Tests for Conditions D/E/F embedding model (Issue #4)."""

    # TODO: test forward produces correct output shape
    # TODO: test freeze_propagation freezes encoder + prop layers
    # TODO: test freeze_transformer freezes aggregator encoder
    # TODO: test both freezes simultaneously (Condition: frozen everything = no grad)
    pass


# ---------------------------------------------------------------------------
# Tests: PretrainedTextEmbeddingNet (Condition A)
# ---------------------------------------------------------------------------

class TestPretrainedTextEmbedding:
    """Tests for Condition A embedding model (Issue #4)."""

    # TODO: test forward with tokenized text produces correct output shape
    # TODO: test freeze/unfreeze
    pass


# ---------------------------------------------------------------------------
# Tests: Matching variants
# ---------------------------------------------------------------------------

class TestPretrainedTreeMatching:
    """Tests for Conditions D/E/F matching model (Issue #5)."""

    # TODO: test forward with even number of graphs
    # TODO: test freeze_propagation / freeze_transformer
    pass


class TestPretrainedTextMatching:
    """Tests for Condition A matching model (Issue #5)."""

    # TODO: test forward with paired text input
    pass


class TestPretrainedNoPropMatching:
    """Tests for Condition B matching model (Issue #5)."""

    # TODO: test forward produces correct output shape
    pass
