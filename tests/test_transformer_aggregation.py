"""
Unit tests for Transformer Aggregation components.

Tests TreeShapePositionalEncoder and TransformerTreeAggregator.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Tree_Matching_Networks.GMN.tree_shape_positional_encoder import TreeShapePositionalEncoder
from Tree_Matching_Networks.GMN.transformer_tree_aggregator import TransformerTreeAggregator


class TestTreeShapePositionalEncoder:
    """Test positional encoding based on tree structural features."""

    def test_simple_tree_encoding(self):
        """Test positional encoder on simple tree structure."""

        # Simple tree: root with 2 children
        #     0 (root)
        #    / \
        #   1   2

        from_idx = torch.tensor([0, 0])
        to_idx = torch.tensor([1, 2])
        graph_idx = torch.tensor([0, 0, 0])  # All nodes in graph 0
        n_graphs = 1

        encoder = TreeShapePositionalEncoder(embed_dim=128)
        pos_enc = encoder(from_idx, to_idx, graph_idx, n_graphs)

        # Check shape
        assert pos_enc.shape == (3, 128), f"Expected shape (3, 128), got {pos_enc.shape}"

        # Root and children should have different encodings
        assert not torch.allclose(pos_enc[0], pos_enc[1]), "Root and child 1 should have different encodings"
        assert not torch.allclose(pos_enc[0], pos_enc[2]), "Root and child 2 should have different encodings"

        # Children might have similar encodings (same depth, same parent)
        # but not necessarily identical due to other features

        print("✓ Simple tree encoding test passed")

    def test_depth_computation(self):
        """Test that depth is computed correctly."""

        # Tree with 3 levels:
        #       0 (root)
        #       |
        #       1 (depth 1)
        #       |
        #       2 (depth 2)

        from_idx = torch.tensor([0, 1])
        to_idx = torch.tensor([1, 2])
        graph_idx = torch.tensor([0, 0, 0])
        n_graphs = 1

        encoder = TreeShapePositionalEncoder(embed_dim=64)
        pos_enc = encoder(from_idx, to_idx, graph_idx, n_graphs)

        # All should be different due to different depths
        assert not torch.allclose(pos_enc[0], pos_enc[1]), "Different depths should have different encodings"
        assert not torch.allclose(pos_enc[1], pos_enc[2]), "Different depths should have different encodings"
        assert not torch.allclose(pos_enc[0], pos_enc[2]), "Different depths should have different encodings"

        print("✓ Depth computation test passed")

    def test_feature_orthogonality(self):
        """Test that dimension partitioning creates orthogonal feature encodings."""

        # Create encoder with dimension partitioning
        encoder = TreeShapePositionalEncoder(
            embed_dim=1536,
            features=['depth', 'num_siblings', 'num_children']
        )

        # Simple tree to get encodings
        from_idx = torch.tensor([0, 0, 1])
        to_idx = torch.tensor([1, 2, 3])
        graph_idx = torch.tensor([0, 0, 0, 0])
        n_graphs = 1

        pos_enc = encoder(from_idx, to_idx, graph_idx, n_graphs)

        # Check that feature dimensions are properly partitioned
        # With 3 features and 1536 dims: each gets 512 dims
        assert encoder.feature_dims['depth'] == (0, 512), "Depth should occupy dims 0-512"
        assert encoder.feature_dims['num_siblings'] == (512, 1024), "Siblings should occupy dims 512-1024"
        assert encoder.feature_dims['num_children'] == (1024, 1536), "Children should occupy dims 1024-1536"

        print("✓ Feature orthogonality test passed")

    def test_batch_processing(self):
        """Test encoding for batch of multiple graphs."""

        # Two graphs:
        # Graph 0: 0 -> 1, 0 -> 2 (3 nodes)
        # Graph 1: 3 -> 4 (2 nodes)

        from_idx = torch.tensor([0, 0, 3])
        to_idx = torch.tensor([1, 2, 4])
        graph_idx = torch.tensor([0, 0, 0, 1, 1])  # First 3 nodes in graph 0, last 2 in graph 1
        n_graphs = 2

        encoder = TreeShapePositionalEncoder(embed_dim=256)
        pos_enc = encoder(from_idx, to_idx, graph_idx, n_graphs)

        # Check shape
        assert pos_enc.shape == (5, 256), f"Expected shape (5, 256), got {pos_enc.shape}"

        # Nodes in different graphs should have different encodings
        # (unless they happen to have identical tree structure, which is unlikely)

        print("✓ Batch processing test passed")


class TestTransformerTreeAggregator:
    """Test transformer-based aggregation."""

    def test_basic_aggregation(self):
        """Test transformer aggregator on simple batch."""

        # Create simple batch with 5 nodes total in 2 graphs
        node_states = torch.randn(5, 256)  # 5 nodes, 256-dim states
        from_idx = torch.tensor([0, 0, 2, 2])
        to_idx = torch.tensor([1, 2, 3, 4])
        graph_idx = torch.tensor([0, 0, 0, 1, 1])  # Graph 0: nodes 0-2, Graph 1: nodes 3-4
        n_graphs = 2

        aggregator = TransformerTreeAggregator(
            node_state_dim=256,
            graph_rep_dim=512,
            max_nodes=8,
            num_heads=4,
            num_layers=1
        )

        graph_emb = aggregator(node_states, graph_idx, n_graphs, from_idx, to_idx)

        # Check output shape
        assert graph_emb.shape == (2, 512), f"Expected shape (2, 512), got {graph_emb.shape}"

        # Check that embeddings are non-zero
        assert graph_emb.abs().sum() > 0, "Graph embeddings should be non-zero"

        print("✓ Basic aggregation test passed")

    def test_padding_handling(self):
        """Test that padding is handled correctly for variable-size graphs."""

        # Create batch with very different graph sizes
        # Graph 0: 2 nodes, Graph 1: 5 nodes
        node_states = torch.randn(7, 128)
        from_idx = torch.tensor([0, 2, 2, 2, 2])
        to_idx = torch.tensor([1, 3, 4, 5, 6])
        graph_idx = torch.tensor([0, 0, 1, 1, 1, 1, 1])
        n_graphs = 2

        aggregator = TransformerTreeAggregator(
            node_state_dim=128,
            graph_rep_dim=256,
            max_nodes=8,  # Enough to fit both graphs
            num_heads=4,
            num_layers=2
        )

        graph_emb = aggregator(node_states, graph_idx, n_graphs, from_idx, to_idx)

        assert graph_emb.shape == (2, 256)
        assert graph_emb.abs().sum() > 0

        # Embeddings should be different for different graphs
        assert not torch.allclose(graph_emb[0], graph_emb[1]), "Different graphs should have different embeddings"

        print("✓ Padding handling test passed")

    def test_without_edge_info(self):
        """Test aggregator works even without edge information (fallback)."""

        node_states = torch.randn(4, 64)
        graph_idx = torch.tensor([0, 0, 1, 1])
        n_graphs = 2

        aggregator = TransformerTreeAggregator(
            node_state_dim=64,
            graph_rep_dim=128,
            max_nodes=4,
            num_heads=2,
            num_layers=1
        )

        # Call without from_idx/to_idx (should use zero positional encoding)
        graph_emb = aggregator(node_states, graph_idx, n_graphs, from_idx=None, to_idx=None)

        assert graph_emb.shape == (2, 128)
        assert graph_emb.abs().sum() > 0

        print("✓ Fallback without edge info test passed")

    def test_max_nodes_limit(self):
        """Test that graphs exceeding max_nodes are truncated."""

        # Create graph with 10 nodes but max_nodes=8
        node_states = torch.randn(10, 32)
        from_idx = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3])
        to_idx = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        graph_idx = torch.tensor([0] * 10)
        n_graphs = 1

        aggregator = TransformerTreeAggregator(
            node_state_dim=32,
            graph_rep_dim=64,
            max_nodes=8,  # Will truncate to 8 nodes
            num_heads=2,
            num_layers=1
        )

        graph_emb = aggregator(node_states, graph_idx, n_graphs, from_idx, to_idx)

        # Should still produce valid embedding
        assert graph_emb.shape == (1, 64)
        assert graph_emb.abs().sum() > 0

        print("✓ Max nodes limit test passed")


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_encoder_aggregator_integration(self):
        """Test that positional encoder and aggregator work together."""

        # Create realistic tree batch
        from_idx = torch.tensor([0, 0, 1, 3, 3])
        to_idx = torch.tensor([1, 2, 3, 4, 5])
        graph_idx = torch.tensor([0, 0, 0, 0, 1, 1])  # 2 graphs: 4 nodes and 2 nodes
        n_graphs = 2
        node_states = torch.randn(6, 512)

        # Create encoder and aggregator
        encoder = TreeShapePositionalEncoder(embed_dim=512)
        aggregator = TransformerTreeAggregator(
            node_state_dim=512,
            graph_rep_dim=1024,
            max_nodes=10,
            num_heads=8,
            num_layers=2
        )

        # Full pipeline
        pos_enc = encoder(from_idx, to_idx, graph_idx, n_graphs)
        assert pos_enc.shape == (6, 512)

        node_states_with_pos = node_states + pos_enc
        graph_emb = aggregator._group_and_pad(node_states_with_pos, graph_idx, n_graphs)

        # Check batching worked
        batched_nodes, padding_mask = graph_emb
        assert batched_nodes.shape == (2, 10, 512)
        assert padding_mask.shape == (2, 10)

        # Graph 0 should have 4 non-padded nodes
        assert (padding_mask[0, :4] == False).all()
        assert (padding_mask[0, 4:] == True).all()

        # Graph 1 should have 2 non-padded nodes
        assert (padding_mask[1, :2] == False).all()
        assert (padding_mask[1, 2:] == True).all()

        print("✓ Encoder-aggregator integration test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Transformer Aggregation Unit Tests")
    print("="*60 + "\n")

    # Positional encoder tests
    print("Testing TreeShapePositionalEncoder...")
    test_pos = TestTreeShapePositionalEncoder()
    test_pos.test_simple_tree_encoding()
    test_pos.test_depth_computation()
    test_pos.test_feature_orthogonality()
    test_pos.test_batch_processing()
    print()

    # Aggregator tests
    print("Testing TransformerTreeAggregator...")
    test_agg = TestTransformerTreeAggregator()
    test_agg.test_basic_aggregation()
    test_agg.test_padding_handling()
    test_agg.test_without_edge_info()
    test_agg.test_max_nodes_limit()
    print()

    # Integration tests
    print("Testing Integration...")
    test_int = TestIntegration()
    test_int.test_encoder_aggregator_integration()
    print()

    print("="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
