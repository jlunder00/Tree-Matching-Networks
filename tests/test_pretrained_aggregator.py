"""
Unit tests for Pre-trained HuggingFace Transformer Aggregation components.

Tests PretrainedTransformerAggregator and all pretrained model variants.
"""

import torch
import pytest
import sys
from pathlib import Path
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Tree_Matching_Networks.GMN.pretrained_transformer_aggregator import (
    PretrainedTransformerAggregator,
    extract_encoder_from_hf_model,
)
from Tree_Matching_Networks.LinguisticTrees.models.pretrained_noprop_embedding import PretrainedNoPropEmbeddingNet
from Tree_Matching_Networks.LinguisticTrees.models.pretrained_noprop_matching import PretrainedNoPropMatchingNet
from Tree_Matching_Networks.LinguisticTrees.models.pretrained_tree_embedding import PretrainedTreeEmbeddingNet
from Tree_Matching_Networks.LinguisticTrees.models.pretrained_tree_matching import PretrainedTreeMatchingNet
from Tree_Matching_Networks.LinguisticTrees.models.pretrained_text_embedding import PretrainedTextEmbeddingNet
from Tree_Matching_Networks.LinguisticTrees.models.pretrained_text_matching import PretrainedTextMatchingNet

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
        "node_state_dim": 1280,
        "n_nodes": 3,  # n_nodes
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
        "node_state_dim": 1280,
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

def get_node_states(n_nodes, node_state_dim):
    return torch.randn(n_nodes, node_state_dim)


# ---------------------------------------------------------------------------
# Tests: PretrainedTransformerAggregator
# ---------------------------------------------------------------------------

class TestPretrainedTransformerAggregator:
    """Tests for th we core pretrained aggregator component (Issue #3)."""

    # test output shape with single graph
    def testShapeSingle(self, simple_tree):
        tree = simple_tree
        agg = PretrainedTransformerAggregator(
            node_state_dim=tree['node_state_dim'], graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64
        )

        
        out = agg(get_node_states(tree['n_nodes'], tree['node_state_dim']), tree['graph_idx'], 1, from_idx = tree['from_idx'], to_idx = tree['to_idx'])
        print(f'Output shape: {out.shape}')
        assert out.shape == (1, 2048)

    # TODO: test output shape with batched graphs
    def testShapeBatch(self, binary_trees_batch):
        agg = PretrainedTransformerAggregator(
            node_state_dim=binary_trees_batch['node_state_dim'], graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64
        )

        
        out = agg(get_node_states(binary_trees_batch['n_nodes'], binary_trees_batch['node_state_dim']), binary_trees_batch['graph_idx'], binary_trees_batch['n_graphs'], from_idx = binary_trees_batch['from_idx'], to_idx = binary_trees_batch['to_idx'])
        print(f'Output shape: {out.shape}')
        assert out.shape == (3, 2048)

    # TODO: test freeze_transformer freezes encoder params only
    # TODO: test input_projection and pos_encoder stay trainable when frozen
    def testFreezeTransformerFreezesEncoderOnly(self, simple_tree):
        agg = PretrainedTransformerAggregator(
            node_state_dim=simple_tree['node_state_dim'], graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64,
            freeze_transformer=True,
            use_cls_token=True
        )
        
        param_tracker = {
			r'^input_projection\.': {
				'should_train': True,
				'seen': False
			},
			r'^pos_encoder\.': {
				'should_train': True,
				'seen': False
			},
			r'^output_projection\.': {
				'should_train': True,
				'seen': False
			},
			r'^output_norm\.': {
				'should_train': True,
				'seen': False
			},
			r'^cls_embedding$': {
				'should_train': True,
				'seen': False
			},
			r'^encoder\.': {
				'should_train': False,
				'seen': False
			}
        }
        for name, param in agg.named_parameters():
            for pattern, data in param_tracker.items():
                if re.match(pattern, name):
                    data['seen'] = True
                    t = param.requires_grad
                    print(f'Parameter {name} is{"" if t else " not"} trainable')
                    assert t == data['should_train'] 

        assert all([v['seen'] for _, v in param_tracker.items()])

    # TODO: test unfreeze_transformer restores requires_grad
    def testUnfreezeTransformerRestoresTrainability(self, simple_tree):
        agg = PretrainedTransformerAggregator(
            node_state_dim=simple_tree['node_state_dim'], graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64,
            freeze_transformer=True,
            use_cls_token=True
        )

        agg.unfreeze_transformer()
        
        param_tracker = {
			r'^input_projection\.': {
				'should_train': True,
				'seen': False
			},
			r'^pos_encoder\.': {
				'should_train': True,
				'seen': False
			},
			r'^output_projection\.': {
				'should_train': True,
				'seen': False
			},
			r'^output_norm\.': {
				'should_train': True,
				'seen': False
			},
			r'^cls_embedding$': {
				'should_train': True,
				'seen': False
			},
			r'^encoder\.': {
				'should_train': True,
				'seen': False
			}
        }
        for name, param in agg.named_parameters():
            for pattern, data in param_tracker.items():
                if re.match(pattern, name):
                    data['seen'] = True
                    t = param.requires_grad
                    print(f'Parameter {name} is{"" if t else " not"} trainable')
                    assert t == data['should_train'] 

        assert all([v['seen'] for _, v in param_tracker.items()])

    # TODO: test get_parameter_groups returns correct grouping
    def testParameterGroups(self, simple_tree):
        agg = PretrainedTransformerAggregator(
            node_state_dim=simple_tree['node_state_dim'], graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64,
            use_cls_token=True
        )

        groups = agg.get_parameter_groups(base_lr=1e-4, pretrained_lr_scale=0.1)
        assert len(groups) == 2
        assert groups[0].get('lr', 0) == 1e-4
        assert groups[1].get('lr', 0) == 1e-5



    # TODO: test get_parameter_groups excludes frozen params
    def testFrozenParamGroups(self, simple_tree):
        agg = PretrainedTransformerAggregator(
            node_state_dim=simple_tree['node_state_dim'], graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64,
            freeze_transformer=True,
            use_cls_token=True
        )
        
        groups = agg.get_parameter_groups(base_lr=1e-4, pretrained_lr_scale=0.1)
        assert len(groups) == 1
        assert groups[0].get('lr', 0) == 1e-4


    # TODO: test with CLS token (virtual)
    def testCLSVirtual(self, simple_tree):
        tree = simple_tree
        agg = PretrainedTransformerAggregator(
            node_state_dim=tree['node_state_dim'], graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64,
            use_cls_token=True,
            cls_token_type="virtual"
        )

        
        out = agg(get_node_states(tree['n_nodes'], tree['node_state_dim']), tree['graph_idx'], 1, from_idx = tree['from_idx'], to_idx = tree['to_idx'])
        print(f'Output shape: {out.shape}')
        assert out.shape == (1, 2048)

    # TODO: test with CLS token (root)
    def testCLSRoot(self, simple_tree):
        tree = simple_tree
        agg = PretrainedTransformerAggregator(
            node_state_dim=tree['node_state_dim'], graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64,
            use_cls_token=True,
            cls_token_type="root"
        )

        
        out = agg(get_node_states(tree['n_nodes'], tree['node_state_dim']), tree['graph_idx'], 1, from_idx = tree['from_idx'], to_idx = tree['to_idx'])
        print(f'Output shape: {out.shape}')
        assert out.shape == (1, 2048)

    # TODO: test mean pooling aggregation (no CLS)
    def testNoCLS(self, simple_tree):
        tree = simple_tree
        agg = PretrainedTransformerAggregator(
            node_state_dim=tree['node_state_dim'], graph_rep_dim=2048,
            hf_model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_nodes=64,
            use_cls_token=False
        )

        
        out = agg(get_node_states(tree['n_nodes'], tree['node_state_dim']), tree['graph_idx'], 1, from_idx = tree['from_idx'], to_idx = tree['to_idx'])
        print(f'Output shape: {out.shape}')
        assert out.shape == (1, 2048)
    pass


# ---------------------------------------------------------------------------
# Tests: PretrainedNoPropEmbeddingNet (Condition B)
# ---------------------------------------------------------------------------

class TestPretrainedNoPropEmbedding:
    """Tests for Condition B embedding model (Issue #4)."""

    def test_forward_shape(self, binary_trees_batch, prop_heavy_config):
        model = PretrainedNoPropEmbeddingNet(prop_heavy_config)
        batch = binary_trees_batch
        node_features = torch.randn(batch['n_nodes'], 804)
        edge_features = torch.randn(len(batch['from_idx']), 70)
        out = model(node_features, edge_features,
                    batch['from_idx'], batch['to_idx'],
                    batch['graph_idx'], batch['n_graphs'])
        assert out.shape == (3, 2048)

    def test_freeze_delegates_to_aggregator(self, prop_heavy_config):
        model = PretrainedNoPropEmbeddingNet(prop_heavy_config)
        model.freeze_transformer()
        assert all(not p.requires_grad for p in model.aggregator.encoder.parameters())
        model.unfreeze_transformer()
        assert all(p.requires_grad for p in model.aggregator.encoder.parameters())


# ---------------------------------------------------------------------------
# Tests: PretrainedTreeEmbeddingNet (Conditions D/E/F)
# ---------------------------------------------------------------------------

class TestPretrainedTreeEmbedding:
    """Tests for Conditions D/E/F embedding model (Issue #4)."""

    def test_forward_shape(self, binary_trees_batch, prop_heavy_config):
        model = PretrainedTreeEmbeddingNet(prop_heavy_config)
        batch = binary_trees_batch
        node_features = torch.randn(batch['n_nodes'], 804)
        edge_features = torch.randn(len(batch['from_idx']), 70)
        out = model(node_features, edge_features,
                    batch['from_idx'], batch['to_idx'],
                    batch['graph_idx'], batch['n_graphs'])
        assert out.shape == (3, 2048)

    def test_freeze_propagation(self, prop_heavy_config):
        model = PretrainedTreeEmbeddingNet(prop_heavy_config)
        model.freeze_propagation()
        # Encoder and prop layers should be frozen
        assert all(not p.requires_grad for p in model._encoder.parameters())
        for layer in model._prop_layers:
            assert all(not p.requires_grad for p in layer.parameters())
        # Aggregator should still be trainable
        assert any(p.requires_grad for p in model._aggregator.parameters())

    def test_freeze_transformer(self, prop_heavy_config):
        model = PretrainedTreeEmbeddingNet(prop_heavy_config)
        model.freeze_transformer()
        # HF encoder in aggregator should be frozen
        assert all(not p.requires_grad for p in model._aggregator.encoder.parameters())
        # GNN should still be trainable
        assert any(p.requires_grad for p in model._encoder.parameters())

    def test_freeze_both(self, prop_heavy_config):
        model = PretrainedTreeEmbeddingNet(prop_heavy_config)
        model.freeze_propagation()
        model.freeze_transformer()
        # Only projections, pos_encoder, output layers should be trainable
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0, "Some params should still be trainable"
        for name in trainable:
            assert not name.startswith('_encoder.'), f"Encoder param {name} should be frozen"
            assert '_aggregator.encoder.' not in name, f"HF encoder param {name} should be frozen"


# ---------------------------------------------------------------------------
# Tests: PretrainedTextEmbeddingNet (Condition A)
# ---------------------------------------------------------------------------

class TestPretrainedTextEmbedding:
    """Tests for Condition A embedding model (Issue #4)."""

    def test_forward_shape(self, text_config):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = PretrainedTextEmbeddingNet(text_config, tokenizer)
        batch_encoding = tokenizer(
            ['The cat sat on the mat.', 'A dog ran in the park.'],
            padding=True, truncation=True, return_tensors='pt'
        )
        out = model(batch_encoding)
        assert out.shape == (2, 2048)

    def test_freeze_unfreeze(self, text_config):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = PretrainedTextEmbeddingNet(text_config, tokenizer)
        model.freeze_transformer()
        assert all(not p.requires_grad for p in model.transformer.parameters())
        # Projection should still be trainable
        assert all(p.requires_grad for p in model.projection.parameters())
        model.unfreeze_transformer()
        assert all(p.requires_grad for p in model.transformer.parameters())


# ---------------------------------------------------------------------------
# Tests: Matching variants
# ---------------------------------------------------------------------------

class TestPretrainedTreeMatching:
    """Tests for Conditions D/E/F matching model (Issue #5)."""

    def test_forward_shape(self, prop_heavy_config):
        model = PretrainedTreeMatchingNet(prop_heavy_config)
        # Matching needs even number of graphs (pairs)
        from_idx = []
        to_idx = []
        for i in range(1, 8):
            from_idx.append((i - 1) // 2)
            to_idx.append(i)
        for i in range(9, 16):
            from_idx.append(8 + (i - 9) // 2)
            to_idx.append(i)
        for i in range(17, 24):
            from_idx.append(16 + (i - 17) // 2)
            to_idx.append(i)
        for i in range(25, 32):
            from_idx.append(24 + (i - 25) // 2)
            to_idx.append(i)

        n_nodes = 32
        graph_idx = torch.cat([
            torch.full((8,), 0, dtype=torch.long),
            torch.full((8,), 1, dtype=torch.long),
            torch.full((8,), 2, dtype=torch.long),
            torch.full((8,), 3, dtype=torch.long),
        ])
        node_features = torch.randn(n_nodes, 804)
        edge_features = torch.randn(len(from_idx), 70)
        out = model(node_features, edge_features,
                    torch.tensor(from_idx), torch.tensor(to_idx),
                    graph_idx, 4)
        assert out.shape == (4, 2048)

    def test_freeze_propagation(self, prop_heavy_config):
        model = PretrainedTreeMatchingNet(prop_heavy_config)
        model.freeze_propagation()
        assert all(not p.requires_grad for p in model.gmn._encoder.parameters())

    def test_freeze_transformer(self, prop_heavy_config):
        model = PretrainedTreeMatchingNet(prop_heavy_config)
        model.freeze_transformer()
        assert all(not p.requires_grad for p in model.gmn._aggregator.encoder.parameters())


class TestPretrainedTextMatching:
    """Tests for Condition A matching model (Issue #5)."""

    def test_forward_shape(self, text_config):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = PretrainedTextMatchingNet(text_config, tokenizer)
        # Matching needs even batch size (pairs)
        batch_encoding = tokenizer(
            ['The cat sat.', 'A dog ran.', 'Birds flew high.', 'Fish swam deep.'],
            padding=True, truncation=True, return_tensors='pt'
        )
        out = model(batch_encoding)
        assert out.shape == (4, 2048)


class TestPretrainedNoPropMatching:
    """Tests for Condition B matching model (Issue #5)."""

    def test_forward_shape(self, binary_trees_batch, prop_heavy_config):
        model = PretrainedNoPropMatchingNet(prop_heavy_config)
        batch = binary_trees_batch
        node_features = torch.randn(batch['n_nodes'], 804)
        edge_features = torch.randn(len(batch['from_idx']), 70)
        out = model(node_features, edge_features,
                    batch['from_idx'], batch['to_idx'],
                    batch['graph_idx'], batch['n_graphs'])
        assert out.shape == (3, 2048)
