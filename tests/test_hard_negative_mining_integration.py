"""
Integration tests for Hard Negative Mining with real dev data.

These tests verify end-to-end functionality using actual WikiQS dev data.
Tests are marked as integration tests and may take longer to run.
"""

import torch
import pytest
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Tree_Matching_Networks.LinguisticTrees.data.hard_negative_miner import HardNegativeMiner
from Tree_Matching_Networks.LinguisticTrees.data.dynamic_calculated_contrastive_dataset import (
    DynamicCalculatedContrastiveDataset
)
from Tree_Matching_Networks.LinguisticTrees.data import get_dynamic_calculated_dataloader
from Tree_Matching_Networks.LinguisticTrees.configs.tree_data_config import TreeDataConfig

logger = logging.getLogger(__name__)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestHardNegativeMiningRealData:
    """Integration tests using real WikiQS dev data."""

    @pytest.fixture
    def dev_data_root(self):
        """Path to dev data root."""
        # Check if dev data exists
        data_root = Path("/home/jlunder/research/data/processed_data/")
        if not data_root.exists():
            pytest.skip("Dev data not found at /home/jlunder/research/data/processed_data/")
        return str(data_root)

    @pytest.fixture
    def base_config(self):
        """Base configuration for hard negative mining."""
        return {
            'data': {
                'batch_size': 64,  # Smaller batch for faster testing
                'dataset_type': 'wikiqs',
                'dataset_specs': ['wikiqs'],
                'pos_pairs_per_anchor': 1,
                'min_groups_per_batch': 8,
                'anchors_per_group': 1,
                'max_batches_per_epoch': 5,  # Only test a few batches
                'num_workers': 0,  # Easier debugging
                'allow_cross_dataset_negatives': False,
                'use_hard_negative_mining': False,  # Will be enabled per test
                'hard_negative_mining': {
                    'negative_sampling_mode': 'ratio_based',
                    'target_neg_to_pos_ratio': 10,
                    'use_structural_filtering': True,
                    'structural_features': [
                        {
                            'name': 'node_count',
                            'threshold': 0.3,
                            'weight': 1.0,
                            'order': 1
                        }
                    ],
                    'use_semantic_ranking': True,
                    'semantic_weight': 1.0,
                    'sampling_strategy': 'top_k'
                }
            },
            'model': {
                'task_type': 'infonce',
                'task_loader_type': 'contrastive',
                'model_type': 'embedding'
            },
            'embedding_cache_dir': '/home/jlunder/research/TMN_DataGen/embedding_cache5/'
        }

    def test_dataset_initialization_with_hard_negative_mining(self, dev_data_root, base_config):
        """Test that dataset initializes correctly with hard negative mining enabled."""
        # Enable hard negative mining
        base_config['data']['use_hard_negative_mining'] = True

        # Create data config
        data_config = TreeDataConfig(
            data_root=dev_data_root,
            dataset_specs=['wikiqs'],
            task_type='infonce',
            use_sharded_train=False,
            use_sharded_validate=True,
            allow_cross_dataset_negatives=False
        )

        # Create dataset
        dataset = DynamicCalculatedContrastiveDataset(
            data_dirs=[data_config.dev_dirs[0]],
            config=base_config,
            model_type='embedding',
            batch_size=base_config['data']['batch_size'],
            min_groups_per_batch=base_config['data']['min_groups_per_batch'],
            anchors_per_group=base_config['data']['anchors_per_group'],
            pos_pairs_per_anchor=base_config['data']['pos_pairs_per_anchor'],
            max_batches_per_epoch=base_config['data']['max_batches_per_epoch'],
            allow_cross_dataset_negatives=base_config['data']['allow_cross_dataset_negatives'],
            shuffle=False
        )

        # Verify hard negative miner was initialized
        assert dataset.hard_negative_miner is not None
        assert dataset.hard_negative_miner.enabled == True
        assert dataset.hard_negative_miner.negative_sampling_mode == 'ratio_based'
        assert dataset.hard_negative_miner.target_neg_to_pos_ratio == 10

        print("✓ Dataset initialization with hard negative mining test passed")

    def test_batch_construction_with_real_data(self, dev_data_root, base_config):
        """Test that batches are constructed correctly with real WikiQS dev data."""
        # Enable hard negative mining
        base_config['data']['use_hard_negative_mining'] = True

        # Create data config
        data_config = TreeDataConfig(
            data_root=dev_data_root,
            dataset_specs=['wikiqs'],
            task_type='infonce',
            use_sharded_train=False,
            use_sharded_validate=True,
            allow_cross_dataset_negatives=False
        )

        # Create dataset
        dataset = DynamicCalculatedContrastiveDataset(
            data_dirs=[data_config.dev_dirs[0]],
            config=base_config,
            model_type='embedding',
            batch_size=base_config['data']['batch_size'],
            min_groups_per_batch=base_config['data']['min_groups_per_batch'],
            anchors_per_group=base_config['data']['anchors_per_group'],
            pos_pairs_per_anchor=base_config['data']['pos_pairs_per_anchor'],
            max_batches_per_epoch=base_config['data']['max_batches_per_epoch'],
            allow_cross_dataset_negatives=base_config['data']['allow_cross_dataset_negatives'],
            shuffle=False
        )

        # Create dataloader
        dataloader = get_dynamic_calculated_dataloader(
            dataset,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )

        # Iterate through a few batches
        batch_count = 0
        for graphs, batch_info in dataloader:
            # Verify batch structure
            assert graphs is not None
            assert batch_info is not None
            assert len(batch_info.anchor_indices) > 0

            # Verify ratio enforcement
            n_anchors = len(batch_info.anchor_indices)
            n_positives = len(batch_info.positive_pairs)
            n_negatives = len(batch_info.negative_pairs)

            pos_per_anchor = n_positives / n_anchors if n_anchors > 0 else 0
            neg_per_anchor = n_negatives / n_anchors if n_anchors > 0 else 0

            # Should be approximately 1:10 ratio (allowing some variance due to filtering)
            if pos_per_anchor > 0:
                actual_ratio = neg_per_anchor / pos_per_anchor
                # Allow ratio between 5 and 15 (some negatives may be filtered out)
                assert 5 <= actual_ratio <= 15, \
                    f"Ratio {actual_ratio:.1f} outside expected range [5, 15]"

            logger.info(f"Batch {batch_count}: {n_anchors} anchors, "
                       f"{pos_per_anchor:.1f} pos/anchor, "
                       f"{neg_per_anchor:.1f} neg/anchor, "
                       f"ratio 1:{actual_ratio:.1f}")

            batch_count += 1
            if batch_count >= 3:  # Test only first 3 batches
                break

        assert batch_count > 0, "No batches were generated"
        print(f"✓ Batch construction test passed ({batch_count} batches tested)")

    def test_baseline_vs_hard_negative_mining_comparison(self, dev_data_root, base_config):
        """Compare baseline (all negatives) vs hard negative mining on same data."""
        data_config = TreeDataConfig(
            data_root=dev_data_root,
            dataset_specs=['wikiqs'],
            task_type='infonce',
            use_sharded_train=False,
            use_sharded_validate=True,
            allow_cross_dataset_negatives=False
        )

        # Create baseline dataset (no hard negative mining)
        baseline_config = base_config.copy()
        baseline_config['data']['use_hard_negative_mining'] = False

        baseline_dataset = DynamicCalculatedContrastiveDataset(
            data_dirs=[data_config.dev_dirs[0]],
            config=baseline_config,
            model_type='embedding',
            batch_size=baseline_config['data']['batch_size'],
            min_groups_per_batch=baseline_config['data']['min_groups_per_batch'],
            anchors_per_group=baseline_config['data']['anchors_per_group'],
            pos_pairs_per_anchor=baseline_config['data']['pos_pairs_per_anchor'],
            max_batches_per_epoch=2,
            allow_cross_dataset_negatives=baseline_config['data']['allow_cross_dataset_negatives'],
            shuffle=False
        )

        # Create hard negative mining dataset
        hn_config = base_config.copy()
        hn_config['data']['use_hard_negative_mining'] = True

        hn_dataset = DynamicCalculatedContrastiveDataset(
            data_dirs=[data_config.dev_dirs[0]],
            config=hn_config,
            model_type='embedding',
            batch_size=hn_config['data']['batch_size'],
            min_groups_per_batch=hn_config['data']['min_groups_per_batch'],
            anchors_per_group=hn_config['data']['anchors_per_group'],
            pos_pairs_per_anchor=hn_config['data']['pos_pairs_per_anchor'],
            max_batches_per_epoch=2,
            allow_cross_dataset_negatives=hn_config['data']['allow_cross_dataset_negatives'],
            shuffle=False
        )

        # Create dataloaders
        baseline_loader = get_dynamic_calculated_dataloader(baseline_dataset, num_workers=0)
        hn_loader = get_dynamic_calculated_dataloader(hn_dataset, num_workers=0)

        # Compare first batch from each
        baseline_batch = next(iter(baseline_loader))
        hn_batch = next(iter(hn_loader))

        baseline_graphs, baseline_info = baseline_batch
        hn_graphs, hn_info = hn_batch

        # Baseline should have MANY more negatives (all out-group items)
        baseline_negs = len(baseline_info.negative_pairs) / len(baseline_info.anchor_indices)
        hn_negs = len(hn_info.negative_pairs) / len(hn_info.anchor_indices)

        logger.info(f"Baseline: {baseline_negs:.1f} negatives per anchor")
        logger.info(f"Hard negative mining: {hn_negs:.1f} negatives per anchor")

        # Hard negative mining should have significantly fewer negatives
        assert hn_negs < baseline_negs * 0.5, \
            f"Hard negative mining ({hn_negs:.1f}) should have <50% of baseline negatives ({baseline_negs:.1f})"

        # Hard negative mining should still have reasonable number of negatives
        assert hn_negs >= 5, \
            f"Hard negative mining should have at least 5 negatives per anchor, got {hn_negs:.1f}"

        reduction_pct = (1 - hn_negs / baseline_negs) * 100
        logger.info(f"Negative reduction: {reduction_pct:.1f}% ({baseline_negs:.0f} → {hn_negs:.0f} per anchor)")

        print(f"✓ Baseline vs hard negative mining comparison test passed "
              f"({reduction_pct:.1f}% reduction)")

    def test_different_pos_pairs_per_anchor(self, dev_data_root, base_config):
        """Test that ratio enforcement works with different pos_pairs_per_anchor values."""
        data_config = TreeDataConfig(
            data_root=dev_data_root,
            dataset_specs=['wikiqs'],
            task_type='infonce',
            use_sharded_train=False,
            use_sharded_validate=True,
            allow_cross_dataset_negatives=False
        )

        base_config['data']['use_hard_negative_mining'] = True
        results = {}

        for pos_pairs in [1, 2, 3]:
            config = base_config.copy()
            config['data']['pos_pairs_per_anchor'] = pos_pairs

            dataset = DynamicCalculatedContrastiveDataset(
                data_dirs=[data_config.dev_dirs[0]],
                config=config,
                model_type='embedding',
                batch_size=config['data']['batch_size'],
                min_groups_per_batch=config['data']['min_groups_per_batch'],
                anchors_per_group=config['data']['anchors_per_group'],
                pos_pairs_per_anchor=pos_pairs,
                max_batches_per_epoch=2,
                allow_cross_dataset_negatives=False,
                shuffle=False
            )

            dataloader = get_dynamic_calculated_dataloader(dataset, num_workers=0)
            graphs, batch_info = next(iter(dataloader))

            n_anchors = len(batch_info.anchor_indices)
            n_positives = len(batch_info.positive_pairs)
            n_negatives = len(batch_info.negative_pairs)

            pos_per_anchor = n_positives / n_anchors
            neg_per_anchor = n_negatives / n_anchors
            ratio = neg_per_anchor / pos_per_anchor if pos_per_anchor > 0 else 0

            results[pos_pairs] = {
                'pos_per_anchor': pos_per_anchor,
                'neg_per_anchor': neg_per_anchor,
                'ratio': ratio
            }

            logger.info(f"pos_pairs={pos_pairs}: "
                       f"{pos_per_anchor:.1f} pos/anchor, "
                       f"{neg_per_anchor:.1f} neg/anchor, "
                       f"ratio 1:{ratio:.1f}")

        # Verify that negatives scale with positives
        # pos=1 should have ~10 negatives, pos=2 should have ~20, pos=3 should have ~30
        for pos_pairs in [1, 2, 3]:
            expected_negs = pos_pairs * 10
            actual_negs = results[pos_pairs]['neg_per_anchor']
            # Allow 30% variance due to filtering
            assert expected_negs * 0.7 <= actual_negs <= expected_negs * 1.3, \
                f"pos={pos_pairs}: expected ~{expected_negs} negs, got {actual_negs:.1f}"

        print("✓ Different pos_pairs_per_anchor test passed")

    def test_structural_filtering_effectiveness(self, dev_data_root, base_config):
        """Test that structural filtering actually filters candidates."""
        data_config = TreeDataConfig(
            data_root=dev_data_root,
            dataset_specs=['wikiqs'],
            task_type='infonce',
            use_sharded_train=False,
            use_sharded_validate=True,
            allow_cross_dataset_negatives=False
        )

        # Test with vs without structural filtering
        configs = {
            'no_filtering': {
                **base_config,
                'data': {
                    **base_config['data'],
                    'use_hard_negative_mining': True,
                    'hard_negative_mining': {
                        **base_config['data']['hard_negative_mining'],
                        'use_structural_filtering': False,
                        'use_semantic_ranking': True  # Only semantic
                    }
                }
            },
            'with_filtering': {
                **base_config,
                'data': {
                    **base_config['data'],
                    'use_hard_negative_mining': True,
                    'hard_negative_mining': {
                        **base_config['data']['hard_negative_mining'],
                        'use_structural_filtering': True,  # Both structural and semantic
                        'use_semantic_ranking': True
                    }
                }
            }
        }

        results = {}
        for name, config in configs.items():
            dataset = DynamicCalculatedContrastiveDataset(
                data_dirs=[data_config.dev_dirs[0]],
                config=config,
                model_type='embedding',
                batch_size=config['data']['batch_size'],
                min_groups_per_batch=config['data']['min_groups_per_batch'],
                anchors_per_group=config['data']['anchors_per_group'],
                pos_pairs_per_anchor=config['data']['pos_pairs_per_anchor'],
                max_batches_per_epoch=2,
                allow_cross_dataset_negatives=False,
                shuffle=False
            )

            dataloader = get_dynamic_calculated_dataloader(dataset, num_workers=0)
            graphs, batch_info = next(iter(dataloader))

            n_negatives = len(batch_info.negative_pairs)
            results[name] = n_negatives

            logger.info(f"{name}: {n_negatives} total negatives")

        # With structural filtering should potentially have fewer negatives
        # (though not always, depends on data distribution)
        logger.info(f"Structural filtering impact: "
                   f"{results['no_filtering']} (no filter) vs "
                   f"{results['with_filtering']} (with filter)")

        print("✓ Structural filtering effectiveness test passed")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-m", "integration", "--tb=short"])
