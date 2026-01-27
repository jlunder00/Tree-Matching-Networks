# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

"""
Hard negative mining for contrastive learning based on tree structure similarity.

Implements efficient two-stage hard negative selection:
- Stage 1: Coarse structural filtering (eliminate dissimilar items)
- Stage 2: Fine semantic ranking (rank by similarity, select TOP-K most similar)

Hard negatives = out-group items that are MOST similar to anchor
(Forces model to learn subtle semantic differences, not just group membership)
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional
import logging
import random

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Mine hard negatives based on structural and semantic similarity.

    Two-stage process:
    1. Structural filtering: Eliminate items outside similarity thresholds (fast)
    2. Semantic ranking: Rank survivors by semantic similarity, select TOP-K (expensive)

    All operations use GPU-accelerated PyTorch where possible.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hard negative miner from config.

        Args:
            config: Full experiment config dict containing 'data' section
        """
        data_config = config.get('data', {})
        self.enabled = data_config.get('use_hard_negative_mining', False)

        if not self.enabled:
            logger.info("Hard negative mining disabled - using original sampling")
            return

        hn_config = data_config.get('hard_negative_mining', {})

        # Core parameters
        self.negative_sampling_mode = hn_config.get('negative_sampling_mode', 'fixed')
        self.max_negatives_per_anchor = hn_config.get('max_negatives_per_anchor', 10)
        self.target_neg_to_pos_ratio = hn_config.get('target_neg_to_pos_ratio', 10)
        self.sampling_strategy = hn_config.get('sampling_strategy', 'top_k')

        # Stage 1: Structural filtering (optional but recommended)
        self.use_structural_filtering = hn_config.get('use_structural_filtering', True)
        self.structural_features = hn_config.get('structural_features', [])

        # Stage 2: Semantic ranking (always used if hard negative mining enabled)
        self.use_semantic_ranking = hn_config.get('use_semantic_ranking', True)
        self.semantic_weight = hn_config.get('semantic_weight', 1.0)

        # Sort structural features by order for multi-stage filtering
        self.structural_features = sorted(
            self.structural_features,
            key=lambda x: x.get('order', 0)
        )

        # Validate configuration
        self._validate_config()

        # Log configuration
        logger.info(f"Hard negative mining enabled:")
        logger.info(f"  Negative sampling mode: {self.negative_sampling_mode}")
        if self.negative_sampling_mode == 'fixed':
            logger.info(f"    Max negatives per anchor: {self.max_negatives_per_anchor} (fixed)")
        else:  # ratio_based
            logger.info(f"    Target neg:pos ratio: 1:{self.target_neg_to_pos_ratio}")
            logger.info(f"    (Actual negatives = positives × {self.target_neg_to_pos_ratio})")
        logger.info(f"  Sampling strategy: {self.sampling_strategy}")
        logger.info(f"  Structural filtering: {self.use_structural_filtering}")
        if self.use_structural_filtering:
            logger.info(f"    Features: {len(self.structural_features)}")
            for feat in self.structural_features:
                logger.info(f"      Order {feat.get('order', 0)}: {feat['name']} "
                           f"(threshold={feat['threshold']}, weight={feat.get('weight', 1.0)})")
        logger.info(f"  Semantic ranking: {self.use_semantic_ranking}")
        if self.use_semantic_ranking:
            logger.info(f"    Weight: {self.semantic_weight}")

    def _validate_config(self):
        """Validate feature configurations."""
        valid_structural = {'node_count', 'max_depth', 'edge_count', 'avg_depth', 'leaf_ratio'}

        for feat in self.structural_features:
            if 'name' not in feat:
                raise ValueError("Each structural feature must have 'name' field")
            if feat['name'] not in valid_structural:
                raise ValueError(f"Unknown structural feature: {feat['name']}. "
                               f"Valid: {valid_structural}")
            if 'threshold' not in feat:
                raise ValueError(f"Feature {feat['name']} missing 'threshold'")

    def extract_tree_features(self, trees: List[Dict]) -> Dict[str, Any]:
        """
        Extract structural features and mean-pooled embeddings from trees.

        Args:
            trees: List of tree dictionaries with 'node_features', 'from_idx', 'to_idx'

        Returns:
            Dict of features:
                - Scalar structural features: [n_trees] tensors
                - mean_pooled_embeddings: [n_trees, 768] tensor (if semantic ranking enabled)
                - device: torch.device
        """
        n_trees = len(trees)
        # Force CPU to avoid CUDA re-initialization issues in DataLoader workers
        # Feature extraction is lightweight, GPU not needed here
        device = torch.device('cpu')

        features = {
            'n_trees': n_trees,
            'device': device
        }

        # Allocate tensors for structural features
        if self.use_structural_filtering:
            node_counts = torch.zeros(n_trees, dtype=torch.float32, device=device)
            max_depths = torch.zeros(n_trees, dtype=torch.float32, device=device)
            edge_counts = torch.zeros(n_trees, dtype=torch.float32, device=device)
            avg_depths = torch.zeros(n_trees, dtype=torch.float32, device=device)
            leaf_ratios = torch.zeros(n_trees, dtype=torch.float32, device=device)

            for i, tree in enumerate(trees):
                node_features = tree.get('node_features', [])
                from_idx = tree.get('from_idx', [])
                to_idx = tree.get('to_idx', [])

                n_nodes = len(node_features)
                n_edges = len(from_idx)

                node_counts[i] = n_nodes
                edge_counts[i] = n_edges

                if to_idx:
                    # Max depth from to_idx
                    max_depths[i] = max(to_idx) + 1 if to_idx else 1

                    # Average depth
                    avg_depths[i] = sum(to_idx) / len(to_idx) if to_idx else 0

                    # Leaf ratio
                    has_outgoing = set(from_idx)
                    n_leaves = n_nodes - len(has_outgoing)
                    leaf_ratios[i] = n_leaves / n_nodes if n_nodes > 0 else 0
                else:
                    max_depths[i] = 1
                    avg_depths[i] = 0
                    leaf_ratios[i] = 1.0

            features['node_count'] = node_counts
            features['max_depth'] = max_depths
            features['edge_count'] = edge_counts
            features['avg_depth'] = avg_depths
            features['leaf_ratio'] = leaf_ratios

        # Extract mean-pooled embeddings for semantic ranking
        if self.use_semantic_ranking:
            embeddings_list = []

            for tree in trees:
                node_features = tree.get('node_features', [])

                if len(node_features) > 0:
                    # Convert to tensor if needed
                    if isinstance(node_features, list):
                        node_tensor = torch.tensor(node_features, dtype=torch.float32, device=device)
                    else:
                        node_tensor = torch.as_tensor(node_features, dtype=torch.float32, device=device)

                    # Mean pool across nodes: [n_nodes, 768] -> [768]
                    mean_embedding = node_tensor.mean(dim=0)
                else:
                    # Empty tree - use zero vector
                    mean_embedding = torch.zeros(768, dtype=torch.float32, device=device)

                embeddings_list.append(mean_embedding)

            # Stack into [n_trees, 768]
            features['mean_pooled_embeddings'] = torch.stack(embeddings_list)

        return features

    def select_hard_negatives(
        self,
        anchor_indices: List[int],
        features: Dict[str, Any],
        global_group_assignment: Dict[int, str],
        global_dataset_assignment: Dict[int, str],
        allow_cross_dataset: bool,
        positives_per_anchor: float = 1.0
    ) -> List[Tuple[int, int]]:
        """
        Select hard negatives using two-stage filtering:
        1. Structural filtering: Eliminate dissimilar items (coarse, fast)
        2. Semantic ranking: Rank survivors by similarity, select TOP-K (fine, expensive)

        Args:
            anchor_indices: List of anchor tree indices in batch
            features: Extracted features from extract_tree_features()
            global_group_assignment: Dict mapping tree idx -> group ID
            global_dataset_assignment: Dict mapping tree idx -> dataset name
            allow_cross_dataset: Whether to allow cross-dataset negatives
            positives_per_anchor: Number of positives per anchor (for ratio_based mode)

        Returns:
            List of (anchor_idx, negative_idx) tuples
        """
        if not self.enabled:
            # Fallback to original sampling - ALL different group trees
            return self._original_negative_sampling(
                anchor_indices, global_group_assignment,
                global_dataset_assignment, allow_cross_dataset
            )

        # Compute effective max_negatives based on mode
        if self.negative_sampling_mode == 'ratio_based':
            # Enforce ratio: negatives = positives × target_ratio
            effective_max_negatives = int(positives_per_anchor * self.target_neg_to_pos_ratio)
            logger.debug(f"Ratio-based mode: {positives_per_anchor:.1f} pos × {self.target_neg_to_pos_ratio} "
                        f"= {effective_max_negatives} negatives per anchor")
        else:  # fixed mode
            effective_max_negatives = self.max_negatives_per_anchor

        negative_pairs = []
        device = features['device']

        # Statistics for logging
        stats = {
            'total_initial': 0,
            'total_after_structural': 0,
            'total_final': 0
        }

        for anchor_idx in anchor_indices:
            anchor_gid = global_group_assignment[anchor_idx]
            anchor_dataset = global_dataset_assignment[anchor_idx]

            # Initial candidate pool: all trees from different groups
            if allow_cross_dataset:
                candidates = [i for i, gid in global_group_assignment.items()
                            if gid != anchor_gid]
            else:
                candidates = [i for i, gid in global_group_assignment.items()
                            if gid != anchor_gid and global_dataset_assignment[i] == anchor_dataset]

            if not candidates:
                continue

            stats['total_initial'] += len(candidates)

            # Convert to tensor for GPU operations
            candidates_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

            # Track similarity scores (accumulated across features)
            similarity_scores = torch.zeros(len(candidates), device=device)

            # STAGE 1: Structural Filtering (eliminate dissimilar)
            if self.use_structural_filtering:
                for feat_config in self.structural_features:
                    if len(candidates_tensor) == 0:
                        break

                    feature_name = feat_config['name']
                    threshold = feat_config['threshold']
                    weight = feat_config.get('weight', 1.0)

                    # Get anchor and candidate values
                    anchor_val = features[feature_name][anchor_idx]
                    candidate_vals = features[feature_name][candidates_tensor]

                    # Compute relative difference
                    rel_diff = torch.abs(candidate_vals - anchor_val) / (anchor_val + 1e-6)

                    # FILTER: Keep only items within threshold (similar enough)
                    mask = rel_diff <= threshold
                    candidates_tensor = candidates_tensor[mask]
                    similarity_scores = similarity_scores[mask]

                    # Accumulate similarity scores (1 - difference = higher for more similar)
                    similarity_scores += weight * (1.0 - rel_diff[mask])

                stats['total_after_structural'] += len(candidates_tensor)

            # STAGE 2: Semantic Ranking (rank by similarity, select TOP-K)
            if self.use_semantic_ranking and len(candidates_tensor) > 0:
                # Get mean-pooled embeddings
                anchor_emb = features['mean_pooled_embeddings'][anchor_idx]
                candidate_embs = features['mean_pooled_embeddings'][candidates_tensor]

                # Compute cosine similarity
                semantic_sim = F.cosine_similarity(
                    anchor_emb.unsqueeze(0),
                    candidate_embs,
                    dim=1
                )

                # Add to similarity scores (higher = more similar = harder negative)
                similarity_scores += self.semantic_weight * semantic_sim

            # Select TOP-K most similar = hardest negatives
            if len(candidates_tensor) > 0:
                n_to_sample = min(effective_max_negatives, len(candidates_tensor))

                if self.sampling_strategy == 'top_k':
                    # Select top K by similarity score
                    if n_to_sample < len(candidates_tensor):
                        _, top_indices = torch.topk(similarity_scores, n_to_sample)
                        selected = candidates_tensor[top_indices]
                    else:
                        selected = candidates_tensor

                elif self.sampling_strategy == 'weighted':
                    # Sample with probability proportional to similarity
                    probs = F.softmax(similarity_scores, dim=0)
                    selected_indices = torch.multinomial(probs, n_to_sample, replacement=False)
                    selected = candidates_tensor[selected_indices]

                else:  # random or other
                    # Random sample from survivors
                    if n_to_sample < len(candidates_tensor):
                        perm = torch.randperm(len(candidates_tensor), device=device)
                        selected = candidates_tensor[perm[:n_to_sample]]
                    else:
                        selected = candidates_tensor

                for neg_idx in selected:
                    negative_pairs.append((anchor_idx, neg_idx.item()))

                stats['total_final'] += len(selected)
            else:
                # Fallback: No survivors after filtering - use random sampling
                logger.warning(f"Anchor {anchor_idx}: Over-filtered, falling back to random")
                candidates_list = [i for i, gid in global_group_assignment.items()
                                 if gid != anchor_gid]
                if candidates_list:
                    n_to_sample = min(effective_max_negatives, len(candidates_list))
                    sampled = random.sample(candidates_list, n_to_sample)
                    for neg_idx in sampled:
                        negative_pairs.append((anchor_idx, neg_idx))
                    stats['total_final'] += len(sampled)

        # Log summary statistics
        if len(anchor_indices) > 0:
            avg_initial = stats['total_initial'] / len(anchor_indices)
            avg_after_struct = stats['total_after_structural'] / len(anchor_indices) if self.use_structural_filtering else avg_initial
            avg_final = stats['total_final'] / len(anchor_indices)

            logger.debug(f"Hard negative mining: {avg_initial:.1f} initial -> "
                        f"{avg_after_struct:.1f} after structural -> "
                        f"{avg_final:.1f} final negatives/anchor")

            # Log reduction percentage
            if avg_initial > 0:
                reduction_pct = (1 - avg_final / avg_initial) * 100
                logger.debug(f"Negative reduction: {reduction_pct:.1f}% "
                           f"({avg_initial:.0f} -> {avg_final:.0f} per anchor)")

        return negative_pairs

    def _original_negative_sampling(
        self,
        anchor_indices: List[int],
        global_group_assignment: Dict[int, str],
        global_dataset_assignment: Dict[int, str],
        allow_cross_dataset: bool
    ) -> List[Tuple[int, int]]:
        """Fallback to original negative sampling (all out-group items)."""
        negative_pairs = []

        for anchor_idx in anchor_indices:
            anchor_gid = global_group_assignment[anchor_idx]
            anchor_dataset = global_dataset_assignment[anchor_idx]

            if allow_cross_dataset:
                candidates = [i for i, gid in global_group_assignment.items()
                            if gid != anchor_gid]
            else:
                candidates = [i for i, gid in global_group_assignment.items()
                            if gid != anchor_gid and global_dataset_assignment[i] == anchor_dataset]

            for neg_idx in candidates:
                negative_pairs.append((anchor_idx, neg_idx))

        return negative_pairs
