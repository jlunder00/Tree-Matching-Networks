# data/dynamic_calculated_contrastive_dataset.py
import json
import gc
import math
import random
import logging
from collections import defaultdict, deque
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# Import your feature extractor and tree-to-graph conversion utility.
from TMN_DataGen import FeatureExtractor
try:
    from .data_utils import convert_tree_to_graph_data, GraphData
except ImportError:
    from data_utils import convert_tree_to_graph_data, GraphData

logger = logging.getLogger(__name__)


def get_feature_config(config: Dict) -> Dict:
    """Create feature extractor config."""
    return {
        'feature_extraction': {
            'word_embedding_model': config.get('word_embedding_model', 'bert-base-uncased'),
            'use_gpu': config.get('use_gpu', True) and torch.cuda.is_available(),
            'cache_embeddings': True,
            'embedding_cache_dir': config.get('embedding_cache_dir', 'embedding_cache'),
            'do_not_store_word_embeddings': False,
            'is_runtime': True,
            'shard_size': config.get('cache_shard_size', 10000),
            # 'num_workers': config.get('cache_workers', 1),
            'num_workers': 1,
        },
        'verbose': config.get('verbose', 'normal')
    }


@dataclass
class BatchInfo:
    """Tracks pairing information for a batch."""
    group_indices: List[int]            # numeric indices (if available) for groups
    group_ids: List[str]                # group IDs present in the batch
    anchor_indices: List[int]           # global indices of anchor trees in the final list
    positive_pairs: List[Tuple[int, int]]  # (anchor index, positive candidate index)
    negative_pairs: List[Tuple[int, int]]  # (anchor index, negative candidate index)


class DynamicCalculatedContrastiveDataset(IterableDataset):
    """
    An IterableDataset that loads sharded JSON files containing groups of trees and
    dynamically builds batches for InfoNCE contrastive learning.
    
    This dataset uses precomputed counts (via _counts files) to track how many trees are available in
    each group and then calculates, for each batch, exactly how many trees to take from each group
    so that every tree is used exactly as many times as desired:
      - anchors_per_group (A)
      - pos_pairs_per_anchor (P)
      - neg_pairs_per_anchor (N)
      - overall batch is defined in terms of total pairs desired.
    
    It works as follows:
      1. Using the batch size (in terms of pairs) and pairing parameters, it computes:
           X = ceil(batch_pairs / (P+N))   # total anchors needed
           G = ceil(X / A)                 # number of groups required
           R = A * (1 + P)                 # trees needed per group for positive pairing
      2. The dataset maintains a buffer of groups (loaded from files) where each group is stored
         as a dict: { 'group_id': ..., 'group_idx': ..., 'trees': [tree_item, ...] }.
      3. When forming a batch, it selects G groups that have at least R trees. For each such group,
         it removes the first R trees from its buffer. The first A trees become anchors and the next A*P
         become the positive candidates (each anchor gets P positives).
      4. Negative pairs are formed by sampling, for each anchor, N trees from the union of trees in the
         other groups in the batch.
      5. The trees from all groups are concatenated into a single list and the pairing indices are stored
         in a BatchInfo object.
      6. Leftover trees in any group remain in the buffer to be used in later batches.
    """
    def __init__(self, 
                 data_dir: str,
                 config: Dict,
                 batch_pairs: int,
                 anchors_per_group: int = 2,
                 pos_pairs_per_anchor: int = 2,
                 neg_pairs_per_anchor: int = 4,
                 min_groups_per_batch: int = 2,
                 shuffle_files: bool = False,
                 prefetch_factor: int = 2,
                 max_active_files: int = 2,
                 recycle_leftovers: bool = True):
        """
        Args:
          data_dir: Directory containing the shard JSON files.
          config: Configuration dictionary.
          batch_pairs: Desired number of pairs in a batch (each pair consists of 2 trees).
          anchors_per_group (A): Number of anchors to sample per group.
          pos_pairs_per_anchor (P): Number of positive pairs per anchor.
          neg_pairs_per_anchor (N): Number of negative pairs per anchor.
          min_groups_per_batch: Minimum number of groups required in a batch.
          recycle_leftovers: If True, leftover trees that do not meet R are kept for future batches.
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.batch_pairs = batch_pairs
        self.A = anchors_per_group
        self.P = pos_pairs_per_anchor
        self.N = neg_pairs_per_anchor
        self.min_groups = min_groups_per_batch
        self.shuffle_files = shuffle_files
        self.prefetch_factor = prefetch_factor
        self.max_active_files = max_active_files
        self.recycle_leftovers = recycle_leftovers
        self.requires_embeddings = True

        # Calculate the number of anchors and groups required.
        # Each anchor will yield (P + N) pairs.
        self.total_anchors_needed = math.ceil(self.batch_pairs / (self.P + self.N))
        self.groups_needed = max(self.min_groups, math.ceil(self.total_anchors_needed / self.A))
        # For positive pairing within a group, each group must supply:
        self.R = self.A * (1 + self.P)  # trees per group

        # Gather shard files (assume naming like part_*_shard_*.json)
        self.data_files = sorted(
            [f for f in self.data_dir.glob("part_*_shard_*.json")
             if not f.name.endswith('_counts.json')],
            key=lambda x: (int(x.stem.split('_')[1]), int(x.stem.split('_shard_')[1]))
        )
        if not self.data_files:
            self.data_files = sorted(self.data_dir.glob("part_*.json"))
        if self.shuffle_files:
            random.shuffle(self.data_files)
        
        # Load counts from _counts files (used only for __len__ estimation)
        self.file_counts = []
        for file in self.data_files:
            count_file = file.parent / f"{file.stem}_counts.json"
            if count_file.exists():
                with open(count_file) as f:
                    self.file_counts.append(json.load(f))
            else:
                with open(file) as f:
                    data = json.load(f)
                    self.file_counts.append({
                        'n_groups': len(data.get('groups', [])),
                        'trees_per_group': [len(g.get('trees', [])) for g in data.get('groups', [])]
                    })
                    
        # # Initialize feature extractor if needed.
        # with open(self.data_files[0]) as f:
        #     data = json.load(f)
        #     self.requires_embeddings = data.get('requires_word_embeddings', False)
        #     if self.requires_embeddings:
        #         feat_config = get_feature_config(config)
        #         self.embedding_extractor = FeatureExtractor(feat_config)
        
        # A buffer mapping group_id to a dict { 'group_idx': int, 'trees': [tree_item, ...] }
        self.group_buffers: Dict[str, Dict] = {}
        # A deque of files not yet processed.
        self.file_queue = deque(self.data_files)
        self.active_files = deque(maxlen=self.max_active_files)
    
    def _load_embeddings(self, tree: Dict) -> torch.Tensor:
        """Load or generate embeddings for a tree if required."""
        if not tree.get('node_features_need_word_embs_prepended', False):
            return torch.tensor(tree['node_features'])
            
        embeddings = []
        for word, lemma in tree['node_texts']:
            emb = None
            if lemma in self.embedding_extractor.embedding_cache:
                emb = self.embedding_extractor.embedding_cache[lemma]
            elif word in self.embedding_extractor.embedding_cache:
                emb = self.embedding_extractor.embedding_cache[word]
            if emb is None:
                emb = self.embedding_extractor.get_word_embedding(lemma)
            embeddings.append(emb)
        word_embeddings = torch.stack(embeddings)
        node_features = torch.tensor(tree['node_features'])
        return torch.cat([word_embeddings, node_features], dim=-1)
    
    def _add_groups_from_file(self, file: Path):
        """Load a file and add its groups to the buffer."""
        try:
            with open(file) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading file {file}: {e}")
            return
        
        self.active_files.append(file)
        for group in data.get('groups', []):
            if not group.get('trees'):
                continue
            group_id = group.get('group_id')
            group_idx = group.get('group_idx', 0)
            trees = group.get('trees', [])[:self.config.get('max_group_size', 32)]
            processed_trees = []
            for tree_idx, tree in enumerate(trees):
                # Process embeddings if required.
                if tree.get('node_features_need_word_embs_prepended', False):
                    tree = dict(tree)
                    tree['node_features'] = self._load_embeddings(tree)
                processed_trees.append({
                    'tree': tree,
                    'group_id': group_id,
                    'group_idx': group_idx,
                    'tree_idx': tree_idx,
                    'text': tree.get('text', '')
                })
            # If the group is already in the buffer, append new trees.
            if group_id in self.group_buffers:
                self.group_buffers[group_id]['trees'].extend(processed_trees)
            else:
                self.group_buffers[group_id] = {
                    'group_idx': group_idx,
                    'trees': processed_trees
                }
    
    def _fill_buffer(self):
        """Keep loading files until we have at least the required number of groups with enough trees."""
        while len([g for g in self.group_buffers.values() if len(g['trees']) >= self.R]) < self.groups_needed and self.file_queue:
            file = self.file_queue.popleft()
            self._add_groups_from_file(file)
            # Optionally clean up if too many active files
            if len(self.active_files) >= self.max_active_files:
                self.active_files.popleft()
                gc.collect()
    
    def _generate_batch(self) -> Optional[Tuple[GraphData, BatchInfo]]:
        """
        Using the current group_buffers, select groups that have at least R trees,
        remove R trees from each, and build the batch.
        """
        eligible_groups = [ (gid, info) for gid, info in self.group_buffers.items() if len(info['trees']) >= self.R ]
        if len(eligible_groups) < self.groups_needed:
            return None  # not enough groups
        
        # Randomly select the required number of groups.
        selected = random.sample(eligible_groups, self.groups_needed)
        batch_trees = []         # All trees that will be used in this batch.
        batch_group_ids = []     # Track group ids in order.
        batch_group_indices = [] # Track corresponding numeric group indices.
        anchor_indices = []      # Global indices of anchors in batch_trees.
        positive_pairs = []      # (anchor_index, positive_index)
        negative_pairs = []      # (anchor_index, negative_index)
        
        # For each selected group, remove R trees from its buffer.
        # In each group, the first A trees are anchors; the next A*P are positives.
        group_start_indices = {}  # map group_id to the starting global index for its trees in this batch.
        for gid, info in selected:
            trees = info['trees']
            used = trees[:self.R]
            # Remove used trees from the buffer.
            if len(trees) > self.R:
                self.group_buffers[gid]['trees'] = trees[self.R:]
            else:
                # If not recycling leftovers, remove the group entirely.
                if not self.recycle_leftovers:
                    del self.group_buffers[gid]
                else:
                    self.group_buffers[gid]['trees'] = []
            group_start_indices[gid] = len(batch_trees)
            batch_trees.extend(used)
            batch_group_ids.append(gid)
            batch_group_indices.append(info['group_idx'])
        
        # Now, within each group’s segment in batch_trees, designate the first A as anchors
        # and partition the following A*P trees as positive candidates (each anchor gets P positives).
        for gid, info in selected:
            start = group_start_indices[gid]
            group_trees = batch_trees[start : start + self.R]
            # The first A trees are anchors.
            for i in range(self.A):
                global_anchor_idx = start + i
                anchor_indices.append(global_anchor_idx)
                # For positive pairs, assign the P positive candidates for this anchor.
                pos_candidates = group_trees[self.A + i*self.P : self.A + (i+1)*self.P]
                for pos_tree in pos_candidates:
                    global_pos_idx = batch_trees.index(pos_tree)  # not efficient but groups are small
                    positive_pairs.append((global_anchor_idx, global_pos_idx))
        
        # Build the negative pool: for each anchor, pool together all trees from groups other than its own.
        # First, record which global index belongs to which group.
        global_group_assignment = {}
        for gid, info in selected:
            start = group_start_indices[gid]
            for j in range(self.R):
                global_group_assignment[start + j] = gid
        
        # For each anchor, sample N negatives from trees whose group is different.
        for anchor_idx in anchor_indices:
            anchor_gid = global_group_assignment[anchor_idx]
            # Build negative candidate indices: all indices in batch_trees whose group != anchor_gid.
            neg_candidates = [i for i, gid in global_group_assignment.items() if gid != anchor_gid]
            if not neg_candidates:
                continue
            sampled = random.sample(neg_candidates, min(self.N, len(neg_candidates)))
            for neg_idx in sampled:
                negative_pairs.append((anchor_idx, neg_idx))
        
        # Finally, convert the list of trees to GraphData.
        graphs = convert_tree_to_graph_data([item['tree'] for item in batch_trees])
        batch_info = BatchInfo(
            group_indices=batch_group_indices,
            group_ids=batch_group_ids,
            anchor_indices=anchor_indices,
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs
        )
        return graphs, batch_info

    def __iter__(self):
        """Iterate over the dataset, yielding (GraphData, BatchInfo) batches."""
        worker_info = get_worker_info()
        if worker_info is not None:
            # Split files across workers.
            files = list(self.data_files)
            files = files[worker_info.id::worker_info.num_workers]
            self.file_queue = deque(files)
            if self.requires_embeddings and not hasattr(self, "embedding_extractor"):
                feat_config = get_feature_config(self.config)
                self.embedding_extractor = FeatureExtractor(feat_config)
        else:
            # In single-worker mode, ensure the extractor is initialized.
            if self.requires_embeddings and not hasattr(self, "embedding_extractor"):
                feat_config = get_feature_config(self.config)
                self.embedding_extractor = FeatureExtractor(feat_config)
        # Loop until files are exhausted and buffer cannot be refilled.
        while True:
            self._fill_buffer()
            batch = self._generate_batch()
            if batch is not None:
                yield batch
            else:
                # If no batch can be produced and no more files remain, exit.
                if not self.file_queue:
                    break
                else:
                    # Otherwise, load more files.
                    continue

    def __len__(self):
        """A rough length estimate (number of pairs) based on counts."""
        total_trees = sum(sum(fc['trees_per_group']) for fc in self.file_counts)
        est_pairs_per_tree = self.P + self.N
        return total_trees // est_pairs_per_tree


def get_dynamic_calculated_dataloader(dataset: DynamicCalculatedContrastiveDataset,
                                      num_workers: int = 4,
                                      pin_memory: bool = True) -> DataLoader:
    """
    Returns a DataLoader for the DynamicCalculatedContrastiveDataset.
    Since each __iter__ call yields a complete batch (GraphData, BatchInfo), we use batch_size=1
    and a collate_fn that unwraps the singleton.
    """
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=dataset.prefetch_factor,
        persistent_workers=True
    )

