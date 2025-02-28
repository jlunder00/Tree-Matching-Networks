# data/dynamic_calculated_contrastive_dataset.py
import numpy as np
import json
import gc
import math
import random
import logging
from collections import defaultdict, deque
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import copy

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# Import your feature extractor and tree-to-graph conversion utility.
from TMN_DataGen import FeatureExtractor
try:
    from .data_utils import convert_tree_to_graph_data, GraphData, get_min_groups_trees_per_group, get_min_groups_pairs_per_anchor
except ImportError:
    from data_utils import convert_tree_to_graph_data, GraphData, get_min_groups_trees_per_group, get_min_groups_pairs_per_anchor

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
            'num_workers': config.get('cache_workers', 4),
            # 'num_workers': 1,
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
    pair_indices: List[Tuple[int, int, bool]]
    anchor_positive_indexes: Dict
    anchor_negative_indexes: Dict


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
           X = ceil(batch_size / (P+N))   # total anchors needed
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
                 batch_size: int = 512, #either maximum or minimum batch size based on if get min groups is passed ceil = True or not. Max if false, min if true.
                 anchors_per_group: int = 1,
                 pos_pairs_per_anchor: int = 1,
                 # neg_pairs_per_anchor: int = 4,
                 # min_groups_per_batch: int = 2,
                 shuffle_files: bool = False,
                 prefetch_factor: int = 2,
                 max_active_files: int = 2,
                 recycle_leftovers: bool = True,
                 allow_cross_dataset_negatives: bool = True,
                 model_type: str = 'matching'):
        """
        Args:
          data_dir: Directory containing the shard JSON files.
          config: Configuration dictionary.
          batch_size: Desired number of pairs in a batch (each pair consists of 2 trees).
          anchors_per_group (A): Number of anchors to sample per group.
          pos_pairs_per_anchor (P): Number of positive pairs per anchor.
          neg_pairs_per_anchor (N): Number of negative pairs per anchor.
          min_groups_per_batch: Minimum number of groups required in a batch.
          recycle_leftovers: If True, leftover trees that do not meet R are kept for future batches.
          allow_cross_dataset_negatives: Allow negatives from other datasets.
          model_type: 'matching' or 'embedding'
        """
        # Convert single directory to list
        if isinstance(data_dir, str):
            self.data_dirs = [Path(data_dir)]
        else:
            self.data_dirs = [Path(dir) for dir in data_dir]
        self.config = config
        self.batch_size = batch_size
        self.A = anchors_per_group
        self.P = pos_pairs_per_anchor
        # self.N = neg_pairs_per_anchor
        # self.min_groups = min_groups_per_batch
        self.shuffle_files = shuffle_files
        self.prefetch_factor = prefetch_factor
        self.max_active_files = max_active_files
        self.recycle_leftovers = recycle_leftovers
        self.requires_embeddings = True
        self.model_type = model_type
        self._batches_provided = 0
        self.allow_cross_dataset_negatives = allow_cross_dataset_negatives

        if self.model_type == 'matching':
            #adjusts the batch size to use the entirety of the calculated integer number of groups. Rounding up with ceil = True means
            #batch size will be >= what was passed, otherwise it will be <= what was passed
            self.groups_needed, self.batch_size = get_min_groups_pairs_per_anchor(self.A, self.P, self.batch_size)
            #need to get the number of negative pairs per group based on this

            # For positive pairing within a group, each group must supply:
            self.R = self.A + self.P  # trees per group, need the number of anchors plus the number of positive pairs, eg, one more positive pair per anchor means one more non anchor tree from the group. P > 0 always.
            #R is the number of trees provided by each group, and P is the number of positive pairs per anchor, so we can find how many positive pairs that makes per group, and then use the number of groups needed to find the overall number of positive pairs
            self.positive_pairs_per_group = self.P * self.A
            self.positive_pairs_total = self.positive_pairs_per_group * self.groups_needed
            #we know the adjusted batch size from using the found number of groups and the number of positive pairs, so we can find the number of negative pairs
            self.N = self.batch_size - self.positive_pairs_total

            # Calculate the number of anchors required
            self.total_anchors_needed = self.A * self.groups_needed
        else:
            self.additional_trees_needed = self.P-(self.A-1) if self.P > (self.A-1) else 0
            self.R = self.A + self.additional_trees_needed
            self.groups_needed = int(self.batch_size/self.R) #number of groups needed to fulfil batch size embeddings
            self.batch_pair_size = int(self.A*(self.A+self.additional_trees_needed)*(self.groups_needed**2) - (self.A**2)*self.groups_needed)
            self.positive_pairs_per_group = self.P * self.A
            self.positive_pairs_total = self.positive_pairs_per_group * self.groups_needed
            self.batch_size = int(self.groups_needed * self.R)
            self.N = self.batch_pair_size - self.positive_pairs_total

        #all we need to know is the number of trees each group has to provide, which are anchors, which are not, since we have the number of groups

        # Gather shard files (assume naming like part_*_shard_*.json)
        self.data_files = []
        for data_dir in self.data_dirs:
            files = sorted(
                [f for f in data_dir.glob("part_*_shard_*.json")
                 if not f.name.endswith('_counts.json')],
                key=lambda x: (int(x.stem.split('_')[1]), int(x.stem.split('_shard_')[1]))
            )
            if not self.data_files:
                self.data_files = sorted(data_dir.glob("part_*.json"))
            self.data_files.extend(files)
            
        if self.shuffle_files:
            random.shuffle(self.data_files)

        # Store dataset origin for each file
        self.file_to_dataset = {}
        for data_dir in self.data_dirs:
            for file in [f for f in self.data_files if str(f).startswith(str(data_dir))]:
                self.file_to_dataset[file] = data_dir.name
        
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
        with open(self.data_files[0]) as f:
            data = json.load(f)
            self.requires_embeddings = data.get('requires_word_embeddings', False)
            if self.requires_embeddings:
                feat_config = get_feature_config(config)
                self.embedding_extractor = FeatureExtractor(feat_config)
        
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
        dataset_origin = self.file_to_dataset.get('file', 'unknown')

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
                    'text': tree.get('text', ''),
                    'dataset': dataset_origin
                })
            # If the group is already in the buffer, append new trees.
            if group_id in self.group_buffers:
                self.group_buffers[group_id]['trees'].extend(processed_trees)
            else:
                self.group_buffers[group_id] = {
                    'group_idx': group_idx,
                    'trees': processed_trees,
                    'dataset': dataset_origin
                }
    
    def _fill_buffer(self):
        """Keep loading files until we have at least the required number of groups with enough trees."""
        while len([g for g in self.group_buffers.values() if len(g['trees']) >= self.R]) < self.groups_needed and self.file_queue:
            print('populating buffer...')
            file = self.file_queue.popleft()
            self._add_groups_from_file(file)
            # Optionally clean up if too many active files
            if len(self.active_files) >= self.max_active_files:
                self.active_files.popleft()
                gc.collect()

    def _randomly_select_trees(self, trees, n):
        arr = np.array(trees, dtype=object)

        valid_indices = np.where(arr != None)[0]

        selected_indices = np.random.choice(valid_indices, size=n, replace=False)
        selected_trees, indices = arr[selected_indices].tolist(), selected_indices.tolist()
        return selected_trees, indices

    def reset_epoch(self):
        """Reset dataset state for new epoch"""
        # Reset file queue
        if not self.file_queue:
            self.file_queue = deque(self.data_files)
            if self.shuffle_files:
                files = list(self.file_queue)
                random.shuffle(files)
                self.file_queue = deque(files)
        
        # Clear buffers
        self.group_buffers.clear()
        self.active_files.clear()
        
        # Reset batch count
        self._batches_provided = 0
    
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
        selected_gids = [gid for gid, info in selected]
        eligable_groups_not_already_selected_not_np = [(gid, info) for gid, info in eligible_groups if gid not in selected_gids]
        eligable_groups_not_already_selected = np.array(eligable_groups_not_already_selected_not_np)
        batch_trees = []         # All trees that will be used in this batch.
        batch_group_ids = []     # Track group ids in order.
        batch_group_indices = [] # Track corresponding numeric group indices.
        batch_datasets = []      # Track dataset origins
        anchor_indices = []      # Global indices of anchors in batch_trees.
        positive_pairs = []      # (anchor_index, positive_index)
        negative_pairs = []      # (anchor_index, negative_index)
        
        # For each selected group, remove R trees from its buffer.
        # In each group, the first A trees are anchors; the next A*P are positives.
        group_start_indices = {}  # map group_id to the starting global index for its trees in this batch.
        group_to_dataset = {}
        for gid, info in selected:
            trees = info['trees']
            dataset_origin = info.get('dataset', 'unknown')
            group_to_dataset[gid] = dataset_origin
            batch_datasets.append(dataset_origin)
            num_trees_not_none = len([t for t in trees if t is not None])
            # Remove used trees from the buffer.-
            new_group_index = -1
            while num_trees_not_none < self.R: #there werent enough trees in the group beforehand means in the first place, drop the group AND SELECT A NEW ONE
                print(f"Not enough trees in group {gid} beforehand! dropping and trying another")
                del self.group_buffers[gid]
                if new_group_index >= 0:
                    print(f"deleting {gid} from eligable replacement groups")
                    eligable_groups_not_already_selected = np.delete(eligable_groups_not_already_selected, new_group_index)
                if len(eligable_groups_not_already_selected) < 1:
                    print("No more eligable groups!")
                    return None
                
                eligable_indexes = np.where(eligable_groups_not_already_selected)[0]
                new_group_index = np.random.choice(eligable_indexes, 1, replace=False)
                gid, info = eligable_groups_not_already_selected[new_group_index]
                trees = info['trees']
                dataset_origin = info.get('dataset', 'unknown')
                group_to_dataset[gid] = dataset_origin
                num_trees_not_none = len([t for t in trees if t is not None])

            used, used_indices = self._randomly_select_trees(copy.deepcopy(trees), self.R)
            # used = trees[:self.R]
            trees_after = [trees[i] if i not in used_indices else None for i in range(len(trees))]
            num_trees_not_none_after = len([t for t in trees_after if t is not None])
            if num_trees_not_none_after > self.R: #dont need to worry about recycling
                self.group_buffers[gid]['trees'] = trees_after

            elif not self.recycle_leftovers or len(trees_after) == 0: #dont ever recycle if we cleanly used all the trees. delete once used entirely
                # If not recycling leftovers, remove the group entirely.
                del self.group_buffers[gid]
            else: #if recycling leftovers, figure out how many to add back and add them back
                difference = self.R - num_trees_not_none_after # number of trees we need to keep to recycle with enough for another selection
                if difference > len(used_indices): #not enough to make up the difference (should be mathematically impossible but oh well)
                    del self.group_buffers[gid]
                
                #add back as many as we need to make it recyclable (dont need to randomize as selection was random)
                add_back, add_back_indices = [copy.deepcopy(t) for t in used[:difference]], used_indices[:difference]
                for idx, tree in zip(add_back_indices, add_back):
                    trees_after[idx] = tree

                self.group_buffers[gid]['trees'] = trees_after 
            group_start_indices[gid] = len(batch_trees)
            batch_trees.extend(used)
            batch_group_ids.append(gid)
            batch_group_indices.append(info['group_idx'])
            if new_group_index >= 0:
                eligable_groups_not_already_selected = np.delete(eligable_groups_not_already_selected, new_group_index)
        
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
                # for pos_tree in pos_candidates:
                #     global_pos_idx = batch_trees.index(pos_tree)  # not efficient but groups are small
                for j in range(len(pos_candidates)):
                    global_pos_idx = start + self.A + i * self.P + j
                    positive_pairs.append((global_anchor_idx, global_pos_idx))
        
        # Build the negative pool: for each anchor, pool together all trees from groups other than its own.
        # First, record which global index belongs to which group.
        global_group_assignment = {}
        global_dataset_assignment = {}
        for gid, info in selected:
            start = group_start_indices[gid]
            dataset = group_to_dataset[gid]
            for j in range(self.R):
                global_group_assignment[start + j] = gid
                global_dataset_assignment[start + j] = dataset
        
        # For each anchor, sample N negatives from trees whose group is different.
        for anchor_idx in anchor_indices:
            anchor_gid = global_group_assignment[anchor_idx]
            anchor_dataset = global_dataset_assignment[anchor_idx]
            # Build negative candidate indices: all indices in batch_trees whose group != anchor_gid.
            if self.allow_cross_dataset_negatives:
                neg_candidates = [i for i, gid in global_group_assignment.items() if gid != anchor_gid]
            else:
                neg_candidates = [i for i, gid in global_group_assignment.items() if gid != anchor_gid and global_dataset_assignment[i] == anchor_dataset]
            if not neg_candidates:
                continue
            # sampled = random.sample(neg_candidates, min(self.N, len(neg_candidates)))
            for neg_idx in neg_candidates:
                negative_pairs.append((anchor_idx, neg_idx))
        
        batch_info = BatchInfo(
            group_indices=batch_group_indices,
            group_ids=batch_group_ids,
            anchor_indices=anchor_indices,
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs,
            pair_indices = [],
            anchor_positive_indexes = {},
            anchor_negative_indexes = {}
        )

        if self.model_type == 'embedding': #format as pairs for graph matching model with attention network
            # Finally, convert the list of trees to GraphData.
            all_batch_graphs = [convert_tree_to_graph_data([item['tree']]) for item in batch_trees]
            # graphs = [convert_tree_to_graph_data([item['tree']]) for item in batch_trees] #this would be a graph data object per tree
        else: #do pairs for cross attention model
            # For each anchor, we need its tree multiple times (for each pos/neg pair)
            all_batch_graphs = []
            all_pair_info = []  # Track which indices form pairs
            anchor_positive_indexes = {}
            
                # Add positive pairs
            for pair_idx, (tree_idx1, tree_idx2) in enumerate(batch_info.positive_pairs):
                tree1, tree2 = copy.deepcopy(batch_trees[tree_idx1]['tree']), copy.deepcopy(batch_trees[tree_idx2]['tree'])
                # We need a deep copy if the same tree appears multiple times
                all_batch_graphs.append(convert_tree_to_graph_data([tree1, tree2]))
                # Record these will be paired with graph_idx [2i, 2i+1]
                all_pair_info.append((pair_idx * 2, pair_idx * 2 + 1, True))  # True = positive pair
                if tree_idx1 in anchor_positive_indexes.keys():
                    anchor_positive_indexes[tree_idx1].append(pair_idx)
                else:
                    anchor_positive_indexes[tree_idx1] = [pair_idx]
                
                
            anchor_negative_indexes = {}
            # Add negative pairs  
            for pair_idx, (tree_idx1, tree_idx2) in enumerate(batch_info.negative_pairs):
                tree1, tree2 = copy.deepcopy(batch_trees[tree_idx1]['tree']), copy.deepcopy(batch_trees[tree_idx2]['tree'])
                # We need a deep copy if the same tree appears multiple times
                all_batch_graphs.append(convert_tree_to_graph_data([tree1, tree2]))
                # Record these will be paired with graph_idx [2i, 2i+1]
                all_pair_info.append((pair_idx * 2, pair_idx * 2 + 1, False))  # False = negative pair
                if tree_idx1 in anchor_negative_indexes.keys():
                    anchor_negative_indexes[tree_idx1].append(pair_idx)
                else:
                    anchor_negative_indexes[tree_idx1] = [pair_idx]

            # graphs = convert_tree_to_graph_data(all_batch_trees)  # This creates sequential graph_idx
            
            # Update batch info with the new indices
            batch_info.pair_indices = all_pair_info
            batch_info.anchor_positive_indexes = anchor_positive_indexes
            batch_info.anchor_negative_indexes = anchor_negative_indexes
            
        # Convert all trees into single GraphData with correct indexing
        graphs = self.collate_graphs(all_batch_graphs)

        return graphs, batch_info

    def collate_graphs(self, graphs):
        from_idx = []
        to_idx = []
        graph_idx = []
        node_features = []
        edge_features = []
        last_graph_idx = 0
        for i, g in enumerate(graphs):
            from_idx.append(g.from_idx)
            to_idx.append(g.to_idx)
            for j in range(g.n_graphs):
                n_nodes = len(g.graph_idx[g.graph_idx == j])
                graph_idx.append(torch.ones(n_nodes, dtype=torch.int64)*last_graph_idx)
                last_graph_idx += 1
            node_features.append(g.node_features)
            edge_features.append(g.edge_features)

        graph_data = GraphData(
            from_idx=torch.cat(from_idx),
            to_idx=torch.cat(to_idx),
            node_features=torch.cat(node_features),
            edge_features=torch.cat(edge_features),
            graph_idx=torch.cat(graph_idx),
            n_graphs=sum(b.n_graphs for b in graphs)
        )
        return graph_data

    def __iter__(self):
        """Iterate over the dataset, yielding (GraphData, BatchInfo) batches."""
        worker_info = get_worker_info()
        if worker_info is not None:
            # Split files across workers.
            files = list(self.data_files)
            files = files[worker_info.id::worker_info.num_workers]
            self.file_queue = deque(files)

        max_batches = self.config.get('max_batches_per_epoch', 1000)
        # Loop until files are exhausted and buffer cannot be refilled.
        while self._batches_provided < max_batches:
            self._fill_buffer()
            batch = self._generate_batch()
            if batch is not None:
                self._batches_provided += 1
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
        return total_trees // self.batch_size


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
        persistent_workers=True if num_workers > 0 else False
    )

