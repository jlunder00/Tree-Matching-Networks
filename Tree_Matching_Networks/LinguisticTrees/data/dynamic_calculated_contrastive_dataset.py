# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#Legacy dataset: paired_groups_dataset and the associated text level loss can handle infonce for single sentence style data, and for other types
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
from typing import List, Tuple, Dict, Optional, Callable
import copy
import uuid
from functools import wraps

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

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


def dataloader_handler(key: str):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        wrapper.dataloader_handler = True
        wrapper.handler_key = key
        return wrapper

    return decorator
                
            


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
    strict_matching: bool
    labeled: bool


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
           R = A * (1 + P)                 # tralbuquerqueees needed per group for positive pairing
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
                 model_type: str = 'matching',
                 strict_matching: bool = False,
                 labeled: bool = False,
                 allow_text_files = False,
                 text_mode: bool = False,
                 tokenizer = None,
                 max_length: int = 512):
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

        self._dataloader_handlers = {}
        for name in dir(self):
            method = getattr(self, name)
            if getattr(method, "dataloader_handler", False):
                self._dataloader_handlers[method.handler_key] = method

        # Convert single directory to list
        if isinstance(data_dir, str):
            self.data_dirs = [Path(data_dir)]
        else:
            self.data_dirs = [Path(dir) for dir in data_dir]

        self.labeled = labeled #means that positive/negative is already defined, and we are given groups that contain a pair of tree groups.
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
        self.strict_matching = strict_matching
        self.positive_pairing_ratio = config['data'].get('positive_pairing_ratio', 1.0)
        self.ensure_positives_in_batch = config['data'].get("ensure_positives_in_batch", True)
        if not 0.0 <= self.positive_pairing_ratio <= 1.0:
            raise ValueError("positive_pairing_ratio must be between 0.0 and 1.0")
        
        self.allow_text_files = allow_text_files
        self.text_mode = text_mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.allow_text_files and not self.text_mode:
            logger.warning("text mode not activated when text files are allowed. Forcing text mode in dataloader")
            self.text_mode = True

        if self.model_type == 'matching' and self.strict_matching:
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
            json_files = list(data_dir.glob("part_*_shard_*.json"))
            txt_files = []
            if self.allow_text_files and self.text_mode:
                txt_files = list(data_dir.glob("part_*_shard_*.txt"))
            files = sorted(
                [f for f in json_files + txt_files
                 if not f.name.endswith('_counts.json')],
                key=lambda x: (int(x.stem.split('_')[1]), int(x.stem.split('_shard_')[1]) if '_shard_' in x.stem else 0)
            )
            if not self.data_files:
                json_files = list(data_dir.glob("part_*.json"))
                txt_files = []
                if self.allow_text_files and self.text_mode:
                    txt_files = list(data_dir.glob("part_*.txt"))
                self.data_files = sorted(json_files + txt_files)
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
        # print(f"counting: {self.data_files}\n{len(self.data_files)}")
        for file in self.data_files:
            count_file = file.parent / f"{file.stem}_counts.json"
            if file.suffix == '.txt':
                self.file_counts.append({
                    'n_groups': 10000,
                    'trees_per_group': [10]*10000
                })
                continue
            if count_file.exists():
                with open(count_file) as f:
                    self.file_counts.append(json.load(f))
            else:
                # print(f"shit its being loaded {file.suffix}")
                data = self._dataloader_handlers[file.suffix](file)
                self.file_counts.append({
                    'n_groups': len(data.get('groups', [])),
                    'trees_per_group': [len(g.get('trees', [])) for g in data.get('groups', [])]
                })
                    
        print("counting done")
        # # Initialize feature extractor if needed.
        data = self._dataloader_handlers[self.data_files[0].suffix](self.data_files[0])
        print(self.text_mode)
        print(data.get('requires_word_embeddings', f"no word embeddings???: {data}"))
        self.requires_embeddings = not self.text_mode
        # data.get('requires_word_embeddings', False) and not self.text_mode
        if self.requires_embeddings:
            feat_config = get_feature_config(config)
            self.embedding_extractor = FeatureExtractor(feat_config)
        
        # A buffer mapping group_id to a dict { 'group_idx': int, 'trees': [tree_item, ...] }
        self.group_buffers: Dict[str, Dict] = {}
        # A deque of files not yet processed.
        self.file_queue = deque(self.data_files)
        self.active_files = deque(maxlen=self.max_active_files)
    

    def get_pairing_ratio(self):
        return self.positive_pairing_ratio

    def set_pairing_ratio(self, r):
        self.positive_pairing_ratio = r 


    def get_text_mode(self):
        return self.text_mode

    @dataloader_handler(".json")
    def parse_json(self, file):
        with open(file) as f:
            data = json.load(f)
        return data

    @dataloader_handler(".txt")
    def parse_wikiqs_file(self, file):
        data = []
        filtered_too_long = 0
        max_char_length = self.config['data'].get("max_text_chars", 500)
        with open(file) as f:
            for i, line in enumerate(f):
                trees = []
                line = line.strip()
                if not line:
                    continue
                questions = [
                    q.strip()[2:] # Remove "q:" prefix
                    for q in line.strip().split('\t')
                    if q.startswith('q:') or q.startswith('a:')
                ]
                original_count = len(questions)
                questions = [q for q in questions if len(q) < max_char_length]
                filtered_too_long += (original_count - len(questions))
                if len(questions) < 2:
                    continue
                rejoined_line = ' '.join(questions)

                trees = [{'text': t.strip()} for t in questions]
                data.append({
                    'group_id': str(uuid.uuid4()),
                    'trees': trees,
                    'trees_b': trees,
                    'text': rejoined_line.strip(),
                    'text_b': rejoined_line.strip(),
                })
        print(f"{filtered_too_long} texts removed for being too long")
        return {
            "groups": data,
            "version": "1.0",
            "requires_word_embeddings": False,
            "format": "infonce-text"
        }
    
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
            data = self._dataloader_handlers[file.suffix](file)
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
                if tree.get('node_features_need_word_embs_prepended', False) and not self.text_mode:
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

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text using provided tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when text mode enabled")

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        actual_length = len([t for t in encoded['input_ids'][0] if t != self.tokenizer.pad_token_id])
        if actual_length >= self.max_length:
            print(f"TRUNCATED: text was {actual_length}+ tokens, cut to {self.max_length}")
            print(f"{text}")

        
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def _tokenize_pair_separate(self, text_a: str, text_b: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Tokenize a pair of texts separately for matching models."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when text_mode is enabled")
            
        # Tokenize each sequence separately
        encoded_a = self.tokenizer(
            text_a,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        encoded_b = self.tokenizer(
            text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return (
            {k: v.squeeze(0) for k, v in encoded_a.items()},
            {k: v.squeeze(0) for k, v in encoded_b.items()}
        )

    
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


        # NEW: Apply mixed pairing strategy to rearrange pairing
        batch_trees, positive_pairs = self._apply_mixed_pairing_strategy(batch_trees, positive_pairs, anchor_indices)
        
        # Update anchor_indices to reflect new positions
        old_to_new = {old_idx: new_idx for new_idx, (old_idx, tree) in enumerate(zip(range(len(batch_trees)), batch_trees))}
        anchor_indices = [old_to_new.get(idx, idx) for idx in anchor_indices]
        
        batch_info = BatchInfo(
            group_indices=batch_group_indices,
            group_ids=batch_group_ids,
            anchor_indices=anchor_indices,
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs,
            pair_indices = [],
            anchor_positive_indexes = {},
            anchor_negative_indexes = {},
            strict_matching = self.strict_matching,
            labeled = self.labeled
        )

        if self.text_mode:
            model_type = self.config['model'].get('model_type', 'embedding')
            if model_type == 'matching':
                all_pair_info = []  # Track which indices form pairs
                anchor_positive_indexes = {}
                # For matching models, we need to tokenize pairs separately
                # Collect pairs based on the batch structure
                text_list = []
                
                # Create pairs from batch_trees based on positive pairs
                for pair_idx, (anchor_idx, pos_idx) in enumerate(positive_pairs):
                    text_a = batch_trees[pair_idx*2].get('text', '')
                    text_b = batch_trees[pair_idx*2+1].get('text', '')
                    all_pair_info.append((anchor_idx, pos_idx, True))
                    text_list.extend([text_a, text_b])
                    if anchor_idx in anchor_positive_indexes.keys():
                        anchor_positive_indexes[anchor_idx].append(pair_idx)
                    else:
                        anchor_positive_indexes[anchor_idx] = [pair_idx]

                if batch_info.strict_matching:
                    anchor_negative_indexes = {}
                    for pair_idx, (anchor_idx, neg_idx) in enumerate(batch_info.negative_pairs):
                        text_a, text_b = batch_trees[anchor_idx].get('text', ''), batch_trees[neg_idx].get('text', '')
                        text_list.extend([text_a, text_b])
                        all_pair_info.append(((pair_idx+len(batch_info.positive_pairs)) * 2, (pair_idx+len(batch_info.positive_pairs)) * 2 + 1, False))  # False = negative pair
                        if anchor_idx in anchor_negative_indexes.keys():
                            anchor_negative_indexes[anchor_idx].append(pair_idx)
                        else:
                            anchor_negative_indexes[anchor_idx] = [pair_idx]

                    batch_info.anchor_negative_indexes = anchor_negative_indexes
                
                
                text_encodings = []

                for text in text_list:
                    encoding = self._tokenize_text(text)
                    text_encodings.append(encoding)

                # for text_a, text_b in text_pairs:
                #     encoding_a, encoding_b = self._tokenize_pair_separate(text_a, text_b)
                #     batch_encoding_a_list.append(encoding_a)
                #     batch_encoding_b_list.append(encoding_b)
                # 
                # if batch_encoding_a_list:
                #     batch_encoding_a = {k: torch.stack([enc[k] for enc in batch_encoding_a_list]) 
                #                        for k in batch_encoding_a_list[0].keys()}
                #     batch_encoding_b = {k: torch.stack([enc[k] for enc in batch_encoding_b_list]) 
                #                        for k in batch_encoding_b_list[0].keys()}
                #     output = (batch_encoding_a, batch_encoding_b)
                if text_encodings:
                    batch_encoding = {k: torch.stack([enc[k] for enc in text_encodings]) 
                         for k in text_encodings[0].keys()}
                    output = batch_encoding
                else:
                    # Empty batch fallback
                    output = {
                        'input_ids': torch.zeros((0, self.max_length), dtype=torch.long),
                        'attention_mask': torch.zeros((0, self.max_length), dtype=torch.long),
                        'token_type_ids': torch.zeros((0, self.max_length), dtype=torch.long)
                    }

                batch_info.pair_indices = all_pair_info
                batch_info.anchor_positive_indexes = anchor_positive_indexes
            else:

                text_inputs = [self._tokenize_text(tree.get('text', '')) for tree in batch_trees]
                if text_inputs:
                    batch_encoded = {k: torch.cat([p[k] for p in text_inputs]) for k in text_inputs[0].keys()}
                else:
                    # Empty batch fallback
                    batch_encoded = {
                        'input_ids': torch.zeros((0, self.max_length), dtype=torch.long),
                        'attention_mask': torch.zeros((0, self.max_length), dtype=torch.long),
                        'token_type_ids': torch.zeros((0, self.max_length), dtype=torch.long)
                    }
                output = batch_encoded
        else:
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
                    # tree1, tree2 = copy.deepcopy(batch_trees[tree_idx1]['tree']), copy.deepcopy(batch_trees[tree_idx2]['tree'])
                    tree1, tree2 = copy.deepcopy(batch_trees[pair_idx*2]['tree']), copy.deepcopy(batch_trees[pair_idx*2+1]['tree'])
                    # We need a deep copy if the same tree appears multiple times
                    all_batch_graphs.append(convert_tree_to_graph_data([tree1, tree2]))
                    # Record these will be paired with graph_idx [2i, 2i+1]
                    # all_pair_info.append((pair_idx * 2, pair_idx * 2 + 1, True))  # True = positive pair
                    all_pair_info.append((tree_idx1, tree_idx2, True))  # True = positive pair
                    if tree_idx1 in anchor_positive_indexes.keys():
                        anchor_positive_indexes[tree_idx1].append(pair_idx)
                    else:
                        anchor_positive_indexes[tree_idx1] = [pair_idx]
                    

                if batch_info.strict_matching:    
                    anchor_negative_indexes = {}
                    # Add negative pairs  
                    for pair_idx, (tree_idx1, tree_idx2) in enumerate(batch_info.negative_pairs):
                        tree1, tree2 = copy.deepcopy(batch_trees[tree_idx1]['tree']), copy.deepcopy(batch_trees[tree_idx2]['tree'])
                        # We need a deep copy if the same tree appears multiple times
                        all_batch_graphs.append(convert_tree_to_graph_data([tree1, tree2]))
                        # Record these will be paired with graph_idx [2i, 2i+1]
                        all_pair_info.append(((pair_idx+len(batch_info.positive_pairs)) * 2, (pair_idx+len(batch_info.positive_pairs)) * 2 + 1, False))  # False = negative pair
                        if tree_idx1 in anchor_negative_indexes.keys():
                            anchor_negative_indexes[tree_idx1].append(pair_idx)
                        else:
                            anchor_negative_indexes[tree_idx1] = [pair_idx]

                    batch_info.anchor_negative_indexes = anchor_negative_indexes
                # graphs = convert_tree_to_graph_data(all_batch_trees)  # This creates sequential graph_idx
                
                # Update batch info with the new indices
                batch_info.pair_indices = all_pair_info
                batch_info.anchor_positive_indexes = anchor_positive_indexes
                
            # Convert all trees into single GraphData with correct indexing
            output = self.collate_graphs(all_batch_graphs)

        return output, batch_info

    def _apply_mixed_pairing_strategy(self, batch_trees, positive_pairs, anchor_indices):
        """
        Shift only a proportion of positive items to create loose pairs based on ratio.
        
        Args:
            batch_trees: List alternating [anchor_0, pos_0, anchor_1, pos_1, ...]
            positive_pairs: List of (anchor_idx, pos_idx) tuples
            anchor_indices: List of anchor indices [0, 2, 4, 6, ...]
        
        Returns:
            rearranged_trees: Reordered batch_trees  
            updated_positive_pairs: Updated positive pairs
        """
        if self.positive_pairing_ratio >= 1.0:
            return batch_trees, positive_pairs
        
        total_pairs = len(positive_pairs)
        num_direct_pairs = int(total_pairs * self.positive_pairing_ratio)
        num_loose_pairs = total_pairs - num_direct_pairs
        
        if num_loose_pairs == 0:
            return batch_trees, positive_pairs
        
        # Randomly choose which pairs remain direct (adjacent)
        direct_pair_indices = set(random.sample(range(total_pairs), num_direct_pairs))
        
        # Collect positive items that need to be shifted (for loose pairs)
        loose_positive_items = []
        loose_positive_positions = []  # Their current positions in batch_trees
        
        for i, (anchor_idx, pos_idx) in enumerate(positive_pairs):
            if i not in direct_pair_indices:  # This pair should become loose
                loose_positive_items.append(batch_trees[pos_idx])
                loose_positive_positions.append(pos_idx)
        
        if not loose_positive_items:
            return batch_trees, positive_pairs
        
        # Shift the loose positive items with wrapping
        shift_amount = random.randint(1, len(loose_positive_items)) if len(loose_positive_items) > 1 else 1
        shifted_items = loose_positive_items[-shift_amount:] + loose_positive_items[:-shift_amount]
        
        # Create new batch with shifted items
        new_batch_trees = batch_trees.copy()
        for i, new_item in enumerate(shifted_items):
            target_position = loose_positive_positions[i]
            new_batch_trees[target_position] = new_item
        
        # Update positive_pairs - need to track where each shifted item ended up
        updated_positive_pairs = []
        
        for i, (anchor_idx, pos_idx) in enumerate(positive_pairs):
            if i in direct_pair_indices:
                # Direct pair stays unchanged
                updated_positive_pairs.append((anchor_idx, pos_idx))
            else:
                # Loose pair - find where this positive item moved to
                original_item = batch_trees[pos_idx]
                
                # Find the new position of this item after shifting
                original_loose_index = loose_positive_positions.index(pos_idx)
                new_loose_index = (original_loose_index + shift_amount) % len(loose_positive_items)
                new_pos_idx = loose_positive_positions[new_loose_index]
                
                updated_positive_pairs.append((anchor_idx, new_pos_idx))
        
        return new_batch_trees, updated_positive_pairs

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

        max_batches = self.config['data'].get('max_batches_per_epoch', 250)
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
        print(total_trees)
        worker_info = get_worker_info()
        mult = 6
        if worker_info is not None:
            mult = worker_info.num_workers
        

        return min(total_trees // self.batch_size, self.config['data'].get('max_batches_per_epoch', 250)*mult)
        


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

