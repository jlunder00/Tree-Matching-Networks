# data/grouped_tree_dataset.py
from typing import Iterator, Tuple, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import torch
import json
import logging
import random
from collections import deque, defaultdict
import gc
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from TMN_DataGen import FeatureExtractor
try:
    from .data_utils import convert_tree_to_graph_data, GraphData
except:
    from data_utils import convert_tree_to_graph_data, GraphData

logger = logging.getLogger(__name__)

@dataclass
class TreeGroup:
    """Container for a group of related trees"""
    group_id: str
    trees: List[Dict]   # List of related trees
    text: str           # Original text

def get_feature_config(config: Dict) -> Dict:
    """Create feature extractor config"""
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
        },
        'verbose': config.get('verbose', 'normal')
    }

@dataclass
class BatchInfo:
    """Track batch sample info for coverage tracking"""
    group_indices: List[int]
    group_ids: List[str]  # Group IDs in batch
    anchor_indices: List[int]  # Trees used as anchors
    positive_pairs: List[Tuple[int, int]]  # Positive pair indices
    negative_pairs: List[Tuple[int, int]]  # Negative pair indices

class GroupedTreeDataset(IterableDataset):
    def __init__(self, 
                 data_dir: str,
                 config: Dict,
                 max_group_size: int = 32,
                 shuffle_files: bool = False,
                 num_workers: Optional[int] = None,
                 prefetch_factor: int = 2,
                 max_active_files: int = 2):
        self.data_dir = Path(data_dir)
        self.config = config
        self.max_group_size = max_group_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.max_active_files = max_active_files

        # Get shard files
        self.data_files = sorted(
            [f for f in self.data_dir.glob("part_*_shard_*.json")
             if not f.name.endswith('_counts.json')],
            key=lambda x: (int(x.stem.split('_')[1]), 
                         int(x.stem.split('_shard_')[1]))
        )
        if not self.data_files:
            self.data_files = sorted(self.data_dir.glob("part_*.json"))

        # Load counts
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
                        'n_groups': len(data['groups']),
                        'trees_per_group': [len(g['trees']) for g in data['groups']]
                    })

        # Initialize feature extractor
        with open(self.data_files[0]) as f:
            data = json.load(f)
            self.requires_embeddings = data['requires_word_embeddings']
            if self.requires_embeddings:
                feature_config = get_feature_config(config)
                self.embedding_extractor = FeatureExtractor(feature_config)

        if shuffle_files:
            random.shuffle(self.data_files)

    def _load_embeddings(self, tree: Dict) -> torch.Tensor:
        """Load embeddings for tree nodes"""
        if not tree['node_features_need_word_embs_prepended']:
            return torch.tensor(tree['node_features'])
            
        embeddings = []
        for word, lemma in tree['node_texts']:
            emb = None
            # Try lemma first
            if lemma in self.embedding_extractor.embedding_cache:
                emb = self.embedding_extractor.embedding_cache[lemma]
            # Try word form
            elif word in self.embedding_extractor.embedding_cache:
                emb = self.embedding_extractor.embedding_cache[word]
            # Generate and cache if needed
            if emb is None:
                emb = self.embedding_extractor.get_word_embedding(lemma)
            embeddings.append(emb)

        word_embeddings = torch.stack(embeddings)
        node_features = torch.tensor(tree['node_features'])
        return torch.cat([word_embeddings, node_features], dim=-1)

    def __iter__(self):
        """Iterate over trees yielding (tree, group_id) pairs"""
        worker_info = get_worker_info()
        if worker_info:
            files_to_process = self.data_files[worker_info.id::worker_info.num_workers]
        else:
            files_to_process = self.data_files

        active_files = deque(maxlen=self.max_active_files)
        
        for file in files_to_process:
            with open(file) as f:
                data = json.load(f)
            active_files.append(file)

            # Process each group
            for group_idx, group in enumerate(data['groups']):
                if not group['trees']:
                    continue
                group_id = group['group_id']
                trees = group['trees'][:self.max_group_size]

                # Yield each tree with group info
                for tree_idx, tree in enumerate(trees):
                    if tree['node_features_need_word_embs_prepended']:
                        tree = dict(tree)  # Make copy
                        tree['node_features'] = self._load_embeddings(tree)
                    yield {
                        'tree': tree, 
                        'group_id': group_id,
                        'group_idx': group_idx,
                        'tree_idx': tree_idx,
                        'text': tree['text']
                    }

            if len(active_files) >= self.max_active_files:
                old_file = active_files.popleft()
                gc.collect()

    def __len__(self):
        return sum(sum(counts['trees_per_group']) for counts in self.file_counts)

class ContrastiveBatchCollator:
    def __init__(self,
                 pos_pairs_per_anchor: int = 2,
                 neg_pairs_per_anchor: int = 4,
                 min_groups_per_batch: int = 4,
                 anchors_per_group: int = 2):
        self.pos_per_anchor = pos_pairs_per_anchor
        self.neg_per_anchor = neg_pairs_per_anchor 
        self.min_groups = min_groups_per_batch
        self.anchors_per_group = anchors_per_group

    def __call__(self, batch: List[Dict]) -> Tuple[GraphData, BatchInfo]:
        # Group by group_id
        groups = defaultdict(list)
        group_ids = {}
        for item in batch:
            groups[item['group_idx']].append(item)
            group_ids[item['group_idx']] = item['group_id']


        if len(groups) < self.min_groups:
            raise ValueError(f"Need at least {self.min_groups} groups")

        all_trees = []
        tree_indices = {}  # Map (group_id, position) to index in all_trees
        anchors = []
        pos_pairs = []
        neg_pairs = []

        # Process each group
        for group_idx, items in groups.items():
            group_trees = []

            # Add trees to batch
            for item in items:
                idx = len(all_trees)
                all_trees.append(item['tree'])
                tree_indices[(group_idx, item['tree_idx'])] = idx
                group_trees.append(idx)

            # Sample anchors
            n_anchors = min(len(group_trees), self.anchors_per_group)
            group_anchors = random.sample(group_trees, n_anchors)
            anchors.extend(group_anchors)

            # Add anchors and create positive pairs
            for anchor in group_anchors:
                pos_candidates = [idx for idx in group_trees if all_trees[idx]['text'] != all_trees[anchor]['text']]
                if pos_candidates:
                    pos_indices = random.sample(
                        pos_candidates,
                        min(len(pos_candidates), self.pos_per_anchor)
                    )
                    pos_pairs.extend([(anchor, pos) for pos in pos_indices])

        # Create negative pairs across groups
        for anchor in anchors:
            # Get anchor's group
            anchor_group = None
            for (g_idx, _), idx in tree_indices.items():
                if idx == anchor:
                    anchor_group = g_idx
                    break
                    
            # Get trees from other groups
            neg_candidates = []
            for (g_idx, _), idx in tree_indices.items():
                if g_idx != anchor_group:
                    neg_candidates.append(idx)
                    
            if neg_candidates:
                neg_indices = random.sample(
                    neg_candidates,
                    min(len(neg_candidates), self.neg_per_anchor)
                )
                neg_pairs.extend([(anchor, neg) for neg in neg_indices])

        graphs = convert_tree_to_graph_data(all_trees)
        batch_info = BatchInfo(
            group_indices=list(groups.keys()),
            group_ids=list(group_ids.values()),
            anchor_indices=anchors,
            positive_pairs=pos_pairs,
            negative_pairs=neg_pairs
        )

        return graphs, batch_info

def get_dataloader(dataset: GroupedTreeDataset,
                  batch_size: int,
                  pos_pairs_per_anchor: int = 2,
                  neg_pairs_per_anchor: int = 4,
                  min_groups_per_batch: int = 4,
                  anchors_per_group: int = 2,
                  **kwargs) -> DataLoader:
    """Get DataLoader with contrastive batch sampling"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=ContrastiveBatchCollator(
            pos_pairs_per_anchor=pos_pairs_per_anchor,
            neg_pairs_per_anchor=neg_pairs_per_anchor,
            min_groups_per_batch=min_groups_per_batch,
            anchors_per_group=anchors_per_group
        ),
        **kwargs
    )


# # data/grouped_tree_dataset.py
# from dataclasses import dataclass
# from typing import Dict, List, Optional
# import torch
# from torch.utils.data import Dataset, DataLoader
# from TMN_DataGen import FeatureExtractor, count_up_groups
# from pathlib import Path
# from .batch_utils import ContrastiveBatchCollator
# import json
# import logging
# import random
# import multiprocessing as mp

# logger = logging.getLogger(__name__)

# @dataclass
# class TreeGroup:
#     """Container for a group of related trees"""
#     group_id: str
#     trees: List[Dict]   # List of related trees
#     text: str           # Original text

# def get_feature_config(config: Dict) -> Dict:
#     """Create feature extractor config"""
#     return {
#         'feature_extraction': {
#             'word_embedding_model': config.get('word_embedding_model', 'bert-base-uncased'),
#             'use_gpu': config.get('use_gpu', True) and torch.cuda.is_available(),
#             'cache_embeddings': True,
#             'embedding_cache_dir': config.get('embedding_cache_dir', 'embedding_cache'),
#             'do_not_store_word_embeddings': False,
#             'is_runtime': True,
#             'shard_size': config.get('cache_shard_size', 10000),
#             'num_workers': config.get('cache_workers', 4),
#         },
#         'verbose': config.get('verbose', 'normal')
#     }

# class GroupedTreeDataset(IterableDataset):
#     """Dataset that handles groups of related trees for contrastive learning"""
#     
#     def __init__(self, 
#                  data_dir: str,
#                  config: Dict,
#                  max_group_size: int = 32,
#                  shuffle_files: bool = False,
#                  num_workers: int = None,
#                  prefetch_factor: int = 2,
#                  max_active_files: int = 2
#                  ):
#         """Initialize dataset
#         
#         Args:
#             data_path: Path to JSON data file
#             config: Configuration dict
#             max_group_size: Maximum trees to keep per group
#         """
#         self.data_dir = Path(data_dir)
#         self.config = config
#         self.max_group_size = max_group_size
#         self.num_workers = num_workers or mp.cpu_count() // 2 - 1
#         self.prefetch_factor = prefetch_factor
#         self.shuffle_files = shuffle_files
#         self.max_active_files = max_active_files
#         

#         # Gather all shard files
#         self.data_files = sorted(
#             [f for f in self.data_dir.glob("part_*_shard_*.json") 
#              if not f.name.endswith('_counts.json')],
#             key=lambda x: (int(x.stem.split('_')[1]), 
#                            int(x.stem.split('_shard_')[1]))
#         )
#         
#         # Fallback to regular partitions if no shards found
#         if not self.data_files:
#             self.data_files = sorted(
#                 [f for f in self.data_dir.glob("part_*.json")
#                  if not f.name.endswith('_counts.json')],
#                 key=lambda x: int(x.stem.split('_')[1])
#             )
#         
#         if shuffle_files:
#             from random import shuffle
#             shuffle(self.data_files)

#         # Pre-compute number of groups and number of trees per file/group using count files
#         self.file_group_counts = []
#         for file in self.data_files:
#             count_file = file.parent / f"{file.stem}_counts.json"
#             if count_file.exists():
#                 with open(count_file) as f:
#                     self.file_group_counts.append(json.load(f))
#             else:
#                 with open(file) as f:
#                     self.file_group_counts.append(count_up_groups(json.load(f)))
#         
#         self.total_
#         # Load data
#         with open(self.data_path) as f:
#             data = json.load(f)
#             
#         # Initialize embedding extractor
#         self.requires_embeddings = data['requires_word_embeddings']
#         if self.requires_embeddings:
#             feature_config = get_feature_config(config)
#             self.embedding_extractor = FeatureExtractor(feature_config)
#             
#         # Process groups
#         self.groups = []
#         for group in data['groups']:
#             if not group['trees']:  # Skip empty groups
#                 continue
#                 
#             # Limit group size
#             trees = group['trees']
#             if len(trees) > self.max_group_size:
#                 trees = random.sample(trees, self.max_group_size)
#                 
#             self.groups.append(TreeGroup(
#                 group_id=group['group_id'],
#                 trees=trees,
#                 text=group['text']
#             ))
#             
#         # Build lookup indices
#         self._build_indices()
#         logger.info(f"Loaded {len(self.groups)} groups with {len(self)} total trees")
#         
#     def _build_indices(self):
#         """Build indices for efficient lookup"""
#         self.group_boundaries = []  # (start_idx, end_idx) for each group
#         self.tree_to_group = {}    # Map tree idx to group idx
#         self.tree_to_text = {}     # Map tree idx to original text
#         
#         curr_idx = 0
#         for group_idx, group in enumerate(self.groups):
#             n_trees = len(group.trees)
#             self.group_boundaries.append((curr_idx, curr_idx + n_trees))
#             
#             for i in range(n_trees):
#                 tree_idx = curr_idx + i
#                 self.tree_to_group[tree_idx] = group_idx
#                 self.tree_to_text[tree_idx] = group.trees[i]['text']
#             curr_idx += n_trees
#             
#     def _load_word_embeddings(self, tree: Dict) -> torch.Tensor:
#         """Load or compute word embeddings for a tree"""
#         node_features = torch.tensor(tree['node_features'])
#         if not tree['node_features_need_word_embs_prepended']:
#             return node_features
#             
#         # Get embeddings for each word
#         embeddings = []
#         for word, lemma in tree['node_texts']:
#             # Try lemma first, then word form
#             emb = None
#             if lemma in self.embedding_extractor.embedding_cache:
#                 emb = self.embedding_extractor.embedding_cache[lemma]
#             elif word in self.embedding_extractor.embedding_cache:
#                 emb = self.embedding_extractor.embedding_cache[word]
#             if emb is None:
#                 emb = self.embedding_extractor.get_word_embedding(lemma)
#             embeddings.append(emb)
#             
#         word_embeddings = torch.stack(embeddings)
#         return torch.cat([word_embeddings, node_features], dim=-1)
#         
#     def __len__(self) -> int:
#         return sum(len(g.trees) for g in self.groups)
#         
#     def __getitem__(self, idx: int) -> Dict:
#         """Get single tree and its group info"""
#         group_idx = self.tree_to_group[idx]
#         group = self.groups[group_idx]
#         start_idx, end_idx = self.group_boundaries[group_idx]
#         relative_idx = idx - start_idx
#         
#         tree = group.trees[relative_idx]
#         if self.requires_embeddings:
#             node_features = self._load_word_embeddings(tree)
#         else:
#             node_features = torch.tensor(tree['node_features'])
#             
#         tree = dict(tree)  # Make a copy
#         tree['node_features'] = node_features
#         
#         return {
#             'tree': tree,
#             'group_idx': group_idx,
#             'group_id': group.group_id,
#             'tree_idx': relative_idx
#         }

#     def get_dataloader(self, batch_size: int, pos_pairs_per_anchor:int, neg_pairs_per_anchor:int, min_groups_per_batch:int, anchors_per_group:int, **kwargs):
#         """Get DataLoader with contrastive batch sampling"""
#         return DataLoader(
#             self,
#             batch_size=batch_size,
#             collate_fn=ContrastiveBatchCollator(
#                 pos_pairs_per_anchor=pos_pairs_per_anchor,
#                 neg_pairs_per_anchor=neg_pairs_per_anchor,
#                 min_groups_per_batch=min_groups_per_batch,
#                 anchors_per_group=anchors_per_group
#             ),
#             **kwargs
#         )
