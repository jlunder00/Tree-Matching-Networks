# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

# data/paired_groups_dataset.py
import numpy as np
import json
import gc
import copy
import random
import logging
from collections import defaultdict, deque
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set, Union, Any, Callable
import uuid
from functools import wraps

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from TMN_DataGen import FeatureExtractor
try:
    from .data_utils import convert_tree_to_graph_data, GraphData, normalize
except ImportError:
    from data_utils import convert_tree_to_graph_data, GraphData, normalize

logger = logging.getLogger(__name__)

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
class PairedGroupBatchInfo:
    """Tracks information for a batch of paired groups."""
    # Group information
    group_indices: List[int]                           # Group indices in batch
    group_ids: List[str]                               # Group identifiers
    group_labels: List[float]                          # Label for each group
    
    # Tree indexing
    trees_a_indices: List[List[int]]                   # Indices of trees in set A per group
    trees_b_indices: List[List[int]]                   # Indices of trees in set B per group
    tree_to_group_map: Dict[int, int]                  # Maps tree index to group index
    tree_to_set_map: Dict[int, str]                    # Maps tree index to 'a' or 'b'
    
    # Pair information
    pair_indices: List[Tuple[int, int, float]]         # (tree_a_idx, tree_b_idx, label)
    anchor_indices: List[int]                          # Global indices of anchor trees
    
    # Settings
    strict_matching: bool                              # Whether strict matching is enabled
    contrastive_mode: bool                             # Whether in contrastive mode
    
    def get_subgroup_trees(self, group_idx: int, set_id: str) -> List[int]:
        """Get tree indices for a specific subgroup."""
        if set_id == 'a':
            return self.trees_a_indices[group_idx]
        else:
            return self.trees_b_indices[group_idx]
    
    def get_all_subgroup_indices(self) -> List[Tuple[int, str]]:
        """Get all (group_idx, set_id) combinations."""
        result = []
        for g_idx in range(len(self.group_indices)):
            result.append((g_idx, 'a'))
            result.append((g_idx, 'b'))
        return result

class PairedGroupsDatasetBase(IterableDataset):
    """
    Base class for datasets loading groups with paired tree sets.
    
    Each group contains:
    - 'trees': First set of trees (set A)
    - 'trees_b': Second set of trees (set B)
    - 'label': Label indicating similarity/relationship
    """
    
    def __init__(self, 
                 data_dir: str,
                 config: Dict,
                 batch_size: int = 32,
                 shuffle_files: bool = False,
                 prefetch_factor: int = 2,
                 max_active_files: int = 2,
                 contrastive_mode: bool = False,
                 min_trees_per_group: int = 1,
                 filter_labels: Optional[Set[float]] = None,
                 label_map: Optional[Dict] = {},
                 label_norm: Optional[Dict[str, Tuple[float, float]]] = None,
                 text_mode: bool = False,
                 tokenizer = None,
                 max_length: int = 512,
                 allow_text_files: bool = False,
                 **kwargs):
        """
        Initialize the base dataset.
        
        Args:
            data_dir: Directory containing data files
            config: Configuration dictionary
            batch_size: Target batch size (meaning depends on model type)
            shuffle_files: Whether to shuffle files
            prefetch_factor: Prefetch factor for data loading
            max_active_files: Maximum number of files to keep in memory
            contrastive_mode: Whether to use contrastive loss
            min_trees_per_group: Minimum number of trees required per subgroup
            filter_labels: Set of labels to keep (filter out others)
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
            
        self.label_map = label_map
        self.label_norm = label_norm
        self.config = config
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.prefetch_factor = prefetch_factor
        self.max_active_files = max_active_files
        self.contrastive_mode = contrastive_mode
        self.min_trees_per_group = min_trees_per_group
        self.filter_labels = filter_labels
        self._batches_provided = 0
        
        # Calculate batch parameters - will be done in subclasses
        self.target_groups = None  # Set in subclasses
        self.adjusted_batch_size = None  # Set in subclasses
        self.self_pair = None
        
        self.allow_text_files = allow_text_files
        self.text_mode = text_mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.allow_text_files and not self.text_mode:
            logger.warning("text mode not activated when text files are allowed. Forcing text mode in dataloader")
            self.text_mode = True
        
        # Gather files
        self.data_files = []
        for data_dir in self.data_dirs:
            all_files = list(data_dir.glob("part_*_shard_*.json")) + [] if not self.allow_text_files else list(data_dir.glob("part_*_shard_*.txt"))
            files = sorted(
                [f for f in all_files 
                 if not f.name.endswith('_counts.json')],
                key=lambda x: (int(x.stem.split('_')[1]), 
                             int(x.stem.split('_shard_')[1]))
            )
            if not files:
                all_files = list(data_dir.glob("part_*.json")) + [] if not self.allow_text_files else list(data_dir.glob("part_*.txt"))
                files = sorted(
                    [f for f in all_files
                     if not f.name.endswith('_counts.json')],
                    key=lambda x: int(x.stem.split('_')[1])
                )
            self.data_files.extend(files)
            
        if not self.data_files:
            raise ValueError(f"No data files found in {self.data_dirs}")
            
        if self.shuffle_files:
            random.shuffle(self.data_files)
            
        # Load a sample file to check for embedding requirements
        data = self._dataloader_handlers[file.suffix](self.data_files[0])
        self.requires_embeddings = data.get('requires_word_embeddings', False) and not self.text_mode
            
        if self.requires_embeddings:
            feat_config = self._get_feature_config()
            self.embedding_extractor = FeatureExtractor(feat_config)

        # Load count files for better estimation
        self.file_counts = {}
        for file in self.data_files:
            count_file = file.parent / f"{file.stem}_counts.json"
            if count_file.exists():
                with open(count_file) as f:
                    self.file_counts[file] = json.load(f)
        
        # Calculate actual averages if count files exist
        if self.file_counts:
            total_trees_a = 0
            total_trees_b = 0
            total_groups = 0
            
            for count_data in self.file_counts.values():
                total_groups += count_data.get('n_groups', 0)
                trees_per_group = count_data.get('trees_per_group', [])
                trees_b_per_group = count_data.get('trees_b_per_group', []) if not self.self_pair else trees_per_group
                if self.self_pair is None and trees_b_per_group == 0 and trees_per_group > 0:
                    self.self_pair = True
                
                total_trees_a += sum(trees_per_group)
                total_trees_b += sum(trees_b_per_group)
            if self.self_pair is None:
                self.self_pair = False

            if self.self_pair:
                for file, count_data in self.file_counts.items():
                    if 'trees_per_group' in count_data:
                        count_data['trees_b_per_group'] = count_data['trees_per_group']
            
            if total_groups > 0:
                self.avg_trees_per_subgroup_a = total_trees_a / total_groups
                self.avg_trees_per_subgroup_b = total_trees_b / total_groups
                logger.info(f"Calculated average trees per subgroup from count files: "
                           f"A: {self.avg_trees_per_subgroup_a:.2f}, "
                           f"B: {self.avg_trees_per_subgroup_b:.2f}")
            else:
                self.avg_trees_per_subgroup_a = kwargs.get('avg_trees_per_subgroup', 5)
                self.avg_trees_per_subgroup_b = kwargs.get('avg_trees_per_subgroup', 5)
        else:
            self.avg_trees_per_subgroup_a = kwargs.get('avg_trees_per_subgroup', 5)
            self.avg_trees_per_subgroup_b = kwargs.get('avg_trees_per_subgroup', 5)
        
        # Store group counts from loaded files
        self.group_counts = {}  # Will be populated during _add_groups_from_file
            
        # Buffer for groups
        self.group_buffers = {}
        # Queue of files to process
        self.file_queue = deque(self.data_files)
        self.active_files = deque(maxlen=self.max_active_files)

    @dataloader_handler(".json")
    def parse_json(self, file):
        with open(file) as f:
            data = json.load(f)
        return data

    @dataloader_handler(".txt")
    def parse_wikiqs_file(self, file):
        data = []
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
        return {
            "groups": data,
            "version": "1.0",
            "requires_word_embeddings": False,
            "format": "infonce-text"
        }


    def get_text_mode(self):
        return self.text_mode

    def reset_epoch(self):
        if not self.file_queue:
            self.file_queue = deque(self.data_files)
            if self.shuffle_files:
                files = list(self.file_queue)
                random.shuffle(files)
                self.file_queue = deque(files)
        self.group_buffers.clear()
        self.active_files.clear()
        self._batches_provided = 0
        
    def _get_feature_config(self) -> Dict:
        """Create feature extractor config."""
        return {
            'feature_extraction': {
                'word_embedding_model': self.config.get('word_embedding_model', 'bert-base-uncased'),
                'use_gpu': self.config.get('use_gpu', True) and torch.cuda.is_available(),
                'cache_embeddings': True,
                'embedding_cache_dir': self.config.get('embedding_cache_dir', 'embedding_cache'),
                'do_not_store_word_embeddings': False,
                'is_runtime': True,
            },
            'verbose': self.config.get('verbose', 'normal')
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
        
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def _tokenize_pair(self, text_a: str, text_b: str) -> Dict[str, torch.Tensor]:
        """Tokenize a pair of texts using the provided tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when text_mode is enabled")
            
        # Handle tokenization with padding and truncation
        encoded = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {k: v.squeeze(0) for k, v in encoded.items()}
        
    def _add_groups_from_file(self, file: Path):
        """Load a file and add its groups to the buffer."""
        try:
            data = self._dataloader_handlers[file.suffix](file)
        except Exception as e:
            logger.error(f"Error loading file {file}: {e}")
            return
        
        self.active_files.append(file)
        
        for group in data.get('groups', []):
            if not group.get('trees') or not group.get('trees_b'):
                continue
                
            group_id = group.get('group_id', f"group_{len(self.group_buffers)}")
            group_idx = group.get('group_idx', 0)
            label_raw = group.get('label', 1.0 if self.contrastive_mode else 0.0)
            if isinstance(label_raw, str):
                if label_raw not in self.label_map.keys():
                    print(label_raw)
                    print(list(group.keys()))
                    print(group.get('text', 'wrong key'))
                    print(group.get('text_b', 'wrong key'))
                    continue
                label_raw = self.label_map[label_raw]
            label = float(label_raw)
            if self.label_norm is not None:
                label = normalize(label, self.label_norm['old'][0], self.label_norm['old'][1], self.label_norm['new'][0], self.label_norm['new'][1])
            
            # Filter by label if needed
            if self.filter_labels is not None and label not in self.filter_labels:
                continue
                
            # For contrastive mode, only consider positive pairs within groups
            if not self.config['data'].get('allow_negatives', False) and self.contrastive_mode and label < 0:
                continue
            if not self.config['data'].get('allow_neutrals', False) and self.contrastive_mode and label == 0:
                continue
            if not self.config['data'].get('allow_positives', True) and self.contrastive_mode and label > 0:
                continue

            # Count trees in each subgroup
            trees_a_count = len(group.get('trees', []))
            trees_b_count = len(group.get('trees_b', [])) if not self.self_pair else trees_a_count

            # Check if both subgroups have enough trees
            if (trees_a_count < self.min_trees_per_group or
                trees_b_count < self.min_trees_per_group):
                continue

            # Store tree counts for better batch calculation
            # print(group_id)
            # print(trees_a_count)
            # print(trees_b_count)
            self.group_counts[group_id] = {
                'trees_a': trees_a_count,
                'trees_b': trees_b_count
            }
                
            # Process trees from set A
            trees_a = []
            for tree_idx, tree in enumerate(group.get('trees', [])):
                if tree.get('node_features_need_word_embs_prepended', False) and self.requires_embeddings:
                    tree = dict(tree)
                    tree['node_features'] = self._load_embeddings(tree)
                trees_a.append({
                    'tree': tree,
                    'group_id': group_id,
                    'group_idx': group_idx,
                    'tree_idx': tree_idx,
                    'text': tree.get('text', ''),
                })
                
            # Process trees from set B
            trees_b = []
            k = 'trees_b' if not self.self_pair else 'trees_a'
            for tree_idx, tree in enumerate(group.get(k, [])):
                if tree.get('node_features_need_word_embs_prepended', False) and self.requires_embeddings:
                    tree = dict(tree)
                    tree['node_features'] = self._load_embeddings(tree)
                trees_b.append({
                    'tree': tree,
                    'group_id': group_id,
                    'group_idx': group_idx,
                    'tree_idx': tree_idx,
                    'text': tree.get('text', ''),
                })
                
            # Add to buffer
            self.group_buffers[group_id] = {
                'group_idx': group_idx,
                'label': label,
                'trees_a': trees_a,
                'trees_b': trees_b
            }
            
    def _fill_buffer(self):
        """Keep loading files until we have enough groups."""
        while len(self.group_buffers) < self.target_groups and self.file_queue:
            file = self.file_queue.popleft()
            self._add_groups_from_file(file)
            
            # Clean up if too many active files
            if len(self.active_files) >= self.max_active_files:
                self.active_files.popleft()
                gc.collect()
    
    def _sort_trees_by_size(self, trees: List[Dict]) -> List[Dict]:
        """Sort trees by node count (proxy for information density)."""
        return sorted(trees, 
                    key=lambda t: len(t['tree'].get('node_features', [])), 
                    reverse=True)
    
    def _generate_batch(self):
        """
        Template method to generate a batch.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _generate_batch")
    
    def _collate_graphs(self, graphs):
        """Collate multiple GraphData instances into one."""
        from_idx = []
        to_idx = []
        graph_idx = []
        node_features = []
        edge_features = []
        last_graph_idx = 0
        
        for g in graphs:
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
            n_graphs=last_graph_idx
        )
        return graph_data
    
    def __iter__(self):
        """Iterate over batches."""
        worker_info = get_worker_info()
        if worker_info is not None:
            # Split files across workers
            files = list(self.file_queue)
            files = files[worker_info.id::worker_info.num_workers]
            self.file_queue = deque(files)
        
        max_batches = self.config['data'].get('max_batches_per_epoch', 250)
        
        while self._batches_provided < max_batches:
            self._fill_buffer()
            batch = self._generate_batch()
            
            if batch is not None:
                self._batches_provided += 1
                yield batch
            elif not self.file_queue:
                break
            else:
                continue
    
    def __len__(self):
        """
        Estimate total number of batches.
        Uses actual counts when available, falls back to estimates otherwise.
        """
        max_batches = self.config['data'].get('max_batches_per_epoch', 250)
        
        print(self.group_counts)
        if hasattr(self, 'file_counts') and self.file_counts:
            # If we have actual group counts, use them for better estimation
            if isinstance(self, StrictMatchingDataset):
                total_pairs = 0
                # For strict matching, count pairs
                if hasattr(self, 'contrastive_mode') and self.contrastive_mode:
                    for count_data in self.file_counts.values():
                        file_pairs = sum(
                            (count_data['trees_per_group'][i] * sum(count_data['trees_b_per_group'])) - count_data['trees_per_group'][i] ** 2
                            for i in range(count_data['n_groups'])
                        )
                        total_pairs += file_pairs
                else:
                    for count_data in self.file_counts.values():
                        file_pairs = sum(
                            count_data['trees_per_group'][i] * count_data['trees_b_per_group'][i] 
                            for i in range(count_data['n_groups']) 
                        )
                        total_pairs +=  file_pairs
                    
                return min(total_pairs // self.batch_size, max_batches)
            else:
                # For non-strict, count trees
                total_trees = 0
                for count_data in self.file_counts.values():
                    if isinstance(self, MatchingDataset):
                        file_trees = sum(
                            max(count_data['trees_per_group'][i], count_data['trees_b_per_group'][i])*2
                            for i in range(count_data['n_groups'])
                        )
                    else:
                        file_trees = sum(
                            count_data['trees_per_group'][i] + count_data['trees_b_per_group'][i]
                            for i in range(count_data['n_groups'])
                        )
                    total_trees += file_trees
                return min(total_trees // self.batch_size, max_batches)
        else:
            # Fall back to simple estimate
            # return min(max_batches, 100)
            return max_batches


class StrictMatchingDataset(PairedGroupsDatasetBase):
    """
    Dataset for strict matching model. 
    Creates explicit pairs for all combinations of trees in subgroups.
    Batch size is interpreted as number of pairs.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_pairs = self.batch_size
        logger.info(f"Strict matching - Target pairs: {self.target_pairs}")
    

    def _create_all_pairs(self, trees_a: List[Dict], trees_b: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Create all possible pairs between trees_a and trees_b."""
        if not trees_a or not trees_b:
            return []
            
        # Create all pairs
        pairs = []
        for i, tree_a in enumerate(trees_a):
            for j, tree_b in enumerate(trees_b):
                if self.self_pair and i == j: #same tree, skip
                    continue
                pairs.append((tree_a, tree_b))
                
        return pairs
        
    def _generate_batch(self):
        """Generate a batch with explicit pairs for strict matching."""
        if not self.group_buffers:
            return None
            
        # Prepare batch data
        batch_trees = []  # All trees
        group_indices = []  # Group indices
        group_ids = []  # Group IDs
        group_labels = []  # Group labels
        trees_a_indices = []  # Indices of trees from set A for each group
        trees_b_indices = []  # Indices of trees from set B for each group
        tree_to_group_map = {}  # Maps tree index to group index
        tree_to_set_map = {}  # Maps tree index to set ('a' or 'b')
        pair_indices = []  # (tree_a_idx, tree_b_idx, label)
        anchor_indices = []  # Global indices of anchor trees
        
        # Select groups while tracking pair count
        selected_groups = []
        pair_count = 0
        
        # Sort groups by their pair contribution (product of tree counts)
        # This helps us efficiently pack groups into batches
        group_contributions = []
        for group_id, group_info in self.group_buffers.items():
            n_pairs = len(group_info['trees_a']) * len(group_info['trees_b'])
            group_contributions.append((group_id, group_info, n_pairs))
        
        # Sort by contribution (smallest first for better batch packing)
        group_contributions.sort(key=lambda x: x[2])
        
        # Greedy algorithm to fill batch without exceeding target
        for group_id, group_info, contribution in group_contributions:
            if pair_count + contribution > self.target_pairs and selected_groups:
                continue
                
            selected_groups.append((group_id, group_info))
            pair_count += contribution
            
            # Stop if we're close enough to target
            if pair_count >= self.target_pairs * 0.9:
                break
                
        if not selected_groups and self.group_buffers:
            # If no suitable groups found, take the smallest
            group_id, group_info, _ = min(group_contributions, key=lambda x: x[2])
            selected_groups.append((group_id, group_info))
        # # Select groups while tracking pair count
        # selected_groups = []
        # pair_count = 0
        # target_count = self.target_pairs

        # sorted_groups = sorted(
        #     self.group_buffers.items(),
        #     key=lambda x: len(x[1]['trees_a']) * len(x[1]['trees_b'])
        # )
        # 
        # for group_id, group_info in sorted_groups:
        #     # Calculate contribution of this group (all pairs between trees_a and trees_b)
        #     contribution = len(group_info['trees_a']) * len(group_info['trees_b'])
        #     
        #     # Add group if it won't exceed our target
        #     if pair_count + contribution > target_count and selected_groups:
        #         continue
        #     selected_groups.append((group_id, group_info))
        #     pair_count += contribution
        #         
        #     # Stop if we have enough groups
        #     if pair_count >= target_count * 0.9:
        #         break
                
        if not selected_groups and self.group_buffers:
            group_id = next(iter(self.group_buffers))
            selected_groups.append((group_id, self.group_buffers[group_id]))

        # Process each selected group
        for batch_group_idx, (group_id, group_info) in enumerate(selected_groups):
            group_indices.append(batch_group_idx)
            group_ids.append(group_id)
            group_labels.append(group_info['label'])
            
            # Get trees from both sets
            trees_a = group_info['trees_a']
            k = 'trees_b' if not self.self_pair else 'trees_a'
            trees_b = group_info[k]
            
            # Create all pairs
            pairs = self._create_all_pairs(trees_a, trees_b)
            
            # For each pair, add both trees to batch
            group_a_indices = []
            group_b_indices = []
            
            for tree_a, tree_b in pairs:
                # Add trees to batch
                tree_a_idx = len(batch_trees)
                batch_trees.append(tree_a)
                anchor_indices.append(tree_a_idx)  # Consider all set A trees as anchors
                
                tree_b_idx = len(batch_trees)
                batch_trees.append(tree_b)
                
                # Track indices
                group_a_indices.append(tree_a_idx)
                group_b_indices.append(tree_b_idx)
                
                # Update mappings
                tree_to_group_map[tree_a_idx] = batch_group_idx
                tree_to_group_map[tree_b_idx] = batch_group_idx
                
                tree_to_set_map[tree_a_idx] = 'a'
                tree_to_set_map[tree_b_idx] = 'b'
                
                # Add pair
                pair_indices.append((tree_a_idx, tree_b_idx, group_info['label']))
            
            trees_a_indices.append(group_a_indices)
            trees_b_indices.append(group_b_indices)
            
            # Remove processed group from buffer
            del self.group_buffers[group_id]
        
        # Create batch info
        batch_info = PairedGroupBatchInfo(
            group_indices=group_indices,
            group_ids=group_ids,
            group_labels=group_labels,
            trees_a_indices=trees_a_indices,
            trees_b_indices=trees_b_indices,
            tree_to_group_map=tree_to_group_map,
            tree_to_set_map=tree_to_set_map,
            pair_indices=pair_indices,
            anchor_indices=anchor_indices,
            strict_matching=True,
            contrastive_mode=self.contrastive_mode
        )

        # If text_mode is enabled, process texts individually to preserve indexing
        if self.text_mode:
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
            data_output = batch_encoded
        else:
            # Create GraphData
            all_trees = [item['tree'] for item in batch_trees]
            
            # For strict matching, create GraphData for all pairs
            all_graph_data = []
            
            # Track processed pairs to avoid duplicates
            processed_pairs = set()
            
            for i, (a_idx, b_idx, _) in enumerate(pair_indices):
                if (a_idx, b_idx) in processed_pairs:
                    continue
                
                tree_a = batch_trees[a_idx]['tree']
                tree_b = batch_trees[b_idx]['tree']
                pair_graphs = convert_tree_to_graph_data([tree_a, tree_b])
                all_graph_data.append(pair_graphs)
                processed_pairs.add((a_idx, b_idx))
                
            graphs = self._collate_graphs(all_graph_data)
            data_output = graphs
        
        
        return data_output, batch_info


class NonStrictDataset(PairedGroupsDatasetBase):
    """
    Base class for non-strict datasets (embedding and matching).
    Batch size is interpreted as number of trees.
    """
    
    def __init__(self, *args, max_skips=5, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Calculate batch parameters for non-strict
        self.max_skips = max_skips
        avg_trees_per_group = self.avg_trees_per_subgroup_a + self.avg_trees_per_subgroup_b
        self.target_groups = max(1, self.batch_size // (avg_trees_per_group))
        logger.info(f"Non-strict - Target groups: {self.target_groups}")
        logger.info(f"based on avg {avg_trees_per_group:.2f} trees per group")

    def _calculate_group_contribution(self, group_info):
        raise NotImplementedError("Subclasses must implement _calculate_group_contribution")
    
    def _select_groups(self):
        """
        Greedy algorithm to select groups without exceeding batch size.
        Uses actual tree counts from each group instead of averages.
        """
        selected_groups = []
        current_count = 0
        target_count = self.batch_size
        skips = 0

        # Make a copy of keys to avoid modification issues
        group_keys = list(self.group_buffers.keys())
        random.shuffle(group_keys)  # Randomize for better variety
        
        # Sort groups by their contribution for more efficient packing
        sorted_groups = []
        for group_id in group_keys:
            group_info = self.group_buffers[group_id]
            contribution = self._calculate_group_contribution(group_info)
            sorted_groups.append((group_id, group_info, contribution))
        
        # Sort by contribution (smallest first for better batch packing)
        sorted_groups.sort(key=lambda x: x[2])
        
        # Try to fill the batch as close as possible to target
        for group_id, group_info, contribution in sorted_groups:
            # Skip if adding this group would exceed target (unless it's our first group)
            if current_count + contribution > target_count and selected_groups:
                skips += 1
                if skips >= self.max_skips:
                    break
                continue
                
            selected_groups.append((group_id, group_info))
            current_count += contribution
            del self.group_buffers[group_id]
            
            # Stop if we're close enough to target
            if current_count >= target_count * 0.9:
                break
        
        # If we couldn't find any suitable groups, take the smallest available
        if not selected_groups and self.group_buffers:
            smallest_group = min(sorted_groups, key=lambda x: x[2])
            group_id, group_info, _ = smallest_group
            selected_groups.append((group_id, group_info))
            del self.group_buffers[group_id]

        logger.debug(f"Selected {len(selected_groups)} groups with {current_count} " 
                   f"trees out of target {target_count}")
        return selected_groups

    def _process_trees(self, selected_groups):
        """Process trees from selected groups."""
        batch_trees = []
        group_indices = []
        group_ids = []
        group_labels = []
        trees_a_indices = []
        trees_b_indices = []
        tree_to_group_map = {}
        tree_to_set_map = {}
        anchor_indices = []
        
        # Process each selected group
        for batch_group_idx, (group_id, group_info) in enumerate(selected_groups):
            group_indices.append(batch_group_idx)
            group_ids.append(group_id)
            group_labels.append(group_info['label'])
            
            # Process trees from set A
            group_a_indices = []
            for tree in group_info['trees_a']:
                tree_idx = len(batch_trees)
                batch_trees.append(tree)
                group_a_indices.append(tree_idx)
                tree_to_group_map[tree_idx] = batch_group_idx
                tree_to_set_map[tree_idx] = 'a'
                anchor_indices.append(tree_idx)  # Consider all set A trees as anchors
            
            # Process trees from set B
            group_b_indices = []
            k = 'trees_b' if not self.self_pair else 'trees_a'
            for tree in group_info[k]:
                tree_idx = len(batch_trees)
                batch_trees.append(tree)
                group_b_indices.append(tree_idx)
                tree_to_group_map[tree_idx] = batch_group_idx
                tree_to_set_map[tree_idx] = 'b'
            
            trees_a_indices.append(group_a_indices)
            trees_b_indices.append(group_b_indices)
            
            # # Remove processed group from buffer
            # del self.group_buffers[group_id]
            
        return (
            batch_trees, 
            group_indices, 
            group_ids, 
            group_labels, 
            trees_a_indices,
            trees_b_indices,
            tree_to_group_map,
            tree_to_set_map,
            anchor_indices
        )

    def _process_final_data(self, batch_trees, pair_indices):
        """
        Process batch data into the format expected by the model.
        To be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _process_data")
    
    def _process_final_text_data(self, batch_trees):
        """Shared text processing logic"""
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
        return batch_encoded
    
    def _prepare_pairs(self, batch_data):
        """
        Prepare pairs for the batch.
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _prepare_pairs")
    
    def _generate_batch(self):
        """
        Template method for generating a batch.
        Handles the overall flow but delegates pair creation to subclasses.
        """
        if not self.group_buffers:
            return None
            
        # 1. Select groups
        selected_groups = self._select_groups()
        
        # 2. Process trees from selected groups
        batch_data = self._process_trees(selected_groups)
        (
            batch_trees, 
            group_indices, 
            group_ids, 
            group_labels, 
            trees_a_indices,
            trees_b_indices,
            tree_to_group_map,
            tree_to_set_map,
            anchor_indices
        ) = batch_data
        
        # 3. Prepare pairs (implemented by subclasses)
        pair_indices = self._prepare_pairs(batch_data)
        
        data_output = self._process_final_data(batch_trees, pair_indices)
        # if self.text_mode:
        #     text_inputs = [self._tokenize_text(tree.get('text', '')) for tree in batch_trees]
        #     if text_inputs:
        #         batch_encoded = {k: torch.cat([p[k] for p in text_inputs]) for k in text_inputs[0].keys()}
        #     else:
        #         # Empty batch fallback
        #         batch_encoded = {
        #             'input_ids': torch.zeros((0, self.max_length), dtype=torch.long),
        #             'attention_mask': torch.zeros((0, self.max_length), dtype=torch.long),
        #             'token_type_ids': torch.zeros((0, self.max_length), dtype=torch.long)
        #         }
        #     data_output = batch_encoded
        #     
        # else:


        #     # 4. Create GraphData
        #     all_trees = [item['tree'] for item in batch_trees]
        #     data_output = convert_tree_to_graph_data(all_trees)
        
        # 5. Create batch info
        batch_info = PairedGroupBatchInfo(
            group_indices=group_indices,
            group_ids=group_ids,
            group_labels=group_labels,
            trees_a_indices=trees_a_indices,
            trees_b_indices=trees_b_indices,
            tree_to_group_map=tree_to_group_map,
            tree_to_set_map=tree_to_set_map,
            pair_indices=pair_indices,
            anchor_indices=anchor_indices,
            strict_matching=False,
            contrastive_mode=self.contrastive_mode
        )
        
        return data_output, batch_info


class EmbeddingDataset(NonStrictDataset):
    """
    Dataset for embedding model.
    Tracks tree-to-group mapping for loss function.
    Doesn't create explicit pairs - pair formation happens in loss function.
    """

    def _calculate_group_contribution(self, batch_info):
        b_key = 'trees_b' if not self.self_pair else 'trees_a'
        return len(batch_info['trees_a']) + len(batch_info[b_key]) #the number of trees provided to the embedder from this batch is the sum of how many trees there are
    
    def _prepare_pairs(self, batch_data):
        """
        For embedding model, we don't need explicit pairs for the model itself.
        However, we track potential pairs for the loss function.
        """
        (
            batch_trees, 
            group_indices, 
            group_ids, 
            group_labels, 
            trees_a_indices,
            trees_b_indices,
            tree_to_group_map,
            tree_to_set_map,
            anchor_indices
        ) = batch_data
        
        # Record pairs for loss function (mostly for contrastive learning)
        pair_indices = []
        
        # Within-group positive pairs
        for g_idx in range(len(group_indices)):
            # Create positive pairs between trees_a and trees_b within same group
            for a_idx in trees_a_indices[g_idx]:
                for b_idx in trees_b_indices[g_idx]:
                    pair_indices.append((a_idx, b_idx, group_labels[g_idx]))
        
        # For contrastive mode, add negative pairs across groups
        if self.contrastive_mode:
            for g1_idx in range(len(group_indices)):
                for g2_idx in range(len(group_indices)):
                    if g1_idx != g2_idx:  # Different groups
                        # Sample representative trees from each group
                        a_trees = trees_a_indices[g1_idx]
                        b_trees = trees_b_indices[g2_idx]
                        
                        # Create negative pairs (limited to prevent explosion)
                        a_sample = random.sample(a_trees, min(2, len(a_trees)))
                        b_sample = random.sample(b_trees, min(2, len(b_trees)))
                        
                        for a_idx in a_sample:
                            for b_idx in b_sample:
                                pair_indices.append((a_idx, b_idx, 0.0))  # Negative pair
        
        return pair_indices

    def _process_final_data(self, batch_trees, pair_indices):
        """Process data for embedding model"""
        if self.text_mode:
            return self._process_final_text_data(batch_trees)
        else:
            # All trees individually for embedding
            all_trees = [item['tree'] for item in batch_trees]
            return convert_tree_to_graph_data(all_trees)


class MatchingDataset(NonStrictDataset):
    """
    Dataset for matching model without strict matching.
    Creates minimal set of pairs that covers all trees.
    """
    
    def _calculate_group_contribution(self, batch_info):
        b_key = 'trees_b' if not self.self_pair else 'trees_a'
        n = max(len(batch_info['trees_a']), len(batch_info[b_key])) #pairing necessitates an nxn matrix. must find n, n*2 is contribution of a group towards total trees sent to embedder
        return 2*n

    def _create_minimal_pairs(self, trees_a, trees_b, indices_a, indices_b):
        """
        Create minimal set of (index) pairs between trees_a and trees_b that covers all trees.
        Uses size-based pairing strategy.
        """
        if not indices_a or not indices_b:
            return []
            
        # Sort by tree size
        sorted_a = [(i, len(trees_a[i]['tree'].get('node_features', []))) 
                   for i in indices_a]
        sorted_a.sort(key=lambda x: x[1], reverse=True)
        
        sorted_b = [(i, len(trees_b[i]['tree'].get('node_features', []))) 
                   for i in indices_b]
        sorted_b.sort(key=lambda x: x[1], reverse=True)
        if self.self_pair and len(sorted_b) > 1:
            for i in range(0, len(sorted_b)-1, 2):
                sorted_b[i], sorted_b[i+1] = sorted_b[i+1], sorted_b[i]

            if len(sorted_b) % 2 == 1 and len(sorted_b) > 2:
                swap_idx = random.randrange(len(sorted_b) - 1)
                sorted_b[-1], sorted_b[swap_idx] = sorted_b[swap_idx], sorted_b[-1]

            bl = len(sorted_b)
            pshifts = [i for i in range(-bl, bl+1) if i % 2 == 0]

            ws = []
            mp = bl/2
            for s in pshifts:
                d = abs(abs(s) - mp)
                w = 1 / (d+1)
                ws.append(w)

            total = sum(ws)
            ws = [wi/total for wi in ws]

            shift = random.choices(pshifts, weights=ws, k=1)[0]

            sorted_b = sorted_b[shift:] + sorted_b[:shift]

        
        # Create pairs to cover all trees
        pairs = []
        
        # 1. First pass: pair trees until one set is exhausted
        for i in range(min(len(sorted_a), len(sorted_b))):
            pairs.append((sorted_a[i][0], sorted_b[i][0]))
            
        # 2. Second pass: handle extras from longer set
        if len(sorted_a) > len(sorted_b):
            # More A trees than B trees - pair remaining A with B trees (duplicate B)
            for i in range(len(sorted_b), len(sorted_a)):
                b_idx = i % len(sorted_b)  # Cycle through B trees
                pairs.append((sorted_a[i][0], sorted_b[b_idx][0]))
        elif len(sorted_b) > len(sorted_a):
            # More B trees than A trees - pair remaining B with A trees (duplicate A)
            for i in range(len(sorted_a), len(sorted_b)):
                a_idx = i % len(sorted_a)  # Cycle through A trees
                pairs.append((sorted_a[a_idx][0], sorted_b[i][0]))
                
        return pairs
    
    def _prepare_pairs(self, batch_data):
        """
        For matching model, create minimal pairs covering all trees.
        """
        (
            batch_trees, 
            group_indices, 
            group_ids, 
            group_labels, 
            trees_a_indices,
            trees_b_indices,
            tree_to_group_map,
            tree_to_set_map,
            anchor_indices
        ) = batch_data
        
        # Create pairs that cover all trees
        pair_indices = []
        
        # Process each group to create minimal covering pairs
        for g_idx in range(len(group_indices)):
            # Create minimal pairs between trees_a and trees_b within same group
            a_indices = trees_a_indices[g_idx]
            b_indices = trees_b_indices[g_idx]
            
            minimal_pairs = self._create_minimal_pairs(
                batch_trees, batch_trees, a_indices, b_indices
            )
            
            for a_idx, b_idx in minimal_pairs:
                pair_indices.append((a_idx, b_idx, group_labels[g_idx]))
        
        # For contrastive mode, add negative pairs across groups
        if self.contrastive_mode:
            for g1_idx in range(len(group_indices)):
                for g2_idx in range(len(group_indices)):
                    if g1_idx != g2_idx:  # Different groups
                        # Sample representative trees from each group
                        a_indices = trees_a_indices[g1_idx]
                        b_indices = trees_b_indices[g2_idx]
                        
                        # Create a few negative pairs
                        a_sample = random.sample(a_indices, min(2, len(a_indices)))
                        b_sample = random.sample(b_indices, min(2, len(b_indices)))
                        
                        for a_idx in a_sample:
                            for b_idx in b_sample:
                                pair_indices.append((a_idx, b_idx, 0.0))  # Negative pair
        
        return pair_indices


    def _process_final_data(self, batch_trees, pair_indices):
        """Process data for matching model"""
        if self.text_mode:
            return self._process_final_text_data(batch_trees)
        else:
            # Process tree pairs for matching (similar to StrictMatchingDataset)
            all_graph_data = []
            processed_pairs = set()
            
            for a_idx, b_idx, _ in pair_indices:
                if (a_idx, b_idx) in processed_pairs:
                    continue
                
                tree_a = batch_trees[a_idx]['tree']
                tree_b = batch_trees[b_idx]['tree']
                pair_graphs = convert_tree_to_graph_data([tree_a, tree_b])
                all_graph_data.append(pair_graphs)
                processed_pairs.add((a_idx, b_idx))
                
            return self._collate_graphs(all_graph_data)


def create_paired_groups_dataset(
    data_dir, config, model_type='matching', 
    strict_matching=False, contrastive_mode=False,
    text_mode=False, tokenizer=None, max_length=512,
    **kwargs
):
    """Factory function to create the appropriate dataset type."""
    
    base_args = {
        'data_dir': data_dir,
        'config': config,
        'contrastive_mode': contrastive_mode,
        'text_mode': text_mode,
        'tokenizer': tokenizer,
        'max_length': max_length,
        **kwargs
    }
    
    if model_type == 'matching' and strict_matching:
        return StrictMatchingDataset(**base_args)
    elif model_type == 'embedding':
        return EmbeddingDataset(**base_args)
    else:  # model_type == 'matching' and not strict_matching
        return MatchingDataset(**base_args)


def get_paired_groups_dataloader(dataset, num_workers=4, pin_memory=True, persistent_workers=True):
    """Create a DataLoader for PairedGroupsDataset."""
    return DataLoader(
        dataset,
        batch_size=1,  # Already batched by the dataset
        collate_fn=lambda x: x[0],  # Unwrap the singleton
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=dataset.prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 and persistent_workers else False
    )



