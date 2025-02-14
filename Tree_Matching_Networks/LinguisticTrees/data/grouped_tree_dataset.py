# data/grouped_tree_dataset.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from TMN_DataGen import FeatureExtractor
from pathlib import Path
from batch_utils import ContrastiveBatchCollator
import json
import logging
import random

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

class GroupedTreeDataset(Dataset):
    """Dataset that handles groups of related trees for contrastive learning"""
    
    def __init__(self, 
                 data_path: str,
                 config: Dict,
                 max_group_size: int = 32):
        """Initialize dataset
        
        Args:
            data_path: Path to JSON data file
            config: Configuration dict
            max_group_size: Maximum trees to keep per group
        """
        self.data_path = Path(data_path)
        self.config = config
        self.max_group_size = max_group_size
        
        # Load data
        with open(self.data_path) as f:
            data = json.load(f)
            
        # Initialize embedding extractor
        self.requires_embeddings = data['requires_word_embeddings']
        if self.requires_embeddings:
            feature_config = get_feature_config(config)
            self.embedding_extractor = FeatureExtractor(feature_config)
            
        # Process groups
        self.groups = []
        for group in data['groups']:
            if not group['trees1']:  # Skip empty groups
                continue
                
            # Limit group size
            trees = group['trees1']
            if len(trees) > self.max_group_size:
                trees = random.sample(trees, self.max_group_size)
                
            self.groups.append(TreeGroup(
                group_id=group['group_id'],
                trees=trees,
                text=group['text1']
            ))
            
        # Build lookup indices
        self._build_indices()
        logger.info(f"Loaded {len(self.groups)} groups with {len(self)} total trees")
        
    def _build_indices(self):
        """Build indices for efficient lookup"""
        self.group_boundaries = []  # (start_idx, end_idx) for each group
        self.tree_to_group = {}    # Map tree idx to group idx
        self.tree_to_text = {}     # Map tree idx to original text
        
        curr_idx = 0
        for group_idx, group in enumerate(self.groups):
            n_trees = len(group.trees)
            self.group_boundaries.append((curr_idx, curr_idx + n_trees))
            
            for i in range(n_trees):
                tree_idx = curr_idx + i
                self.tree_to_group[tree_idx] = group_idx
                self.tree_to_text[tree_idx] = group.trees[i]['text']
            curr_idx += n_trees
            
    def _load_word_embeddings(self, tree: Dict) -> torch.Tensor:
        """Load or compute word embeddings for a tree"""
        node_features = torch.tensor(tree['node_features'])
        if not tree['node_features_need_word_embs_prepended']:
            return node_features
            
        # Get embeddings for each word
        embeddings = []
        for word, lemma in tree['node_texts']:
            # Try lemma first, then word form
            emb = None
            if lemma in self.embedding_extractor.embedding_cache:
                emb = self.embedding_extractor.embedding_cache[lemma]
            elif word in self.embedding_extractor.embedding_cache:
                emb = self.embedding_extractor.embedding_cache[word]
            if emb is None:
                emb = self.embedding_extractor.get_word_embedding(lemma)
            embeddings.append(emb)
            
        word_embeddings = torch.stack(embeddings)
        return torch.cat([word_embeddings, node_features], dim=-1)
        
    def __len__(self) -> int:
        return sum(len(g.trees) for g in self.groups)
        
    def __getitem__(self, idx: int) -> Dict:
        """Get single tree and its group info"""
        group_idx = self.tree_to_group[idx]
        group = self.groups[group_idx]
        start_idx, end_idx = self.group_boundaries[group_idx]
        relative_idx = idx - start_idx
        
        tree = group.trees[relative_idx]
        if self.requires_embeddings:
            node_features = self._load_word_embeddings(tree)
        else:
            node_features = torch.tensor(tree['node_features'])
            
        tree = dict(tree)  # Make a copy
        tree['node_features'] = node_features
        
        return {
            'tree': tree,
            'group_idx': group_idx,
            'group_id': group.group_id,
            'tree_idx': relative_idx
        }

    def get_dataloader(self, batch_size: int, pos_pairs_per_anchor:int, neg_pairs_per_anchor:int, min_groups_per_batch:int, anchors_per_group:int, **kwargs):
        """Get DataLoader with contrastive batch sampling"""
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=ContrastiveBatchCollator(
                pos_pairs_per_anchor=pos_pairs_per_anchor,
                neg_pairs_per_anchor=neg_pairs_per_anchor,
                min_groups_per_batch=min_groups_per_batch,
                anchors_per_group=anchors_per_group
            ),
            **kwargs
        )
