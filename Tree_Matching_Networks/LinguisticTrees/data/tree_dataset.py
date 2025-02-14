# data/tree_dataset.py
from ...GMN.dataset import GraphSimilarityDataset
from .data_utils import convert_tree_to_graph_data
import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import logging
from ..utils.memory_utils import MemoryMonitor
from TMN_DataGen import FeatureExtractor
import random
import math

logger = logging.getLogger(__name__)

def _get_feature_extractor_config(config: Dict) -> Dict:
    """Create feature extractor config with proper cache settings"""
    feature_config = {
        'feature_extraction': {
            'word_embedding_model': config.get('word_embedding_model', 'bert-base-uncased'),
            'use_gpu': config.get('use_gpu', True) and torch.cuda.is_available(),
            'cache_embeddings': True,  # Always use cache for dataset loading
            'embedding_cache_dir': config.get('embedding_cache_dir', 'embedding_cache'),
            'do_not_store_word_embeddings': False,  # Need embeddings during runtime
            'is_runtime': True,
            'shard_size': config.get('cache_shard_size', 10000),
            'num_workers': config.get('cache_workers', 4),
        },
        'verbose': config.get('verbose', 'normal')
    }
    return feature_config

@dataclass
class TreeGroup:
    """Container for a group of related trees"""
    group_id: str
    trees1: List[Dict]
    trees2: List[Dict]
    text1: str
    text2: str

class GroupedTreeDataset(Dataset):
    """Dataset that handles groups of related trees for contrastive learning"""
    
    def __init__(self, 
                 data_path: str,
                 config: Dict,
                 max_group_size: int = 32,
                 min_positives: int = 2,
                 max_positives: int = 8):
        """Initialize dataset
        
        Args:
            data_path: Path to JSON data file
            config: Configuration dict
            max_group_size: Maximum number of trees to keep per group
            min_positives: Minimum positive pairs per anchor
            max_positives: Maximum positive pairs per anchor
        """
        self.data_path = Path(data_path)
        self.config = config
        self.max_group_size = max_group_size
        self.min_positives = min_positives
        self.max_positives = max_positives
        
        # Load groups
        with open(self.data_path) as f:
            data = json.load(f)
            
        self.requires_embeddings = data['requires_word_embeddings']
        if self.requires_embeddings:
            feature_config = _get_feature_extractor_config(self.config)
            self.embedding_extractor = FeatureExtractor(feature_config)
            
        # Process and store groups
        self.groups: List[TreeGroup] = []
        self._process_groups(data['groups'])
        
        logger.info(f"Loaded {len(self.groups)} groups")
        
    def _process_groups(self, raw_groups: List[Dict]):
        """Process raw groups into TreeGroup objects"""
        for group in raw_groups:
            # Skip empty groups
            if not group['trees1'] or not group['trees2']:
                continue
                
            # Limit group size if needed
            if len(group['trees1']) > self.max_group_size:
                indices = random.sample(range(len(group['trees1'])), self.max_group_size)
                trees1 = [group['trees1'][i] for i in indices]
                trees2 = [group['trees2'][i] for i in indices]
            else:
                trees1 = group['trees1']
                trees2 = group['trees2']
                
            self.groups.append(TreeGroup(
                group_id=group['group_id'],
                trees1=trees1,
                trees2=trees2,
                text1=group['text1'],
                text2=group['text2']
            ))
            
    def _load_word_embeddings(self, tree: Dict) -> torch.Tensor:
        """Load or compute word embeddings for a tree"""
        node_features = torch.tensor(tree['node_features'])
        if not tree['node_features_need_word_embs_prepended']:
            return node_features
            
        # Get embeddings for each word
        embeddings = []
        for word, lemma in tree['node_texts']:
            # Try lemma first, then word form
            if lemma in self.embedding_extractor.embedding_cache:
                emb = self.embedding_extractor.embedding_cache[lemma]
            elif word in self.embedding_extractor.embedding_cache:
                emb = self.embedding_extractor.embedding_cache[word]

            if emb is None:
                emb = self.embedding_extractor.get_word_embedding(lemma)
            if emb is None:
                emb = self.embedding_extractor.get_word_embedding(word)
            embeddings.append(emb)
            
        word_embeddings = torch.stack(embeddings)
        return torch.cat([word_embeddings, node_features], dim=-1)
        
    def __len__(self) -> int:
        return len(self.groups)
        
    def __getitem__(self, idx: int) -> TreeGroup:
        """Get a single group"""
        return self.groups[idx]

class StreamingGroupedTreeDataset(IterableDataset):
    """Streaming version of GroupedTreeDataset for large datasets"""
    
    def __init__(self,
                 data_dir: str,
                 config: Dict,
                 max_group_size: int = 32,
                 min_positives: int = 2,
                 max_positives: int = 8,
                 num_workers: Optional[int] = None,
                 buffer_size: int = 1000):
        """Initialize streaming dataset
        
        Args:
            data_dir: Directory containing data shards
            config: Configuration dict
            max_group_size: Maximum trees per group
            min_positives: Minimum positive pairs per anchor
            max_positives: Maximum positive pairs per anchor
            num_workers: Number of worker processes
            buffer_size: Size of prefetch buffer
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = config
        self.max_group_size = max_group_size
        self.min_positives = min_positives
        self.max_positives = max_positives
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        
        # Get all shard files
        self.shard_files = sorted(list(self.data_dir.glob("*.json")))
        if not self.shard_files:
            raise ValueError(f"No shards found in {data_dir}")
            
        # Initialize embedding extractor if needed
        with open(self.shard_files[0]) as f:
            data = json.load(f)
            self.requires_embeddings = data['requires_word_embeddings']
            
        if self.requires_embeddings:
            feature_config = _get_feature_extractor_config(self.config)
            self.embedding_extractor = FeatureExtractor(feature_config)
            
        logger.info(f"Found {len(self.shard_files)} shards")
        
    def _load_shard(self, shard_path: Path) -> List[TreeGroup]:
        """Load and process a single shard"""
        with open(shard_path) as f:
            data = json.load(f)
            
        groups = []
        for group in data['groups']:
            if not group['trees1'] or not group['trees2']:
                continue
                
            if len(group['trees1']) > self.max_group_size:
                indices = random.sample(range(len(group['trees1'])), self.max_group_size)
                trees1 = [group['trees1'][i] for i in indices]
                trees2 = [group['trees2'][i] for i in indices]
            else:
                trees1 = group['trees1']
                trees2 = group['trees2']
                
            groups.append(TreeGroup(
                group_id=group['group_id'],
                trees1=trees1,
                trees2=trees2,
                text1=group['text1'],
                text2=group['text2']
            ))
            
        return groups
        
    def __iter__(self):
        """Iterate over groups across shards"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process loading
            shard_files = self.shard_files
        else:
            # Split shards among workers
            per_worker = int(math.ceil(len(self.shard_files) / worker_info.num_workers))
            worker_id = worker_info.id
            shard_files = self.shard_files[worker_id:worker_id + per_worker]
            
        # Load groups from shards
        for shard in shard_files:
            groups = self._load_shard(shard)
            for group in groups:
                yield group

class TreeMatchingDataset(Dataset):
    """Base dataset class for tree matching"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def __len__(self):
        raise NotImplementedError("Derived classes must implement __len__()")
        
    def __getitem__(self, idx):
        raise NotImplementedError("Derived classes must implement __getitem__()")
        
    @staticmethod
    def collate_fn(batch):
        """Convert batch of samples to model input format"""
        raise NotImplementedError("Derived classes must implement collate_fn()")


