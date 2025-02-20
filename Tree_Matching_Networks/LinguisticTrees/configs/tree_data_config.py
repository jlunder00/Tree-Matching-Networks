#configs/tree_data_config.py
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

@dataclass
class TreeDataConfig:
    """Configuration for tree data paths and variants"""
    
    # Root data directory
    data_root: str = '/home/jlunder/research/Tree-Matching-Networks/data/processed_data'
    
    # Dataset type and task
    dataset_type: Literal['snli', 'semeval', 'para50m', 'wikiqs'] = 'snli'
    task_type: Literal['entailment', 'similarity', 'info_nce'] = 'entailment'
    
    # SpaCy model variant (trf, lg, sm)
    spacy_variant: Literal['trf', 'lg', 'sm'] = 'trf'
    use_sharded_train: bool = True
    use_sharded_validate: bool = True
    
    def __post_init__(self):
        """Convert data_root to absolute path and validate config"""
        self.data_root = Path(self.data_root).resolve()
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration combinations"""
        if self.dataset_type == 'snli' and self.task_type == 'similarity':
            raise ValueError("SNLI dataset can only be used with entailment task")
            
    def _get_split_path(self, split: str) -> Path:
        """Internal helper to construct split path"""
        if self.dataset_type == 'snli': 
            base_dir = f'snli_1.0_{split}_converted_{self.spacy_variant}' 
        elif self.dataset_type == 'semeval':
            base_dir = f'semeval_{split}_converted_{self.spacy_variant}'
        elif self.dataset_type == 'para50m':
            base_dir = f'para_50m_{split}_converted_{self.spacy_variant}' 
        elif self.dataset_type == 'wikiqs':
            base_dir = f'wikiqs_{split}_converted_{self.spacy_variant}'


        if split == 'train' and self.use_sharded_train:
            base_dir += '_sharded'
        if split == 'dev' and self.use_sharded_validate:
            base_dir += '_sharded'
        return self.data_root / split / base_dir

        # if self.dataset_type == 'snli':
        #     return self.data_root / split / f'snli_1.0_{split}_converted_{self.spacy_variant}'
        # else:  # semeval
        #     return self.data_root / split / f'semeval_{split}_converted_{self.spacy_variant}'
            
    @property 
    def train_path(self) -> Path:
        return self._get_split_path('train')
        
    @property
    def dev_path(self) -> Path:
        return self._get_split_path('dev')
        
    @property
    def test_path(self) -> Path:
        return self._get_split_path('test')
    
    def validate_paths(self):
        """Verify data paths exist"""
        for split in ['train', 'dev', 'test']:
            path = self._get_split_path(split)
            print(f"Checking path: {path}")
            if not path.exists():
                raise ValueError(f"Data path does not exist: {path}")
