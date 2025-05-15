# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#configs/tree_data_config.py
from dataclasses import dataclass
from typing import Literal, List
from pathlib import Path

@dataclass
class TreeDataConfig:
    """Configuration for tree data paths and variants"""
    
    # Root data directory
    data_root: str = '/home/jlunder/research/Tree-Matching-Networks/data/processed_data'
    
    # Multiple datasets with optional category specification
    # Format: ["dataset_name", "dataset_name/category", ...]
    dataset_specs: List[str] = None
    
    # Keep single dataset_type for backward compatibility
    dataset_type: Literal['snli', 'semeval', 'para50m', 'wikiqs', 'amazonqa_multiple', 'amazonqa_single', 'patentmatch_balanced', 'patentmatch_ultrabalanced'] = 'snli'
    task_type: Literal['entailment', 'similarity', 'info_nce', 'binary'] = 'entailment'
    
    # SpaCy model variant (trf, lg, sm)
    spacy_variant: Literal['trf', 'lg', 'sm'] = 'trf'
    use_sharded_train: bool = True
    use_sharded_validate: bool = True
    use_sharded_test: bool = True
    
    # New option to allow cross-dataset negatives
    allow_cross_dataset_negatives: bool = True
    
    def __post_init__(self):
        """Initialize and validate configuration"""
        self.data_root = Path(self.data_root).resolve()
        
        # Initialize dataset_specs if not provided
        if self.dataset_specs is None:
            self.dataset_specs = [self.dataset_type]
            
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration combinations"""
        for spec in self.dataset_specs:
            base_dataset = spec.split('/')[0] if '/' in spec else spec
            if base_dataset == 'snli' and self.task_type == 'similarity':
                raise ValueError(f"SNLI dataset ({spec}) can only be used with entailment task")
    
    def _resolve_dataset_path(self, split: str, dataset_spec: str) -> Path:
        """Resolve path for a specific dataset and split"""
        # Parse dataset spec (may include category)
        if '/' in dataset_spec:
            base_dataset, category = dataset_spec.split('/', 1)
        else:
            base_dataset, category = dataset_spec, None
            
        # Handle different dataset types
        if base_dataset == 'snli':
            base_dir = f'snli_1.0_{split}_converted_{self.spacy_variant}'
        elif base_dataset == 'semeval':
            base_dir = f'semeval_{split}_converted_{self.spacy_variant}'
        elif base_dataset == 'para50m':
            base_dir = f'para_50m_{split}_converted_{self.spacy_variant}'
        elif base_dataset == 'wikiqs':
            base_dir = f'wikiqs_{split}_converted_{self.spacy_variant}'
        elif base_dataset == 'amazonqa_multiple':
            base_dir = f'amazonqa_multiple_converted_{self.spacy_variant}'
        elif base_dataset == 'amazonqa_single':
            base_dir = f'amazonqa_single_converted_{self.spacy_variant}'
        elif base_dataset == 'patentmatch_balanced':
            base_dir = f'patentmatch_balanced_{split}_converted_{self.spacy_variant}'
        elif base_dataset == 'patentmatch_ultrabalanced':
            base_dir = f'patentmatch_ultrabalanced_{split}_converted_{self.spacy_variant}'
        else:
            # Generic fallback
            base_dir = f'{base_dataset}_{split}_converted_{self.spacy_variant}'
            
        # Add sharded suffix if appropriate
        if split == 'train' and self.use_sharded_train:
            base_dir += '_sharded'
        if split == 'dev' and self.use_sharded_validate:
            base_dir += '_sharded'
        if split == 'test' and self.use_sharded_test:
            base_dir += '_sharded'

        if base_dataset == 'amazonqa_multiple':
            if category:
                return self.data_root / split / base_dir / f"QA_{category}.json"
        elif base_dataset == 'amazonqa_single':
            if category:
                return self.data_root / split / base_dir / f"qa_{category}.json"
            
        return self.data_root / split / base_dir
    
    def get_split_paths(self, split: str) -> List[Path]:
        """Get all paths for a specific split"""
        paths = []
        for spec in self.dataset_specs:
            path = self._resolve_dataset_path(split, spec)
            if path.exists():
                paths.append(path)
            else:
                print(f"Warning: Path does not. exist: {path}")
        return paths
    
    @property 
    def train_paths(self) -> List[Path]:
        return self.get_split_paths('train')
        
    @property
    def dev_paths(self) -> List[Path]:
        return self.get_split_paths('dev')
        
    @property
    def test_paths(self) -> List[Path]:
        return self.get_split_paths('test')
    
    # Keep compatibility with old code
    @property 
    def train_path(self) -> Path:
        paths = self.get_split_paths('train')
        if not paths:
            raise ValueError("No valid train paths found")
        return paths[0]
        
    @property
    def dev_path(self) -> Path:
        paths = self.get_split_paths('dev')
        if not paths:
            raise ValueError("No valid dev paths found")  
        return paths[0]
        
    @property
    def test_path(self) -> Path:
        paths = self.get_split_paths('test')
        if not paths:
            raise ValueError("No valid test paths found")
        return paths[0]
    
    def validate_paths(self):
        """Verify data paths exist"""
        for split in ['train', 'dev', 'test']:
            paths = self.get_split_paths(split)
            if not paths:
                raise ValueError(f"No valid paths found for split '{split}'")

