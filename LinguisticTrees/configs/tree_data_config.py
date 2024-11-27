from dataclasses import dataclass
from typing import Literal
from pathlib import Path

@dataclass
class TreeDataConfig:
    """Configuration for tree data paths and variants"""
    
    # Root data directory - can be absolute or relative
    data_root: str = '/home/jlunder/research/data/processed_data'
    # data_root: str = 'data/processed_data'
    
    # SpaCy model variant (trf, lg, sm)
    spacy_variant: Literal['trf', 'lg', 'sm'] = 'trf'
    
    def __post_init__(self):
        """Convert data_root to absolute path if needed"""
        self.data_root = Path(self.data_root).resolve()
    
    @property
    def train_path(self) -> Path:
        return self.data_root / 'train' / f'snli_1.0_train_converted_{self.spacy_variant}'
    
    @property
    def dev_path(self) -> Path:
        return self.data_root / 'dev' / f'snli_1.0_dev_converted_{self.spacy_variant}'
        
    @property
    def test_path(self) -> Path:
        return self.data_root / 'test' / f'snli_1.0_test_converted_{self.spacy_variant}'
    
    def validate_paths(self):
        """Verify data paths exist"""
        # Print the paths we're checking for debugging
        for path in [self.train_path, self.dev_path, self.test_path]:
            print(f"Checking path: {path}")
            if not path.exists():
                raise ValueError(f"Data path does not exist: {path}")

