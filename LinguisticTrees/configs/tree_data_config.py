from pathlib import Path
from typing import Literal
from dataclasses import dataclass

@dataclass
class TreeDataConfig:
    """Configuration for tree data paths and variants"""
    
    # Root data directory
    data_root: str = 'data/processed_data'
    
    # SpaCy model variant (trf, lg, sm)
    spacy_variant: Literal['trf', 'lg', 'sm'] = 'trf'
    
    @property
    def train_path(self) -> Path:
        return Path(self.data_root) / 'train' / f'snli_1.0_train_converted_{self.spacy_variant}'
    
    @property
    def dev_path(self) -> Path:
        return Path(self.data_root) / 'dev' / f'snli_1.0_dev_converted_{self.spacy_variant}'
        
    @property
    def test_path(self) -> Path:
        return Path(self.data_root) / 'test' / f'snli_1.0_test_converted_{self.spacy_variant}'
    
    def validate_paths(self):
        """Verify data paths exist"""
        for path in [self.train_path, self.dev_path, self.test_path]:
            if not path.exists():
                raise ValueError(f"Data path does not exist: {path}")
