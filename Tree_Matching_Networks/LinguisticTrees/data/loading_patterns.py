#data/loading_patterns.py
from enum import Enum
from typing import List, Iterator
from pathlib import Path
import random

class PartitionLoadingPattern(Enum):
    SEQUENTIAL = "sequential"  # Load partitions in order
    RANDOM = "random"         # Load partitions randomly
    ROUND_ROBIN = "round_robin"  # Cycle through partitions evenly
    WEIGHTED = "weighted"     # Load partitions based on size

class PartitionLoader:
    """Handles different partition loading patterns"""
    
    def __init__(self, 
                 partition_files: List[Path],
                 pattern: PartitionLoadingPattern,
                 partition_sizes: dict = None):
        self.partition_files = partition_files
        self.pattern = pattern
        self.partition_sizes = partition_sizes
        self._current_idx = 0
        
    def __iter__(self) -> Iterator[Path]:
        if self.pattern == PartitionLoadingPattern.SEQUENTIAL:
            yield from self._sequential_loading()
        elif self.pattern == PartitionLoadingPattern.RANDOM:
            yield from self._random_loading()
        elif self.pattern == PartitionLoadingPattern.ROUND_ROBIN:
            yield from self._round_robin_loading()
        elif self.pattern == PartitionLoadingPattern.WEIGHTED:
            yield from self._weighted_loading()
            
    def _sequential_loading(self) -> Iterator[Path]:
        yield from self.partition_files
        
    def _random_loading(self) -> Iterator[Path]:
        files = list(self.partition_files)
        random.shuffle(files)
        yield from files
        
    def _round_robin_loading(self) -> Iterator[Path]:
        while True:
            yield self.partition_files[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self.partition_files)
            
    def _weighted_loading(self) -> Iterator[Path]:
        if not self.partition_sizes:
            raise ValueError("Partition sizes required for weighted loading")
            
        # Convert counts to probabilities
        total = sum(self.partition_sizes.values())
        weights = [self.partition_sizes[f]/total for f in self.partition_files]
        
        while True:
            yield random.choices(self.partition_files, weights=weights, k=1)[0]
