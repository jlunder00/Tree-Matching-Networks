from typing import Iterator, Tuple
from pathlib import Path
import json
import torch
import logging
from collections import deque
import gc
from .tree_dataset import TreeMatchingDataset
from .loading_patterns import PartitionLoader, PartitionLoadingPattern
from .data_utils import convert_tree_to_graph_data, GraphData
from ..utils.memory_utils import MemoryMonitor
from torch.utils.data import DataLoader
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)


class MultiPartitionTreeDataset(TreeMatchingDataset):
    def __init__(self, 
                 data_dir: str, 
                 config,
                 loading_pattern: str = "sequential",
                 num_workers: int = None,
                 prefetch_factor: int = 2):
        super().__init__(config)
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers or mp.cpu_count() - 1
        self.prefetch_factor = prefetch_factor
        
        # Initialize partition loader with multiprocessing
        self.partition_loader = DataLoader(
            self._get_partition_paths(),
            batch_size=None,  # Load one partition at a time
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )
        
    def _get_partition_paths(self):
        """Get sorted list of partition files"""
        return sorted(
            self.data_dir.glob("part_*.json"),
            key=lambda x: int(x.stem.split('_')[1])
        )
        
    @staticmethod
    def _load_partition(file_path):
        """Worker function to load partition"""
        with open(file_path) as f:
            return json.load(f)
            
    def pairs(self, batch_size: int):
        """Generate batches using multiprocessing"""
        # Create process pool
        with mp.Pool(self.num_workers) as pool:
            for partition_file in self.partition_loader:
                # Load partition in parallel
                partition_data = pool.apply_async(
                    self._load_partition,
                    (partition_file,)
                ).get()
                
                # Create batches
                pairs = partition_data['graph_pairs']
                labels = torch.tensor(partition_data['labels'])
                
                for start_idx in range(0, len(labels), batch_size):
                    end_idx = min(start_idx + batch_size, len(labels))
                    batch_pairs = pairs[start_idx:end_idx]
                    batch_labels = labels[start_idx:end_idx]
                    
                    # Convert batch to GraphData in parallel
                    graph_data = pool.apply_async(
                        convert_tree_to_graph_data,
                        (batch_pairs, self.config)
                    ).get()
                    
                    yield graph_data, batch_labels


class DynamicPartitionTreeDataset(TreeMatchingDataset):
    """Dataset that dynamically loads/unloads partitions to manage memory"""
    
    def __init__(self, 
                 data_dir: str, 
                 config, 
                 max_partitions_in_memory: int = 2,
                 loading_pattern: str = "sequential"):
        """Initialize dataset
        
        Args:
            data_dir: Directory containing partition files
            config: Dataset configuration
            max_partitions_in_memory: Maximum number of partitions to keep in memory
            loading_pattern: One of "sequential", "random", "round_robin", "weighted"
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = config
        self.max_partitions = max_partitions_in_memory
        
        try:
            self.loading_pattern = PartitionLoadingPattern(loading_pattern)
        except ValueError:
            raise ValueError(f"Invalid loading pattern: {loading_pattern}")
        
        # Get all partition files
        self.partition_files = sorted(
            self.data_dir.glob("part_*.json"),
            key=lambda x: int(x.stem.split('_')[1])
        )
        
        if not self.partition_files:
            raise ValueError(f"No partition files found in {data_dir}")
            
        # Initialize partition cache
        self.partition_cache = {}  # file -> data
        self.partition_queue = deque()  # Track loaded partitions
        
        # Read sizes
        self.partition_sizes = {}
        total_examples = 0
        for pf in self.partition_files:
            with open(pf) as f:
                size = len(json.load(f)['labels'])
                self.partition_sizes[pf] = size
                total_examples += size
                
        logger.info(f"Found {len(self.partition_files)} partitions "
                   f"with {total_examples} total examples")
        
        # Create partition loader
        self.partition_loader = PartitionLoader(
            self.partition_files,
            self.loading_pattern,
            self.partition_sizes
        )
                   
    def _load_partition(self, partition_file: Path) -> None:
        """Load a partition into memory"""
        if partition_file in self.partition_cache:
            return
            
        # If cache is full, remove oldest partition
        while (len(self.partition_cache) >= self.max_partitions and 
               self.partition_queue):
            old_file = self.partition_queue.popleft()
            del self.partition_cache[old_file]
            gc.collect()  # Help free memory
            
        # Load new partition
        with open(partition_file) as f:
            self.partition_cache[partition_file] = json.load(f)
        self.partition_queue.append(partition_file)
        
        logger.debug(f"Loaded partition {partition_file.name}, "
                    f"cache size: {len(self.partition_cache)}")
        MemoryMonitor.log_memory(prefix=f'After loading {partition_file.name}: ')

    def pairs(self, batch_size: int) -> Iterator[Tuple[GraphData, torch.Tensor]]:
        """Generate batches of tree pairs"""
        while True:
            for partition_file in self.partition_loader:
                try:
                    # Load partition if needed
                    self._load_partition(partition_file)
                    partition_data = self.partition_cache[partition_file]
                    
                    pairs = partition_data['graph_pairs']
                    labels = torch.tensor(partition_data['labels'])
                    
                    # Create batches
                    indices = list(range(0, len(labels), batch_size))
                    for start_idx in indices:
                        end_idx = min(start_idx + batch_size, len(labels))
                        batch_pairs = [pairs[i] for i in range(start_idx, end_idx)]
                        batch_labels = labels[start_idx:end_idx]
                        
                        graph_data = convert_tree_to_graph_data(
                            batch_pairs,
                            self.config
                        )
                        yield graph_data, batch_labels
                        
                except Exception as e:
                    logger.error(f"Error processing partition {partition_file}: {e}")
                    continue

    def __len__(self) -> int:
        return sum(self.partition_sizes.values())
