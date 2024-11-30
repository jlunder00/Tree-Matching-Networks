#data/partition_dataset.py
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
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)

# class MultiPartitionTreeDataset(TreeMatchingDataset):
#     def __init__(self, 
#                  data_dir: str, 
#                  config,
#                  num_workers: int = None,
#                  prefetch_factor: int = 2):
#         super().__init__(config)
#         self.data_dir = Path(data_dir)
#         # self.num_workers = num_workers or mp.cpu_count() - 1
#         self.num_workers = num_workers 
#         self.prefetch_factor = prefetch_factor
#         
#         # Get all data files (regular partitions or shards)
#         self.data_files = sorted(
#             [f for f in self.data_dir.glob("part_*_shard_*.json") 
#              if not f.name.endswith('_pair_count.json')],
#             key=lambda x: (int(x.stem.split('_')[1]), 
#                          int(x.stem.split('_shard_')[1]))
#         )
#         
#         # Fall back to regular partitions if no shards found
#         if not self.data_files:
#             self.data_files = sorted(
#                 [f for f in self.data_dir.glob("part_*.json")
#                  if not f.name.endswith('_pair_count.json')],
#                 key=lambda x: int(x.stem.split('_')[1])
#             )
#         self._total_pairs = 0
#         for pf in self.data_files:
#             # Look for count file
#             count_file = pf.parent / f"{pf.stem}_pair_count.json"
#             if count_file.exists():
#                 with open(count_file) as f:
#                     self._total_pairs += json.load(f)['n_pairs']
#             else:
#                 logger.warning(f"Count file not found for {pf.name}, loading full file")
#                 with open(pf) as f:
#                     self._total_pairs += len(json.load(f)['labels'])

#         # Initialize partition index and cache
#         self.current_partition = None
#         self.current_pairs = None
#         self.current_labels = None
#         
#     def __len__(self):
#         """Total number of pairs across all partitions"""
#         return self._total_pairs
#     
#     def __getitem__(self, idx):
#         """Get a single pair"""
#         # Map global index to the correct partition and local index
#         cumulative_pairs = 0
#         for partition_idx, file in enumerate(self.data_files):
#             # Check if the current file contains the desired index
#             count_file = file.parent / f"{file.stem}_pair_count.json"
#             if count_file.exists():
#                 with open(count_file) as f:
#                     n_pairs = json.load(f)['n_pairs']
#             else:
#                 # Fallback to load the file and calculate pair count
#                 with open(file) as f:
#                     n_pairs = len(json.load(f)['labels'])

#             if cumulative_pairs + n_pairs > idx:
#                 local_idx = idx - cumulative_pairs
#                 break
#             cumulative_pairs += n_pairs
#         else:
#             raise IndexError(f"Index {idx} is out of bounds for dataset with {cumulative_pairs} pairs")

#         # Load the partition if not already loaded
#         if self.current_partition != partition_idx:
#             if self.current_pairs is not None:
#                 del self.current_pairs
#             if self.current_labels is not None:
#                 del self.current_labels
#             gc.collect()

#             partition_file = self.data_files[partition_idx]
#             with open(partition_file) as f:
#                 logger.debug(f"Opening {partition_file}")
#                 data = json.load(f)
#                 logger.debug(f"Loaded {partition_file}")
#                 self.current_pairs = data['graph_pairs']
#                 self.current_labels = data['labels']
#                 self.current_partition = partition_idx

#         # Return the requested item
#         return (self.current_pairs[local_idx],
#                 torch.tensor(self.current_labels[local_idx]))

#     # def __getitem__(self, idx):
#     #     """Get a single pair"""
#     #     if self.current_partition is None or idx >= len(self.current_labels):
#     #         # Load new partition
#     #         partition_idx = idx // self.config['data']['batch_size']
#     #         if self.current_partition != partition_idx:
#     #             if self.current_pairs is not None:
#     #                 del self.current_pairs
#     #             if self.current_labels is not None:
#     #                 del self.current_labels
#     #             gc.collect()

#     #         partition_file = self.data_files[partition_idx % len(self.data_files)]
#     #         with open(partition_file) as f:
#     #             logger.debug(f"opening {partition_file}")
#     #             data = json.load(f)
#     #             logger.debug(f"loaded {partition_file}")
#     #             self.current_pairs = data['graph_pairs']
#     #             self.current_labels = data['labels']
#     #             self.current_partition = partition_idx
#     #             
#     #     local_idx = idx % len(self.current_labels)
#     #     return (self.current_pairs[local_idx], 
#     #             torch.tensor(self.current_labels[local_idx]))
#     
#     @staticmethod
#     def collate_fn(batch):
#         """Collate batch of pairs"""
#         pairs, labels = zip(*batch)
#         logger.debug("converting tree to graph data")
#         graph_data = convert_tree_to_graph_data(list(pairs))
#         logger.debug("converted tree to graph data...stacking")
#         labels = torch.stack(labels)
#         logger.debug("stacked")
#         return graph_data, labels
    

# class MultiPartitionTreeDataset(TreeMatchingDataset, IterableDataset):
#     """An IterableDataset implementation for multi-partition datasets."""
#     
#     def __init__(self, data_dir: str, config, shuffle_files: bool = False,
#                  num_workers: int = None,
#                  prefetch_factor: int = 2):
#         super().__init__(config)
#         self.data_dir = Path(data_dir)
#         # self.num_workers = num_workers 
#         self.num_workers = num_workers or mp.cpu_count() // 2 - 1
#         self.prefetch_factor = prefetch_factor
#         self.shuffle_files = shuffle_files

#         # Gather all shard files
#         self.data_files = sorted(
#             [f for f in self.data_dir.glob("part_*_shard_*.json") 
#              if not f.name.endswith('_pair_count.json')],
#             key=lambda x: (int(x.stem.split('_')[1]), 
#                          int(x.stem.split('_shard_')[1]))
#         )
#         
#         # Fallback to regular partitions if no shards found
#         if not self.data_files:
#             self.data_files = sorted(
#                 [f for f in self.data_dir.glob("part_*.json")
#                  if not f.name.endswith('_pair_count.json')],
#                 key=lambda x: int(x.stem.split('_')[1])
#             )

#         if shuffle_files:
#             from random import shuffle
#             shuffle(self.data_files)

#         # Pre-compute file sizes using count files
#         self.file_sizes = []
#         for file in self.data_files:
#             count_file = file.parent / f"{file.stem}_pair_count.json"
#             if count_file.exists():
#                 with open(count_file) as f:
#                     self.file_sizes.append(json.load(f)['n_pairs'])
#             else:
#                 with open(file) as f:
#                     self.file_sizes.append(len(json.load(f)['labels']))

#         self.total_pairs = sum(self.file_sizes)
#         logger.info(f"Dataset initialized with {len(self.data_files)} files "
#                     f"and {self.total_pairs} total pairs.")

#     def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
#         """Yield data pairs from the dataset."""
#         for file_idx, file in enumerate(self.data_files):
#             logger.debug(f"Opening {file.name}")
#             with open(file) as f:
#                 data = json.load(f)
#             
#             pairs = data['graph_pairs']
#             labels = data['labels']
#             
#             for i in range(len(labels)):
#                 graph_data = convert_tree_to_graph_data([pairs[i]])
#                 label = torch.tensor(labels[i], dtype=torch.float32)
#                 yield graph_data, label

#     def __len__(self):
#         """Return total number of pairs."""
#         return self.total_pairs

#     @staticmethod
#     def collate_fn(batch):
#         """Collate batch of pairs"""
#         # pairs, labels = zip(*batch)
#         # logger.debug("converting tree to graph data")
#         # graph_data = convert_tree_to_graph_data(list(pairs))
#         # logger.debug("converted tree to graph data...stacking")
#         # labels = torch.stack(labels)
#         # logger.debug("stacked")
#         # return graph_data, labels
#         graph_data = GraphData(
#             from_idx=torch.cat([b[0].from_idx for b in batch]),
#             to_idx=torch.cat([b[0].to_idx for b in batch]),
#             node_features=torch.cat([b[0].node_features for b in batch]),
#             edge_features=torch.cat([b[0].edge_features for b in batch]),
#             graph_idx=torch.cat([b[0].graph_idx for b in batch]),
#             n_graphs=sum(b[0].n_graphs for b in batch)
#         )
#         # Stack labels
#         labels = torch.stack([b[1] for b in batch])
#         return graph_data, labels


#     def pairs(self, batch_size: int):
#         """Get optimized dataloader iterator"""
#         logger.debug("creating dataloader")
#         return DataLoader(
#             self,
#             batch_size=batch_size,
#             collate_fn=self.collate_fn,
#             num_workers=self.num_workers,
#             prefetch_factor=self.prefetch_factor,
#             persistent_workers=False,
#             pin_memory=True,
#             drop_last=False,
#             # shuffle=True,
#             # timeout=3600
#         )

class MultiPartitionTreeDataset(IterableDataset):
    """An IterableDataset implementation for multi-partition datasets."""
    
    def __init__(self, data_dir: str, config, shuffle_files: bool = False,
                 num_workers: int = None, prefetch_factor: int = 2,
                 max_active_files: int = 2):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = config
        self.num_workers = num_workers or mp.cpu_count() // 2 - 1
        self.prefetch_factor = prefetch_factor
        self.shuffle_files = shuffle_files
        self.max_active_files = max_active_files

        # Gather all shard files
        self.data_files = sorted(
            [f for f in self.data_dir.glob("part_*_shard_*.json") 
             if not f.name.endswith('_pair_count.json')],
            key=lambda x: (int(x.stem.split('_')[1]), 
                           int(x.stem.split('_shard_')[1]))
        )
        
        # Fallback to regular partitions if no shards found
        if not self.data_files:
            self.data_files = sorted(
                [f for f in self.data_dir.glob("part_*.json")
                 if not f.name.endswith('_pair_count.json')],
                key=lambda x: int(x.stem.split('_')[1])
            )

        if shuffle_files:
            from random import shuffle
            shuffle(self.data_files)

        # Pre-compute file sizes using count files
        self.file_sizes = []
        for file in self.data_files:
            count_file = file.parent / f"{file.stem}_pair_count.json"
            if count_file.exists():
                with open(count_file) as f:
                    self.file_sizes.append(json.load(f)['n_pairs'])
            else:
                with open(file) as f:
                    self.file_sizes.append(len(json.load(f)['labels']))

        self.total_pairs = sum(self.file_sizes)
        logger.info(f"Dataset initialized with {len(self.data_files)} files "
                    f"and {self.total_pairs} total pairs.")

    def __iter__(self) -> Iterator[Tuple[GraphData, torch.Tensor]]:
        """Yield data pairs from the dataset."""
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading
            files_to_process = self.data_files
        else:
            # Multi-process: Split files across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files_to_process = self.data_files[worker_id::num_workers]

        active_files = deque(maxlen=self.max_active_files)
        for file in files_to_process:
            logger.debug(f"Worker opening file: {file.name}")
            with open(file) as f:
                data = json.load(f)

            # Add file to active list
            active_files.append(file)

            pairs = data['graph_pairs']
            labels = data['labels']
            
            for i in range(len(labels)):
                # Convert each pair into GraphData format
                graph_data = convert_tree_to_graph_data([pairs[i]])
                label = torch.tensor(labels[i], dtype=torch.float32)
                yield graph_data, label

            # Clear memory if max_active_files is exceeded
            if len(active_files) >= self.max_active_files:
                old_file = active_files.popleft()
                logger.debug(f"Releasing memory for file: {old_file.name}")
                gc.collect()

    def __len__(self):
        """Return total number of pairs."""
        return self.total_pairs

    @staticmethod
    def collate_fn(batch):
        """Collate batch of pairs."""
        graph_data = GraphData(
            from_idx=torch.cat([b[0].from_idx for b in batch]),
            to_idx=torch.cat([b[0].to_idx for b in batch]),
            node_features=torch.cat([b[0].node_features for b in batch]),
            edge_features=torch.cat([b[0].edge_features for b in batch]),
            graph_idx=torch.cat([b[0].graph_idx for b in batch]),
            n_graphs=sum(b[0].n_graphs for b in batch)
        )
        # Stack labels
        labels = torch.stack([b[1] for b in batch])
        return graph_data, labels

    def pairs(self, batch_size: int):
        """Get optimized dataloader iterator."""
        logger.debug("Creating dataloader")
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=False,
            pin_memory=True,
            drop_last=False
        )


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
