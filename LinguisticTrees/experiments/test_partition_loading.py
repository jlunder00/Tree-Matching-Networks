import logging
from pathlib import Path
import time
from ..configs.tree_data_config import TreeDataConfig
from ..data.partition_datasets import MultiPartitionTreeDataset
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def test_loading_patterns():
    """Test different partition loading patterns"""
    
    data_config = TreeDataConfig(
        data_root='data/processed_data',
        spacy_variant='trf'  # Change to match your data
    )
    
    patterns = ["sequential", "random", "round_robin", "weighted"]
    batch_size = 32
    
    for pattern in patterns:
        logger.info(f"\nTesting {pattern} loading pattern:")
        
        dataset = MultiPartitionTreeDataset(
            data_config.dev_path,
            config=data_config,
            loading_pattern=pattern
        )
        
        # Track partition order
        loaded_partitions = []
        start_time = time.time()
        
        # Test loading a few batches
        for i, (graphs, labels) in enumerate(dataset.pairs(batch_size)):
            if i == 0:
                logger.info(f"First batch shapes:")
                logger.info(f"Nodes: {graphs.node_features.shape}")
                logger.info(f"Labels: {labels.shape}")
            
            MemoryMonitor.log_memory(step=i)
            
            if i >= 10:  # Test first 10 batches
                break
                
        duration = time.time() - start_time
        logger.info(f"{pattern} loading took {duration:.2f}s for 10 batches")

def test_memory_clearing():
    """Test memory clearing between partitions"""
    data_config = TreeDataConfig(
        data_root='data/processed_data',
        spacy_variant='trf'
    )
    
    dataset = MultiPartitionTreeDataset(
        data_config.dev_path,
        config=data_config,
        loading_pattern='sequential'
    )
    
    logger.info("\nTesting memory clearing:")
    initial_mem = MemoryMonitor.get_memory_usage()
    
    for i, (graphs, labels) in enumerate(dataset.pairs(32)):
        if i % 10 == 0:
            current_mem = MemoryMonitor.get_memory_usage()
            logger.info(f"Batch {i} memory delta: "
                       f"{current_mem['ram_used_gb'] - initial_mem['ram_used_gb']:.2f}GB")
        
        if i >= 30:  # Test a few partition transitions
            break

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_loading_patterns()
    test_memory_clearing()
