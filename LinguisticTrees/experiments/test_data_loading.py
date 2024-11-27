import logging
from pathlib import Path
from ..configs.tree_data_config import TreeDataConfig
from ..data.partition_datasets import MultiPartitionTreeDataset
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading with memory monitoring"""
    
    # Configure data paths
    data_config = TreeDataConfig(
        data_root='data/processed_data',
        spacy_variant='trf'  # Change to match your data
    )
    data_config.validate_paths()
    
    # Create dataset
    dataset = MultiPartitionTreeDataset(
        data_config.dev_path,  # Start with dev data
        config=data_config
    )
    
    # Test batch iteration
    batch_size = 32
    logger.info("Testing batch iteration...")
    
    for i, (graphs, labels) in enumerate(dataset.pairs(batch_size)):
        if i == 0:
            logger.info(f"First batch shapes:")
            logger.info(f"Node features: {graphs.node_features.shape}")
            logger.info(f"Edge features: {graphs.edge_features.shape}")
            logger.info(f"Labels: {labels.shape}")
            
        MemoryMonitor.log_memory(step=i, prefix=f'Batch {i}: ')
        
        if i >= 5:  # Test first few batches
            break
            
    logger.info("Data loading test complete!")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_data_loading()
