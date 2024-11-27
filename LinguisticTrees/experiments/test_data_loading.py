import logging
from pathlib import Path
from ..configs.tree_data_config import TreeDataConfig
from ..data.partition_datasets import MultiPartitionTreeDataset
from ..utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading with memory monitoring"""
    
    # Configure data paths - adjust this path to match your setup
    data_root = '/home/jlunder/research/data/processed_data'  # Change this to your actual path
    
    logger.info(f"Using data root: {data_root}")
    
    data_config = TreeDataConfig(
        data_root=data_root,
        spacy_variant='trf'  # Change if using different model
    )
    
    # This will print the paths it's checking
    data_config.validate_paths()
    
    # Create dataset using dev data first
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    test_data_loading()
