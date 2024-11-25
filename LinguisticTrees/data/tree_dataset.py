#data/tree_dataset.py
from GMN.dataset import GraphSimilarityDataset
from .data_utils import convert_tree_to_graph_data
import json
import torch

class TreeMatchingDataset(GraphSimilarityDataset):
    """Dataset for tree matching using TMN_DataGen output"""
    
    def __init__(self, data_path, config):
        self.config = config
        with open(data_path) as f:
            self.data = json.load(f)
            
        # Convert labels to tensor
        self.labels = torch.tensor(self.data['labels'])
        
    def pairs(self, batch_size):
        """Generate batches of tree pairs"""
        n_samples = len(self.labels)
        indices = torch.randperm(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Get batch of tree pairs
            batch_pairs = [self.data['graph_pairs'][i] for i in batch_indices]
            batch_labels = self.labels[batch_indices]
            
            # Convert to GraphData format
            graph_data = convert_tree_to_graph_data(
                batch_pairs, 
                self.config
            )
            
            yield graph_data, batch_labels
