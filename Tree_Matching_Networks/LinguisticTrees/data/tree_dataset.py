# data/tree_dataset.py
from ...GMN.dataset import GraphSimilarityDataset
from .data_utils import convert_tree_to_graph_data
import json
import torch
from torch.utils.data import Dataset, DataLoader

class TreeMatchingDataset(Dataset):
    """Base dataset class for tree matching"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def __len__(self):
        raise NotImplementedError("Derived classes must implement __len__()")
        
    def __getitem__(self, idx):
        raise NotImplementedError("Derived classes must implement __getitem__()")
        
    @staticmethod
    def collate_fn(batch):
        """Convert batch of samples to model input format"""
        raise NotImplementedError("Derived classes must implement collate_fn()")


