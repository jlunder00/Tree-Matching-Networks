#data/tree_dataset.py
from GMN.dataset import GraphSimilarityDataset
from .data_utils import convert_tree_to_graph_data
import json
import torch

class TreeMatchingDataset:
    """Base dataset class for tree matching"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.labels = None

    def pairs(self, batch_size):
        raise NotImplementedError("Derived classes must implement pairs()")

