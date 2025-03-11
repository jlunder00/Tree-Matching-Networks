# model/tree_aggregator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Callable, Union

class AggregationStrategies:
    """Collection of strategies for aggregating sentence embeddings into text embeddings"""
    
    @staticmethod
    def mean_pooling(embeddings: torch.Tensor) -> torch.Tensor:
        """Simple mean pooling of embeddings"""
        return torch.mean(embeddings, dim=0)
    
    @staticmethod
    def max_pooling(embeddings: torch.Tensor) -> torch.Tensor:
        """Max pooling of embeddings"""
        return torch.max(embeddings, dim=0)[0]
    
    @staticmethod
    def attention_pooling(embeddings: torch.Tensor) -> torch.Tensor:
        """Attention-based pooling of embeddings"""
        # Calculate attention scores
        attn_weights = F.softmax(
            torch.matmul(embeddings, embeddings.mean(dim=0, keepdim=True).transpose(0, 1)) / 
            torch.sqrt(torch.tensor(embeddings.shape[1], dtype=torch.float)),
            dim=0
        )
        # Apply attention weights
        return torch.sum(embeddings * attn_weights, dim=0)

    @classmethod
    def get_strategy(cls, name: str) -> Callable:
        """Get aggregation strategy by name"""
        strategies = {
            'mean': cls.mean_pooling,
            'max': cls.max_pooling,
            'attention': cls.attention_pooling
        }
        if name not in strategies:
            raise ValueError(f"Unknown aggregation strategy: {name}")
        return strategies[name]


class TreeAggregator(nn.Module):
    """Aggregates sentence embeddings into text embeddings"""
    
    def __init__(self, aggregation_strategy: str = 'mean'):
        super().__init__()
        self.aggregate = AggregationStrategies.get_strategy(aggregation_strategy)
    
    def forward(self, 
                embeddings: torch.Tensor, 
                batch_info: 'PairedGroupBatchInfo') -> torch.Tensor:
        """
        Aggregate sentence embeddings into text embeddings
        
        Args:
            embeddings: [n_trees, embedding_dim] tensor with all tree embeddings
            batch_info: Batch information with group mappings
            
        Returns:
            [n_groups*2, embedding_dim] tensor with text embeddings
        """
        # Create text embeddings for each subgroup
        text_embeddings = []
        
        # Process each group
        for group_idx in batch_info.group_indices:
            # Get embeddings for trees in set A
            trees_a_indices = batch_info.trees_a_indices[group_idx]
            trees_a_embeddings = embeddings[trees_a_indices]
            text_a_embedding = self.aggregate(trees_a_embeddings)
            text_embeddings.append(text_a_embedding)
            
            # Get embeddings for trees in set B
            trees_b_indices = batch_info.trees_b_indices[group_idx]
            trees_b_embeddings = embeddings[trees_b_indices]
            text_b_embedding = self.aggregate(trees_b_embeddings)
            text_embeddings.append(text_b_embedding)
        
        return torch.stack(text_embeddings)
