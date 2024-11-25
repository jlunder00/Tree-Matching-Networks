# LinguisticTrees/tree_matching/config.py
from dataclasses import dataclass
from typing import List

@dataclass
class MatchingConfig:
    node_feature_dim: int = 768  # BERT embedding dim
    edge_feature_dim: int = 64
    node_hidden_dim: int = 128
    n_prop_layers: int = 5
    dropout: float = 0.1
