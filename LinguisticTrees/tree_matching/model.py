# LinguisticTrees/tree_matching/model.py
from COMMON.models.COMMON.model import Net as BaseNet
import torch.nn as nn

class TreeMatchingNet(BaseNet):
    """Adaptation of GMN for dependency trees"""
    def __init__(self):
        super().__init__()
        # Add any tree-specific model adaptations here
        # For now, using base GMN architecture
