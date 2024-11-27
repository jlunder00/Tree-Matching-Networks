#training/metrics.py
import torch
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

class TreeMatchingMetrics:
    """Metrics for tree matching evaluation"""
    
    @staticmethod
    def compute_accuracy(similarities, labels):
        """Compute matching accuracy"""
        predictions = (similarities > 0).float()
        return (predictions == labels).float().mean()
    
    @staticmethod
    def compute_f1(similarities, labels):
        """Compute F1 score"""
        predictions = (similarities > 0).cpu().numpy()
        labels = labels.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, 
            predictions, 
            average='binary'
        )
        return precision, recall, f1
    
    @staticmethod
    def compute_entailment_metrics(similarities, labels):
        """Compute metrics specific to entailment"""
        # Convert -1,0,1 to class indices
        label_indices = labels + 1  # Convert to 0,1,2
        
        # Compute per-class accuracy
        accuracies = []
        for i in range(3):
            mask = (label_indices == i)
            if mask.sum() > 0:
                acc = ((similarities[mask] > 0).float() == (labels[mask] > 0).float()).mean()
                accuracies.append(acc)
                
        return {
            'contradiction_acc': accuracies[0],
            'neutral_acc': accuracies[1],
            'entailment_acc': accuracies[2]
        }

    @classmethod
    def compute_all_metrics(cls, similarities, labels):
        """Compute all metrics"""
        accuracy = cls.compute_accuracy(similarities, labels)
        precision, recall, f1 = cls.compute_f1(similarities, labels)
        entailment_metrics = cls.compute_entailment_metrics(similarities, labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            **entailment_metrics
        }
