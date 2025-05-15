# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

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
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, 
            predictions, 
            average=None
        )
        return precision, recall
    
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
                acc = ((similarities[mask] > 0).float() == (labels[mask] > 0).float()).float().mean()
                accuracies.append(acc)
                
        return {
            'contradiction_acc': accuracies[0],
            'neutral_acc': accuracies[1],
            'entailment_acc': accuracies[2]
        }

    @staticmethod
    def compute_similarity_metrics(predictions, labels):
        """Compute metrics specific to similarity task"""
        predictions = predictions.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        
        from scipy.stats import pearsonr, spearmanr
        # Pearson correlation
        pearson_corr, _ = pearsonr(predictions, labels)
        
        # Spearman correlation
        spearman_corr, _ = spearmanr(predictions, labels)
        
        # Mean squared error
        mse = np.mean((predictions - labels) ** 2)
        
        return {
            'correlation': float(pearson_corr),
            'spearman': float(spearman_corr),
            'mse': float(mse)
        }

    @staticmethod
    def compute_task_metrics(predictions, labels, task_type='entailment'):
        """Compute metrics based on task type"""
        if task_type == 'similarity':
            return TreeMatchingMetrics.compute_similarity_metrics(predictions, labels)
        else:
            # Existing entailment metrics
            # accuracy = (predictions == labels).float().mean()
            accuracy = TreeMatchingMetrics.compute_accuracy(predictions, labels)
            precision, recall = TreeMatchingMetrics.compute_f1(predictions, labels)
            entailment_metrics = TreeMatchingMetrics.compute_entailment_metrics(predictions, labels)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                # 'f1': f1,
                **entailment_metrics
            }

    @classmethod
    def compute_all_metrics(cls, predictions, labels, task_type='entailment'):
        """Compute all metrics"""
        if task_type == 'similarity':
            return cls.compute_similarity_metrics(predictions, labels)
        else:
            # Existing entailment metrics
            # accuracy = (predictions == labels).float().mean()
            accuracy = cls.compute_accuracy(predictions, labels)
            precision, recall= cls.compute_f1(predictions, labels)
            entailment_metrics = cls.compute_entailment_metrics(predictions, labels)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                # 'f1': f1,
                **entailment_metrics
            }
