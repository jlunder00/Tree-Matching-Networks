# LinguisticTrees/experiments/metrics.py
from typing import Dict, List
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(all_labels: List[int], all_preds: List[int]) -> Dict:
    """Compute detailed metrics for NLI task"""
    labels = np.array(all_labels)
    preds = np.array(all_preds)
    
    # Per-class metrics
    results = {}
    for cls in np.unique(labels):
        cls_preds = preds[labels == cls]
        cls_correct = (cls_preds == cls).sum()
        results[f'class_{cls}_accuracy'] = cls_correct / len(cls_preds)
    
    # Overall metrics
    results['accuracy'] = (labels == preds).mean()
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    results['confusion_matrix'] = cm
    
    return results

def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """Plot confusion matrix with pretty formatting"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.close()
