# LinguisticTrees/experiments/eval_tree_matching.py
import torch
from COMMON.src.utils.config import cfg
from COMMON.src.evaluation_metric import *
from sklearn.metrics import precision_recall_fscore_support

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            graphs, labels = batch
            graphs = graphs.to(device)
            labels = labels.to(device)
            
            outputs = model(graphs)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted')
    
    accuracy = sum(1 for x,y in zip(all_labels, all_preds) if x == y) / len(all_labels)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
