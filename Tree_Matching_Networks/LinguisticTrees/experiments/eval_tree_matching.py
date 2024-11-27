#experiments/eval_tree_matching.py
import torch
from pathlib import Path
from ..models.tree_matching import TreeMatchingNet
from ..data.tree_dataset import TreeMatchingDataset
from ..training.metrics import TreeMatchingMetrics
import json
import wandb

def evaluate_model(model, test_loader, config):
    """Evaluate model on test set"""
    model.eval()
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (graphs, labels) in enumerate(test_loader):
            # Move to device
            graphs = graphs.to(config.device)
            labels = labels.to(config.device)
            
            # Forward pass
            outputs = model(
                graphs.node_features,
                graphs.edge_features,
                graphs.from_idx,
                graphs.to_idx,
                graphs.graph_idx,
                graphs.n_graphs
            )
            
            all_similarities.append(outputs)
            all_labels.append(labels)
    
    # Concatenate results
    similarities = torch.cat(all_similarities)
    labels = torch.cat(all_labels)
    
    # Compute metrics
    metrics = TreeMatchingMetrics.compute_all_metrics(
        similarities, 
        labels
    )
    
    return metrics

def main():
    # Load config
    config = load_config()
    
    # Initialize WandB
    wandb.init(
        project=config.wandb.project_name,
        config=config,
        tags=['evaluation']
    )
    
    # Load model
    model = TreeMatchingNet(config)
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    
    # Create test dataset
    test_dataset = TreeMatchingDataset(
        config.data.test_path,
        config
    )
    
    # Evaluate
    metrics = evaluate_model(model, test_dataset, config)
    
    # Log results
    wandb.log(metrics)
    
    # Save results
    results_path = Path(config.results_dir) / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()
