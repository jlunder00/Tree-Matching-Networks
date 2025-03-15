# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

#evaluation script for testing model

# experiments/eval_aggregated.py
import torch
import yaml
import numpy as np
import logging
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.stats import pearsonr, spearmanr
try:
    from ..configs.default_tree_config import get_tree_config
    from ..configs.tree_data_config import TreeDataConfig
    from ..models.tree_matching import TreeMatchingNet
    from ..models.tree_embedding import TreeEmbeddingNet
    from ..training.experiment import ExperimentManager
    from ..data.paired_groups_dataset import create_paired_groups_dataset, get_paired_groups_dataloader
    from ..training.loss_handlers import LOSS_HANDLERS
    from ..data.data_utils import GraphData
except:
    from Tree_Matching_Networks.LinguisticTrees.configs.default_tree_config import get_tree_config
    from Tree_Matching_Networks.LinguisticTrees.configs.tree_data_config import TreeDataConfig
    from Tree_Matching_Networks.LinguisticTrees.models.tree_matching import TreeMatchingNet
    from Tree_Matching_Networks.LinguisticTrees.models.tree_embedding import TreeEmbeddingNet
    from Tree_Matching_Networks.LinguisticTrees.training.experiment import ExperimentManager
    from Tree_Matching_Networks.LinguisticTrees.data.paired_groups_dataset import create_paired_groups_dataset, get_paired_groups_dataloader
    from Tree_Matching_Networks.LinguisticTrees.training.loss_handlers import LOSS_HANDLERS
    from Tree_Matching_Networks.LinguisticTrees.data.data_utils import GraphData

logger = logging.getLogger(__name__)

def load_model_from_checkpoint(checkpoint_path, base_config=None, override_config=None):
    """Load model from checkpoint"""
    checkpoint, manager, config, override = ExperimentManager.load_checkpoint(
        checkpoint_path, 
        base_config,
        override_config)
    
    config = get_tree_config(
        task_type='aggregative',
        base_config=base_config, 
        override_config=override_config
    ) 


    # Create appropriate model based on config
    if config['model'].get('model_type', 'matching') == 'embedding':
        model = TreeEmbeddingNet(config)
    else:
        model = TreeMatchingNet(config)
    
    # Load state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config, checkpoint.get('epoch', 0)

def compute_metrics(predictions, labels, task_type):
    """Compute evaluation metrics based on task type"""

    metrics = {}
    
    if task_type == 'entailment':
        # Convert to integer classes if not already
        pred_classes = np.array([int(p) + 1 for p in predictions])  # -1,0,1 → 0,1,2
        true_classes = np.array([int(l) + 1 for l in labels])       # -1,0,1 → 0,1,2
        if 1 in pred_classes:
            print("found something")
        if 0 in predictions:
            print("found another thing")
        
        metrics['accuracy'] = accuracy_score(true_classes, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_classes, pred_classes, average='weighted')
        
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
        
        # Class-specific metrics
        class_names = ["contradiction", "neutral", "entailment"]
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(true_classes, pred_classes, average=None)
            
        for i, cls in enumerate(class_names):
            metrics[f'{cls}_precision'] = float(precision_per_class[i])
            metrics[f'{cls}_recall'] = float(recall_per_class[i])
            metrics[f'{cls}_f1'] = float(f1_per_class[i])
            metrics[f'{cls}_support'] = int(support_per_class[i])
            
        # Confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)
        metrics['confusion_matrix'] = cm.tolist()
        
    elif task_type == 'similarity':
        # Calculate correlation coefficients

        preds = np.array(predictions)
        labs = np.array(labels)
        metrics['mse'] = float(np.mean((preds - labs) ** 2))
        metrics['pearson'], pearson_p = pearsonr(preds, labs)
        metrics['spearman'], spearman_p = spearmanr(preds, labs)
        metrics['pearson'] = float(metrics['pearson'])
        metrics['spearman'] = float(metrics['spearman'])
        metrics['pearson_p'] = float(pearson_p)
        metrics['spearman_p'] = float(spearman_p)
        
    else:  # binary
        # Binary classification metrics
        pred_classes = np.array([1 if p > 0.5 else 0 for p in predictions])
        true_classes = np.array([1 if l > 0.5 else 0 for l in labels])
        
        metrics['accuracy'] = float(accuracy_score(true_classes, pred_classes))
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_classes, pred_classes, average='binary')
            
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)
        metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

def get_confusion_matrix_plot(cm, class_names):
    """Create confusion matrix plot and return the figure"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    return fig

def get_scatter_plot(labels, predictions, pearson_corr):
    """Create scatter plot and return the figure"""
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(labels, predictions, alpha=0.5)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs Ground Truth (Pearson={pearson_corr:.4f})')
    plt.grid(True)
    
    # Add identity line
    min_val = min(min(labels), min(predictions))
    max_val = max(max(labels), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    return fig

@torch.no_grad()
def evaluate_model(model, test_loader, loss_fn, device, config):
    """Evaluate model on test set"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    n_batches = len(test_loader) if hasattr(test_loader, '__len__') else None
    pbar = tqdm(
        enumerate(test_loader),
        total=n_batches,
        desc=f'Evaluating'
    )
    
    for batch_idx, (graphs, batch_info) in pbar:
        # Move data to device
        graphs = GraphData(
            node_features=graphs.node_features.to(device, non_blocking=True),
            edge_features=graphs.edge_features.to(device, non_blocking=True),
            from_idx=graphs.from_idx.to(device, non_blocking=True),
            to_idx=graphs.to_idx.to(device, non_blocking=True),
            graph_idx=graphs.graph_idx.to(device, non_blocking=True),
            n_graphs=graphs.n_graphs
        )
        
        # Forward pass to get tree embeddings
        embeddings = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
        
        # Compute loss and get predictions
        # The loss function handles aggregating tree embeddings to text embeddings
        _, predictions, _ = loss_fn(embeddings, batch_info)
        
        # Handle different prediction formats
        if isinstance(predictions, torch.Tensor):
            all_predictions.extend(predictions.cpu().tolist())
        elif isinstance(predictions, tuple) and len(predictions) > 0:
            # For the case when predictions might be similarity matrices
            if hasattr(batch_info, 'group_labels'):
                all_predictions.extend(predictions[0].cpu().tolist())
        
        # Store true labels
        all_labels.extend(batch_info.group_labels)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels, config['model']['task_type'])
    
    return metrics, all_predictions, all_labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate tree matching models")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=None,
                      help="Batch size for evaluation (groups per batch)")
    parser.add_argument("--num_workers", type=int, default=2,
                      help="Number of data loading workers")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                      help="Directory to save results")
    parser.add_argument("--use_wandb", action="store_true", 
                      help="Log results to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="tree-embedding",
                      help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                      help="WandB run name (default: checkpoint filename)")
    parser.add_argument("--wandb_tags", type=str, default=None,
                      help="Comma-separated list of tags for WandB")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None,
                        help='The root data directory, containing dev, test, and train folders with dataset folders inside')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load model and config from checkpoint
    logger.info(f"Loading model from {args.checkpoint}")
    
    base_config = None
    base_config_path= args.config if args.config is not None else '/home/jlunder/research/Tree-Matching-Networks/Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/aggregative_config.yaml'
    override_config = None
    with open(base_config_path, 'r') as fin:
        base_config = yaml.safe_load(fin)
    if args.override:
        with open(args.override, 'r') as fin:
            override_config = yaml.safe_load(fin)
    if args.config is None:
        base_config = None

    model, config, checkpoint_epoch = load_model_from_checkpoint(args.checkpoint, base_config, override_config)
    
    # Override config with evaluation settings
    config['data']['shuffle_files'] = False  # Important for sequential evaluation
    config['data']['batch_size'] = args.batch_size if args.batch_size is not None else config['data']['batch_size']
    config['data']['num_workers_val'] = args.num_workers
    
    # Get task type and setup
    task_type = config['model']['task_type']
    dataset_type = config['data']['dataset_type']
    task_loader_type = config['model'].get('task_loader_type', 'aggregative')
    logger.info(f"Evaluating {task_type} model with {task_loader_type} loader")
    
    # Create data config
    if args.data_root:
        data_config = TreeDataConfig(
            data_root = args.data_root,
            dataset_type=config['data']['dataset_type'],
            task_type=task_type,
            use_sharded_train=True,
            use_sharded_validate=False,
            use_sharded_test=True
        )
    else:
        data_config = TreeDataConfig(
            dataset_type=config['data']['dataset_type'],
            task_type=task_type,
            use_sharded_train=True,
            use_sharded_validate=False,
            use_sharded_test=True
        )
    label_map = {'-': 1.0, 'entailment':1.0, 'neutral':0.0, 'contradiction':-1.0, '0': 0.0, '0.0':0.0, 0:0.0, '1':1.0, '1.0':1.0, 1:1.0}
    if task_type == 'similarity' or dataset_type == 'semeval':
        label_norm = {'old':(0, 5), 'new':(-1, 1)}
    elif task_type == 'binary' or dataset_type == 'patentmatch_balanced':
        label_norm = {'old':(0, 1), 'new':(-1, 1)}
    else:
        label_norm = None
    
    # Create test dataset with sequential processing
    test_dataset = create_paired_groups_dataset(
        data_dir=[str(path) for path in data_config.test_paths],
        config=config,
        model_type=config['model'].get('model_type', 'matching'),
        strict_matching=config['data'].get('strict_matching', False),
        contrastive_mode=config['data'].get('contrastive_mode', False),
        batch_size=config['data']['batch_size'],
        shuffle_files=False,  # No shuffling for evaluation
        max_active_files=4,
        min_trees_per_group=1,
        label_map=label_map,
        label_norm=label_norm
    )
    
    # Create sequential dataloader
    test_loader = get_paired_groups_dataloader(
        test_dataset, 
        # num_workers=config['data']['num_workers_val'],
        num_workers=0,
        persistent_workers=False
    )
    
    # Create appropriate loss function
    loss_loader = 'other' if task_loader_type != 'aggregative' else task_loader_type
    
    loss_fn = LOSS_HANDLERS[task_type][loss_loader](
        device=config['device'],
        temperature=config['model'].get('temperature', 0.07),
        aggregation=config['model'].get('aggregation', 'attention'),
        threshold=config['model'].get("threshold", 0.5),
        num_classes=config['model'].get("num_classes", 3),
        classifier_input_dim=config['model'].get("graph_rep_dim", 1792) * 2,
        classifier_hidden_dims=config['model'].get("classifier_hidden_dims", [512]),
        positive_infonce_weight=config['model'].get("positive_infonce_weight", 1.0),
        inverse_infonce_weight=config['model'].get("inverse_infonce_weight", 0.25),
        midpoint_infonce_weight=config['model'].get("midpoint_infonce_weight", 0.25),
        thresh_low=config['model'].get("thresh_low", -1),
        thresh_high=config['model'].get("thresh_high", 0)
    )
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb_name = args.wandb_name or Path(args.output_dir).stem
        wandb_tags = list(set(args.wandb_tags.split(",") if args.wandb_tags else []) | set([*config['wandb'].get('tags', [])]))
        wandb_tags.extend([task_type, config['data']['dataset_type']])
        
        wandb.init(
            project=args.wandb_project if args.wandb_project else config['wandb']['project'],
            name=wandb_name,
            config={
                "task_type": task_type,
                "dataset": config['data']['dataset_type'],
                "model_type": config['model'].get('model_type', 'matching'),
                "batch_size": args.batch_size,
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint_epoch,
                "evaluation": True
            },
            tags=wandb_tags
        )
    
    # Move model to device
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Evaluate model
    logger.info("Starting evaluation...")
    metrics, predictions, labels = evaluate_model(model, test_loader, loss_fn, device, config)
    
    # Print results
    logger.info("Evaluation results:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            logger.info(f"{k}: {v}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_file = output_dir / f"{Path(args.checkpoint).stem}_metrics.json"
    metrics['checkpoint_path'] = args.checkpoint
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Save predictions and labels
    predictions_file = output_dir / f"{Path(args.checkpoint).stem}_predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump({
            'predictions': predictions,
            'labels': labels
        }, f, indent=2)
    logger.info(f"Saved predictions to {predictions_file}")
    
    # Log to wandb if enabled
    if args.use_wandb:
        # Log metrics
        wandb_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        wandb.log(wandb_metrics)
        
        # Generate and log plots
        if task_type == 'entailment':
            # Plot confusion matrix
            class_names = ["contradiction", "neutral", "entailment"]
            cm_fig = get_confusion_matrix_plot(
                np.array(metrics['confusion_matrix']),
                class_names
            )
            wandb.log({"confusion_matrix": wandb.Image(cm_fig)})
            plt.close(cm_fig)
            
            # Create a table of class metrics
            class_metrics = []
            for i, cls in enumerate(class_names):
                class_metrics.append([
                    cls, 
                    metrics[f'{cls}_precision'], 
                    metrics[f'{cls}_recall'], 
                    metrics[f'{cls}_f1'],
                    metrics[f'{cls}_support']
                ])
            
            class_table = wandb.Table(
                data=class_metrics,
                columns=["Class", "Precision", "Recall", "F1", "Support"]
            )
            wandb.log({"class_metrics": class_table})
            
        elif task_type == 'similarity':
            # Scatter plot
            scatter_fig = get_scatter_plot(labels, predictions, metrics["pearson"])
            wandb.log({"scatter_plot": wandb.Image(scatter_fig)})
            plt.close(scatter_fig)
            
            # Histogram of predictions vs true values
            wandb.log({
                "prediction_distribution": wandb.Histogram(predictions),
                "label_distribution": wandb.Histogram(labels)
            })
            
        elif task_type == 'binary':
            # Plot confusion matrix
            class_names = ["negative", "positive"]
            cm_fig = get_confusion_matrix_plot(
                np.array(metrics['confusion_matrix']),
                class_names
            )
            wandb.log({"confusion_matrix": wandb.Image(cm_fig)})
            plt.close(cm_fig)
            
            # Create ROC curve
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve([int(l > 0.5) for l in labels], predictions)
            roc_auc = auc(fpr, tpr)
            
            roc_fig, ax = plt.subplots(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            wandb.log({"roc_curve": wandb.Image(roc_fig)})
            plt.close(roc_fig)
            
        # Log sample predictions
        sample_size = min(100, len(predictions))
        sample_idx = np.random.choice(len(predictions), sample_size, replace=False)
        
        samples = []
        for i in sample_idx:
            samples.append([i, labels[i], predictions[i]])
        
        samples_table = wandb.Table(
            data=samples,
            columns=["Index", "True Label", "Prediction"]
        )
        wandb.log({"prediction_samples": samples_table})
        
        # Finish wandb run
        wandb.finish()
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
