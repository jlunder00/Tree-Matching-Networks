# train_unified.py
import torch.multiprocessing as mp
import wandb
import torch
from pathlib import Path
import logging
import argparse
from datetime import datetime
import sys
import yaml
import os
from typing import Optional, Dict, Any

# Import necessary modules based on your project structure
try:
    from ..configs.default_tree_config import get_tree_config
    from ..configs.tree_data_config import TreeDataConfig
    from ..data import create_paired_groups_dataset, DynamicCalculatedContrastiveDataset
    from ..data import get_paired_groups_dataloader, get_dynamic_calculated_dataloader
    from ..models.tree_matching import TreeMatchingNet
    from ..models.tree_embedding import TreeEmbeddingNet
    from ..models.bert_embedding import BertEmbeddingNet
    from ..training.experiment import ExperimentManager
    from ..training.train import train_epoch
    from ..training.validation import validate_epoch
    from ..utils.memory_utils import MemoryMonitor
except ImportError:
    # Alternative import paths for direct script execution
    from Tree_Matching_Networks.LinguisticTrees.configs.default_tree_config import get_tree_config
    from Tree_Matching_Networks.LinguisticTrees.configs.tree_data_config import TreeDataConfig
    from Tree_Matching_Networks.LinguisticTrees.data import create_paired_groups_dataset, DynamicCalculatedContrastiveDataset
    from Tree_Matching_Networks.LinguisticTrees.data import get_paired_groups_dataloader, get_dynamic_calculated_dataloader
    from Tree_Matching_Networks.LinguisticTrees.models.tree_matching import TreeMatchingNet
    from Tree_Matching_Networks.LinguisticTrees.models.tree_embedding import TreeEmbeddingNet
    from Tree_Matching_Networks.LinguisticTrees.models.bert_embedding import BertEmbeddingNet
    from Tree_Matching_Networks.LinguisticTrees.training.experiment import ExperimentManager
    from Tree_Matching_Networks.LinguisticTrees.training.train import train_epoch
    from Tree_Matching_Networks.LinguisticTrees.training.validation import validate_epoch
    from Tree_Matching_Networks.LinguisticTrees.utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def train_unified(args):
    """
    Unified training workflow supporting both contrastive and aggregative approaches.
    
    This function handles the complete training pipeline, from configuration and
    initialization to the training loop and evaluation.
    
    Args:
        args: Command-line arguments including config paths, training mode, etc.
    """
    #==========================================================================
    # Phase 1: Configuration & Experiment Setup
    #==========================================================================
    
    # 1.1 Initial config loading
    logger.info(f"Initializing unified training with mode: {args.mode}")
    base_config = None
    override_config = None
    
    # 1.2 Load base config
    base_config_path = args.config
    if base_config_path:
        with open(base_config_path, 'r') as fin:
            base_config = yaml.safe_load(fin)
    else:
        # Default config paths based on mode
        if args.mode == 'contrastive':
            base_config_path = Path('/home/jlunder/research/Tree-Matching-Networks/Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/contrastive_config.yaml')
        else:  # aggregative
            base_config_path = Path('/home/jlunder/research/Tree-Matching-Networks/Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/aggregative_config.yaml')
        with open(base_config_path, 'r') as fin:
            base_config = yaml.safe_load(fin)
    
    # 1.3 Load override config if provided
    if args.override:
        with open(args.override, 'r') as fin:
            override_config = yaml.safe_load(fin)
    
    # 1.4 Handle resuming from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        
        # When resuming, we can either use the config from checkpoint or passed config
        if args.config is None:
            logger.info("Using configuration from checkpoint")
            base_config = None  # Will be loaded from checkpoint
        
        checkpoint, experiment, base_config, override_config = ExperimentManager.load_checkpoint(
            args.resume, base_config, override_config
        )
        
        if args.resume_with_epoch:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
    else:
        # Create fresh experiment
        task_type = args.task_type if args.task_type else base_config['model']['task_type']
        experiment = ExperimentManager(task_type, base_config, override_config)
    
    # 1.5 Get final merged configuration
    logger.info("Finalizing configuration")
    config = get_tree_config(
        task_type=args.task_type if args.task_type else 'infonce',
        base_config=base_config,
        override_config=override_config
    )
    
    # 1.6 Initialize WandB
    is_sweep_run = wandb.run is not None and wandb.run.name is not None
    
    if not is_sweep_run:
        wandb.init(
            project=config['wandb']['project'],
            name=f"{args.mode}_{experiment.timestamp}",
            config=config,
            tags=[args.mode, *config['wandb'].get('tags', [])]
        )
    else:
        # Update experiment tags if in a sweep
        wandb.run.tags = list(set(wandb.run.tags) | set([args.mode, *config['wandb'].get('tags', [])]))
    
    #==========================================================================
    # Phase 2: Data Configuration
    #==========================================================================
    
    # 2.1 Set up data configuration
    logger.info("Setting up data configuration")
    
    # 2.2 Handle custom data root if provided
    if args.data_root:
        data_config = TreeDataConfig(
            data_root=args.data_root,
            dataset_specs=config.get('data', {}).get('dataset_specs', 
                                                   [config.get('data', {}).get('dataset_type', 'wikiqs')]),
            task_type=config.get('model', {}).get('task_type', 'infonce'),  # Empty for flexibility
            use_sharded_train=True,
            use_sharded_validate=True,
            allow_cross_dataset_negatives=config.get('data', {}).get('allow_cross_dataset_negatives', True)
        )
    else:
        data_config = TreeDataConfig(
            dataset_specs=config.get('data', {}).get('dataset_specs', 
                                                   [config.get('data', {}).get('dataset_type', 'wikiqs')]),
            task_type=config.get('model', {}).get('task_type', 'infonce'),  # Empty for flexibility
            use_sharded_train=True,
            use_sharded_validate=True,
            allow_cross_dataset_negatives=config.get('data', {}).get('allow_cross_dataset_negatives', True)
        )
    
    # 2.3 Set up common dataset parameters
    logger.info("Preparing dataset parameters")
    text_mode = config.get('text_mode', False)
    allow_text_files = config.get('allow_text_files', False)
    # Force text_mode if allow_text_files is enabled
    if allow_text_files and not text_mode:
        logger.warning("allow_text_files is True but text_mode is False. Forcing text_mode to True")
        text_mode = True
        config['text_mode'] = True
    tokenizer = None
    
    #==========================================================================
    # Phase 3: Model Initialization
    #==========================================================================
    
    # 3.1 Initialize text model or graph model based on config
    logger.info(f"Initializing model (text_mode: {text_mode})")
    
    if text_mode:
        # Text mode - initialize BERT or other transformer model
        from transformers import AutoTokenizer
        
        tokenizer_path = config['model']['bert'].get('tokenizer_path', 'bert-base-uncased')
        model = BertEmbeddingNet(config).to(config['device']) 
        logger.info(f"Loading tokenizer from {tokenizer_path}")

        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        # Graph mode - initialize TreeMatchingNet or TreeEmbeddingNet
        model_type = config['model'].get('model_type', 'matching')
        logger.info(f"Creating {model_type} graph model")
        
        if model_type == 'embedding':
            model = TreeEmbeddingNet(config).to(config['device'])
        else:
            model = TreeMatchingNet(config).to(config['device'])
    
    # 3.2 Initialize optimizer
    logger.info("Initializing optimizer")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    # 3.3 Load checkpoint state if resuming
    if args.resume:
        logger.info("Loading model and optimizer state from checkpoint")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Log best metrics from checkpoint
        if 'best_metrics' in checkpoint:
            for k, v in checkpoint['best_metrics'].items():
                wandb.run.summary[f"best_{k}"] = v
    
    #==========================================================================
    # Phase 4: Dataset Creation
    #==========================================================================
    
    # 4.1 Set up label mapping
    logger.info("Creating datasets")
    label_map = {'-': 1.0, 'entailment':1.0, 'neutral':0.0, 'contradiction':-1.0, 
                '0': 0.0, '0.0':0.0, 0:0.0, '1':1.0, '1.0':1.0, 1:1.0}
    
    # 4.2 Configure label normalization if needed
    label_norm = None
    task_type = config['model']['task_type']
    dataset_type = config['data']['dataset_type']
    
    if task_type == 'similarity' or dataset_type == 'semeval':
        label_norm = {'old':(0, 5), 'new':(-1, 1)}
    elif task_type == 'binary' or dataset_type == 'patentmatch_balanced':
        label_norm = {'old': (0, 1), 'new': (-1, 1)}
    
    # 4.3 Create appropriate datasets based on training mode
    if args.mode == 'contrastive':
        # Contrastive mode - use DynamicCalculatedContrastiveDataset
        logger.info("Creating contrastive datasets")
        
        train_dataset = DynamicCalculatedContrastiveDataset(
            data_dir=[str(path) for path in data_config.train_paths],
            config=config,
            batch_size=config['data']['batch_size'],  
            anchors_per_group=config['data'].get('anchors_per_group', 1),
            pos_pairs_per_anchor=config['data'].get('pos_pairs_per_anchor', 1),
            shuffle_files=True,
            prefetch_factor=config['data'].get('prefetch_factor', 2),
            max_active_files=4,
            allow_cross_dataset_negatives=data_config.allow_cross_dataset_negatives,
            recycle_leftovers=True,
            model_type=config['model'].get('model_type', 'matching'),
            strict_matching=config['data'].get('strict_matching', False),
            text_mode=text_mode,
            allow_text_files=allow_text_files,
            tokenizer=tokenizer,
            max_length=config.get('max_length', 512)
        )
        
        val_dataset = DynamicCalculatedContrastiveDataset(
            data_dir=[str(path) for path in data_config.dev_paths],
            config=config,
            batch_size=config['data']['batch_size'],
            anchors_per_group=config['data'].get('anchors_per_group', 1),
            pos_pairs_per_anchor=config['data'].get('pos_pairs_per_anchor', 1),
            shuffle_files=True,
            prefetch_factor=config['data'].get('prefetch_factor', 2),
            max_active_files=4,
            allow_cross_dataset_negatives=data_config.allow_cross_dataset_negatives,
            recycle_leftovers=True,
            model_type=config['model'].get('model_type', 'matching'),
            strict_matching=config['data'].get('strict_matching', False),
            text_mode=text_mode,
            allow_text_files=allow_text_files,
            tokenizer=tokenizer,
            max_length=config.get('max_length', 512)
        )
    else:
        # Aggregative mode - use paired groups dataset
        logger.info("Creating aggregative datasets")
        
        train_dataset = create_paired_groups_dataset(
            data_dir=[str(path) for path in data_config.train_paths],
            config=config,
            model_type=config['model'].get('model_type', 'matching'),
            strict_matching=config['data'].get('strict_matching', False),
            contrastive_mode=config['data'].get('contrastive_mode', False),
            batch_size=config['data']['batch_size'],
            shuffle_files=True,
            prefetch_factor=config['data'].get('prefetch_factor', 2),
            max_active_files=4,
            min_trees_per_group=1,
            label_map=label_map,
            label_norm=label_norm,
            text_mode=text_mode,
            allow_text_files=allow_text_files,
            tokenizer=tokenizer,
            max_length=config.get('max_length', 512)
        )
        
        val_dataset = create_paired_groups_dataset(
            data_dir=[str(path) for path in data_config.dev_paths],
            config=config,
            model_type=config['model'].get('model_type', 'matching'),
            strict_matching=config['data'].get('strict_matching', False),
            contrastive_mode=config['data'].get('contrastive_mode', False),
            batch_size=config['data']['batch_size'],
            shuffle_files=True,
            prefetch_factor=config['data'].get('prefetch_factor', 2),
            max_active_files=4,
            min_trees_per_group=1,
            label_map=label_map,
            label_norm=label_norm,
            text_mode=text_mode,
            allow_text_files=allow_text_files,
            tokenizer=tokenizer,
            max_length=config.get('max_length', 512)
        )
    
    #==========================================================================
    # Phase 5: Training Loop
    #==========================================================================
    
    # 5.1 Initialize training state
    logger.info("Starting training loop")
    best_val_loss = float('inf') if not args.resume else checkpoint.get('best_val_loss', float('inf'))
    patience_counter = 0
    
    # 5.2 Training epochs
    for epoch in range(start_epoch, config['train']['n_epochs']):
        logger.info(f"Starting epoch {epoch}/{config['train']['n_epochs']}")
        
        # 5.3 Train for one epoch
        train_metrics = train_epoch(model, train_dataset, optimizer, config, epoch)
        
        # 5.4 Validate
        val_metrics = validate_epoch(model, val_dataset, config, epoch)
        
        # 5.5 Log metrics
        metrics = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)
        
        # 5.6 Save checkpoint
        experiment.save_checkpoint(
            model, optimizer, epoch, metrics
        )
        
        # 5.7 Early stopping based on validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            experiment.save_best_model(
                model, optimizer, epoch,
                {**metrics, 'best_val_loss': best_val_loss}
            )
            patience_counter = 0
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['train']['patience'] and not args.ignore_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    wandb.finish()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified training for tree matching models')
    parser.add_argument('--config', type=str,
                      help='Base config path', default=None)
    parser.add_argument('--override', type=str,
                      help='Override config path', default=None)
    parser.add_argument('--resume', type=str,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    parser.add_argument('--resume_with_epoch', action='store_true',
                        help='Pickup from same epoch number as previous run (for resuming crashed runs)')
    parser.add_argument('--ignore_patience', action='store_true',
                        help='Ignore early stopping patience')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root data directory with dev/test/train folders')
    parser.add_argument('--mode', type=str, choices=['contrastive', 'aggregative'],
                      help='Training mode', default='contrastive')
    parser.add_argument('--task_type', type=str, default=None,
                      help='Task type (infonce, similarity, entailment, etc.)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        train_unified(args)
    except Exception as e:
        logger.exception("Training failed with error:")
        raise
