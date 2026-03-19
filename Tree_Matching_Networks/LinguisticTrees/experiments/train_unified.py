# train_unified.py
import torch.multiprocessing as mp
import random
import wandb
import torch
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime
import sys
import yaml
import os
from typing import Optional, Dict, Any

# Note: Initial seed setting moved to train function for configurability

# Import necessary modules based on your project structure
try:
    from ..configs.default_tree_config import get_tree_config
    from ..configs.tree_data_config import TreeDataConfig
    from ..data import create_paired_groups_dataset, DynamicCalculatedContrastiveDataset
    from ..data import get_paired_groups_dataloader, get_dynamic_calculated_dataloader
    from ..models.tree_matching import TreeMatchingNet
    from ..models.tree_embedding import TreeEmbeddingNet
    from ..models.bert_embedding import BertEmbeddingNet
    from ..models.bert_matching import BertMatchingNet
    from ..models.pretrained_text_embedding import PretrainedTextEmbeddingNet
    from ..models.pretrained_text_matching import PretrainedTextMatchingNet
    from ..models.pretrained_tree_embedding import PretrainedTreeEmbeddingNet
    from ..models.pretrained_tree_matching import PretrainedTreeMatchingNet
    from ..models.pretrained_noprop_embedding import PretrainedNoPropEmbeddingNet
    from ..models.pretrained_noprop_matching import PretrainedNoPropMatchingNet
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
    from Tree_Matching_Networks.LinguisticTrees.models.bert_matching import BertMatchingNet
    from Tree_Matching_Networks.LinguisticTrees.models.pretrained_text_embedding import PretrainedTextEmbeddingNet
    from Tree_Matching_Networks.LinguisticTrees.models.pretrained_text_matching import PretrainedTextMatchingNet
    from Tree_Matching_Networks.LinguisticTrees.models.pretrained_tree_embedding import PretrainedTreeEmbeddingNet
    from Tree_Matching_Networks.LinguisticTrees.models.pretrained_tree_matching import PretrainedTreeMatchingNet
    from Tree_Matching_Networks.LinguisticTrees.models.pretrained_noprop_embedding import PretrainedNoPropEmbeddingNet
    from Tree_Matching_Networks.LinguisticTrees.models.pretrained_noprop_matching import PretrainedNoPropMatchingNet
    from Tree_Matching_Networks.LinguisticTrees.training.experiment import ExperimentManager
    from Tree_Matching_Networks.LinguisticTrees.training.train import train_epoch
    from Tree_Matching_Networks.LinguisticTrees.training.validation import validate_epoch
    from Tree_Matching_Networks.LinguisticTrees.utils.memory_utils import MemoryMonitor

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Additional determinism for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_unified(args):
    """
    Unified training workflow supporting both contrastive and aggregative approaches.
    
    This function handles the complete training pipeline, from configuration and
    initialization to the training loop and evaluation.
    
    Args:
        args: Command-line arguments including config paths, training mode, etc.
    """
    
    logger.info(f"Initializing unified training with mode: {args.mode}")
    base_config = None
    override_config = None
    
    base_config_path = args.config
    if base_config_path:
        with open(base_config_path, 'r') as fin:
            base_config = yaml.safe_load(fin)
    else:
        if args.mode == 'contrastive':
            base_config_path = Path('/home/jlunder/research/Tree-Matching-Networks/Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/contrastive_config.yaml')
        else:  # aggregative
            base_config_path = Path('/home/jlunder/research/Tree-Matching-Networks/Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/aggregative_config.yaml')
        with open(base_config_path, 'r') as fin:
            base_config = yaml.safe_load(fin)
    
    if args.override:
        with open(args.override, 'r') as fin:
            override_config = yaml.safe_load(fin)
    
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
        task_type = args.task_type if args.task_type else base_config['model']['task_type']
        experiment = ExperimentManager(task_type, base_config, override_config)
    
    logger.info("Finalizing configuration")
    config = get_tree_config(
        task_type=args.task_type if args.task_type else 'infonce',
        base_config=base_config,
        override_config=override_config
    )

    # Override n_epochs if specified on command line
    if args.n_epochs is not None:
        logger.info(f"Overriding n_epochs from config ({config['train']['n_epochs']}) to command-line value ({args.n_epochs})")
        config['train']['n_epochs'] = args.n_epochs

    # Override max_batches_per_epoch if specified on command line
    if args.max_batches_per_epoch is not None:
        logger.info(f"Overriding max_batches_per_epoch from config ({config['data']['max_batches_per_epoch']}) to command-line value ({args.max_batches_per_epoch})")
        config['data']['max_batches_per_epoch'] = args.max_batches_per_epoch

    # Override batch_size if specified on command line
    if args.batch_size is not None:
        logger.info(f"Overriding batch_size from config ({config['data']['batch_size']}) to command-line value ({args.batch_size})")
        config['data']['batch_size'] = args.batch_size

    # Override temperature if specified on command line
    if args.temperature is not None:
        old_temp = config['model'].get('temperature', 'not set')
        logger.info(f"Overriding temperature from config ({old_temp}) to command-line value ({args.temperature})")
        config['model']['temperature'] = args.temperature

    # Override target_neg_to_pos_ratio if specified on command line
    if args.target_neg_to_pos_ratio is not None:
        if 'hard_negative_mining' in config['data']:
            old_ratio = config['data']['hard_negative_mining'].get('target_neg_to_pos_ratio', 'not set')
            logger.info(f"Overriding target_neg_to_pos_ratio from config ({old_ratio}) to command-line value ({args.target_neg_to_pos_ratio})")
            config['data']['hard_negative_mining']['target_neg_to_pos_ratio'] = args.target_neg_to_pos_ratio
        else:
            logger.warning("Cannot override target_neg_to_pos_ratio: hard_negative_mining not in config")

    # Override pos_pairs_per_anchor if specified on command line
    if args.pos_pairs_per_anchor is not None:
        old_pos = config['data'].get('pos_pairs_per_anchor', 'not set')
        logger.info(f"Overriding pos_pairs_per_anchor from config ({old_pos}) to command-line value ({args.pos_pairs_per_anchor})")
        config['data']['pos_pairs_per_anchor'] = args.pos_pairs_per_anchor

    # Override anchors_per_group if specified on command line
    if args.anchors_per_group is not None:
        old_anchors = config['data'].get('anchors_per_group', 'not set')
        logger.info(f"Overriding anchors_per_group from config ({old_anchors}) to command-line value ({args.anchors_per_group})")
        config['data']['anchors_per_group'] = args.anchors_per_group

    if args.bidirectional_anchor_pairs is not None:
        old_bidirectional = config['data'].get('bidirectional_anchor_pairs', 'not set')
        logger.info(f"Overriding bidirectional_anchor_pairs from config ({old_bidirectional}) to command-line value ({args.bidirectional_anchor_pairs})")
        config['data']['bidirectional_anchor_pairs'] = args.bidirectional_anchor_pairs

    if args.allow_anchor_anchor_pairing is not None:
        old_allow = config['data'].get('allow_anchor_anchor_pairing', 'not set')
        logger.info(f"Overriding allow_anchor_anchor_pairing from config ({old_allow}) to command-line value ({args.allow_anchor_anchor_pairing})")
        config['data']['allow_anchor_anchor_pairing'] = args.allow_anchor_anchor_pairing

    is_sweep_run = wandb.run is not None and wandb.run.name is not None

    # Prepare WandB run name
    if args.wandb_name:
        run_name = f"{args.wandb_name}_{experiment.timestamp}"
    else:
        run_name = f"{args.mode}_{experiment.timestamp}"

    # Prepare WandB tags
    wandb_tags = [args.mode, *config['wandb'].get('tags', [])]
    if args.wandb_tags:
        additional_tags = [tag.strip() for tag in args.wandb_tags.split(',')]
        wandb_tags.extend(additional_tags)

    # Determine WandB project name (command-line override or config)
    wandb_project = args.wandb_project if args.wandb_project else config['wandb']['project']

    if not is_sweep_run:
        wandb.init(
            project=wandb_project,
            name=run_name,
            config=config,
            tags=wandb_tags
        )
    else:
        # Update experiment tags if in a sweep
        wandb.run.tags = list(set(wandb.run.tags) | set(wandb_tags))
    
    logger.info("Setting up data configuration")
    
    if args.data_root:
        data_config = TreeDataConfig(
            data_root=args.data_root,
            dataset_specs=config.get('data', {}).get('dataset_specs',
                                                   [config.get('data', {}).get('dataset_type', 'wikiqs')]),
            task_type=config.get('model', {}).get('task_type', 'infonce'),  # Empty for flexibility
            use_sharded_train=True,
            use_sharded_validate=True,
            use_full_suffix=args.use_full_suffix,
            allow_cross_dataset_negatives=config.get('data', {}).get('allow_cross_dataset_negatives', True)
        )
    else:
        data_config = TreeDataConfig(
            dataset_specs=config.get('data', {}).get('dataset_specs',
                                                   [config.get('data', {}).get('dataset_type', 'wikiqs')]),
            task_type=config.get('model', {}).get('task_type', 'infonce'),  # Empty for flexibility
            use_sharded_train=True,
            use_sharded_validate=True,
            use_full_suffix=args.use_full_suffix,
            allow_cross_dataset_negatives=config.get('data', {}).get('allow_cross_dataset_negatives', True)
        )
    
    logger.info("Preparing dataset parameters")
    text_mode = config.get('text_mode', False)
    allow_text_files = config.get('allow_text_files', False)

    # Force text_mode if allow_text_files is enabled
    if allow_text_files and not text_mode:
        logger.warning("allow_text_files is True but text_mode is False. Forcing text_mode to True")
        text_mode = True
        config['text_mode'] = True
    tokenizer = None
    


    logger.info(f"Initializing model (text_mode: {text_mode})")

    model_name = config['model'].get('name', '')
    model_type = config['model'].get('model_type', 'matching')

    if model_name == 'pretrained_text':
        # Condition A: pretrained HF on text
        from transformers import AutoTokenizer
        hf_model_name = config['model']['pretrained']['model_name']
        logger.info(f"Loading pretrained text model: {hf_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if model_type == 'embedding':
            model = PretrainedTextEmbeddingNet(config, tokenizer).to(config['device'])
        else:
            model = PretrainedTextMatchingNet(config, tokenizer).to(config['device'])
        config['model_name'] = 'bert'

    elif model_name == 'pretrained_noprop':
        # Condition B: pretrained HF on raw node features
        logger.info("Loading pretrained no-propagation model")
        if model_type == 'embedding':
            model = PretrainedNoPropEmbeddingNet(config).to(config['device'])
        else:
            model = PretrainedNoPropMatchingNet(config).to(config['device'])
        config['model_name'] = 'graph'

    elif model_name == 'pretrained_tree':
        # Conditions D, E, F: GNN + pretrained HF aggregator
        logger.info("Loading pretrained tree model")
        if model_type == 'embedding':
            model = PretrainedTreeEmbeddingNet(config).to(config['device'])
        else:
            model = PretrainedTreeMatchingNet(config).to(config['device'])
        config['model_name'] = 'graph'

        # Apply freeze config
        freeze_config = config['model'].get('freeze', {})
        if freeze_config.get('freeze_propagation', False):
            model.freeze_propagation()
            logger.info("Froze GNN propagation layers (Condition F)")
        if freeze_config.get('freeze_transformer', False):
            model.freeze_transformer()
            logger.info("Froze pretrained transformer (Condition E)")

    elif text_mode:
        from transformers import AutoTokenizer

        tokenizer_path = config['model']['bert'].get('tokenizer_path', 'bert-base-uncased')
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if model_type == 'embedding':
            model = BertEmbeddingNet(config, tokenizer).to(config['device'])
        else:
            model = BertMatchingNet(config, tokenizer).to(config['device'])
        config['model_name'] = 'bert'

    else:
        logger.info(f"Creating {model_type} graph model")

        if model_type == 'embedding':
            model = TreeEmbeddingNet(config).to(config['device'])
        else:
            model = TreeMatchingNet(config).to(config['device'])
        config['model_name'] = 'graph'

    logger.info("Initializing optimizer")
    base_lr = config['train']['learning_rate']
    pretrained_lr_scale = config['train'].get('pretrained_lr_scale', 1.0)

    if hasattr(model, 'get_parameter_groups'):
        param_groups = model.get_parameter_groups(base_lr, pretrained_lr_scale)
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=config['train']['weight_decay']
        )
        logger.info(f"Optimizer: {len(param_groups)} param groups, pretrained_lr_scale={pretrained_lr_scale}")
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=config['train']['weight_decay']
        )

    if args.resume:
        logger.info("Loading model and optimizer state from checkpoint")
        strict_load = config['model'].get('strict_checkpoint_load', True)
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'], strict=strict_load
        )
        if missing:
            logger.info(f"Missing keys (expected for pretrained components): {len(missing)} keys")
            for key in missing[:5]:
                logger.info(f"  Missing: {key}")
            if len(missing) > 5:
                logger.info(f"  ... and {len(missing) - 5} more")
        if unexpected:
            logger.info(f"Unexpected keys (from previous aggregator): {len(unexpected)} keys")
            for key in unexpected[:5]:
                logger.info(f"  Unexpected: {key}")
            if len(unexpected) > 5:
                logger.info(f"  ... and {len(unexpected) - 5} more")

        # Only load optimizer state if checkpoint architecture matches
        if not missing and not unexpected:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.info("Skipping optimizer state load due to architecture mismatch")

        # Log best metrics from checkpoint
        if 'best_metrics' in checkpoint:
            for k, v in checkpoint['best_metrics'].items():
                wandb.run.summary[f"best_{k}"] = v
    
    logger.info("Creating datasets")
    label_map = {'-': 1.0, 'entailment':1.0, 'neutral':0.0, 'contradiction':-1.0, 
                '0': 0.0, '0.0':0.0, 0:0.0, '1':1.0, '1.0':1.0, 1:1.0}
    
    label_norm = None
    task_type = config['model']['task_type']
    dataset_type = config['data']['dataset_type']
    
    if task_type == 'similarity' or dataset_type == 'semeval':
        label_norm = {'old':(0, 5), 'new':(-1, 1)}
    elif task_type == 'binary' or dataset_type == 'patentmatch_balanced':
        label_norm = {'old': (0, 1), 'new': (-1, 1)}
    
    prefetch = config['data'].get("prefetch_factor", 2)

    if args.mode == 'contrastive':
        logger.info("Creating contrastive datasets")
        
        train_dataset = DynamicCalculatedContrastiveDataset(
            data_dir=[str(path) for path in data_config.train_paths],
            config=config,
            batch_size=config['data']['batch_size'],
            anchors_per_group=config['data'].get('anchors_per_group', 1),
            pos_pairs_per_anchor=config['data'].get('pos_pairs_per_anchor', 1),
            bidirectional_anchor_pairs=config['data'].get('bidirectional_anchor_pairs', True),
            allow_anchor_anchor_pairing=config['data'].get('allow_anchor_anchor_pairing', True),
            shuffle_files=True,
            prefetch_factor=prefetch if prefetch > 0 else None,
            max_active_files=4,
            allow_cross_dataset_negatives=data_config.allow_cross_dataset_negatives,
            recycle_leftovers=True,
            model_type=config['model'].get('model_type', 'matching'),
            strict_matching=config['data'].get('strict_matching', False),
            text_mode=text_mode,
            allow_text_files=allow_text_files,
            tokenizer=tokenizer,
            max_length=config['model'].get('bert', {}).get('max_position_embeddings', 512)
        )
        
        val_dataset = DynamicCalculatedContrastiveDataset(
            data_dir=[str(path) for path in data_config.dev_paths],
            config=config,
            batch_size=config['data']['batch_size'],
            anchors_per_group=config['data'].get('anchors_per_group', 1),
            pos_pairs_per_anchor=config['data'].get('pos_pairs_per_anchor', 1),
            bidirectional_anchor_pairs=config['data'].get('bidirectional_anchor_pairs', True),
            allow_anchor_anchor_pairing=config['data'].get('allow_anchor_anchor_pairing', True),
            shuffle_files=True,
            prefetch_factor=prefetch if prefetch > 0 else None,
            max_active_files=4,
            allow_cross_dataset_negatives=data_config.allow_cross_dataset_negatives,
            recycle_leftovers=True,
            model_type=config['model'].get('model_type', 'matching'),
            strict_matching=config['data'].get('strict_matching', False),
            text_mode=text_mode,
            allow_text_files=allow_text_files,
            tokenizer=tokenizer,
            max_length=config['model'].get('bert', {}).get('max_position_embeddings', 512)
        )
    else:
        logger.info("Creating aggregative datasets")
        
        train_dataset = create_paired_groups_dataset(
            data_dir=[str(path) for path in data_config.train_paths],
            config=config,
            model_type=config['model'].get('model_type', 'matching'),
            strict_matching=config['data'].get('strict_matching', False),
            contrastive_mode=config['data'].get('contrastive_mode', False),
            batch_size=config['data']['batch_size'],
            shuffle_files=True,
            prefetch_factor=prefetch if prefetch > 0 else None,
            max_active_files=4,
            min_trees_per_group=1,
            label_map=label_map,
            label_norm=label_norm,
            text_mode=text_mode,
            allow_text_files=allow_text_files,
            tokenizer=tokenizer,
            max_length=config['model'].get('bert', {}).get('max_position_embeddings', 512)
        )
        
        val_dataset = create_paired_groups_dataset(
            data_dir=[str(path) for path in data_config.dev_paths],
            config=config,
            model_type=config['model'].get('model_type', 'matching'),
            strict_matching=config['data'].get('strict_matching', False),
            contrastive_mode=config['data'].get('contrastive_mode', False),
            batch_size=config['data']['batch_size'],
            shuffle_files=True,
            prefetch_factor=prefetch if prefetch > 0 else None,
            max_active_files=4,
            min_trees_per_group=1,
            label_map=label_map,
            label_norm=label_norm,
            text_mode=text_mode,
            allow_text_files=allow_text_files,
            tokenizer=tokenizer,
            max_length=config['model'].get('bert', {}).get('max_position_embeddings', 512)
        )
    
    logger.info("Starting training loop")
    best_val_loss = float('inf') if not args.resume else checkpoint.get('best_val_loss', float('inf'))
    patience_counter = 0

    # Seed management configuration
    base_seed = config['train'].get('base_seed', 42)
    reset_epoch_enabled = config['train'].get('reset_epoch_enabled', True)

    logger.info(f"Seed configuration: base_seed={base_seed}, reset_epoch_enabled={reset_epoch_enabled}")

    # Set initial seed ONCE - random state will evolve naturally from here
    set_seed(base_seed)

    if args.resume_with_epoch and args.mode == 'contrastive':
        r = random.uniform(0, 1)
        train_dataset.set_pairing_ratio(r)
        val_dataset.set_pairing_ratio(r)
        print("using random for first epoch")
    for epoch in range(start_epoch, config['train']['n_epochs']):
        logger.info(f"Starting epoch {epoch}/{config['train']['n_epochs']}")

        # Reset datasets for new epoch to get fresh random data selection
        # This uses the current random state (which evolves each epoch)
        if reset_epoch_enabled:
            if hasattr(train_dataset, 'reset_epoch'):
                train_dataset.reset_epoch()
                logger.debug("Train dataset reset for new epoch (new random data selection)")
            if hasattr(val_dataset, 'reset_epoch'):
                val_dataset.reset_epoch()
                logger.debug("Validation dataset reset for new epoch (new random data selection)")
        else:
            logger.debug("Dataset reset disabled - using continuous data stream (OLD BEHAVIOR)")

        if args.mode == 'contrastive':
            print(f"RATIO IS: {train_dataset.get_pairing_ratio()}")
        
        train_metrics = train_epoch(model, train_dataset, optimizer, config, epoch)
        
        val_metrics = validate_epoch(model, val_dataset, config, epoch)
        if args.mode == 'contrastive':
            if epoch + 1 < 16 and epoch+1 % 4 == 0 and train_dataset.get_pairing_ratio() > 0.0:
                r = train_dataset.get_pairing_ratio()
                train_dataset.set_pairing_ratio(max(r-0.24, 0.0))
                val_dataset.set_pairing_ratio(max(r-0.24, 0.0))
            elif epoch+ 1 < 20: #random per epoch
                r = random.uniform(0, 1) #random ratio
                train_dataset.set_pairing_ratio(r)
                val_dataset.set_pairing_ratio(r)
                print("random for epoch")
            else: #random per batch
                r = -1
                train_dataset.set_pairing_ratio(r)
                val_dataset.set_pairing_ratio(r)
                print("random per batch")
        
        metrics = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'base_seed': base_seed,
            'reset_epoch_enabled': reset_epoch_enabled
        }
        wandb.log(metrics)

        # Save regular checkpoint only every N epochs to save space
        checkpoint_save_frequency = config['train'].get('checkpoint_save_frequency', 1)
        if epoch % checkpoint_save_frequency == 0 or epoch == config['train']['n_epochs'] - 1:
            experiment.save_checkpoint(
                model, optimizer, epoch, metrics
            )
            logger.info(f"Saved checkpoint for epoch {epoch}")
        
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
    parser.add_argument('--n_epochs', type=int, default=None,
                      help='Override n_epochs from config (for testing/experimentation)')
    parser.add_argument('--max_batches_per_epoch', type=int, default=None,
                      help='Override max_batches_per_epoch from config (for testing/experimentation)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Override batch_size from config (for consistency across models)')
    parser.add_argument('--wandb_name', type=str, default=None,
                      help='Custom WandB run name (timestamp will be appended)')
    parser.add_argument('--wandb_tags', type=str, default=None,
                      help='Comma-separated additional tags for WandB')
    parser.add_argument('--wandb_project', type=str, default=None,
                      help='Override WandB project name from config')

    # Hyperparameter sweep arguments
    parser.add_argument('--temperature', type=float, default=None,
                      help='Override temperature from config (for contrastive learning)')
    parser.add_argument('--target_neg_to_pos_ratio', type=int, default=None,
                      help='Override target_neg_to_pos_ratio from config (hard negative mining)')
    parser.add_argument('--pos_pairs_per_anchor', type=int, default=None,
                      help='Override pos_pairs_per_anchor from config (contrastive dataset)')
    parser.add_argument('--anchors_per_group', type=int, default=None,
                      help='Override anchors_per_group from config (contrastive dataset)')
    parser.add_argument('--bidirectional_anchor_pairs', type=lambda x: x.lower() == 'true', default=None,
                      help='Override bidirectional_anchor_pairs from config (true/false)')
    parser.add_argument('--allow_anchor_anchor_pairing', type=lambda x: x.lower() == 'true', default=None,
                      help='Override allow_anchor_anchor_pairing from config (true/false)')
    parser.add_argument('--use_full_suffix', action='store_true',
                      help='Use full SNLI training dataset (applies _full suffix to train split only, not dev/test)')

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        train_unified(args)
    except Exception as e:
        logger.exception("Training failed with error:")
        raise
