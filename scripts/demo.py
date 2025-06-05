# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#!/usr/bin/env python
"""
Demo script showing end-to-end pipeline for Tree Matching Networks and BERT models.
Supports both tree-based and BERT-based models for entailment task prediction.
"""

import argparse
import os
import sys
import json
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# --- Imports from TMN_DataGen and Tree Matching Networks ---
from TMN_DataGen.dataset_generator import DatasetGenerator, TreeGroup, generate_group_id
from TMN_DataGen.utils.text_preprocessing import BasePreprocessor, SentenceSplitter
from TMN_DataGen.parsers.multi_parser import MultiParser
from TMN_DataGen import FeatureExtractor

from Tree_Matching_Networks.LinguisticTrees.training.experiment import ExperimentManager
from Tree_Matching_Networks.LinguisticTrees.configs.default_tree_config import get_tree_config
from Tree_Matching_Networks.LinguisticTrees.models.tree_matching import TreeMatchingNet
from Tree_Matching_Networks.LinguisticTrees.models.tree_embedding import TreeEmbeddingNet
from Tree_Matching_Networks.LinguisticTrees.models.bert_matching import BertMatchingNet
from Tree_Matching_Networks.LinguisticTrees.models.bert_embedding import BertEmbeddingNet
from Tree_Matching_Networks.LinguisticTrees.models import TreeAggregator
from Tree_Matching_Networks.LinguisticTrees.data.data_utils import convert_tree_to_graph_data, GraphData
from Tree_Matching_Networks.LinguisticTrees.data.paired_groups_dataset import PairedGroupBatchInfo

def parse_input(input_arg: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Parse input text pairs from file or command line."""
    pairs = []
    labels = []
    
    if os.path.isfile(input_arg):
        import csv
        with open(input_arg, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row or len(row) < 2:
                    continue
                text_a = row[0].strip()
                text_b = row[1].strip()
                label = row[2].strip() if len(row) >= 3 and row[2].strip() != '' else ''
                pairs.append((text_a, text_b))
                labels.append(label)
    else:
        row = input_arg.split("\t")
        if len(row) < 2:
            raise ValueError("Input must have at least two tab-separated texts.")
        text_a = row[0].strip()
        text_b = row[1].strip()
        label = row[2].strip() if len(row) >= 3 and row[2].strip() != '' else 'entailment'
        pairs.append((text_a, text_b))
        labels.append(label)
    
    return pairs, labels

def load_model_from_checkpoint(checkpoint_path: str, base_config=None, override_config=None):
    """Load model from checkpoint and determine if it's tree or BERT based."""
    checkpoint, manager, config, override = ExperimentManager.load_checkpoint(
        checkpoint_path, base_config, override_config
    )
    
    config = get_tree_config(
        task_type='aggregative',
        base_config=base_config, 
        override_config=override_config
    )
    
    # Determine model type based on config
    is_text_mode = config.get('text_mode', False)
    model_type = config['model'].get('model_type', 'matching')
    
    if is_text_mode:
        # Load BERT model
        from transformers import AutoTokenizer
        tokenizer_path = config['model']['bert'].get('tokenizer_path', 'bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        if model_type == 'embedding':
            model = BertEmbeddingNet(config, tokenizer)
        else:
            model = BertMatchingNet(config, tokenizer)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, config, checkpoint.get('epoch', 0), tokenizer
    else:
        # Load tree model
        if model_type == 'embedding':
            model = TreeEmbeddingNet(config)
        else:
            model = TreeMatchingNet(config)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, config, checkpoint.get('epoch', 0), None

def run_tree_demo(text_pairs: List[Tuple[str, str]], labels: List[str], 
                  checkpoint_path: str, spacy_model: Optional[str] = None) -> Dict:
    """Run demo using tree-based models."""
    print("=== Running Tree-based Model Demo ===")
    
    # Load model and config
    model, config, epoch, _ = load_model_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Load TMN_DataGen configs
    import yaml
    from omegaconf import OmegaConf
    
    datagen_config_dir = Path("/home/jlunder/research/TMN_DataGen/TMN_DataGen/configs")
    configs = {}
    for config_file in ["default_package_config.yaml", "default_parser_config.yaml", 
                       "default_preprocessing_config.yaml", "default_feature_config.yaml", 
                       "default_output_format.yaml", "default_merge_config.yaml"]:
        with open(datagen_config_dir / config_file, 'r') as f:
            configs.update(yaml.safe_load(f))
    
    d_config = OmegaConf.create(configs)
    
    if spacy_model:
        d_config.parser['parsers']['spacy']['model_name'] = spacy_model
    
    # Preprocess and parse
    preprocessor = BasePreprocessor(d_config)
    splitter = SentenceSplitter()
    
    grouped_metadata = []
    grouped_sentences = []
    
    for i, (text_a, text_b) in enumerate(text_pairs):
        sentences_a = [preprocessor.preprocess(s) for s in splitter.split(text_a)]
        sentences_b = [preprocessor.preprocess(s) for s in splitter.split(text_b)]
        
        group_id = generate_group_id()
        metadata = {
            "group_id": group_id,
            "text": text_a,
            "text_clean": '.'.join(sentences_a),
            "text_b": text_b,
            "text_b_clean": '.'.join(sentences_b),
            "label": labels[i]
        }
        grouped_metadata.append(metadata)
        grouped_sentences.append((sentences_a, sentences_b))
    
    # Parse into trees
    sentence_groups = []
    for sentences_a, sentences_b in grouped_sentences:
        sentence_groups.extend([sentences_a, sentences_b])
    
    vocabs = [set()]
    multi_parser = MultiParser(d_config, pkg_config=configs, vocabs=vocabs, 
                              logger=None, max_concurrent=0, num_workers=8)
    all_tree_groups = multi_parser.parse_all(sentence_groups, show_progress=False)
    
    # Create TreeGroup objects and convert to InfoNCE
    tree_groups = []
    for i, meta in enumerate(grouped_metadata):
        tg = TreeGroup(
            group_id=meta["group_id"],
            original_text=meta["text"],
            trees=all_tree_groups[2*i],
            original_text_b=meta["text_b"],
            trees_b=all_tree_groups[2*i+1],
            label=meta["label"]
        )
        tree_groups.append(tg)
    
    generator = DatasetGenerator(num_workers=1)
    _, _ = generator._load_configs(*configs.values(), 'normal', override_pkg_config=configs)
    generator.config = d_config
    infonce_data = generator._convert_to_infonce_format(tree_groups, is_paired=True)
    
    # Load embeddings if needed
    requires_embeddings = True  # Assume needed for tree models
    feature_config = {
        'feature_extraction': {
            'word_embedding_model': config.get('word_embedding_model', 'bert-base-uncased'),
            'use_gpu': config.get('use_gpu', True) and torch.cuda.is_available(),
            'cache_embeddings': True,
            'embedding_cache_dir': config.get('embedding_cache_dir', 'embedding_cache'),
            'do_not_store_word_embeddings': False,
            'is_runtime': True,
        },
        'verbose': config.get('verbose', 'normal')
    }
    
    if requires_embeddings:
        embedding_extractor = FeatureExtractor(feature_config)
    
    def load_embeddings(tree: Dict) -> torch.Tensor:
        if not tree.get('node_features_need_word_embs_prepended', False):
            return torch.tensor(tree['node_features'])
        
        embeddings = []
        for word, lemma in tree['node_texts']:
            emb = None
            if lemma in embedding_extractor.embedding_cache:
                emb = embedding_extractor.embedding_cache[lemma]
            elif word in embedding_extractor.embedding_cache:
                emb = embedding_extractor.embedding_cache[word]
            if emb is None:
                emb = embedding_extractor.get_word_embedding(lemma)
            embeddings.append(emb)
        word_embeddings = torch.stack(embeddings)
        node_features = torch.tensor(tree['node_features'])
        return torch.cat([word_embeddings, node_features], dim=-1)
    
    def make_trees_square(trees_a, trees_b):
        sorted_a = sorted(trees_a, key=lambda x: len(x['tree']['node_texts']), reverse=True)
        sorted_b = sorted(trees_b, key=lambda x: len(x['tree']['node_texts']), reverse=True)
        
        if len(sorted_a) < len(sorted_b):
            small, large = sorted_a, sorted_b
            small_label = 'a'
        else:
            small, large = sorted_b, sorted_a
            small_label = 'b'
        
        original_order = small.copy()
        i = 0
        while len(small) < len(large):
            element_to_duplicate = original_order[i % len(original_order)]
            idx = small.index(element_to_duplicate)
            small.insert(idx + 1, element_to_duplicate)
            i += 1
        
        if small_label == 'a':
            return small, large
        else:
            return large, small
    
    # Process batch
    batch_groups = infonce_data["groups"]
    label_map = {'-': 1.0, 'entailment': 1.0, 'neutral': 0.0, 'contradiction': -1.0, 
                '0': 0.0, '0.0': 0.0, 0: 0.0, '1': 1.0, '1.0': 1.0, 1: 1.0}
    
    buffer_groups = []
    for group_idx, group in enumerate(batch_groups):
        group_id = group.get('group_id', None)
        label_raw = group.get('label', -2)
        if isinstance(label_raw, str) and label_raw in label_map:
            label_raw = label_map[label_raw]
        label = float(label_raw)
        
        trees_a = []
        for tree_idx, tree in enumerate(group.get('trees', [])):
            tree = dict(tree)
            tree['node_features'] = load_embeddings(tree)
            trees_a.append({
                'tree': tree,
                'group_id': group_id,
                'group_idx': group_idx,
                'tree_idx': tree_idx,
                'text': tree.get('text', ''),
            })
        
        trees_b = []
        for tree_idx, tree in enumerate(group.get('trees_b', [])):
            tree = dict(tree)
            tree['node_features'] = load_embeddings(tree)
            trees_b.append({
                'tree': tree,
                'group_id': group_id,
                'group_idx': group_idx,
                'tree_idx': tree_idx,
                'text': tree.get('text', ''),
            })
        
        if len(trees_b) != len(trees_a):
            trees_a, trees_b = make_trees_square(trees_a, trees_b)
        
        buffer_groups.append({
            'group_id': group_id,
            'group_idx': group_idx,
            'label': label,
            'trees_a': trees_a,
            'trees_b': trees_b
        })
    
    # Create batch info
    group_ids = []
    group_labels = []
    trees_a_indices = []
    trees_b_indices = []
    batch_trees = []
    anchor_indices = []
    
    for group in buffer_groups:
        group_ids.append(group["group_id"])
        group_labels.append(group["label"])
        indices_a = []
        for tree in group["trees_a"]:
            tree_idx = len(batch_trees)
            indices_a.append(tree_idx)
            batch_trees.append(tree)
            anchor_indices.append(tree_idx)
        
        indices_b = []
        for tree in group["trees_b"]:
            indices_b.append(len(batch_trees))
            batch_trees.append(tree)
        
        trees_a_indices.append(indices_a)
        trees_b_indices.append(indices_b)
    
    # Convert to GraphData
    graph_data = convert_tree_to_graph_data([item['tree'] for item in batch_trees])
    
    batch_info = PairedGroupBatchInfo(
        group_indices=list(range(len(batch_groups))),
        group_ids=group_ids,
        group_labels=group_labels,
        trees_a_indices=trees_a_indices,
        trees_b_indices=trees_b_indices,
        tree_to_group_map={},
        tree_to_set_map={},
        pair_indices=[],
        anchor_indices=anchor_indices,
        strict_matching=False,
        contrastive_mode=False
    )
    
    # Run inference
    graphs = GraphData(
        node_features=graph_data.node_features.to(device),
        edge_features=graph_data.edge_features.to(device),
        from_idx=graph_data.from_idx.to(device),
        to_idx=graph_data.to_idx.to(device),
        graph_idx=graph_data.graph_idx.to(device),
        n_graphs=graph_data.n_graphs
    )
    
    with torch.no_grad():
        embeddings = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
    
    # Aggregate and predict
    aggregator = TreeAggregator('attention')
    text_embeddings = aggregator(embeddings, batch_info)
    
    text_a_embeddings = text_embeddings[0::2]
    text_b_embeddings = text_embeddings[1::2]
    thresh_high = 0.33
    thresh_low = -0.33
    
    sim_mat = torch.nn.functional.cosine_similarity(text_a_embeddings, text_b_embeddings, dim=1)
    predictions = torch.where(sim_mat > thresh_high,
                            torch.tensor(1, device=device),
                            torch.where(sim_mat < thresh_low,
                                      torch.tensor(-1, device=device),
                                      torch.tensor(0, device=device)))
    
    # Calculate accuracy if labels available
    accuracy = None
    if batch_info.group_labels and all(isinstance(label, (int, float)) for label in batch_info.group_labels):
        target = torch.tensor(batch_info.group_labels, device=device).float()
        accuracy = (predictions == target.long()).float().mean().item()
    
    results = {
        'model_type': 'tree',
        'predictions': [pred.item() for pred in predictions],
        'similarities': [sim.item() for sim in sim_mat],
        'accuracy': accuracy,
        'text_pairs': text_pairs,
        'labels': labels
    }
    
    return results

def run_bert_demo(text_pairs: List[Tuple[str, str]], labels: List[str], 
                  checkpoint_path: str) -> Dict:
    """Run demo using BERT-based models."""
    print("=== Running BERT-based Model Demo ===")
    
    # Load model and config
    model, config, epoch, tokenizer = load_model_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Tokenize text pairs
    max_length = config['model']['bert'].get('max_position_embeddings', 512)
    text_encodings = []
    
    for text_a, text_b in text_pairs:
        # For matching models, concatenate and tokenize together
        encoded_a = tokenizer(
            text_a,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encoded_b = tokenizer(
            text_b,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Concatenate for pair processing
        text_encodings.extend([
            {k: v.squeeze(0) for k, v in encoded_a.items()},
            {k: v.squeeze(0) for k, v in encoded_b.items()}
        ])
    
    # Create batch
    if text_encodings:
        batch_encoding = {k: torch.stack([enc[k] for enc in text_encodings]) 
                         for k in text_encodings[0].keys()}
    else:
        batch_encoding = {
            'input_ids': torch.zeros((0, max_length), dtype=torch.long),
            'attention_mask': torch.zeros((0, max_length), dtype=torch.long),
            'token_type_ids': torch.zeros((0, max_length), dtype=torch.long)
        }
    
    # Move to device
    batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
    
    # Run inference
    with torch.no_grad():
        embeddings = model(batch_encoding)
    
    # For BERT matching, we get embeddings for each text, compute similarity
    text_a_embeddings = embeddings[0::2]  # Even indices
    text_b_embeddings = embeddings[1::2]  # Odd indices
    
    thresh_high = 0.33
    thresh_low = -0.33
    
    sim_mat = torch.nn.functional.cosine_similarity(text_a_embeddings, text_b_embeddings, dim=1)
    predictions = torch.where(sim_mat > thresh_high,
                            torch.tensor(1, device=device),
                            torch.where(sim_mat < thresh_low,
                                      torch.tensor(-1, device=device),
                                      torch.tensor(0, device=device)))
    
    # Calculate accuracy if labels available
    accuracy = None
    label_map = {'-': 1.0, 'entailment': 1.0, 'neutral': 0.0, 'contradiction': -1.0}
    numeric_labels = []
    for label in labels:
        if isinstance(label, str) and label in label_map:
            numeric_labels.append(label_map[label])
        elif isinstance(label, (int, float)):
            numeric_labels.append(float(label))
    
    if numeric_labels:
        target = torch.tensor(numeric_labels, device=device).float()
        accuracy = (predictions == target.long()).float().mean().item()
    
    results = {
        'model_type': 'bert',
        'predictions': [pred.item() for pred in predictions],
        'similarities': [sim.item() for sim in sim_mat],
        'accuracy': accuracy,
        'text_pairs': text_pairs,
        'labels': labels
    }
    
    return results

def print_results(results: Dict, model_name: str):
    """Print results in a formatted way."""
    print(f"\n--- {model_name} Results ---")
    
    label_names = {-1: "contradiction", 0: "neutral", 1: "entailment"}
    
    for i, (text_a, text_b) in enumerate(results['text_pairs']):
        print(f"\nPair {i+1}:")
        print(f"  Text A: {text_a}")
        print(f"  Text B: {text_b}")
        if i < len(results['labels']) and results['labels'][i]:
            print(f"  Ground truth: {results['labels'][i]}")
        
        pred_label = label_names.get(results['predictions'][i], results['predictions'][i])
        similarity = results['similarities'][i]
        print(f"  Prediction: {pred_label} (similarity: {similarity:.4f})")
    
    if results['accuracy'] is not None:
        print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(
        description="Demo pipeline for Tree Matching Networks and BERT models"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input text pair (tab-separated) or file path containing text pairs")
    parser.add_argument("--tree_checkpoint", type=str, default=None,
                        help="Path to tree model checkpoint")
    parser.add_argument("--bert_checkpoint", type=str, default=None,
                        help="Path to BERT model checkpoint")
    parser.add_argument("--spacy_model", type=str, default=None,
                        help="SpaCy model to use for tree parsing")
    parser.add_argument("--mode", type=str, choices=['tree', 'bert', 'both'], default='both',
                        help="Which model(s) to run")
    
    args = parser.parse_args()
    
    if args.mode == 'both' and (not args.tree_checkpoint or not args.bert_checkpoint):
        print("Error: Both tree and BERT checkpoints required for 'both' mode")
        sys.exit(1)
    elif args.mode == 'tree' and not args.tree_checkpoint:
        print("Error: Tree checkpoint required for 'tree' mode")
        sys.exit(1)
    elif args.mode == 'bert' and not args.bert_checkpoint:
        print("Error: BERT checkpoint required for 'bert' mode")
        sys.exit(1)
    
    # Parse input
    text_pairs, labels = parse_input(args.input)
    if not text_pairs:
        print("No valid text pairs found in the input.")
        sys.exit(1)
    
    print(f"Found {len(text_pairs)} text pair(s).")
    
    # Run demos based on mode
    if args.mode in ['tree', 'both']:
        try:
            tree_results = run_tree_demo(text_pairs, labels, args.tree_checkpoint, args.spacy_model)
            print_results(tree_results, "Tree Model")
        except Exception as e:
            print(f"Error running tree demo: {e}")
            if args.mode == 'tree':
                sys.exit(1)
    
    if args.mode in ['bert', 'both']:
        try:
            bert_results = run_bert_demo(text_pairs, labels, args.bert_checkpoint)
            print_results(bert_results, "BERT Model")
        except Exception as e:
            print(f"Error running BERT demo: {e}")
            if args.mode == 'bert':
                sys.exit(1)
    
    # Compare results if both models were run
    if args.mode == 'both' and 'tree_results' in locals() and 'bert_results' in locals():
        print("\n=== Model Comparison ===")
        
        agreements = 0
        total = len(text_pairs)
        
        for i in range(total):
            tree_pred = tree_results['predictions'][i]
            bert_pred = bert_results['predictions'][i]
            
            if tree_pred == bert_pred:
                agreements += 1
            else:
                print(f"Disagreement on pair {i+1}: Tree={tree_pred}, BERT={bert_pred}")
        
        agreement_rate = agreements / total if total > 0 else 0
        print(f"\nModel Agreement: {agreements}/{total} ({agreement_rate*100:.1f}%)")
        
        if tree_results['accuracy'] is not None and bert_results['accuracy'] is not None:
            print(f"Tree Model Accuracy: {tree_results['accuracy']*100:.2f}%")
            print(f"BERT Model Accuracy: {bert_results['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
