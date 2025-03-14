

#!/usr/bin/env python
"""
Demo script showing end-to-end pipeline for Tree Matching Networks (entailment task)
using the InfoNCE conversion. This script accepts a checkpoint path and an input (either a 
comma-separated text pair or a file with one pair per line), preprocesses the texts,
parses them into trees, converts these trees into TreeGroup objects, then converts these
groups into the InfoNCE format. Finally, the script loads the model from a checkpoint,
runs inference, aggregates the tree embeddings into text embeddings, computes cosine similarity
between text pairs, and if ground-truth labels are provided, calculates accuracy.
"""

import argparse
import os
import sys
import json
import torch
from typing import Dict

# --- Imports from TMN_DataGen and Tree Matching Networks ---
# (Make sure your PYTHONPATH is set so these modules can be found.)
from TMN_DataGen.dataset_generator import DatasetGenerator, TreeGroup, generate_group_id
from TMN_DataGen.utils.text_preprocessing import BasePreprocessor, SentenceSplitter
from TMN_DataGen.parsers.multi_parser import MultiParser
from TMN_DataGen import FeatureExtractor

from Tree_Matching_Networks.LinguisticTrees.training.experiment import ExperimentManager
from Tree_Matching_Networks.LinguisticTrees.configs.default_tree_config import get_tree_config
from Tree_Matching_Networks.LinguisticTrees.models.tree_matching import TreeMatchingNet
from Tree_Matching_Networks.LinguisticTrees.models.tree_embedding import TreeEmbeddingNet
from Tree_Matching_Networks.LinguisticTrees.models import TreeAggregator
from Tree_Matching_Networks.LinguisticTrees.data.data_utils import convert_tree_to_graph_data
from Tree_Matching_Networks.LinguisticTrees.data.paired_groups_dataset import PairedGroupBatchInfo
from Tree_Matching_Networks.LinguisticTrees.data.data_utils import GraphData

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

# --- Helper functions ---

import os
import csv

def parse_input(input_arg):
    """
    Parse the input provided by the user.
    If input_arg is a file path, read each nonempty line using tab as the delimiter.
    Otherwise, assume it is a single tab-separated text pair.
    Returns a list of (text_a, text_b) tuples and a list of labels.
    """
    pairs = []
    labels = []
    
    if os.path.isfile(input_arg):
        with open(input_arg, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # Skip empty rows
                if not row or len(row) < 2:
                    continue
                text_a = row[0].strip()
                text_b = row[1].strip()
                label = row[2].strip() if len(row) >= 3 and row[2].strip() != '' else ''
                pairs.append((text_a, text_b))
                labels.append(label)
    else:
        # Assume the input is a single tab-separated string.
        row = input_arg.split("\t")
        if len(row) < 2:
            raise ValueError("Input must have at least two tab-separated texts.")
        text_a = row[0].strip()
        text_b = row[1].strip()
        label = float(row[2].strip()) if len(row) >= 3 and row[2].strip() != '' else 1.0
        pairs.append((text_a, text_b))
        labels.append(label)
    return pairs, labels

# def parse_input(input_arg):
#     """
#     Parse the input provided by the user.
#     If input_arg is a file path, read each nonempty line.
#     Otherwise, assume it is a single tab-separated text pair.
#     Returns a list of (text_a, text_b) tuples and a list of labels.
#     """
#     pairs = []
#     labels = []
#     sep = "\t"
#     if os.path.isfile(input_arg):
#         with open(input_arg, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 parts = [p.strip() for p in line.split(sep)]
#                 if len(parts) < 2:
#                     continue
#                 text_a = parts[0]
#                 text_b = parts[1]
#                 label = float(parts[2]) if len(parts) >= 3 else 1.0
#                 pairs.append((text_a, text_b))
#                 labels.append(label)
#     else:
#         # Single input assumed to be tab-separated
#         parts = [p.strip() for p in input_arg.split(sep)]
#         if len(parts) < 2:
#             raise ValueError("Input must have at least two tab-separated texts.")
#         text_a = parts[0]
#         text_b = parts[1]
#         label = float(parts[2]) if len(parts) >= 3 else 1.0
#         pairs.append((text_a, text_b))
#         labels.append(label)
#     return pairs, labels

def get_feature_config(config) -> Dict:
    """Create feature extractor config."""
    return {
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
    
def load_embeddings(tree: Dict, embedding_extractor: FeatureExtractor) -> torch.Tensor:
    """Load or generate embeddings for a tree if required."""
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


# --- Main pipeline ---

def main():
    parser = argparse.ArgumentParser(
        description="Demo pipeline for entailment using InfoNCE conversion of tree groups"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input text pair (comma-separated) or file path containing text pairs")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config override file")
    args = parser.parse_args()

    # 1. Parse input text pairs
    text_pairs, labels = parse_input(args.input)
    if not text_pairs:
        print("No valid text pairs found in the input.")
        sys.exit(1)
    print(f"Found {len(text_pairs)} text pair(s).")

    # 2. Load configuration (allow override if provided)
    base_config = None
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            base_config = yaml.safe_load(f)
    # Use the default get_tree_config function; task type is 'entailment'
    config = get_tree_config(task_type='entailment', base_config=base_config, override_config=None)
    import yaml
    from pathlib import Path
    from omegaconf import OmegaConf

    # Load TMN_DataGen default configurations:
    datagen_config_dir = Path("/home/jlunder/research/TMN_DataGen/TMN_DataGen/configs")
    with open(datagen_config_dir / "default_package_config.yaml", 'r') as f:
        pkg_config = yaml.safe_load(f)
    with open(datagen_config_dir / "default_parser_config.yaml", 'r') as f:
        parser_config = yaml.safe_load(f)
    with open(datagen_config_dir / "default_preprocessing_config.yaml", 'r') as f:
        preprocessing_config = yaml.safe_load(f)
    with open(datagen_config_dir / "default_feature_config.yaml", 'r') as f:
        feature_config = yaml.safe_load(f)
    with open(datagen_config_dir / "default_output_format.yaml", 'r') as f:
        output_config = yaml.safe_load(f)
    with open(datagen_config_dir / "default_merge_config.yaml", 'r') as f:
        merge_config = yaml.safe_load(f)

    # Merge in order (as done in TMN_DataGen’s run.py)
    datagen_config = {}
    datagen_config.update(parser_config)
    datagen_config.update(preprocessing_config)
    datagen_config.update(feature_config)
    datagen_config.update(output_config)
    datagen_config.update(merge_config)
    d_config = OmegaConf.create(datagen_config)

    print("Configuration loaded.")

    # 3. Preprocess texts and create grouped format.
    preprocessor = BasePreprocessor(d_config)
    splitter = SentenceSplitter()

    grouped_metadata = []
    grouped_sentences = []  # List of tuples: (sentences_from_A, sentences_from_B)
    for i, (text_a, text_b) in enumerate(text_pairs):
        text_a_clean = preprocessor.preprocess(text_a)
        text_b_clean = preprocessor.preprocess(text_b)
        sentences_a = splitter.split(text_a_clean)
        sentences_b = splitter.split(text_b_clean)
        group_id = generate_group_id()
        metadata = {
            "group_id": group_id,
            "text": text_a,
            "text_clean": text_a_clean,
            "text_b": text_b,
            "text_b_clean": text_b_clean,
            "label": labels[i]
        }
        grouped_metadata.append(metadata)
        grouped_sentences.append((sentences_a, sentences_b))
    print("Preprocessing and sentence splitting complete.")

    # 4. Prepare a flat list of sentence groups.
    # For each text pair, we add both the sentence list from text A and text B.
    sentence_groups = []
    for sentences_a, sentences_b in grouped_sentences:
        sentence_groups.append(sentences_a)
        sentence_groups.append(sentences_b)

    # 5. Parse sentences into trees using MultiParser.
    vocabs = [set()]  # In practice, vocabs are loaded from a word vector model.
    multi_parser = MultiParser(d_config, pkg_config=pkg_config, vocabs=vocabs, logger=None)
    all_tree_groups = multi_parser.parse_all(sentence_groups, show_progress=False, num_workers=1)
    print("Parsing of sentences into trees complete.")

    # 6. Create TreeGroup objects.
    tree_groups = []
    for i, meta in enumerate(grouped_metadata):
        tg = TreeGroup(
            group_id=meta["group_id"],
            original_text=meta["text"],
            trees=all_tree_groups[2*i],         # trees from text A
            original_text_b=meta["text_b"],
            trees_b=all_tree_groups[2*i+1],       # trees from text B
            label=meta["label"]
        )
        tree_groups.append(tg)
    print("Created tree group objects.")

    # 7. Convert tree groups to InfoNCE format.
    # We use the _convert_to_infonce_format function from DatasetGenerator.
    generator = DatasetGenerator(num_workers=1)
    _, _ = generator._load_configs(parser_config, preprocessing_config, feature_config, output_config, merge_config, 'normal', override_pkg_config=pkg_config)
    generator.config = d_config
    infonce_data = generator._convert_to_infonce_format(tree_groups, is_paired=True)
    print("Converted tree groups to InfoNCE format.")
    # Now infonce_data is a dictionary with keys "version", "format", and "groups"
    # where each group is in the infonce (grouped) format.

    feature_config = get_feature_config(config)
    embedding_extractor = FeatureExtractor(feature_config)
    requires_embeddings = True
    
    # For demonstration, we will now prepare a batch by simply combining all groups.
    # (In a full pipeline, the downstream dataloader would use the infonce format.)
    batch_groups = infonce_data["groups"]

    
    label_map = {'-': 1.0, 'entailment':1.0, 'neutral':0.0, 'contradiction':-1.0, '0': 0.0, '0.0':0.0, 0:0.0, '1':1.0, '1.0':1.0, 1:1.0}
    buffer_groups = []
    for group_idx, group in enumerate(batch_groups):
        group_id = group.get('group_id', None)
        label_raw = group.get('label', -2)
        if isinstance(label_raw, str):
            if label_raw not in label_map.keys():
                print(f"invalid label for group {i}, skipping")
                continue
        label_raw = label_map[label_raw]
        label = float(label_raw)
        trees_a = []
        for tree_idx, tree in enumerate(group.get('trees', [])):
            tree = dict(tree)
            tree['node_features'] = load_embeddings(tree, embedding_extractor)
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
                tree['node_features'] = load_embeddings(tree, embedding_extractor)
                trees_b.append({
                    'tree': tree,
                    'group_id': group_id,
                    'group_idx': group_idx,
                    'tree_idx': tree_idx,
                    'text': tree.get('text', ''),
                })
        buffer_groups.append({
            'group_id': group_id,
            'group_idx': group_idx,
            'label': label,
            'trees_a': trees_a,
            'trees_b': trees_b
        })
    
    # 8. Create a dummy PairedGroupBatchInfo object.
    # Here we extract for each group the original text and the trees converted to graph data.
    group_ids = []
    group_labels = []
    trees_a_indices = []
    trees_b_indices = []
    batch_trees = []
    anchor_indices = []  # designate the first tree from text A as anchor for each group

    for group in buffer_groups:
        group_ids.append(group["group_id"])
        group_labels.append(group["label"])
        indices_a = []
        for tree in group["trees_a"]:
            tree_idx = len(batch_trees)
            # Assume each tree_data is already in graph format
            indices_a.append(tree_idx)
            batch_trees.append(tree)
            anchor_indices.append(tree_idx)
        # In infonce format, trees_b should be present
        indices_b = []
        for tree in group["trees_b"]:
            indices_b.append(len(batch_trees))
            batch_trees.append(tree)
        trees_a_indices.append(indices_a)
        trees_b_indices.append(indices_b)
    print("Batch prepared from infonce groups.")

    # 9. Convert the list of tree graphs into a GraphData object.
    graph_data = convert_tree_to_graph_data([item['tree'] for item in batch_trees])
    # (Assuming convert_tree_to_graph_data returns an object with attributes like node_features, etc.)

    # 10. Create a dummy PairedGroupBatchInfo for the batch.
    batch_info = PairedGroupBatchInfo(
        group_indices=list(range(len(batch_groups))),
        group_ids=group_ids,
        group_labels=group_labels,
        trees_a_indices=trees_a_indices,
        trees_b_indices=trees_b_indices,
        tree_to_group_map={},  # not used in this demo
        tree_to_set_map={},    # not used in this demo
        pair_indices=[],       # left empty for non-strict mode
        anchor_indices=anchor_indices,
        strict_matching=False,
        contrastive_mode=False
    )

    # 11. Load the entailment model from the checkpoint.
    model, model_config, checkpoint_epoch = load_model_from_checkpoint(
        args.checkpoint, base_config, override_config=None
    )
    model.eval()
    device = torch.device(model_config['device'])
    model = model.to(device)
    print(f"Model loaded from checkpoint (epoch {checkpoint_epoch}).")

    # 12. Move graph data to the appropriate device.
    graphs = GraphData(
        node_features = graph_data.node_features.to(device),
        edge_features = graph_data.edge_features.to(device),
        from_idx = graph_data.from_idx.to(device),
        to_idx = graph_data.to_idx.to(device),
        graph_idx = graph_data.graph_idx.to(device),
        n_graphs = graph_data.n_graphs
    )

    # 13. Run the batch through the model.
    with torch.no_grad():
        embeddings = model(
            graphs.node_features,
            graphs.edge_features,
            graphs.from_idx,
            graphs.to_idx,
            graphs.graph_idx,
            graphs.n_graphs
        )
    # predictions = torch.argmax(embeddings, dim=1) - 1
    # predictions = predictions.cpu().tolist()
    print("Model inference complete.")

    # with torch.no_grad():
    #     tree_embeddings, _ = model._encoder(
    #         graphs.node_features,
    #         graphs.edge_features
    #     )
    aggregator = TreeAggregator(model_config.get('aggregation', 'mean'))
    text_embeddings = aggregator(embeddings, batch_info)
    # text_embeddings = text_embeddings.cpu()
    # group_text_embeddings = []
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

    if batch_info.group_labels: 
        target = torch.tensor(batch_info.group_labels, device=device).float()
        with torch.no_grad():
            accuracy = (predictions == target.long()).float().mean().item()

    print("\n--- Predictions ---")
    for i, group_info in enumerate(batch_groups):
        print(f"Text a: {group_info['text']}")
        print(f"Text b: {group_info['text_b']}")
        if group_info['label']:
            print(f"Ground truth label: {group_info['label']}")
        print(f"Predicted label: {str(predictions[i].item())}")
    # for  in zip(group_ids, predicted_group_labels, similarities):
    #     print(f"Group {gid}: Predicted {pred} (cosine similarity: {sim:.4f})")
    if accuracy is not None:
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()



