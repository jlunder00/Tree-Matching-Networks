

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

# --- Imports from TMN_DataGen and Tree Matching Networks ---
# (Make sure your PYTHONPATH is set so these modules can be found.)
from TMN_DataGen.dataset_generator import DatasetGenerator, TreeGroup, generate_group_id
from TMN_DataGen.utils.text_preprocessing import BasePreprocessor, SentenceSplitter
from TMN_DataGen.parsers.multi_parser import MultiParser

from Tree_Matching_Networks.LinguisticTrees.training.experiment import ExperimentManager
from Tree_Matching_Networks.LinguisticTrees.configs.default_tree_config import get_tree_config
from Tree_Matching_Networks.LinguisticTrees.models.tree_matching import TreeMatchingNet
from Tree_Matching_Networks.LinguisticTrees.models.tree_embedding import TreeEmbeddingNet
from Tree_Matching_Networks.LinguisticTrees.data.data_utils import convert_tree_to_graph_data
from Tree_Matching_Networks.LinguisticTrees.data.paired_groups_dataset import PairedGroupBatchInfo

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
                label = float(row[2].strip()) if len(row) >= 3 and row[2].strip() != '' else 1.0
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
    infonce_data = generator._convert_to_infonce_format(tree_groups, is_paired=True)
    print("Converted tree groups to InfoNCE format.")
    # Now infonce_data is a dictionary with keys "version", "format", and "groups"
    # where each group is in the infonce (grouped) format.
    
    # For demonstration, we will now prepare a batch by simply combining all groups.
    # (In a full pipeline, the downstream dataloader would use the infonce format.)
    batch_groups = infonce_data["groups"]
    
    # 8. Create a dummy PairedGroupBatchInfo object.
    # Here we extract for each group the original text and the trees converted to graph data.
    group_ids = []
    group_labels = []
    trees_a_indices = []
    trees_b_indices = []
    batch_trees = []
    anchor_indices = []  # designate the first tree from text A as anchor for each group

    for group in batch_groups:
        group_ids.append(group["group_id"])
        group_labels.append(group["label"])
        indices_a = []
        for tree_data in group["trees"]:
            # Assume each tree_data is already in graph format
            indices_a.append(len(batch_trees))
            batch_trees.append(tree_data)
            anchor_indices.append(len(batch_trees)-1)
        # In infonce format, trees_b should be present
        indices_b = []
        for tree_data in group["trees_b"]:
            indices_b.append(len(batch_trees))
            batch_trees.append(tree_data)
        trees_a_indices.append(indices_a)
        trees_b_indices.append(indices_b)
    print("Batch prepared from infonce groups.")

    # 9. Convert the list of tree graphs into a GraphData object.
    graph_data = convert_tree_to_graph_data(batch_trees)
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
    graph_data.node_features = graph_data.node_features.to(device)
    graph_data.edge_features = graph_data.edge_features.to(device)
    graph_data.from_idx = graph_data.from_idx.to(device)
    graph_data.to_idx = graph_data.to_idx.to(device)
    graph_data.graph_idx = graph_data.graph_idx.to(device)

    # 13. Run the batch through the model.
    with torch.no_grad():
        logits = model(
            graph_data.node_features,
            graph_data.edge_features,
            graph_data.from_idx,
            graph_data.to_idx,
            graph_data.graph_idx,
            graph_data.n_graphs
        )
    predictions = torch.argmax(logits, dim=1) - 1
    predictions = predictions.cpu().tolist()
    print("Model inference complete.")

    # 14. (Optional) Aggregate tree embeddings into text embeddings.
    with torch.no_grad():
        tree_embeddings, _ = model._encoder(
            graph_data.node_features,
            graph_data.edge_features
        )
    aggregator = TreeAggregator(model_config.get('aggregation', 'mean'))
    text_embeddings = aggregator(tree_embeddings, batch_info)
    text_embeddings = text_embeddings.cpu()
    group_text_embeddings = []
    for i in range(len(batch_info.group_indices)):
        emb_a = text_embeddings[2*i]
        emb_b = text_embeddings[2*i+1]
        group_text_embeddings.append((emb_a, emb_b))

    # 15. Compute cosine similarity between subgroup embeddings to predict entailment.
    predicted_group_labels = []
    similarities = []
    for emb_a, emb_b in group_text_embeddings:
        cos_sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        similarities.append(cos_sim)
        if cos_sim > 0.0:
            pred = 1
        elif cos_sim < 0.0:
            pred = -1
        else:
            pred = 0
        predicted_group_labels.append(pred)

    # 16. If ground-truth labels are provided, calculate accuracy.
    correct = sum(1 for p, t in zip(predicted_group_labels, group_labels) if int(p) == int(t))
    accuracy = correct / len(group_labels) if group_labels else None

    print("\n--- Group-level Predictions ---")
    for gid, pred, sim in zip(group_ids, predicted_group_labels, similarities):
        print(f"Group {gid}: Predicted {pred} (cosine similarity: {sim:.4f})")
    if accuracy is not None:
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()



# #!/usr/bin/env python
# """
# Demo script showing end-to-end usage of the Tree Matching Networks model for entailment.
# It accepts either a single comma-separated text pair (optionally with a label) or a file
# with each line containing text_a, text_b, and optionally a label.
# The script then preprocesses the texts, tokenizes and parses them into trees (using TMN_DataGen),
# converts them into grouped format, prepares a batch as in the non-strict matching dataset,
# converts the batch into graph data, runs the model, aggregates tree embeddings into text embeddings,
# and finally uses cosine similarity (with a threshold) to predict an entailment label.
# If ground-truth labels are provided, it also calculates accuracy.
# """

# import argparse
# import os
# import sys
# import json
# import random
# import torch

# # --- Imports from TMN_DataGen and Tree Matching Networks ---
# # (Make sure your PYTHONPATH is set so that these modules are found.)
# from TMN_DataGen.dataset_generator import DatasetGenerator, generate_group_id, convert_tree_to_graph_data
# from TMN_DataGen.utils.text_preprocessing import BasePreprocessor, SentenceSplitter
# from TMN_DataGen.parsers.multi_parser import MultiParser

# # We use the default tree config and model loading functions from the Tree Matching Networks package.
# from Tree_Matching_Networks.LinguisticTrees.configs.default_tree_config import get_tree_config
# from Tree_Matching_Networks.LinguisticTrees.training.experiment import load_model_from_checkpoint
# from Tree_Matching_Networks.LinguisticTrees.models.tree_aggregator import TreeAggregator
# from Tree_Matching_Networks.LinguisticTrees.data.batch_utils import PairedGroupBatchInfo

# # --- Helper functions ---

# def parse_input(input_arg):
#     """
#     Parse the input provided by the user.
#     If input_arg is a file path, read each nonempty line.
#     Otherwise, assume it is a single comma-separated text pair.
#     Returns a list of (text_a, text_b) tuples and a list of labels.
#     """
#     pairs = []
#     labels = []
#     if os.path.isfile(input_arg):
#         with open(input_arg, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 parts = [p.strip() for p in line.split(',')]
#                 if len(parts) < 2:
#                     continue
#                 text_a = parts[0]
#                 text_b = parts[1]
#                 # If a third column is provided, use it as label; otherwise default to 1 (entailment)
#                 label = float(parts[2]) if len(parts) >= 3 else 1.0
#                 pairs.append((text_a, text_b))
#                 labels.append(label)
#     else:
#         # Single input assumed to be comma-separated
#         parts = [p.strip() for p in input_arg.split(',')]
#         if len(parts) < 2:
#             raise ValueError("Input must have at least two comma-separated texts.")
#         text_a = parts[0]
#         text_b = parts[1]
#         label = float(parts[2]) if len(parts) >= 3 else 1.0
#         pairs.append((text_a, text_b))
#         labels.append(label)
#     return pairs, labels

# # --- Main pipeline ---

# def main():
#     parser = argparse.ArgumentParser(
#         description="Demo end-to-end pipeline for Tree Matching Networks (entailment task)"
#     )
#     parser.add_argument("--checkpoint", type=str, required=True,
#                         help="Path to model checkpoint")
#     parser.add_argument("--input", type=str, required=True,
#                         help="Input text pair (comma-separated) or file path containing text pairs")
#     parser.add_argument("--config", type=str, default=None,
#                         help="Optional config override file")
#     args = parser.parse_args()

#     # 1. Parse input text pairs
#     text_pairs, labels = parse_input(args.input)
#     if not text_pairs:
#         print("No valid text pairs found in the input.")
#         sys.exit(1)
#     print(f"Found {len(text_pairs)} text pair(s).")

#     # 2. Load configuration (allow override if provided)
#     base_config = None
#     if args.config is not None:
#         import yaml
#         with open(args.config, 'r') as f:
#             base_config = yaml.safe_load(f)
#     # Use the default get_tree_config function; task type is 'entailment'
#     config = get_tree_config(task_type='entailment', base_config=base_config, override_config=None)
#     print("Configuration loaded.")

#     # 3. Preprocess texts and create grouped format
#     preprocessor = BasePreprocessor(config)
#     splitter = SentenceSplitter()

#     # For each text pair, create a group (with two subgroups for text A and text B)
#     grouped_metadata = []
#     grouped_sentences = []  # Will be a list of tuples: (sentences_from_A, sentences_from_B)
#     for i, (text_a, text_b) in enumerate(text_pairs):
#         text_a_clean = preprocessor.preprocess(text_a)
#         text_b_clean = preprocessor.preprocess(text_b)
#         sentences_a = splitter.split(text_a_clean)
#         sentences_b = splitter.split(text_b_clean)
#         group_id = generate_group_id()
#         metadata = {
#             "group_id": group_id,
#             "text": text_a,
#             "text_clean": text_a_clean,
#             "text_b": text_b,
#             "text_b_clean": text_b_clean,
#             "label": labels[i]
#         }
#         grouped_metadata.append(metadata)
#         grouped_sentences.append((sentences_a, sentences_b))
#     print("Preprocessing and sentence splitting complete.")

#     # 4. Parse sentences into trees using MultiParser.
#     # Flatten the list so that each subgroup becomes an element.
#     sentence_groups = []
#     for sentences_a, sentences_b in grouped_sentences:
#         sentence_groups.append(sentences_a)
#         sentence_groups.append(sentences_b)

#     # Instantiate MultiParser (vocabs can be empty here; in practice they are loaded from a word vector model)
#     vocabs = [set()]
#     multi_parser = MultiParser(config, pkg_config=None, vocabs=vocabs, logger=None)
#     all_tree_groups = multi_parser.parse_all(sentence_groups, show_progress=False, num_workers=1)
#     print("Parsing of sentences into trees complete.")

#     # 5. Reassemble parsed trees into group dictionaries
#     tree_groups = []
#     for i, meta in enumerate(grouped_metadata):
#         # Assume that for each group, the first subgroup (index 2*i) corresponds to text A and the next (2*i+1) to text B.
#         group_dict = {
#             "group_id": meta["group_id"],
#             "text": meta["text"],
#             "trees": all_tree_groups[2*i],
#             "text_b": meta["text_b"],
#             "trees_b": all_tree_groups[2*i+1],
#             "label": meta["label"]
#         }
#         tree_groups.append(group_dict)
#     print("Grouped tree format created.")

#     # 6. For the demo we will treat everything as one batch.
#     # For non-strict matching, we simply list the trees from each subgroup sequentially.
#     batch_trees = []
#     trees_a_indices = []
#     trees_b_indices = []
#     group_ids = []
#     group_labels = []
#     anchor_indices = []  # We designate each tree from text A as an anchor.

#     for group in tree_groups:
#         group_ids.append(group["group_id"])
#         group_labels.append(group["label"])
#         indices_a = []
#         for tree in group["trees"]:
#             indices_a.append(len(batch_trees))
#             batch_trees.append(tree)
#             anchor_indices.append(len(batch_trees) - 1)
#         indices_b = []
#         for tree in group["trees_b"]:
#             indices_b.append(len(batch_trees))
#             batch_trees.append(tree)
#         trees_a_indices.append(indices_a)
#         trees_b_indices.append(indices_b)
#     print("Batch prepared from grouped trees.")

#     # 7. Convert the list of trees into a GraphData object.
#     graph_data = convert_tree_to_graph_data(batch_trees)
#     # (Assumes that convert_tree_to_graph_data returns an object with attributes like node_features, edge_features, etc.)

#     # 8. Create a dummy PairedGroupBatchInfo object for the batch.
#     batch_info = PairedGroupBatchInfo(
#         group_indices=list(range(len(tree_groups))),
#         group_ids=group_ids,
#         group_labels=group_labels,
#         trees_a_indices=trees_a_indices,
#         trees_b_indices=trees_b_indices,
#         tree_to_group_map={},  # Not used in this demo
#         tree_to_set_map={},    # Not used in this demo
#         pair_indices=[],       # For non-strict we leave this empty
#         anchor_indices=anchor_indices,
#         strict_matching=False,
#         contrastive_mode=False
#     )

#     # 9. Load the entailment model from the checkpoint.
#     # The load_model_from_checkpoint function returns (model, config, epoch)
#     model, model_config, checkpoint_epoch = load_model_from_checkpoint(
#         args.checkpoint, base_config, override_config=None
#     )
#     model.eval()
#     device = torch.device(model_config['device'])
#     model = model.to(device)
#     print(f"Model loaded from checkpoint (epoch {checkpoint_epoch}).")

#     # 10. Move graph data to the correct device.
#     # (Assuming GraphData has attributes that are tensors.)
#     graph_data.node_features = graph_data.node_features.to(device)
#     graph_data.edge_features = graph_data.edge_features.to(device)
#     graph_data.from_idx = graph_data.from_idx.to(device)
#     graph_data.to_idx = graph_data.to_idx.to(device)
#     graph_data.graph_idx = graph_data.graph_idx.to(device)

#     # 11. Run the batch through the model.
#     # For entailment the model is typically TreeMatchingNetEntailment.
#     with torch.no_grad():
#         logits = model(
#             graph_data.node_features,
#             graph_data.edge_features,
#             graph_data.from_idx,
#             graph_data.to_idx,
#             graph_data.graph_idx,
#             graph_data.n_graphs
#         )
#     # For entailment, the model’s forward returns logits of shape [n_graphs, 3].
#     # Since our batch was built by listing all trees (first all text A trees then all text B trees for each group),
#     # we assume that the model was designed to process pairs and then the predictions are computed by splitting
#     # the outputs into two halves per group. (In the original eval code, predictions = argmax(logits, dim=1) - 1)
#     predictions = torch.argmax(logits, dim=1) - 1
#     predictions = predictions.cpu().tolist()
#     print("Model inference complete.")

#     # 12. (Optional) Aggregate tree embeddings into text embeddings using the tree aggregator.
#     # In this demo we simulate a text-level prediction by obtaining tree embeddings from the model’s encoder.
#     with torch.no_grad():
#         # Here we assume that the model has an attribute _encoder which returns tree embeddings.
#         tree_embeddings, _ = model._encoder(
#             graph_data.node_features,
#             graph_data.edge_features
#         )
#     # Instantiate the aggregator (using the strategy defined in config; e.g., 'attention')
#     aggregator = TreeAggregator(model_config.get('aggregation', 'mean'))
#     # Aggregate the tree embeddings into one embedding per subgroup.
#     text_embeddings = aggregator(tree_embeddings, batch_info)
#     # Now, for each group, we have two embeddings (one for text A and one for text B).
#     text_embeddings = text_embeddings.cpu()
#     group_text_embeddings = []
#     for i in range(len(batch_info.group_indices)):
#         emb_a = text_embeddings[2*i]
#         emb_b = text_embeddings[2*i+1]
#         group_text_embeddings.append((emb_a, emb_b))
#     
#     # 13. For the entailment task, we compute cosine similarity between the two subgroup embeddings.
#     # Then, we use a simple threshold (e.g., >0: entailment, <0: contradiction, ~0: neutral).
#     predicted_group_labels = []
#     similarities = []
#     for emb_a, emb_b in group_text_embeddings:
#         cos_sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
#         similarities.append(cos_sim)
#         if cos_sim > 0.0:
#             pred = 1
#         elif cos_sim < 0.0:
#             pred = -1
#         else:
#             pred = 0
#         predicted_group_labels.append(pred)

#     # 14. If ground-truth labels are available (from our input), calculate accuracy.
#     correct = 0
#     for pred, true in zip(predicted_group_labels, group_labels):
#         if int(pred) == int(true):
#             correct += 1
#     accuracy = correct / len(group_labels) if group_labels else None

#     print("\n--- Group-level Predictions ---")
#     for gid, pred, sim in zip(group_ids, predicted_group_labels, similarities):
#         print(f"Group {gid}: Predicted {pred} (cosine similarity: {sim:.4f})")
#     if accuracy is not None:
#         print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

# if __name__ == "__main__":
#     main()

