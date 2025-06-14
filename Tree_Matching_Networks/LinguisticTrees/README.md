[//]: # (Authored by: Jason Lunder, github: https://github.com/jlunder00)

# Linguistic Tree Matching Networks

This module contains an adaptation of Graph Matching Networks (GMN) for linguistic dependency trees, specifically focused on natural language inference and semantic similarity tasks. It leverages the TMN_DataGen package for preprocessing text into tree format.

## Architecture Overview

The implementation supports multiple model architectures:

### Tree Matching Networks (TMN)
- **TreeMatchingNet**: Extends GMN with cross-graph attention to compare dependency trees directly
- **TreeEmbeddingNet**: Creates independent embeddings for each tree, then compares

### BERT Baselines
- **BertMatchingNet**: BERT with cross-attention for sentence pair comparison
- **BertEmbeddingNet**: Independent BERT embeddings with similarity comparison

Both tree and BERT models process their respective representations and can be trained with various loss functions for different NLP tasks.

## Configuration System

Flexible configuration system, supporting different tasks and models.

### Base Configuration

The base configuration comes from `configs/default_tree_config.py` which extends GMN's original configuration with tree-specific parameters. You can override it in three ways:

1. Passing a YAML file path to `get_tree_config(base_config_path=...)`
2. Passing an override YAML path for specific parameters
3. Directly updating the config dict

### Task-Specific Configurations

I included pre-defined configurations in `configs/experiment_configs/`:

- [`aggregative_config.yaml`](configs/experiment_configs/aggregative_config.yaml): Used for all text-level training and evaluation
- [`contrastive_config.yaml`](configs/experiment_configs/contrastive_config.yaml): Used for pretraining with contrastive learning

### Important Configuration Parameters

#### Model Configuration
```yaml
model:
  task_type: "similarity" | "entailment" | "binary" | "infonce"
  task_loader_type: "aggregative"  # Use this for all current training
  name: "tree_matching" | "bert_matching"  # Model architecture to use
  model_type: "matching" | "embedding"
  
  # BERT-specific configuration (when name is "bert_matching" or "bert_embedding")
  bert:
    hidden_size: 1024              # Hidden dimension size
    num_hidden_layers: 4           # Number of transformer layers
    num_attention_heads: 16        # Number of attention heads
    intermediate_size: 2048        # Feed-forward network size
    max_position_embeddings: 384   # Maximum sequence length
    tokenizer_path: /path/to/tokenizer/  # Custom tokenizer path
    project: False                 # Whether to project to graph_rep_dim
  
  # Tree/Graph-specific configuration (when name is "tree_matching" or "tree_embedding")
  graph:
    node_feature_dim: 804          # Dimension of node features (BERT + one-hot encodings)
    edge_feature_dim: 70           # Dimension of edge features
    node_state_dim: 1024           # Internal node state dimension
    edge_state_dim: 256            # Internal edge state dimension
    edge_hidden_sizes:             # Defines internal hidden sizes / number of layers in edge propagation MLP
      - 512 
    node_hidden_sizes:             # Defines internal hidden sizes / number of layers in node propagation MLP
      - 1024 
    n_prop_layers: 5               # Number of graph propagations

    graph_rep_dim: 1792            # The final graph embedding dimension
    graph_transform_sizes:         # Defines internal hidden sizes / number of layers in graph aggregation network
      - 1024

    # other parameters in this area defined and detailed in GMN config docs
  
  # For entailment tasks
  thresh_low: -0.33      # Threshold for contradiction class
  thresh_high: 0.33      # Threshold for entailment class
  
  # For contrastive learning
  temperature: 0.05      # Temperature for scaling similarity scores

  # For custom contrastive goals:
  positive_infonce_weight: 1.0  # Value to multiply the loss for the infonce loss based on the similarity matrix in TextLevelContrastiveLoss
  inverse_infonce_weight: 1.0   # Value to multiply the loss for the infonce loss based on the distance matrix in TextLevelContrastiveLoss
  midpoint_infonce_weight: 1.0  # Value to multiply the loss for the infonce loss based on the midpoint matrix in TextLevelContrastiveLoss
```

#### Data Configuration
```yaml
data:
  dataset_type: "snli" | "semeval" | "patentmatch_balanced" | "wikiqs"
  batch_size: 256        # Number of trees per batch
  strict_matching: False # Whether to use strict matching
  contrastive_mode: True # Whether to use contrastive learning
  min_trees_per_group: 1 # Minimum number of trees per group

# BERT models require text mode enabled
text_mode: True          # Enable for BERT models, False for tree models
allow_text_files: False  # Whether to allow raw text input files - BERT can handle this, or can extract text data stored alongside TMN_DataGen prepared data (better for experimental consistency)
```

#### Training Configuration
```yaml
train:
  learning_rate: 0.000001
  n_epochs: 500
  patience: 10           # Early stopping patience
```

#### BERT Tokenizer Setup

For BERT models, you will need to train a custom tokenizer on your corpus (or use one off the shelf):

```bash
python -m Tree_Matching_Networks.LinguisticTrees.data.tokenizer_prep
```
This script isn't integrated into the config system and doesn't even have command line arguments yet, so you'll have to edit the `__main__` section with your data paths prior to running.    
It supports both JSON files (generated by TMN_DataGen) and plain text files.    

##### Key parameters:
- `configs`: List of dictionary configs, one for each JSON dataset you want to use in your corpus, in TreeDataConfig format
- `text_dirs`: Paths to directories containing text files you want to use in your corpus
- `vocab_size`: The size of the vocabulary for the resultant tokenizer
- `min_frequency`: The number of times a token must appear before it is eligable to be added to the tokenizer
- `tokenizer_save_path`: Where to save the trained tokenizer

After setting these correctly and running, you can change your main training configs to reflect your new tokenizer path.    

#### Wandb Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Configure WandB:
```bash
wandb login
```

## Models and Loss Functions

### Model Types

- **TreeMatchingNet**: Uses cross-graph attention to compare trees directly. Processes on pairs.
- **TreeEmbeddingNet**: Creates independent embeddings for each tree.
- **BertMatchingNet**: Bert augmented with cross attention between hidden states after each transformer layer. Processes on pairs.
- **BertEmbeddingNet**: Simple Bert but configurable with the same parameters (Todo: ensure training pipeline supports)

### Loss Types

The loss choice depends on your task type:

1. **Text-Level Losses** (These use aggregated embeddings across multiple sentences from the same text to compare)
   - `TextLevelSimilarityLoss`: For semantic similarity (regression)
   - `TextLevelEntailmentLoss`: For the entailment task
   - `TextLevelBinaryLoss`: For binary classification
   - `TextLevelContrastiveLoss`: For contrastive learning with custom infonce, controlled by weights in config

2. **Strict Sentence Pair Losses** (These compare single embeddings from individual sentences extracted out of texts)
   - `InfoNCELoss`: Contrastive loss - for pretraining
   - `SimilarityLoss`: Direct similarity scoring
   - `EntailmentLoss`: Classification approach

### Task Types

Each task uses a different loss function and evaluation metrics:

- **Similarity**: Uses cosine similarity to predict a continuous similarity score
- **Entailment**: Classifies text pairs as entailment (1), neutral (0), or contradiction (-1)
- **Binary**: Binary classification for tasks like patent matching
- **InfoNCE**: Contrastive learning task for pretraining and primary training

## Data Handling

The `PairedGroupsDataset` is the primary data handler for already pretrained models. It:
1. Loads preprocessed tree data from TMN_DataGen, or line by line text data (for BERT)
2. Handles tree aggregation for text-level processing
3. Supports both direct labeled learning and contrastive learning

The `DynamicCalculatedContrastiveDataset` is the data handler for pretraining models. It:
1. Loads preprocessed tree data from TMN_DataGen, or line by line text data (for BERT)
2. Handles pair organization with variable positive/negative pairing for contrastive pretraining
3. Supports only contrastive learning

## Training

To pretrain a model (recommended if starting from scratch), use:

```bash
python -m Tree_Matching_Networks.LinguisticTrees.experiments.train_unified \
  --config configs/experiment_configs/contrastive_config.yaml \
  --mode constrastive \
  --data_root /path/to/data/folder/ \
  --task_type infonce
```
Note: data folder should contain folders dev, train, and test, each containing folders for each dataset, like:
```
dev/snli_1.0_dev_converted_sm/
test/snl1_1.0_test_converted_sm/
train/snli_1.0_train_converted_sm/
```
So the format is like: `{dataset_name}_{split}_converted_{spacy_variant}`, see [tree_data_config.py](configs/tree_data_config.py) and [Data root dir](#data-root-dir) for more details. TODO: simplify this structure    

And for training a model in primary training/fine tuning:
```bash
python -m Tree_Matching_Networks.LinguisticTrees.experiments.train_unified \
  --config configs/experiment_configs/aggregative_config.yaml \
  --mode aggregative \
  --data_root /path/to/data/folder/ \
  --task_type <one of infonce, similarity, entailment, etc.>
```

Control of the model sizes and data used is done through the config files, more details on how to use those [here](#important-configuration-parameters)

### Training Arguments

- `--config`: Path to configuration file (default: aggregative_config.yaml)
- `--override`: Override specific config parameters
- `--resume`: Path to checkpoint to resume from
- `--resume_with_epoch`: Resume from the same epoch number (for crashed runs)
- `--ignore_patience`: Ignore early stopping patience
- `--data_root`: root location of data directory

## Evaluation

To evaluate a trained model:

```bash
python -m Tree_Matching_Networks.LinguisticTrees.experiments.eval_aggregated \
  --checkpoint /path/to/checkpoint \
  --batch_size 256 \
  --output_dir evaluation_results
```

### Evaluation Arguments

- `--checkpoint`: Path to model checkpoint
- `--batch_size`: Batch size for evaluation (optional)
- `--output_dir`: Directory to save results
- `--use_wandb`: Log results to Weights & Biases
- `--config`: Optional config override
- `--data_root`: root location of data directory

Note: when you give a checkpoint file at `/path/to/checkpoint_dir/checkpoint/model.pt`, the script searches for    
the config at `/path/to/checkpoint_dir/config/config.yaml`. This file must be present for the script to function.    
The config file given is the config the model was trained with. This is usually the correct config to use, but you can    
pass the `--config` argument to override it. Overriding in this way is usually necessary when training a model in multiple stages.

### Data root dir
Important note: the data root directory should contain 3 directories, train, test, and dev.    
Each of these should contain directories that are the specific processed dataset split.    
The directory names of these must follow the patterns laid out in [config/tree_data_config.py](configs/tree_data_config.py).    
The general format is `{base_dataset}_{split}_converted_{spacy_variant}` with `_sharded` added if you   
set "sharded = true" when setting up the tree data config for that split. The base dataset is determined by    
what is entered into the dataset_spec config variable. For each item you include in the dataset_spec,    
data files will be looked for at that path and included as data to be used, meaning multiple datasets can be used at once.   
If you do not define dataset_specs, the base_dataet is determined by the dataset_type config, eg 'snli', and it expects one directory.

## Results

Both models were trained and evaluated on the SNLI entailment task. The results show that Tree Matching Networks significantly outperform BERT models despite similar parameter counts and nearly identical training methods, achieving approximately 60% accuracy on the test set. It is worth noting that this was the same result as with a much, much smaller version of the TMN model, suggesting that there may be some architectural limitation causing a plateau in accuracy that needs to be addressed.

### Performance Comparison

| Model | Parameters | SNLI Test Accuracy | Notes |
|-------|------------|-------------------|-------|
| Tree Matching Network | ~36M | 60.2% | Good at contradiction/entailment separation |
| Tree Matching Network | ~60K | ~60.0% | Original smaller model |
| BERT Matching Network | ~41M | 35.4% | Heavily biased toward entailment prediction |
| EFL + RoBERTa-large from Wang et al., '21 | 355M | ~90% | SOTA baseline |

### Confusion Matrices

**Tree Matching Network Results:**

![TMN Entailment Large Confusion Matrix](tmn_entailment_lg_confusion_matrix.png)

The Large TMN model shows balanced performance across classes with strong contradiction/entailment separation and a lesser ability to separate out neutral items.

![TMN Entailment Small Confusion Matrix](tmn_entailment_sm_confusion_matrix.png)

The Small TMN model shows very similar performance, suggesting a limitation in the architecture.

**BERT Matching Model Results:**

![BERT Matching Entailment Confusion Matrix](bert_matching_entailment_confusion_matrix.png)

The BERT Matching model exhibits severe bias, predicting entailment for nearly all examples, indicating it was unable to learn under the same conditions as the TMN model.

### Key Findings

**TMN Model Strengths:**
- Excellent contradiction vs. entailment separation
- Effective use of structural linguistic information through dependency trees
- Superior performance compared to BERT despite similar parameter count
- Cross-graph attention captures inter-sentence relationships effectively

**Current Limitations:**
- Neutral class confusion
- Larger model versions didn't improve over smaller ones (scaling plateau)
- May need architectural improvements for better parameter utilization

**BERT Matching Model Issues:**
- Severe bias toward predicting entailment
- Poor overall performance despite similar parameter count to TMN  
- Cross-attention modification insufficient to compete with tree structure, or perhaps was detrimental compared to non matching BERT (TBD)

### Scaling Analysis

Despite a 600x parameter increase (60K -> 36M), TMN performance remained virtually unchanged, suggesting fundamental architectural limitations that may require different scaling approaches or architectural improvements such as replacing graph aggregation layers with transformer attention.

## Further Work

Potential further investigation/improvements include:
- **Embedding Networks Comparison**: Compare TreeEmbeddingNet vs standard BERT at identical parameter sizes to test whether cross-attention is fundamental to learning relationships or primarily improves training speed.
- **Transfer Learning Efficacy**: Test whether insights learned by matching networks transfer to embedding networks by loading trained matching network weights into embedding networks.
- **Scaling Architecture Investigation**: Replace graph aggregation layers with multi-headed attention (transformer) operating on post-propagation node states to potentially resolve the scaling plateau issue.
- More extensive pretraining on larger datasets
- Applying to additional tasks like SemEval

## References

This implementation is based on the following papers:
- Li et al. "Graph Matching Networks for Learning the Similarity of Graph Structured Objects" (ICML 2019)
- Wang et al. "Entailment as Few-Shot Learner" (2021)
- Bowman et al. "A large annotated corpus for learning natural language inference" (EMNLP 2015)


