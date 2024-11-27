# Linguistic Tree Matching Networks

This directory contains an adaptation of Graph Matching Networks (GMN) for linguistic dependency trees, specifically focused on natural language inference tasks. It leverages the TMN_DataGen package for preprocessing dependency trees into a GMN-compatible format.

## Setup

1. Install requirements:
```bash
pip install wandb torch numpy sklearn
```

2. Configure WandB:
```bash
wandb login
```

3. Download and preprocess data:
```bash
# Download SNLI dataset (if not already done)
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip

# Generate tree data using TMN_DataGen
python -m TMN_DataGen.run process \
    -if data/snli_1.0/snli_1.0_dev.jsonl \
    -od data/processed_data/dev \
    -sm en_core_web_trf \
    -v normal
```

## Project Structure

```
LinguisticTrees/
├── configs/           # Model and training configurations
├── data/             # Data loading and processing
├── models/           # Model architectures 
├── training/         # Training utilities
└── experiments/      # Training and evaluation scripts
```

## Quick Start

1. Test training with dev dataset:
```bash
python -m LinguisticTrees.experiments.train_tree_matching \
    --config configs/experiment_configs/tree_matching.yaml \
    --data.train_path data/processed_data/dev/final_dataset.json \
    --wandb.tags dev,test-run
```

2. Full training:
```bash
python -m LinguisticTrees.experiments.train_tree_matching \
    --config configs/experiment_configs/tree_matching.yaml
```

## Key Components

- **TreeMatchingNet**: Extends GMN for linguistic tree structures
- **TreeEncoder**: Specialized encoder for linguistic features
- **TreeMatchingDataset**: Handles TMN_DataGen output format

## Implementation Notes

- Tree directionality is preserved through attention mechanisms
- Node features include word embeddings and linguistic features
- Edge features represent dependency relationships
- Labels: -1 (contradiction), 0 (neutral), 1 (entailment)

## Configuration

Key configuration options in `configs/experiment_configs/tree_matching.yaml`:

- `MODEL.node_feature_dim`: Dimension of node features (default: 768 for BERT embeddings)
- `MODEL.edge_feature_dim`: Dimension of edge features
- `MODEL.n_prop_layers`: Number of graph propagation layers
- `TRAIN.learning_rate`: Learning rate
- `TRAIN.batch_size`: Batch size

See configuration files for complete options.

## References

Based on the following papers:
- Li et al. "Graph Matching Networks for Learning the Similarity of Graph Structured Objects" (ICML 2019)
