[//]: # (Authored by: Jason Lunder)
# Tree Matching Networks for NLI

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/jlunder00/Tree-Matching-Networks/blob/main/LICENSE)

This repository contains an adaptation of Graph Matching Networks (GMN) for linguistic dependency trees, focused on natural language inference (NLI) and semantic similarity tasks. The idea is to represent sentences as dependency trees to capture structural information, then apply graph neural network techniques to learn relationships between sentence pairs.

## Overview

This project extends Graph Matching Networks to operate on linguistic trees and includes two main components:

1. **TMN_DataGen**: A package for generating and processing dependency trees from raw text
2. **Tree-Matching-Networks**: The model implementation for training and inference

## Project Structure

```
.
├── GMN/                   # Original Graph Matching Networks code
├── LinguisticTrees/       # Our tree adaptations and training code
│   ├── configs/           # Configuration files
│   ├── data/              # Data loading and processing
│   ├── models/            # Model architecture
│   ├── training/          # Training and evaluation code
│   └── experiments/       # Training and evaluation scripts
├── scripts/               # Demo and utility scripts
└── data/                  # Data directory (see TMN_DataGen)
```

## Installation

1. First, install TMN_DataGen:
   ```bash
   git clone https://github.com/jlunder00/TMN_DataGen.git
   cd TMN_DataGen
   pip install .
   ```

2. Then, install this repository:
   ```bash
   git clone https://github.com/jlunder00/Tree-Matching-Networks.git
   cd Tree-Matching-Networks
   pip install .
   ```

## Required External Resources

Before using the models, you'll need:

1. **SpaCy Model**: For dependency parsing.
   ```bash
   python -m spacy download en_core_web_trf  # or en_core_web_lg/md/sm
   ```

2. **Word2Vec Vocabulary**: For word boundary correction.
   - Download from [Google News Vectors](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors)
   - Set path in TMN_DataGen configuration

3. **Embedding Cache**: Create directory for caching word embeddings:
   ```bash
   mkdir -p /path/to/embedding_cache
   ```
   - Set path in configuration files

## Quick Start

### Running the Demo

Try out the model with the demo script:

```bash
python -m Tree_Matching_Networks.scripts.demo \
  --checkpoint /path/to/best_entailment_model_checkpoint/checkpoints/best_model.pt \
  --input input.tsv
```

See [Demo Instructions](scripts/README.md) for more details.

### Data Processing

To process your own data, use TMN_DataGen:

```bash
python -m TMN_DataGen.run process \
  --input_path your_data.jsonl \
  --out_dir processed_data/your_dataset \
  --dataset_type snli \
  --spacy_model en_core_web_trf
```

See [TMN_DataGen README](https://github.com/jlunder00/TMN_DataGen/blob/main/README.md) for more details.

### Training

Train a model on processed data:

```bash
python -m Tree_Matching_Networks.LinguisticTrees.experiments.train_aggregative \
  --config Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/aggregative_config.yaml
```

See [LinguisticTrees README](Tree_Matching_Networks/LinguisticTrees/README.md) for more configuration options.

### Evaluation

Evaluate a trained model:

```bash
python -m Tree_Matching_Networks.LinguisticTrees.experiments.eval_aggregated \
  --checkpoint /path/to/checkpoint \
  --output_dir evaluation_results
```

## Key Features

- **Tree-Based Representation**: Leverages dependency trees to capture sentence structure
- **Cross-Graph Attention**: Compares sentences using graph matching techniques
- **Flexible Model Configuration**: Supports different tasks and training approaches
- **Contrastive Learning**: Pretrain on large datasets for better transfer
- **Multiple NLP Tasks**: Supports entailment, similarity, and binary classification

## Model Architecture

Our approach adapts Graph Matching Networks to work with linguistic trees:

1. **Text Processing**: Convert sentences to dependency trees using SpaCy/DiaParser
2. **Feature Extraction**: Embed words and dependency relations
3. **Graph Propagation**: Use message passing to capture tree structure
4. **Graph Matching**: Apply cross-graph attention to compare tree pairs
5. **Aggregation**: Pool sentence trees into text-level representations

## License

This project is MIT licensed, as found in the LICENSE file.

## Acknowledgments

This project builds upon:

> Yujia Li, Chenjie Gu, Thomas Dullien, Oriol Vinyals, Pushmeet Kohli. *Graph Matching Networks for Learning the Similarity of Graph Structured Objects*. ICML 2019. [[paper\]](https://arxiv.org/abs/1904.12787)

> Yijie Lin, Mouxing Yang, Jun Yu, Peng Hu, Changqing Zhang, Xi Peng. *Graph Matching with Bi-level Noisy Correspondence*. ICCV, 2023. [[paper]](https://arxiv.org/pdf/2212.04085.pdf)
