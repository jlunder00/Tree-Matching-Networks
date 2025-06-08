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
├── LinguisticTrees/       # My tree adaptations and training code
│   ├── configs/           # Configuration files
│   ├── data/              # Data loading and processing
│   ├── models/            # Model architecture
│   ├── training/          # Training and evaluation code
│   └── experiments/       # Training and evaluation scripts
└── scripts/               # Demo and utility scripts
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
   python -m spacy download en_core_web_sm # or en_core_web_lg/md/trf
   ```

2. **Word2Vec Vocabulary**: For word boundary correction.
   - Download from [Google News Vectors](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors)
   - Set path in TMN_DataGen configuration

3. **Embedding Cache**: Create directory for caching word embeddings:
   ```bash
   mkdir -p /path/to/embedding_cache
   ```
   - Set path in configuration files

4. **Custom BERT Tokenizer** (for BERT models): Train a tokenizer on your text corpus:
   - See [LinguisticTrees README](https://github.com/jlunder00/Tree-Matching-Networks/tree/main/Tree_Matching_Networks/LinguisticTrees#bert-tokenizer-setup) for instructions
   - Set `tokenizer_path` in BERT model configuration

## Quick Start

### Running the Demo

Try out the model with the demo script:

```bash
python -m Tree_Matching_Networks.scripts.demo \
  --mode both \
  --tree_checkpoint /path/to/tmn_entailment_lg_checkpoint/model.pt \
  --config_tmn /path/to/Tree-Matching-Networks/scripts/demo_configs/tmn_config.yaml  \
  --bert_checkpoint /path/to/bert_matching_entailment_checkpoint/model.pt \
  --config_bert /path/to/Tree-Matching-Networks/scripts/demo_configs/bert_config.yaml  \
  --input input.tsv \
  --spacy_model en_core_web_sm
```
Note that occasionally the provided config that comes with a checkpoint may not work in the demo script.    
Providing a config override to an appropriately configured custom config or one such config from Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/ can resolve this issue.

See [Demo Instructions](https://github.com/jlunder00/Tree-Matching-Networks/tree/main/scripts#demo-script-for-tree-matching-networks) for more details.

### Data Processing

To process your own data, use TMN_DataGen:

```bash
python -m TMN_DataGen.run process \
  --input_path your_data.jsonl \
  --out_dir processed_data/your_dataset \
  --dataset_type snli \
  --spacy_model en_core_web_sm
```

See [TMN_DataGen README](https://github.com/jlunder00/TMN_DataGen/tree/main?tab=readme-ov-file#tmn_datagen) for more details.

### Training

Train a model on processed data:

```bash
python -m Tree_Matching_Networks.LinguisticTrees.experiments.train_aggregative \
  --config Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/aggregative_config.yaml
```

See [LinguisticTrees README](https://github.com/jlunder00/Tree-Matching-Networks/tree/main/Tree_Matching_Networks/LinguisticTrees#configuration-system) for more configuration options.

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
- **BERT Baseline Model**: Enhanced BERT model with cross-attention for comparison
- **Flexible Model Configuration**: Supports different tasks and training approaches
- **Contrastive Learning**: Pretrain on large datasets for better transfer
- **Multiple NLP Tasks**: Supports entailment, similarity, and binary classification

## Model Architecture

My approach adapts Graph Matching Networks to work with linguistic trees:

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
