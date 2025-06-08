[//]: # (Authored by: Jason Lunder, github: https://github.com/jlunder00)

# Demo Script for Tree Matching Networks

This demo script demonstrates the full pipeline for Tree Matching Networks, showing how to:
1. Preprocess text
2. Parse it into dependency trees
3. Run inference with a trained model
4. Compare results between TMN and BERT models
5. Evaluate the results

## Prerequisites

Before running the demo, ensure you have:
1. Installed both packages:
   ```bash
   # Install TMN_DataGen
   cd /path/to/TMN_DataGen
   pip install .
   
   # Install Tree-Matching-Networks
   cd /path/to/Tree-Matching-Networks
   pip install .
   ```

2. Followed the instructions at [TMN_DataGen](https://github.com/jlunder00/TMN_DataGen/tree/main?tab=readme-ov-file#required-dependencies) to set up the vocabularies and embedding cache

3. For BERT models: Trained a custom tokenizer on your corpus using the instructions in [LinguisticTrees README](../Tree_Matching_Networks/LinguisticTrees/README.md#training-custom-bert-tokenizer)

## Running the Demo

The demo script takes a tsv file containing pairs with optional labels and runs it through the full pipeline.

### Single Model Demo

**Tree Matching Network**:
```bash
python -m Tree_Matching_Networks.scripts.demo \
  --mode tree \
  --tree_checkpoint /path/to/checkpoint/best_model.pt \
  --input input.tsv \
  --config_tree /path/to/config/file.yaml \
  --spacy_model en_core_web_sm
```

**BERT Model**:
```bash
python -m Tree_Matching_Networks.scripts.demo \
  --mode bert \
  --bert_checkpoint /path/to/bert_checkpoint/best_model.pt \
  --input input.tsv \
  --config_bert /path/to/bert_config/file.yaml
```

### Comparing Both Models

```bash
python -m Tree_Matching_Networks.scripts.demo \
  --mode both \
  --tree_checkpoint /path/to/tmn/checkpoints/best_model.pt \
  --bert_checkpoint /path/to/bert/checkpoints/best_model.pt \
  --input input.tsv \
  --spacy_model en_core_web_sm \
  --config_tmn /path/to/tmn_config.yaml \
  --config_bert /path/to/bert_config.yaml
```

### Command-line Arguments

- `--tree_checkpoint`: Path to Tree Matching Network checkpoint 
- `--bert_checkpoint`: Path to BERT model checkpoint 
- `--input`: Either a tab-separated text pair or a file path containing pairs (required)
- `--config_tmn`: Configuration file override for TMN model 
- `--config_bert`: Configuration file override for BERT model 
- `--mode`: "tree" or "bert" for single model modes, or "both" for model comparison
- `--spacy_model`: Optional spacy model override to use a different model than en_core_web_sm when generating tree node features.

Note that occasionally the provided config that comes with a checkpoint may not work in the demo script.    
Providing a config override to an appropriately configured custom config or one such config from Tree_Matching_Networks/LinguisticTrees/configs/experiment_configs/ can resolve this issue.


### Input Format

You can provide input via file with one pair per line, tab-separated
```
Text A\tText B\toptional_label
Text C\tText D\toptional_label
...
```
Note: putting a literal \\t will not work, you have to put in the literal tab character. In some text editors, there is a special mode for inserting literal characters like tab.

### Example

Using the sample input file provided:

```bash
python demo.py \
  --tree_checkpoint /path/to/best_entailment_model_checkpoint/checkpoints/best_model.pt \
  --mode 
  --input input.tsv
```

## Understanding the Results

The demo outputs:
- The parsed tree structure for each text (TMN models)
- Tokenization information (BERT models)
- Similarity scores between the text pairs
- Predicted entailment labels (if applicable)
- Overall accuracy (if ground truth labels are provided)
- Model comparison results (when using both models)

## Troubleshooting

If you encounter errors related to embedding cache or preprocessing:

1. Check your TMN_DataGen configuration files to ensure paths are set correctly
2. Make sure the embedding cache directory exists and is writable
3. Ensure the Word2Vec vocabulary file is downloaded and properly configured
4. For BERT models: Verify your custom tokenizer exists and the path is correct in your configuration

Refer to the [main LinguisticTrees README](../Tree_Matching_Networks/LinguisticTrees/README.md) for more detailed configuration instructions.
