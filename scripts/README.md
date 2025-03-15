# Demo Script for Tree Matching Networks

This demo script demonstrates the full pipeline for Tree Matching Networks, showing how to:
1. Preprocess text
2. Parse it into dependency trees
3. Run inference with a trained model
4. Evaluate the results

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

2. Followed the instructions at [TMN_DataGen](https://github.com/jlunder00/TMN_DataGen/tree/CSCD584/Submission?tab=readme-ov-file#required-dependencies) to set up the vocabularies and embedding cache

## Running the Demo

The demo script takes a tsv file containing pairs with optional labels and runs it through the full pipeline.

```bash
python -m Tree_Matching_Networks.scripts.demo \
  --checkpoint /path/to/checkpoint/best_model.pt \
  --input input.tsv \
  --config /path/to/config/file.yaml
```

### Command-line Arguments

- `--checkpoint`: Path to a trained model checkpoint (required)
- `--input`: Either a tab-separated text pair or a file path containing pairs (required)
- `--config`: Optional configuration override file

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
  --checkpoint /path/to/best_entailment_model_checkpoint/checkpoints/best_model.pt \
  --input input.tsv
```

## Understanding the Results

The demo outputs:
- The parsed tree structure for each text
- Similarity scores between the text pairs
- Predicted entailment labels (if applicable)
- Overall accuracy (if ground truth labels are provided)

## Troubleshooting

If you encounter errors related to embedding cache or preprocessing:

1. Check your TMN_DataGen configuration files to ensure paths are set correctly
2. Make sure the embedding cache directory exists and is writable
3. Ensure the Word2Vec vocabulary file is downloaded and properly configured

Refer to the [main LinguisticTrees README](../Tree_Matching_Networks/LinguisticTrees/README.md) for more detailed configuration instructions.
