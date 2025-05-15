# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

#Legacy: not updated to handle paired group datasets. will be updated in future for comparison with small versions of bert style models

import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import argparse
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BertBaseline:
    def __init__(self, model_name='bert-base-uncased', device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
    def get_embeddings(self, sentences, batch_size=32):
        """Get BERT embeddings for sentences"""
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                all_embeddings.append(embeddings)
                
        return np.vstack(all_embeddings)
        
    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        return np.sum(emb1 * emb2, axis=1) / (
            np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
        )

def evaluate_dataset(args):
    """Evaluate BERT baseline on dataset"""
    # Load data
    with open(args.data_path) as f:
        data = json.load(f)
        
    # Extract sentence pairs and labels
    sent1, sent2, labels = [], [], []
    tree_data = data['graph_pairs']
    label_data = data['labels']
    for i in range(len(tree_data)):
        tree_item = tree_data[i]
        sent1.append(tree_item[0]['text'])
        sent2.append(tree_item[1]['text'])
        label_item = label_data[i]
        labels.append(label_item)

    
        # labels.append(item['gold_label'])
        
    # Convert labels
    if args.task == 'similarity':
        labels = np.array(labels, dtype=np.float32)
    else:  # entailment
        label_map = {'contradiction': -1, 'neutral': 0, 'entailment': 1}
        labels = np.array([label_map[l] for l in labels])
    
    # Initialize model
    baseline = BertBaseline(args.model_name, args.device)
    
    # Get embeddings
    logger.info("Computing embeddings for first sentences...")
    emb1 = baseline.get_embeddings(sent1)
    logger.info("Computing embeddings for second sentences...")
    emb2 = baseline.get_embeddings(sent2)
    
    # Compute similarities
    similarities = baseline.compute_similarity(emb1, emb2)
    
    # Compute metrics
    results = {}
    if args.task == 'similarity':
        pearson, _ = pearsonr(similarities, labels)
        spearman, _ = spearmanr(similarities, labels)
        mse = np.mean((similarities - labels) ** 2)
        
        results = {
            'pearson': pearson,
            'spearman': spearman,
            'mse': mse
        }
    else:
        # Convert similarities to predictions using thresholds
        predictions = np.zeros_like(similarities)
        predictions[similarities < -0.3] = -1  # contradiction
        predictions[similarities > 0.3] = 1    # entailment
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Save results
    output_path = Path(args.output_dir) / f'bert_baseline_{args.task}_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Print results
    logger.info("\nResults:")
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to dataset')
    parser.add_argument('--task', type=str, required=True,
                      choices=['similarity', 'entailment'],
                      help='Task type')
    parser.add_argument('--model_name', type=str,
                      default='bert-base-uncased',
                      help='BERT model to use')
    parser.add_argument('--output_dir', type=str,
                      default='experiments/bert/',
                      help='output directory for test')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='cuda to use gpu, cpu to use cpu')

    args = parser.parse_args()
    evaluate_dataset(args)


