# models/bert_embedding.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BertEmbeddingNet(nn.Module):
    """BERT-based embedding network compatible with tree embedding framework"""
    
    def __init__(self, config):
        super().__init__()
        
        # Extract config parameters
        hidden_size = config['model'].get('hidden_size', 256)
        num_hidden_layers = config['model'].get('num_hidden_layers', 2)
        num_attention_heads = config['model'].get('num_attention_heads', 4)
        intermediate_size = config['model'].get('intermediate_size', 512)
        vocab_size = config['model'].get('vocab_size', 1200)
        
        # Output dimension matching your TreeEmbeddingNet
        self.graph_rep_dim = config['model'].get('graph_rep_dim', 1792)
        
        # Create a custom config for a small BERT
        self.bert_config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=128,
        )
        
        # Initialize model from scratch with custom config
        self.bert = AutoModel.from_config(self.bert_config)
        
        # Projection layer to match graph_rep_dim for compatibility
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, self.graph_rep_dim),
            nn.LayerNorm(self.graph_rep_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, batch_encoding):
        """
        Forward pass through BERT
        
        Args:
            batch_encoding: Dictionary with keys 'input_ids', 'attention_mask', etc.
            
        Returns:
            Tensor of shape [batch_size, graph_rep_dim]
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=batch_encoding['input_ids'],
            attention_mask=batch_encoding['attention_mask'],
            token_type_ids=batch_encoding.get('token_type_ids', None)
        )
        
        # Get the [CLS] token embedding as sentence representation
        sentence_embeds = outputs.last_hidden_state[:, 0]
        
        # Project to match graph representation dimensions
        graph_embeds = self.projection(sentence_embeds)
        
        return graph_embeds
