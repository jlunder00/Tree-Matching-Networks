# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

# models/bert_embedding.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

class BertEmbeddingNet(nn.Module):
    """BERT-based embedding network compatible with tree embedding framework"""
    
    def __init__(self, config, tokenizer):
        super().__init__()
        
        # Extract config parameters
        hidden_size = config['model']['bert'].get('hidden_size', 256)
        num_hidden_layers = config['model']['bert'].get('num_hidden_layers', 2)
        num_attention_heads = config['model']['bert'].get('num_attention_heads', 4)
        intermediate_size = config['model']['bert'].get('intermediate_size', 512)
        max_position_embeddings = config['model']['bert'].get('max_position_embeddings', 512)
        

        actual_vocab_size = len(tokenizer.vocab)
        
        # Use actual tokenizer vocabulary size instead of config value
        vocab_size = actual_vocab_size
        
        
        # Create a custom config for a small BERT
        self.bert_config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,  # Make this match your max_length
        )
        
        # Initialize model from scratch with custom config
        self.bert = AutoModel.from_config(self.bert_config) 
        
        # Either include a projection or decide to use embeddings directly
        graph_rep_dim = config['model']['graph'].get('graph_rep_dim', 768)
        if graph_rep_dim != hidden_size and config['model']['bert']['project']:
            self.projection = nn.Linear(hidden_size, graph_rep_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, batch_encoding=None, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        """
        Forward pass through BERT - handles both batch_encoding dict and individual kwargs
        
        Args:
            batch_encoding: Dictionary with 'input_ids', 'attention_mask', 'token_type_ids' keys (training pipeline)
            input_ids: Tensor of token IDs (fallback/override)
            attention_mask: Tensor indicating which tokens should be attended to (fallback/override)  
            token_type_ids: Optional tensor indicating token types (fallback/override)
            
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        # For each kwarg, prefer explicit value, fallback to batch_encoding
        if input_ids is None and batch_encoding is not None:
            input_ids = batch_encoding.get('input_ids')
        if attention_mask is None and batch_encoding is not None:
            attention_mask = batch_encoding.get('attention_mask')
        if token_type_ids is None and batch_encoding is not None:
            token_type_ids = batch_encoding.get('token_type_ids')
            
        # Validate required inputs
        if input_ids is None:
            raise ValueError("input_ids must be provided either directly or via batch_encoding")
        if attention_mask is None:
            raise ValueError("attention_mask must be provided either directly or via batch_encoding")
            
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token embedding as sentence representation
        sentence_embeds = outputs.last_hidden_state[:, 0]
        
        # Return embeddings directly or projected
        return self.projection(sentence_embeds)

