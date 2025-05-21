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
            max_position_embeddings=512,  # Make this match your max_length
        )
        
        # Initialize model from scratch with custom config
        self.bert = AutoModel.from_config(self.bert_config) 
        
        # Either include a projection or decide to use embeddings directly
        graph_rep_dim = config['model']['graph'].get('graph_rep_dim', 768)
        if graph_rep_dim != hidden_size:
            self.projection = nn.Linear(hidden_size, graph_rep_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        """
        Forward pass through BERT
        
        Args:
            input_ids: Tensor of token IDs
            attention_mask: Tensor indicating which tokens should be attended to
            token_type_ids: Optional tensor indicating token types
            
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
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

