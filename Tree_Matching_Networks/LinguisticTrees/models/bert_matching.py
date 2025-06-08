# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

# models/bert_matching.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class BertMatchingNet(nn.Module):
    """BERT matching network that works with adjacent pairs like GMN"""
    
    def __init__(self, config, tokenizer):
        super().__init__()
        
        # Extract config parameters
        bert_config = config['model']['bert']
        hidden_size = bert_config.get('hidden_size', 256)
        num_hidden_layers = bert_config.get('num_hidden_layers', 2)
        num_attention_heads = bert_config.get('num_attention_heads', 4)
        intermediate_size = bert_config.get('intermediate_size', 512)
        vocab_size = bert_config.get('vocab_size', 1200)
        actual_vocab_size = len(tokenizer.vocab)
        
        # Use actual tokenizer vocabulary size instead of config value
        vocab_size = actual_vocab_size
        
        # Create BERT config
        self.bert_config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=bert_config.get('max_position_embeddings', 128),
        )
        
        self.bert = AutoModel.from_config(self.bert_config)
        self.bert.gradient_checkpointing_enable()
        
        # Cross-attention layers
        self.n_cross_layers = num_hidden_layers
        if self.n_cross_layers > 0:
            # Determine which BERT layers to insert cross-attention after
            self.cross_attention_positions = self._get_cross_attention_positions(num_hidden_layers)
            # Create cross-attention layers (no learnable parameters)

            self.cross_attention_layers = nn.ModuleList([
                BertGMNStyleCrossAttention(similarity='dotproduct') 
                for _ in range(len(self.cross_attention_positions))
            ])
        
        # Final projection
        graph_rep_dim = config['model']['graph'].get('graph_rep_dim', 768)
        self.projection = nn.Linear(hidden_size, graph_rep_dim)
        # self.projection = nn.Identity()

    def _get_cross_attention_positions(self, num_layers):
        """Determine after which BERT layers to insert cross-attention"""
        if self.n_cross_layers == 0:
            return []
        elif self.n_cross_layers >= num_layers:
            return list(range(num_layers))
        else:
            # Distribute evenly through the layers
            step = num_layers / self.n_cross_layers
            return [int(i * step) for i in range(1, self.n_cross_layers + 1)]

    
    def forward(self, batch_encoding):
        """
        Forward pass that processes adjacent pairs
        
        Args:
            batch_encoding: Dictionary with concatenated sequences
            Expects even indices to be paired with odd indices
            
        Returns:
            Tensor of embeddings [n_pairs*2, hidden_dim]
        """
        batch_size = batch_encoding['input_ids'].shape[0]
        if batch_size % 2 != 0:
            raise ValueError(f"Batch size must be even for pair processing, got {batch_size}")
        
        if self.n_cross_layers == 0:
            # Pure BERT - no cross-attention
            outputs = self.bert(
                input_ids=batch_encoding['input_ids'],
                attention_mask=batch_encoding['attention_mask'],
                token_type_ids=batch_encoding.get('token_type_ids', None)
            )
            cls_embeddings = outputs.last_hidden_state[:, 0]
            return self.projection(cls_embeddings)
        
        # Manual forward pass with interleaved cross-attention
        # Get embeddings
        hidden_states = self.bert.embeddings(
            input_ids=batch_encoding['input_ids'],
            token_type_ids=batch_encoding.get('token_type_ids', None)
        )
        
        attention_mask = batch_encoding['attention_mask']
        # Convert attention mask to the format BERT expects
        extended_attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, batch_encoding['input_ids'].shape
        )
        
        cross_attn_idx = 0
        
        # Process through BERT layers with interleaved cross-attention
        for i, layer in enumerate(self.bert.encoder.layer):
            # Apply BERT layer
            layer_outputs = torch.utils.checkpoint.checkpoint(
                layer,
                hidden_states,
                extended_attention_mask
            )
            hidden_states = layer_outputs[0]
            
            # Apply cross-attention if this is a designated position
            if i in self.cross_attention_positions and cross_attn_idx < len(self.cross_attention_layers):
                hidden_states = self.cross_attention_layers[cross_attn_idx](
                    hidden_states, 
                    attention_mask  # Use original mask for cross-attention
                )
                cross_attn_idx += 1
        
        # Extract CLS tokens and project
        cls_embeddings = hidden_states[:, 0]
        return self.projection(cls_embeddings)

class BertGMNStyleCrossAttention(nn.Module):
    """GMN-style cross-attention with NO learnable parameters"""
    
    def __init__(self, similarity='dotproduct'):
        super().__init__()
        self.similarity = similarity
        
    def get_similarity_fn(self):
        if self.similarity == 'dotproduct':
            return lambda x, y: torch.mm(x, y.transpose(-2, -1))
        elif self.similarity == 'cosine':
            return lambda x, y: F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)
        # Add other similarity functions as needed
        
    def forward(self, hidden_states, attention_mask):
        """
        Unlearned cross-attention between adjacent pairs
        No learnable parameters - just similarity + softmax
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        sim_fn = self.get_similarity_fn()
        
        updated_states = []
        
        for i in range(0, batch_size, 2):
            seq_a = hidden_states[i]      # [seq_len, hidden_size]
            seq_b = hidden_states[i + 1]  # [seq_len, hidden_size]
            mask_a = attention_mask[i]    # [seq_len] 
            mask_b = attention_mask[i + 1] # [seq_len]
            
            # Unlearned cross-attention
            # A attends to B
            sim_ab = sim_fn(seq_a, seq_b)  # [seq_len_a, seq_len_b]
            
            # Apply padding masks
            if mask_b is not None:
                sim_ab = sim_ab.masked_fill(~mask_b.bool().unsqueeze(0), -float('inf'))
            
            attn_weights_ab = F.softmax(sim_ab, dim=-1)  # A -> B weights
            attended_a = torch.mm(attn_weights_ab, seq_b)  # Weighted sum of B for A
            
            # B attends to A  
            sim_ba = sim_fn(seq_b, seq_a)  # [seq_len_b, seq_len_a]
            
            if mask_a is not None:
                sim_ba = sim_ba.masked_fill(~mask_a.bool().unsqueeze(0), -float('inf'))
                
            attn_weights_ba = F.softmax(sim_ba, dim=-1)  # B -> A weights
            attended_b = torch.mm(attn_weights_ba, seq_a)  # Weighted sum of A for B
            
            updated_a = seq_a + attended_a
            updated_b = seq_b + attended_b
            
            updated_states.extend([updated_a, updated_b])
        
        return torch.stack(updated_states)


