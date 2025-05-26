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
        
        # Cross-attention layers
        self.n_cross_layers = config['model']['bert'].get('n_cross_layers', 2)
        self.cross_attention_layers = nn.ModuleList([
            BertCrossAttentionLayer(hidden_size) for _ in range(self.n_cross_layers)
        ])
        
        # Final projection
        graph_rep_dim = config['model']['graph'].get('graph_rep_dim', 768)
        self.projection = nn.Linear(hidden_size, graph_rep_dim)
    
    def forward(self, batch_encoding):
        """
        Forward pass that processes adjacent pairs
        
        Args:
            batch_encoding: Dictionary with concatenated sequences
            Expects even indices to be paired with odd indices
            
        Returns:
            Tensor of embeddings [n_pairs*2, hidden_dim]
        """
        # Get BERT embeddings for all sequences
        outputs = self.bert(
            input_ids=batch_encoding['input_ids'],
            attention_mask=batch_encoding['attention_mask'],
            token_type_ids=batch_encoding.get('token_type_ids', None)
        )
        
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        batch_size = hidden_states.shape[0]
        
        # Ensure even batch size for pairing
        if batch_size % 2 != 0:
            raise ValueError(f"Batch size must be even for pair processing, got {batch_size}")
        
        # Split into pairs: (0,1), (2,3), (4,5), ...
        n_pairs = batch_size // 2
        
        # Apply cross-attention between adjacent pairs
        for cross_layer in self.cross_attention_layers:
            hidden_states = cross_layer(
                hidden_states, 
                batch_encoding['attention_mask'],
                n_pairs
            )
        
        # Extract CLS tokens and project
        cls_embeddings = hidden_states[:, 0]  # [batch_size, hidden_size]
        final_embeddings = self.projection(cls_embeddings)
        
        return final_embeddings

class BertCrossAttentionLayer(nn.Module):
    """Cross-attention layer that works on adjacent pairs"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, hidden_states, attention_mask, n_pairs):
        """
        Apply cross-attention between adjacent pairs
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            n_pairs: Number of pairs (batch_size // 2)
        """
        seq_len = hidden_states.shape[1]
        results = []
        
        for i in range(0, n_pairs * 2, 2):
            # Get pair
            seq_a = hidden_states[i]      # [seq_len, hidden_size]
            seq_b = hidden_states[i + 1]  # [seq_len, hidden_size]
            mask_a = attention_mask[i]    # [seq_len]
            mask_b = attention_mask[i + 1] # [seq_len]
            
            # Cross-attention: A attends to B, B attends to A
            attn_a, _ = self.attention(seq_a.unsqueeze(0), seq_b.unsqueeze(0), seq_b.unsqueeze(0),
                                     key_padding_mask=~mask_b.bool())
            attn_b, _ = self.attention(seq_b.unsqueeze(0), seq_a.unsqueeze(0), seq_a.unsqueeze(0),
                                     key_padding_mask=~mask_a.bool())
            
            # Residual + norm
            seq_a = self.norm1(seq_a + attn_a.squeeze(0))
            seq_b = self.norm1(seq_b + attn_b.squeeze(0))
            
            # FFN
            seq_a = self.norm2(seq_a + self.ffn(seq_a))
            seq_b = self.norm2(seq_b + self.ffn(seq_b))
            
            results.append(seq_a)
            results.append(seq_b)
        
        return torch.stack(results)  # [batch_size, seq_len, hidden_size]

# class CrossAttentionLayer(nn.Module):
#     """Cross-attention layer similar to graph matching attention"""
#     
#     def __init__(self, hidden_size, num_attention_heads=8, dropout=0.1):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_attention_heads = num_attention_heads
#         self.head_dim = hidden_size // num_attention_heads
#         
#         self.query_a = nn.Linear(hidden_size, hidden_size)
#         self.key_b = nn.Linear(hidden_size, hidden_size)
#         self.value_b = nn.Linear(hidden_size, hidden_size)
#         
#         self.query_b = nn.Linear(hidden_size, hidden_size)
#         self.key_a = nn.Linear(hidden_size, hidden_size)
#         self.value_a = nn.Linear(hidden_size, hidden_size)
#         
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm_a = nn.LayerNorm(hidden_size)
#         self.layer_norm_b = nn.LayerNorm(hidden_size)
#         
#     def forward(self, hidden_a, hidden_b, attention_mask_a=None, attention_mask_b=None):
#         batch_size, seq_len_a, hidden_size = hidden_a.shape
#         seq_len_b = hidden_b.shape[1]
#         
#         # A attends to B
#         q_a = self.query_a(hidden_a).view(batch_size, seq_len_a, self.num_attention_heads, self.head_dim).transpose(1, 2)
#         k_b = self.key_b(hidden_b).view(batch_size, seq_len_b, self.num_attention_heads, self.head_dim).transpose(1, 2)
#         v_b = self.value_b(hidden_b).view(batch_size, seq_len_b, self.num_attention_heads, self.head_dim).transpose(1, 2)
#         
#         # Compute attention scores A->B
#         scores_ab = torch.matmul(q_a, k_b.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         if attention_mask_b is not None:
#             scores_ab = scores_ab + attention_mask_b.unsqueeze(1).unsqueeze(1) * -10000.0
#         
#         attn_weights_ab = F.softmax(scores_ab, dim=-1)
#         attn_weights_ab = self.dropout(attn_weights_ab)
#         
#         attended_a = torch.matmul(attn_weights_ab, v_b)
#         attended_a = attended_a.transpose(1, 2).contiguous().view(batch_size, seq_len_a, hidden_size)
#         
#         # B attends to A  
#         q_b = self.query_b(hidden_b).view(batch_size, seq_len_b, self.num_attention_heads, self.head_dim).transpose(1, 2)
#         k_a = self.key_a(hidden_a).view(batch_size, seq_len_a, self.num_attention_heads, self.head_dim).transpose(1, 2)
#         v_a = self.value_a(hidden_a).view(batch_size, seq_len_a, self.num_attention_heads, self.head_dim).transpose(1, 2)
#         
#         scores_ba = torch.matmul(q_b, k_a.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         if attention_mask_a is not None:
#             scores_ba = scores_ba + attention_mask_a.unsqueeze(1).unsqueeze(1) * -10000.0
#             
#         attn_weights_ba = F.softmax(scores_ba, dim=-1)
#         attn_weights_ba = self.dropout(attn_weights_ba)
#         
#         attended_b = torch.matmul(attn_weights_ba, v_a)
#         attended_b = attended_b.transpose(1, 2).contiguous().view(batch_size, seq_len_b, hidden_size)
#         
#         # Residual connections and layer norm
#         output_a = self.layer_norm_a(hidden_a + attended_a)
#         output_b = self.layer_norm_b(hidden_b + attended_b)
#         
#         return output_a, output_b


# class BertMatchingNet(nn.Module):
#     """BERT-based matching network with cross-attention capabilities"""
#     
#     def __init__(self, config, tokenizer):
#         super().__init__()
#         
#         # Extract config parameters
#         bert_config = config['model']['bert']
#         hidden_size = bert_config.get('hidden_size', 256)
#         num_hidden_layers = bert_config.get('num_hidden_layers', 2)
#         num_attention_heads = bert_config.get('num_attention_heads', 4)
#         intermediate_size = bert_config.get('intermediate_size', 512)
#         vocab_size = bert_config.get('vocab_size', 1200)
#         actual_vocab_size = len(tokenizer.vocab)
#         
#         # Use actual tokenizer vocabulary size instead of config value
#         vocab_size = actual_vocab_size
#         
#         # Create BERT config
#         self.bert_config = AutoConfig.from_pretrained(
#             "bert-base-uncased",
#             vocab_size=vocab_size,
#             hidden_size=hidden_size,
#             num_hidden_layers=num_hidden_layers,
#             num_attention_heads=num_attention_heads,
#             intermediate_size=intermediate_size,
#             max_position_embeddings=bert_config.get('max_position_embeddings', 128),
#         )
#         
#         # Two separate BERT encoders for sequence A and B
#         self.encoder_a = AutoModel.from_config(self.bert_config)
#         self.encoder_b = AutoModel.from_config(self.bert_config)
#         
#         # Cross-attention layers (similar to graph matching)
#         self.n_cross_layers = config['model']['graph'].get('n_prop_layers', 1)
#         self.cross_attention_layer = CrossAttentionLayer(hidden_size, num_attention_heads)
#         
#         # Final projection
#         graph_rep_dim = config['model']['graph'].get('graph_rep_dim', 768)
#         if graph_rep_dim != hidden_size and bert_config['project']:
#             self.projection = nn.Linear(hidden_size, graph_rep_dim)
#         else:
#             self.projection = nn.Identity()
#     
#     def forward(self, batch_encoding_a, batch_encoding_b):
#         """
#         Forward pass through BERT matching network
#         
#         Args:
#             batch_encoding_a: Dictionary with keys for sequence A
#             batch_encoding_b: Dictionary with keys for sequence B
#             
#         Returns:
#             Tuple of embeddings (embed_a, embed_b)
#         """
#         # Encode sequences separately first
#         outputs_a = self.encoder_a(
#             input_ids=batch_encoding_a['input_ids'],
#             attention_mask=batch_encoding_a['attention_mask'],
#             token_type_ids=batch_encoding_a.get('token_type_ids', None)
#         )
#         
#         outputs_b = self.encoder_b(
#             input_ids=batch_encoding_b['input_ids'],
#             attention_mask=batch_encoding_b['attention_mask'],
#             token_type_ids=batch_encoding_b.get('token_type_ids', None)
#         )
#         
#         hidden_a = outputs_a.last_hidden_state
#         hidden_b = outputs_b.last_hidden_state
#         
#         # Apply cross-attention layers
#         for _ in range(self.n_cross_layers):
#             hidden_a, hidden_b = self.cross_attention_layer(
#                 hidden_a, hidden_b,
#                 attention_mask_a=batch_encoding_a['attention_mask'],
#                 attention_mask_b=batch_encoding_b['attention_mask']
#             )
#         
#         # Extract final embeddings (CLS tokens)
#         embed_a = hidden_a[:, 0]  # CLS token for sequence A
#         embed_b = hidden_b[:, 0]  # CLS token for sequence B
#         
#         # Apply projection
#         embed_a = self.projection(embed_a)
#         embed_b = self.projection(embed_b)
#         
#         return embed_a, embed_b
