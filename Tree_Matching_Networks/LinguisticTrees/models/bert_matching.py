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
        # self.cross_attention_layer = BertCrossAttentionLayer(hidden_size)
        self.n_cross_layers = num_hidden_layers
        if self.n_cross_layers > 0:
            # Determine which BERT layers to insert cross-attention after
            self.cross_attention_positions = self._get_cross_attention_positions(num_hidden_layers)
            # Create cross-attention layers (no learnable parameters - pure GMN style)

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

    # def forward(self, batch_encoding):
    #     batch_size = batch_encoding['input_ids'].shape[0]
    #     if batch_size % 2 != 0:
    #         raise ValueError(f"Batch size must be even for pair processing, got {batch_size}")
    #     
    #     if self.n_cross_layers == 0:
    #         # Pure BERT
    #         outputs = self.bert(**batch_encoding)
    #         return self.projection(outputs.last_hidden_state[:, 0])
    #     
    #     # Use gradient checkpointing for the entire sequence
    #     return torch.utils.checkpoint.checkpoint(
    #         self._forward_with_cross_attention,
    #         batch_encoding,
    #         use_reentrant=False
    #     )
    # 
    # def _forward_with_cross_attention(self, batch_encoding):
    #     """Checkpointed forward pass"""
    #     hidden_states = self.bert.embeddings(
    #         input_ids=batch_encoding['input_ids'],
    #         token_type_ids=batch_encoding.get('token_type_ids', None)
    #     )
    #     
    #     attention_mask = batch_encoding['attention_mask']
    #     extended_attention_mask = self.bert.get_extended_attention_mask(
    #         attention_mask, batch_encoding['input_ids'].shape
    #     )
    #     
    #     cross_attn_idx = 0
    #     
    #     for i, layer in enumerate(self.bert.encoder.layer):
    #         # Use gradient checkpointing for each BERT layer
    #         layer_outputs = torch.utils.checkpoint.checkpoint(
    #             layer,
    #             hidden_states,
    #             extended_attention_mask,
    #             use_reentrant=False
    #         )

    #         if isinstance(layer_outputs, tuple):
    #             hidden_states = layer_outputs[0]
    #         else:
    #             hidden_states = layer_outputs
    #         
    #         # Apply cross-attention with memory efficiency
    #         if i in self.cross_attention_positions and cross_attn_idx < len(self.cross_attention_layers):
    #             # hidden_states = torch.utils.checkpoint.checkpoint(
    #             #     self.cross_attention_layers[cross_attn_idx],
    #             #     hidden_states,
    #             #     attention_mask,
    #             #     use_reentrant=False
    #             # )
    #             cross_attn_idx += 1
    #     
    #     return self.projection(hidden_states[:, 0])
    
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
        # # Get BERT embeddings for all sequences
        # outputs = self.bert(
        #     input_ids=batch_encoding['input_ids'],
        #     attention_mask=batch_encoding['attention_mask'],
        #     token_type_ids=batch_encoding.get('token_type_ids', None)
        # )
        # 
        # hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        # batch_size = hidden_states.shape[0]
        # 
        # # Ensure even batch size for pairing
        # if batch_size % 2 != 0:
        #     raise ValueError(f"Batch size must be even for pair processing, got {batch_size}")
        # 
        # # Split into pairs: (0,1), (2,3), (4,5), ...
        # # n_pairs = batch_size // 2
        # 
        # # Apply cross-attention between adjacent pairs
        # for _ in range(self.n_cross_layers):
        #     hidden_states = self.cross_attention_layer(
        #         hidden_states, 
        #         batch_encoding['attention_mask']
        #         # n_pairs
        #     )
        # 
        # # Extract CLS tokens and project
        # cls_embeddings = hidden_states[:, 0]  # [batch_size, hidden_size]
        # final_embeddings = self.projection(cls_embeddings)
        # 
        # return final_embeddings

class MemoryEfficientCrossAttention(nn.Module):
    """Memory-efficient cross-attention with chunked processing"""
    
    def __init__(self, similarity='dotproduct', chunk_size=64):
        super().__init__()
        self.similarity = similarity
        self.chunk_size = chunk_size  # Process attention in chunks
    
    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Process pairs in-place to save memory
        for i in range(0, batch_size, 2):
            seq_a = hidden_states[i]
            seq_b = hidden_states[i + 1]
            mask_a = attention_mask[i]
            mask_b = attention_mask[i + 1]
            
            # Chunked attention computation to reduce memory
            attended_a = self._chunked_attention(seq_a, seq_b, mask_b)
            attended_b = self._chunked_attention(seq_b, seq_a, mask_a)
            
            # In-place update to save memory
            hidden_states[i] += attended_a
            hidden_states[i + 1] += attended_b
        
        return hidden_states
    
    def _chunked_attention(self, query_seq, key_seq, key_mask):
        """Compute attention in chunks to reduce memory usage"""
        seq_len, hidden_size = query_seq.shape
        attended_output = torch.zeros_like(query_seq)
        
        # Process in chunks
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            query_chunk = query_seq[start:end]
            
            # Compute similarity for this chunk
            if self.similarity == 'dotproduct':
                sim_chunk = torch.mm(query_chunk, key_seq.transpose(0, 1))
            elif self.similarity == 'cosine':
                sim_chunk = F.cosine_similarity(
                    query_chunk.unsqueeze(1), 
                    key_seq.unsqueeze(0), 
                    dim=-1
                )
            
            # Apply mask
            if key_mask is not None:
                sim_chunk = sim_chunk.masked_fill(~key_mask.bool().unsqueeze(0), -float('inf'))
            
            # Attention weights and output
            attn_weights = F.softmax(sim_chunk, dim=-1)
            attended_chunk = torch.mm(attn_weights, key_seq)
            
            attended_output[start:end] = attended_chunk
            
            # Clean up intermediate tensors
            del sim_chunk, attn_weights, attended_chunk
        
        return attended_output

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


class BertCrossAttentionLayer(nn.Module):
    """Cross-attention layer that mimics GMN cross-graph attention"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=16, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            # nn.Linear(hidden_size * 4, hidden_size * 4),
            # nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )
    
    def forward(self, hidden_states, attention_mask):
        """
        Apply cross-attention between adjacent pairs (GMN style)
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        n_pairs = batch_size // 2
        
        # Process pairs: (0,1), (2,3), (4,5), etc.
        updated_states = []
        
        for i in range(0, batch_size, 2):
            # Get sequences A and B
            seq_a = hidden_states[i]      # [seq_len, hidden_size]
            seq_b = hidden_states[i + 1]  # [seq_len, hidden_size] 
            mask_a = attention_mask[i]    # [seq_len]
            mask_b = attention_mask[i + 1] # [seq_len]
            
            # Cross-attention: A attends to B, B attends to A (GMN style)
            # A queries B
            attn_a, _ = self.attention(
                seq_a.unsqueeze(0), seq_b.unsqueeze(0), seq_b.unsqueeze(0),
                key_padding_mask=~mask_b.bool().unsqueeze(0)
            )
            # B queries A  
            attn_b, _ = self.attention(
                seq_b.unsqueeze(0), seq_a.unsqueeze(0), seq_a.unsqueeze(0),
                key_padding_mask=~mask_a.bool().unsqueeze(0)
            )
            
            # Residual connections and normalization
            seq_a_updated = self.norm1(seq_a + attn_a.squeeze(0))
            seq_b_updated = self.norm1(seq_b + attn_b.squeeze(0))
            
            # Feed-forward networks
            seq_a_final = self.norm2(seq_a_updated + self.ffn(seq_a_updated))
            seq_b_final = self.norm2(seq_b_updated + self.ffn(seq_b_updated))
            
            updated_states.extend([seq_a_final, seq_b_final])
        
        return torch.stack(updated_states)  # [batch_size, seq_len, hidden_size]


# class BertCrossAttentionLayer(nn.Module):
#     """Cross-attention layer that works on adjacent pairs"""
#     
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
#         self.norm1 = nn.LayerNorm(hidden_size)
#         self.norm2 = nn.LayerNorm(hidden_size)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 4),
#             nn.ReLU(),
#             nn.Linear(hidden_size * 4, hidden_size)
#         )
#     
#     def forward(self, hidden_states, attention_mask, n_pairs):
#         """
#         Apply cross-attention between adjacent pairs
#         
#         Args:
#             hidden_states: [batch_size, seq_len, hidden_size]
#             attention_mask: [batch_size, seq_len]
#             n_pairs: Number of pairs (batch_size // 2)
#         """
#         seq_len = hidden_states.shape[1]
#         results = []
#         
#         for i in range(0, n_pairs * 2, 2):
#             # Get pair
#             seq_a = hidden_states[i]      # [seq_len, hidden_size]
#             seq_b = hidden_states[i + 1]  # [seq_len, hidden_size]
#             mask_a = attention_mask[i]    # [seq_len]
#             mask_b = attention_mask[i + 1] # [seq_len]
#             
#             # Cross-attention: A attends to B, B attends to A
#             attn_a, _ = self.attention(seq_a.unsqueeze(0), seq_b.unsqueeze(0), seq_b.unsqueeze(0),
#                                      key_padding_mask=~mask_b.bool())
#             attn_b, _ = self.attention(seq_b.unsqueeze(0), seq_a.unsqueeze(0), seq_a.unsqueeze(0),
#                                      key_padding_mask=~mask_a.bool())
#             
#             # Residual + norm
#             seq_a = self.norm1(seq_a + attn_a.squeeze(0))
#             seq_b = self.norm1(seq_b + attn_b.squeeze(0))
#             
#             # FFN
#             seq_a = self.norm2(seq_a + self.ffn(seq_a))
#             seq_b = self.norm2(seq_b + self.ffn(seq_b))
#             
#             results.append(seq_a)
#             results.append(seq_b)
#         
#         return torch.stack(results)  # [batch_size, seq_len, hidden_size]

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
