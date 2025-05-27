# GMN/attention_layers.py
import torch
import torch.nn as nn
from .graphembeddingnetwork import GraphPropLayer
from .attention_utils import graph_prop_once_attention

class AttentionGraphPropLayer(GraphPropLayer):
    """
    Graph propagation layer with configurable attention mechanisms.
    Can use attention for message computation, aggregation, and/or node updates.
    """
    
    def __init__(self, node_state_dim, edge_state_dim, edge_hidden_sizes, node_hidden_sizes,
                 edge_net_init_scale=0.1, node_update_type='residual', use_reverse_direction=True,
                 reverse_dir_param_different=True, layer_norm=False, prop_type='embedding',
                 # New attention parameters
                 use_message_attention=False, use_aggregation_attention=False, 
                 use_node_update_attention=False, attention_heads=4, name='attention-graph-prop'):
        
        # Store attention config before calling super().__init__
        self.use_message_attention = use_message_attention
        self.use_aggregation_attention = use_aggregation_attention  
        self.use_node_update_attention = use_node_update_attention
        self.attention_heads = attention_heads
        
        # Initialize parent class
        super().__init__(
            node_state_dim, edge_state_dim, edge_hidden_sizes, node_hidden_sizes,
            edge_net_init_scale, node_update_type, use_reverse_direction,
            reverse_dir_param_different, layer_norm, prop_type
        )
    
    def build_model(self):
        # Build standard MLP components first
        super().build_model()
        
        # Add attention components
        if self.use_message_attention:
            self.message_attention = nn.MultiheadAttention(
                embed_dim=self._node_state_dim,
                num_heads=self.attention_heads,
                batch_first=True
            )
            
        if self.use_aggregation_attention:
            self.aggregation_attention = nn.MultiheadAttention(
                embed_dim=self._node_state_dim,
                num_heads=self.attention_heads, 
                batch_first=True
            )
            
        if self.use_node_update_attention:
            self.node_update_attention = nn.MultiheadAttention(
                embed_dim=self._node_state_dim,
                num_heads=self.attention_heads,
                batch_first=True
            )

    def _compute_aggregated_messages(self, node_states, from_idx, to_idx, edge_features=None):
        """Override to use attention-based message passing when enabled"""
        
        if self.use_message_attention or self.use_aggregation_attention:
            # Use attention-based approach
            message_attn = self.message_attention if self.use_message_attention else None
            agg_attn = self.aggregation_attention if self.use_aggregation_attention else None
            
            aggregated_messages = graph_prop_once_attention(
                node_states, from_idx, to_idx, 
                message_attention=message_attn,
                aggregation_attention=agg_attn,
                edge_features=edge_features
            )
        else:
            # Use standard MLP approach
            aggregated_messages = super()._compute_aggregated_messages(
                node_states, from_idx, to_idx, edge_features
            )
            
        # Handle reverse direction
        if self._use_reverse_direction:
            if self.use_message_attention or self.use_aggregation_attention:
                # Attention for reverse direction  
                reverse_message_attn = (self._reverse_message_net if self._reverse_dir_param_different 
                                      else self.message_attention) if self.use_message_attention else None
                reverse_agg_attn = self.aggregation_attention if self.use_aggregation_attention else None
                
                reverse_aggregated_messages = graph_prop_once_attention(
                    node_states, to_idx, from_idx,  # Note: swapped indices
                    message_attention=reverse_message_attn,
                    aggregation_attention=reverse_agg_attn, 
                    edge_features=edge_features
                )
            else:
                # Standard approach for reverse
                from .graphembeddingnetwork import graph_prop_once
                reverse_aggregated_messages = graph_prop_once(
                    node_states, to_idx, from_idx, self._reverse_message_net,
                    aggregation_module=None, edge_features=edge_features
                )
                
            aggregated_messages += reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)

        return aggregated_messages

    def _compute_node_update(self, node_states, node_state_inputs, node_features=None):
        """Override to use attention-based node updates when enabled"""
        
        if self.use_node_update_attention:
            # Prepare inputs for attention
            if self._node_update_type in ('mlp', 'residual'):
                node_state_inputs.append(node_states)
            if node_features is not None:
                node_state_inputs.append(node_features)

            # Stack inputs as sequence for attention
            if len(node_state_inputs) == 1:
                input_sequence = node_state_inputs[0].unsqueeze(1)  # [n_nodes, 1, dim]
            else:
                input_sequence = torch.stack(node_state_inputs, dim=1)  # [n_nodes, seq_len, dim]
            
            # Self-attention over input components
            updated_states, _ = self.node_update_attention(
                query=node_states.unsqueeze(1),  # [n_nodes, 1, dim]
                key=input_sequence,              # [n_nodes, seq_len, dim]
                value=input_sequence             # [n_nodes, seq_len, dim]
            )
            updated_states = updated_states.squeeze(1)  # [n_nodes, dim]
            
            if self._layer_norm:
                updated_states = self.layer_norm2(updated_states)
                
            if self._node_update_type == 'residual':
                return node_states + updated_states
            else:
                return updated_states
        else:
            # Use standard MLP approach
            return super()._compute_node_update(node_states, node_state_inputs, node_features)
