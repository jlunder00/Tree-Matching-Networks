# GMN/attention_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segment import unsorted_segment_sum

def graph_prop_once_attention(node_states, from_idx, to_idx, message_attention, 
                            aggregation_attention=None, edge_features=None):
    """
    Attention-based version of graph_prop_once that replaces MLP message computation
    and optionally replaces summation aggregation.
    """
    n_nodes = node_states.shape[0]
    n_edges = len(from_idx)
    
    if n_edges == 0:
        return torch.zeros_like(node_states)
    
    # Get from and to node states
    from_states = node_states[from_idx]  # [n_edges, node_dim]
    to_states = node_states[to_idx]      # [n_edges, node_dim]
    
    # Message computation with attention instead of MLP
    # Each edge computes its message by attending between from/to nodes
    edge_pairs = torch.stack([from_states, to_states], dim=1)  # [n_edges, 2, node_dim]
    
    # Use from_state as query, attend over [from, to] pair
    messages, _ = message_attention(
        query=from_states.unsqueeze(1),    # [n_edges, 1, node_dim]
        key=edge_pairs,                    # [n_edges, 2, node_dim]  
        value=edge_pairs                   # [n_edges, 2, node_dim]
    )
    messages = messages.squeeze(1)  # [n_edges, node_dim]
    
    # Message aggregation
    if aggregation_attention is not None:
        # Attention-based aggregation instead of sum
        aggregated_messages = torch.zeros(n_nodes, node_states.shape[1], device=node_states.device)
        
        for node_i in range(n_nodes):
            # Get incoming messages for this node
            incoming_mask = (to_idx == node_i)
            if incoming_mask.sum() > 0:
                node_messages = messages[incoming_mask]  # [n_incoming, node_dim]
                
                if node_messages.shape[0] == 1:
                    # Only one message, no need for attention
                    aggregated_messages[node_i] = node_messages[0]
                else:
                    # Multiple messages - use attention to aggregate
                    query = node_messages.mean(0, keepdim=True).unsqueeze(0)  # [1, 1, node_dim]
                    attended, _ = aggregation_attention(
                        query=query,
                        key=node_messages.unsqueeze(0),      # [1, n_incoming, node_dim]
                        value=node_messages.unsqueeze(0)     # [1, n_incoming, node_dim]
                    )
                    aggregated_messages[node_i] = attended.squeeze()
    else:
        # Standard summation aggregation
        aggregated_messages = unsorted_segment_sum(messages, to_idx, n_nodes)
    
    return aggregated_messages


class AttentionGraphAggregator(nn.Module):
    """Graph aggregator using attention pooling instead of standard pooling"""
    
    def __init__(self, node_dim, graph_dim, num_heads=8, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        
        if use_attention:
            self.graph_attention = nn.MultiheadAttention(
                embed_dim=graph_dim, 
                num_heads=num_heads, 
                batch_first=True
            )
            # Learnable query for each graph
            self.graph_query = nn.Parameter(torch.randn(1, graph_dim))
            
            # Project nodes to graph dimension if needed
            if node_dim != graph_dim:
                self.node_projection = nn.Linear(node_dim, graph_dim)
            else:
                self.node_projection = nn.Identity()
        
    def forward(self, node_states, graph_idx, n_graphs):
        """
        Args:
            node_states: [n_total_nodes, node_dim]
            graph_idx: [n_total_nodes] - which graph each node belongs to
            n_graphs: int - number of graphs in batch
        """
        if not self.use_attention:
            # Fallback to mean pooling
            graph_representations = []
            for i in range(n_graphs):
                mask = (graph_idx == i)
                if mask.sum() > 0:
                    graph_repr = node_states[mask].mean(0)
                else:
                    graph_repr = torch.zeros(self.graph_dim, device=node_states.device)
                graph_representations.append(graph_repr)
            return torch.stack(graph_representations)
        
        # Attention-based graph aggregation
        projected_nodes = self.node_projection(node_states)  # [n_total_nodes, graph_dim]
        graph_representations = []
        
        for i in range(n_graphs):
            mask = (graph_idx == i)
            if mask.sum() > 0:
                graph_nodes = projected_nodes[mask]  # [n_nodes_in_graph, graph_dim]
                
                if graph_nodes.shape[0] == 1:
                    # Single node graph
                    graph_repr = graph_nodes[0]
                else:
                    # Multi-node graph - use attention
                    query = self.graph_query.unsqueeze(0)  # [1, 1, graph_dim]
                    graph_repr, _ = self.graph_attention(
                        query=query,
                        key=graph_nodes.unsqueeze(0),    # [1, n_nodes, graph_dim]
                        value=graph_nodes.unsqueeze(0)   # [1, n_nodes, graph_dim]
                    )
                    graph_repr = graph_repr.squeeze()  # [graph_dim]
            else:
                graph_repr = torch.zeros(self.graph_dim, device=node_states.device)
            
            graph_representations.append(graph_repr)
        
        return torch.stack(graph_representations)  # [n_graphs, graph_dim]
