# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

# models/pretrained_noprop_embedding.py
import torch.nn as nn
from ...GMN.pretrained_transformer_aggregator import PretrainedTransformerAggregator


class PretrainedNoPropEmbeddingNet(nn.Module):
    """Condition B: Raw 804-dim node features -> pretrained HF transformer -> graph embedding.

    No GNN propagation. No TreeEncoder. Passes raw node features directly to
    the pretrained transformer aggregator with tree positional encoding.

    Same forward interface as TreeEmbeddingNet/GraphEmbeddingNet:
        forward(node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs)
    """

    def __init__(self, config):
        super().__init__()

        node_feature_dim = config['model']['graph']['node_feature_dim']
        graph_rep_dim = config['model']['graph']['graph_rep_dim']

        pt_config = config['model']['graph']['pretrained_transformer']

        self.aggregator = PretrainedTransformerAggregator(
            node_state_dim=node_feature_dim,
            graph_rep_dim=graph_rep_dim,
            hf_model_name=pt_config['model_name'],
            max_nodes=pt_config.get('max_nodes', 64),
            positional_features=pt_config.get('positional_features', None),
            positional_max_values=pt_config.get('positional_max_values', None),
            use_cls_token=pt_config.get('use_cls_token', False),
            cls_token_type=pt_config.get('cls_token_type', 'virtual'),
            freeze_transformer=pt_config.get('freeze_transformer', False),
        )

    def freeze_transformer(self):
        self.aggregator.freeze_transformer()

    def unfreeze_transformer(self):
        self.aggregator.unfreeze_transformer()

    def get_parameter_groups(self, base_lr, pretrained_lr_scale=0.1):
        return self.aggregator.get_parameter_groups(base_lr, pretrained_lr_scale)

    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
        return self.aggregator(node_features, graph_idx, n_graphs,
                              from_idx=from_idx, to_idx=to_idx)
