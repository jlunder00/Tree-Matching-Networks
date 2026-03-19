# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

# models/pretrained_tree_matching.py
import torch.nn as nn
from ...GMN.graphmatchingnetwork import GraphMatchingNet
from ...GMN.pretrained_transformer_aggregator import PretrainedTransformerAggregator
from .tree_encoder import TreeEncoder


class PretrainedTreeMatchingNet(nn.Module):
    """GNN propagation + pretrained HF aggregator (matching mode).

    Used for Conditions D, E, F.
    Same forward interface as TreeMatchingNet.
    """

    def __init__(self, config):
        super().__init__()

        node_feature_dim = config['model']['graph']['node_feature_dim']
        edge_feature_dim = config['model']['graph']['edge_feature_dim']
        node_state_dim = config['model']['graph']['node_state_dim']
        edge_state_dim = config['model']['graph']['edge_state_dim']

        node_hidden_sizes = list(config['model']['graph']['node_hidden_sizes'])
        node_hidden_sizes.append(node_state_dim * 2)
        edge_hidden_sizes = list(config['model']['graph']['edge_hidden_sizes'])
        edge_hidden_sizes.append(node_state_dim * 2)
        graph_rep_dim = config['model']['graph']['graph_rep_dim']
        edge_net_init_scale = config['model']['graph']['edge_net_init_scale']
        n_prop_layers = config['model']['graph']['n_prop_layers']
        share_prop_params = config['model']['graph']['share_prop_params']
        use_reverse_direction = config['model']['graph']['use_reverse_direction']
        reverse_dir_param_different = config['model']['graph']['reverse_dir_param_different']

        encoder = TreeEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            node_state_dim=node_state_dim,
            edge_state_dim=edge_state_dim,
        )

        pt_config = config['model']['graph']['pretrained_transformer']
        aggregator = PretrainedTransformerAggregator(
            node_state_dim=node_state_dim,
            graph_rep_dim=graph_rep_dim,
            hf_model_name=pt_config['model_name'],
            max_nodes=pt_config.get('max_nodes', 64),
            positional_features=pt_config.get('positional_features', None),
            positional_max_values=pt_config.get('positional_max_values', None),
            use_cls_token=pt_config.get('use_cls_token', False),
            cls_token_type=pt_config.get('cls_token_type', 'virtual'),
            freeze_transformer=pt_config.get('freeze_transformer', False),
        )

        self.gmn = GraphMatchingNet(
            encoder=encoder,
            aggregator=aggregator,
            node_state_dim=node_state_dim,
            edge_state_dim=edge_state_dim,
            edge_hidden_sizes=edge_hidden_sizes,
            node_hidden_sizes=node_hidden_sizes,
            n_prop_layers=n_prop_layers,
            share_prop_params=share_prop_params,
            edge_net_init_scale=edge_net_init_scale,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            prop_type='matching'
        )

        print(f"PretrainedTreeMatchingNet initialized:")
        print(f"  GNN: {node_state_dim}-dim, {n_prop_layers} prop layers")
        print(f"  Aggregator: {pt_config['model_name']} ({aggregator.hf_hidden_dim}-dim)")
        print(f"  Output: {graph_rep_dim}-dim")

    def freeze_propagation(self):
        """Freeze GNN encoder + propagation layers (Condition F)."""
        for param in self.gmn._encoder.parameters():
            param.requires_grad = False
        for layer in self.gmn._prop_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_propagation(self):
        for param in self.gmn._encoder.parameters():
            param.requires_grad = True
        for layer in self.gmn._prop_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def freeze_transformer(self):
        self.gmn._aggregator.freeze_transformer()

    def unfreeze_transformer(self):
        self.gmn._aggregator.unfreeze_transformer()

    def get_parameter_groups(self, base_lr, pretrained_lr_scale=0.1):
        pretrained_params = []
        new_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'gmn._aggregator.encoder.' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
        groups = []
        if new_params:
            groups.append({'params': new_params, 'lr': base_lr})
        if pretrained_params:
            groups.append({'params': pretrained_params, 'lr': base_lr * pretrained_lr_scale})
        return groups

    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs):
        return self.gmn(node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs)
