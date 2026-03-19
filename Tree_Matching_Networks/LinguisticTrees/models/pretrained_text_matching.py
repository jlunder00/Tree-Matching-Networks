# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

# models/pretrained_text_matching.py
import torch
import torch.nn as nn
from transformers import AutoModel
from .bert_matching import BertGMNStyleCrossAttention


class PretrainedTextMatchingNet(nn.Module):
    """Condition A matching: Pre-trained HF transformer on text with cross-attention.

    Same forward interface as BertMatchingNet.
    """

    def __init__(self, config, tokenizer):
        super().__init__()

        pt_config = config['model']['pretrained']
        model_name = pt_config['model_name']

        self.transformer = AutoModel.from_pretrained(model_name)
        hf_hidden_size = self.transformer.config.hidden_size
        num_hidden_layers = self.transformer.config.num_hidden_layers
        graph_rep_dim = config['model']['graph']['graph_rep_dim']

        # Cross-attention at every layer
        self.n_cross_layers = num_hidden_layers
        self.cross_attention_positions = list(range(num_hidden_layers))
        self.cross_attention_layers = nn.ModuleList([
            BertGMNStyleCrossAttention(similarity='dotproduct')
            for _ in range(num_hidden_layers)
        ])

        self.projection = nn.Linear(hf_hidden_size, graph_rep_dim)

        self._frozen = False
        if pt_config.get('freeze_transformer', False):
            self.freeze_transformer()

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = True
        self._frozen = False

    def get_parameter_groups(self, base_lr, pretrained_lr_scale=0.1):
        pretrained_params = []
        new_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('transformer.'):
                pretrained_params.append(param)
            else:
                new_params.append(param)
        groups = []
        if new_params:
            groups.append({'params': new_params, 'lr': base_lr})
        if pretrained_params:
            groups.append({'params': pretrained_params, 'lr': base_lr * pretrained_lr_scale})
        return groups

    def _get_encoder_layers(self):
        """Get the encoder layer list, handling different HF model families."""
        if hasattr(self.transformer, 'encoder'):
            return self.transformer.encoder.layer
        elif hasattr(self.transformer, 'transformer'):
            return self.transformer.transformer.layer
        else:
            raise ValueError(f"Cannot find encoder layers in {type(self.transformer).__name__}")

    def forward(self, batch_encoding):
        batch_size = batch_encoding['input_ids'].shape[0]
        if batch_size % 2 != 0:
            raise ValueError(f"Batch size must be even for pairs, got {batch_size}")

        # Get embeddings from the HF model's embedding layer
        hidden_states = self.transformer.embeddings(
            input_ids=batch_encoding['input_ids'],
            token_type_ids=batch_encoding.get('token_type_ids', None)
        )

        attention_mask = batch_encoding['attention_mask']
        extended_attention_mask = self.transformer.get_extended_attention_mask(
            attention_mask, batch_encoding['input_ids'].shape
        )

        cross_attn_idx = 0
        encoder_layers = self._get_encoder_layers()

        for i, layer in enumerate(encoder_layers):
            layer_outputs = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]

            if i in self.cross_attention_positions and cross_attn_idx < len(self.cross_attention_layers):
                hidden_states = self.cross_attention_layers[cross_attn_idx](
                    hidden_states, attention_mask
                )
                cross_attn_idx += 1

        cls_embeddings = hidden_states[:, 0]
        return self.projection(cls_embeddings)
