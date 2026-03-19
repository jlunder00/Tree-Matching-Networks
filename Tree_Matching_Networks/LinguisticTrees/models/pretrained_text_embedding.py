# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

# models/pretrained_text_embedding.py
import torch
import torch.nn as nn
from transformers import AutoModel


class PretrainedTextEmbeddingNet(nn.Module):
    """Condition A: Pre-trained HuggingFace transformer on text input.

    Same forward interface as BertEmbeddingNet: accepts batch_encoding dict.
    Uses mean pooling over tokens (standard for sentence-transformers).
    """

    def __init__(self, config, tokenizer):
        super().__init__()

        pt_config = config['model']['pretrained']
        model_name = pt_config['model_name']

        self.transformer = AutoModel.from_pretrained(model_name)
        hf_hidden_size = self.transformer.config.hidden_size
        graph_rep_dim = config['model']['graph']['graph_rep_dim']

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

    def forward(self, batch_encoding=None, input_ids=None,
                attention_mask=None, token_type_ids=None, **kwargs):
        if input_ids is None and batch_encoding is not None:
            input_ids = batch_encoding.get('input_ids')
        if attention_mask is None and batch_encoding is not None:
            attention_mask = batch_encoding.get('attention_mask')
        if token_type_ids is None and batch_encoding is not None:
            token_type_ids = batch_encoding.get('token_type_ids')

        if input_ids is None:
            raise ValueError("input_ids required")
        if attention_mask is None:
            raise ValueError("attention_mask required")

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Mean pooling over tokens
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeds = torch.sum(token_embeddings * mask_expanded, 1) / \
                         torch.clamp(mask_expanded.sum(1), min=1e-9)

        return self.projection(sentence_embeds)
