#training/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import TreeMatchingMetrics
# from scipy.stats import pearsonr, spearmanr



class BaseLoss(nn.Module):
    """Base class for loss functions"""
    def __init__(self, device):
        super().__init__()
        self.device = device

    def _send_to_device(self, tensor):
        """Helper to send tensor to device"""
        return tensor.to(self.device, non_blocking=True)

class SimilarityLoss(BaseLoss):
    """Loss function for similarity task"""
    def __init__(self, device, **kwargs):
        super().__init__(device)
        self.margin = nn.Parameter(
            torch.tensor(kwargs.get('margin', 0.1)),
            requires_grad=False
        ).to(device, non_blocking=True)

    def forward(self, x, y, labels):
        """
        Args:
            x: First graph embeddings [batch_size, hidden_dim]
            y: Second graph embeddings [batch_size, hidden_dim]
            labels: Similarity scores [batch_size]
        """
        # Compute cosine similarity
        # similarities = F.cosine_similarity(x, y)
        similarities = self.cosine_similarity(x, y)
        
        # Multiple loss components
        loss = 0.2 * self.pearson_loss(similarities, labels)
        loss = loss +  F.mse_loss(similarities, labels)
        loss = loss + 0.05 * self.variance_loss(similarities)

        clipped_loss = torch.clamp(loss, min=-10.0, max=10.0)
        with torch.no_grad():
            metrics = TreeMatchingMetrics.compute_task_metrics(
                similarities, labels, 'similarity'
            )
        
        # return loss, similarities, metrics
        return clipped_loss, similarities, metrics
    
    def pearson_loss(self, sim, labels):
        # Center the variables
        sim_centered = sim - sim.mean()
        labels_centered = labels - labels.mean()
        
        # Standardize
        sim_std = torch.sqrt(torch.sum(sim_centered ** 2) / (sim.size(0) - 1))
        labels_std = torch.sqrt(torch.sum(labels_centered ** 2) / (labels.size(0) - 1))
        
        # Correlation
        correlation = torch.sum(sim_centered * labels_centered) / ((sim.size(0) - 1) * sim_std * labels_std)
        
        # Loss that encourages high positive correlation
        loss = 1 - correlation
        return loss

    def variance_loss(self, predictions):
        """Loss to encourage predictions to have good variance/spread"""
        return -torch.var(predictions)  # Negative since we want to maximize variance
    
    def cosine_similarity(self, x, y):
        # Compute cosine similarity
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        return torch.sum(x_norm * y_norm, dim=1)

class EntailmentLoss(BaseLoss):
    """Loss function for entailment classification"""
    def __init__(self, device):
        super().__init__(device)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits: Class logits [batch_size, 3]
            labels: Class labels -1,0,1 [batch_size]
        """
        # Convert -1,0,1 to 0,1,2
        label_indices = self._send_to_device(labels + 1)

        logits = torch.clamp(logits, min=-100, max=100)

        loss = self.criterion(logits, label_indices.long())

        clipped_loss = torch.clamp(loss, min=-10.0, max=10.0)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1) - 1
        
        with torch.no_grad():
            metrics = TreeMatchingMetrics.compute_task_metrics(
                predictions, labels, 'entailment'
            )
        
        # return loss, predictions, metrics
        return clipped_loss, predictions, metrics

class InfoNCELoss(BaseLoss):
    """InfoNCE contrastive loss implementation"""
    def __init__(self, device, temperature=0.07):
        super().__init__(device)
        self.temperature = temperature

    def forward(self, embeddings, batch_info):
        """Compute InfoNCE loss from embeddings and batch info
        
        Args:
            embeddings: Graph embeddings for all trees [n_total, hidden_dim]
            batch_info: BatchInfo object with anchor/positive/negative indices
        """
        print("Num positive pairs:", sum(1 for _,_,flag in batch_info.pair_indices if flag))
        print("First few pair indices:", batch_info.pair_indices[:5])
        # Extract pairs
        tree1_embeddings = embeddings[[item[0] for item in batch_info.pair_indices]]
        tree2_embeddings = embeddings[[item[1] for item in batch_info.pair_indices]]
        
        # Compute similarities between all pairs
        raw_sim_matrix = F.cosine_similarity(
            tree1_embeddings.unsqueeze(1),  # [n_anchors, 1, hidden_dim]
            tree2_embeddings.unsqueeze(0),         # [1, n_total, hidden_dim]
            dim=2
        )
        sim_matrix = raw_sim_matrix / self.temperature
        # Create labels matrix - 1 for positive pairs, 0 for all else
        labels = torch.zeros_like(sim_matrix, device=sim_matrix.device)
        positive_rows = torch.tensor([i for i, j, flag in batch_info.pair_indices if flag], dtype=torch.long, device=sim_matrix.device)
        positive_cols = torch.tensor([j for i, j, flag in batch_info.pair_indices if flag], dtype=torch.long, device=sim_matrix.device)
        labels[positive_rows, positive_cols] = 1
        print("Sim matrix shape:", sim_matrix.shape)  
        print("Labels shape:", labels.shape)
        print("Sample similarities and labels:")
        for i in range(min(5, len(positive_rows))):
            r, c = positive_rows[i], positive_cols[i]
            print(f"Anchor {i} positive pair: {sim_matrix[r,c]:.3f}")
            print(f"Anchor {i} negative means: {sim_matrix[r,:].mean():.3f}")
            
        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        # Basic metrics
        with torch.no_grad():
            metrics = {
                'pos_similarity': sim_matrix[labels == 1].mean(),
                'neg_similarity': sim_matrix[labels == 0].mean(),
                'raw_pos_sim': raw_sim_matrix[labels == 1].mean(),
                'raw_neg_sim': raw_sim_matrix[labels == 0].mean(),
            }
            
        return loss, None, metrics

# class TreeMatchingLoss(nn.Module):
#     def __init__(self, device, task_type='entailment', **kwargs):
#         super().__init__()
#         self.task_type = task_type
#         self.device = device
#         if task_type == 'entailment':
#             # self.thresholds = nn.Parameter(
#             #     torch.tensor(kwargs.get('thresholds', [-0.3, 0.3])),
#             #     requires_grad=False
#             # ).to(device, non_blocking=True)
#             self.criterion = nn.CrossEntropyLoss().to(device, non_blocking=True)
#         elif task_type == 'similarity':
#             self.margin = nn.Parameter(
#                 torch.tensor(kwargs.get('margin', 0.1)),
#                 requires_grad=False
#             ).to(device, non_blocking=True)
#         else:
#             raise ValueError(f"Unknown task type: {task_type}")

#     def forward(self, x, y, labels):
#         """
#         Args:
#             x: First set of graph vectors [batch_size, hidden_dim]
#             y: Second set of graph vectors [batch_size, hidden_dim]
#             labels: Either entailment labels (-1,0,1) or similarity scores
#         """

#         if self.task_type == 'entailment':
#             # # Convert labels to class indices (shift -1,0,1 to 0,1,2)
#             # labels_idx = labels + 1
#             # 
#             # # Create logits based on similarity thresholds
#             # logits = torch.zeros(x.size(0), 3, device=x.device)
#             # 
#             # # Contradiction logits (similarity < lower threshold)
#             # logits[:, 0] = torch.sigmoid(-similarity_scores - self.thresholds[0])
#             # 
#             # # Neutral logits (similarity between thresholds)
#             # neutral_mask = (similarity_scores >= self.thresholds[0]) & (similarity_scores <= self.thresholds[1])
#             # logits[:, 1] = torch.sigmoid(neutral_mask.float())
#             # 
#             # # Entailment logits (similarity > upper threshold)
#             # logits[:, 2] = torch.sigmoid(similarity_scores - self.thresholds[1])

#             # # Compute cross entropy loss
#             # loss = F.cross_entropy(logits, labels_idx.long())
#             # 
#             # # Get predictions
#             # predictions = torch.argmax(logits, dim=1) - 1  # Shift back to -1,0,1
#             # # accuracy = (predictions == labels).float().mean()
#              # Convert -1,0,1 labels to 0,1,2 indices
#             label_indices = labels + 1
#             loss = self.criterion(outputs, label_indices.long())
#             predictions = torch.argmax(outputs, dim=1) - 1  # Convert back to -1,0,1
#             with torch.no_grad():
#                 metrics = TreeMatchingMetrics.compute_task_metrics(predictions, labels, 'entailment')
#             
#             return loss, predictions, metrics
#             
#         else:  # similarity task
#             # For similarity task, directly optimize MSE between similarities
#             similarity_scores = self.cosine_similarity(x, y)
#             labels = 2*labels -1
#             # similarity_scores = self.normalize_cosine_sim(similarity_scores)
#             loss = 2.0 * self.pearson_loss(similarity_scores, labels)
#             # loss = self.pearson_loss_with_corrcoef(similarity_scores, labels)
#             loss = loss + 0.1 * F.mse_loss(similarity_scores, labels)

#             l2_reg = 0.01 * (torch.norm(x) + torch.norm(y))
#             loss = loss + l2_reg

#             loss = loss + 0.05 * self.variance_loss(similarity_scores)
#             # accuracy = 1.0 - (similarity_scores - labels).abs().mean()
#             with torch.no_grad():
#                 metrics = TreeMatchingMetrics.compute_task_metrics(similarity_scores, labels, 'similarity')
#             return loss, similarity_scores, metrics




