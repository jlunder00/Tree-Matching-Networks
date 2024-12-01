#training/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import TreeMatchingMetrics
# from scipy.stats import pearsonr, spearmanr


class TreeMatchingLoss(nn.Module):
    def __init__(self, device, task_type='entailment', **kwargs):
        super().__init__()
        self.task_type = task_type
        if task_type == 'entailment':
            self.thresholds = nn.Parameter(
                torch.tensor(kwargs.get('thresholds', [-0.3, 0.3])),
                requires_grad=False
            ).to(device, non_blocking=True)
        elif task_type == 'similarity':
            self.margin = nn.Parameter(
                torch.tensor(kwargs.get('margin', 0.1)),
                requires_grad=False
            ).to(device, non_blocking=True)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def forward(self, x, y, labels):
        """
        Args:
            x: First set of graph vectors [batch_size, hidden_dim]
            y: Second set of graph vectors [batch_size, hidden_dim]
            labels: Either entailment labels (-1,0,1) or similarity scores
        """

        # similarity_scores = self.euclidean_similarity(x, y)
        similarity_scores = self.cosine_similarity(x, y)
        if self.task_type == 'entailment':
            # Convert labels to class indices (shift -1,0,1 to 0,1,2)
            labels_idx = labels + 1
            
            # Create logits based on similarity thresholds
            logits = torch.zeros(x.size(0), 3, device=x.device)
            
            # Contradiction logits (similarity < lower threshold)
            logits[:, 0] = torch.sigmoid(-similarity_scores - self.thresholds[0])
            
            # Neutral logits (similarity between thresholds)
            neutral_mask = (similarity_scores >= self.thresholds[0]) & (similarity_scores <= self.thresholds[1])
            logits[:, 1] = torch.sigmoid(neutral_mask.float())
            
            # Entailment logits (similarity > upper threshold)
            logits[:, 2] = torch.sigmoid(similarity_scores - self.thresholds[1])

            # Compute cross entropy loss
            loss = F.cross_entropy(logits, labels_idx.long())
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1) - 1  # Shift back to -1,0,1
            # accuracy = (predictions == labels).float().mean()
            with torch.no_grad():
                metrics = TreeMatchingMetrics.compute_task_metrics(predictions, labels, 'entailment')
            
            return loss, predictions, metrics
            
        else:  # similarity task
            # For similarity task, directly optimize MSE between similarities
            labels = 2*labels -1
            # similarity_scores = self.normalize_cosine_sim(similarity_scores)
            loss = 2.0 * self.pearson_loss(similarity_scores, labels)
            # loss = self.pearson_loss_with_corrcoef(similarity_scores, labels)
            loss = loss + 0.1 * F.mse_loss(similarity_scores, labels)

            l2_reg = 0.01 * (torch.norm(x) + torch.norm(y))
            loss = loss + l2_reg

            loss = loss + 0.05 * self.variance_loss(similarity_scores)
            # accuracy = 1.0 - (similarity_scores - labels).abs().mean()
            with torch.no_grad():
                metrics = TreeMatchingMetrics.compute_task_metrics(similarity_scores, labels, 'similarity')
            return loss, similarity_scores, metrics

    def normalize_cosine_sim(self, cosine_sim):
        #normalize -1 to 1 cosine sim to be 0-1
        return (cosine_sim + 1)/2

    def variance_loss(self, predictions):
        """Loss to encourage predictions to have good variance/spread"""
        return -torch.var(predictions)  # Negative since we want to maximize variance


    # def pearson_loss(self, sim, labels):
    #     scores_mean = sim.mean()
    #     labels_mean = labels.mean()
    #     numerator = ((sim - scores_mean) * (labels - labels_mean)).sum()
    #     denominator = torch.sqrt(((sim - scores_mean) ** 2).sum() * ((labels - labels_mean) ** 2).sum())
    #     correlation = numerator / (denominator + 1e-8)
    #     return 1-correlation

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

    def euclidean_similarity(self, x, y, scale=1.0):
        """
        Compute Euclidean similarity between two tensors.

        Args:
            x (torch.Tensor): First set of vectors [batch_size, hidden_dim].
            y (torch.Tensor): Second set of vectors [batch_size, hidden_dim].
            scale (float): Scale factor to adjust sensitivity of similarity.

        Returns:
            torch.Tensor: Similarity scores in range [0, 1].
        """
        euclidean_distance = torch.sqrt(torch.sum((x - y) ** 2, dim=1))
        return torch.exp(-euclidean_distance / scale)

    def manhattan_similarity(self, x, y, scale=1.0):
        """
        Compute Manhattan similarity between two tensors.

        Args:
            x (torch.Tensor): First set of vectors [batch_size, hidden_dim].
            y (torch.Tensor): Second set of vectors [batch_size, hidden_dim].
            scale (float): Scale factor to adjust sensitivity of similarity.

        Returns:
            torch.Tensor: Similarity scores in range [0, 1].
        """
        manhattan_distance = torch.sum(torch.abs(x - y), dim=1)
        return torch.exp(-manhattan_distance / scale)

    def chebyshev_similarity(self, x, y, scale=1.0):
        """
        Compute Chebyshev similarity between two tensors.

        Args:
            x (torch.Tensor): First set of vectors [batch_size, hidden_dim].
            y (torch.Tensor): Second set of vectors [batch_size, hidden_dim].
            scale (float): Scale factor to adjust sensitivity of similarity.

        Returns:
            torch.Tensor: Similarity scores in range [0, 1].
        """
        chebyshev_distance = torch.max(torch.abs(x - y), dim=1).values
        return torch.exp(-chebyshev_distance / scale)

    def cosine_similarity(self, x, y):
        # Compute cosine similarity
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        return torch.sum(x_norm * y_norm, dim=1)

    def pearson_loss_with_corrcoef(self, predictions, targets):
        """
        Compute the negative Pearson correlation loss using torch.corrcoef.
        
        Args:
            predictions (torch.Tensor): Predicted values [batch_size].
            targets (torch.Tensor): True target values [batch_size].
            
        Returns:
            torch.Tensor: Negative Pearson correlation loss.
        """
        # Stack predictions and targets to form a 2D tensor
        inputs = torch.stack([predictions, targets])
        
        # Compute correlation matrix
        corr_matrix = torch.corrcoef(inputs)
        
        # Extract Pearson correlation (off-diagonal element)
        pearson_corr = corr_matrix[0, 1]
        
        # Return negative correlation for minimization
        return 1-pearson_corr









# def cosine_similarity(x, y):
#     """Compute cosine similarity between batches of vectors"""
#     x = x.requires_grad_(True)
#     y = y.requires_grad_(True)

#     # Normalize to unit vectors
#     x_norm = F.normalize(x, p=2, dim=1)
#     y_norm = F.normalize(y, p=2, dim=1)
#     # Compute cosine similarity
#     return torch.sum(x_norm * y_norm, dim=1)

# class TreeMatchingLoss(nn.Module):
#     """Loss function that handles both similarity and entailment tasks"""
#     
#     def __init__(self, task_type='entailment', **kwargs):
#         super().__init__()
#         self.task_type = task_type
#         if task_type == 'entailment':
#             # Parameters for entailment bucketing
#             self.thresholds = torch.tensor(
#                     kwargs.get('thresholds', [-0.3, 0.3]),  # Creates 3 buckets
#                     requires_grad=False
#             )
#         elif task_type == 'similarity':
#             # For direct similarity scoring
#             self.margin = torch.tensor(
#                     kwargs.get('margin', 0.1),
#                     requires_grad=False
#             )
#         else:
#             raise ValueError(f"Unknown task type: {task_type}")
#             
#     # def similarity_to_entailment(self, similarity):
#     #     """Convert similarity scores to entailment predictions (-1, 0, 1)"""
#     #     predictions = torch.zeros_like(similarity)
#     #     predictions[similarity < self.thresholds[0]] = -1  # Contradiction
#     #     predictions[similarity > self.thresholds[1]] = 1   # Entailment
#     #     return predictions

#     def forward(self, x, y, labels):
#         """
#         Args:
#             x: First set of graph vectors
#             y: Second set of graph vectors
#             labels: Either entailment labels (-1,0,1) or similarity scores
#         """
#         x = x.requires_grad_(True)
#         y = y.requires_grad_(True)

#         # Compute cosine similarity between graph vectors
#         similarity_scores = cosine_similarity(x, y)
#         
#         if self.task_type == 'entailment':
#             # Convert similarities to entailment predictions
#             # predictions = self.similarity_to_entailment(similarity_scores)
#             
#             
#             # Convert labels to class indices for cross entropy (shift -1,0,1 to 0,1,2)
#             labels_idx = labels + 1
#             
#             # Convert predictions to logits for 3 classes
#             # logits = torch.zeros((predictions.size(0), 3), device=predictions.device, requires_grad=True)
#             # logits[:, 0] = (similarity_scores < self.thresholds[0]).float()  # Contradiction
#             # logits[:, 1] = ((similarity_scores >= self.thresholds[0]) & 
#             #                (similarity_scores <= self.thresholds[1])).float()  # Neutral
#             # logits[:, 2] = (similarity_scores > self.thresholds[1]).float()  # Entailment

#             contradictions = (similarity_scores < self.thresholds[0]).float().unsqueeze(1)
#             neutral = ((similarity_scores >= self.thresholds[0]) & 
#                       (similarity_scores <= self.thresholds[1])).float().unsqueeze(1)
#             entailments = (similarity_scores > self.thresholds[1]).float().unsqueeze(1)

#             # Concatenate to form logits
#             logits = torch.cat([contradictions, neutral, entailments], dim=1)

#             # Cross entropy loss
#             loss = F.cross_entropy(logits, labels_idx.long())
#             
#             predictions = torch.argmax(logits, dim=1) - 1
#             # Compute accuracy 
#             accuracy = (predictions == labels).float().mean()
#             
#             
#             return loss, predictions, accuracy
#             
#         else:  # similarity task
#             # For similarity task, directly compare similarity scores with target
#             loss = F.mse_loss(similarity_scores, labels)
#             accuracy = 1.0 - (similarity_scores - labels).abs().mean()  # Similarity accuracy
#             return loss, similarity_scores, accuracy
