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

        is_embedding_model = len(batch_info.pair_indices) == 0

        if is_embedding_model:
            return self._forward_embedding(embeddings, batch_info)
        else:
            return self._forward_matching(embeddings, batch_info)

    def _infonce_loss_update(self, loss, n_valid_anchors, scaled_sim_matrix, pos_indices, anchor_idx, n_embeddings):
        # For each anchor, we compute separate InfoNCE loss
        # treating one positive as the target and remaining entries as negatives
        for pos_idx in pos_indices:
            # Create target vector (1 for positive, 0 for all others)
            target_vector = torch.zeros(n_embeddings, device=self.device)
            target_vector[pos_idx] = 1
            
            # Get similarities for this anchor
            anchor_similarities = scaled_sim_matrix[anchor_idx]
            
            # Compute cross entropy loss (InfoNCE)
            # This pushes the positive similarity higher than all negatives
            loss += F.cross_entropy(
                anchor_similarities.unsqueeze(0),
                torch.tensor([pos_idx], device=self.device)
            )
            n_valid_anchors += 1
        return loss, n_valid_anchors


    def _forward_embedding(self, embeddings, batch_info):
        """InfoNCE loss computation for embedding model"""
        # Get number of embeddings
        n_embeddings = embeddings.shape[0]
        
        # Create similarity matrix for all embeddings
        # [n_embeddings, n_embeddings]
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1),  # [n_embeddings, 1, hidden_dim]
            embeddings.unsqueeze(0),  # [1, n_embeddings, hidden_dim]
            dim=2
        )
        
        # Scale by temperature
        scaled_sim_matrix = sim_matrix / self.temperature
        
        # Create target matrix - 1 for positive pairs, 0 elsewhere
        targets = torch.zeros_like(scaled_sim_matrix, device=self.device)
        
        # Mark positive pairs (trees from same group, one being an anchor)
        for anchor_idx, pos_idx in batch_info.positive_pairs:
            targets[anchor_idx, pos_idx] = 1

        # For InfoNCE loss, we need a label for each anchor indicating
        # which embedding is its positive pair
        loss = 0
        n_valid_anchors = 0
        
        # Process each anchor separately
        for anchor_idx in batch_info.anchor_indices:
            # Get all positive pairs for this anchor
            pos_indices = [pos_idx for a_idx, pos_idx in batch_info.positive_pairs 
                          if a_idx == anchor_idx]
            
            if not pos_indices:
                continue  # Skip anchors with no positives
                
            loss, n_valid_anchors = self._infonce_loss_update(loss, n_valid_anchors, scaled_sim_matrix, pos_indices, anchor_idx, n_embeddings)
        
        # Average loss across all anchors
        if n_valid_anchors > 0:
            loss = loss / n_valid_anchors
           
        metrics = self._get_metrics(batch_info, sim_matrix)
            
        return loss, None, metrics

    def _get_metrics(self, batch_info, sim_matrix, diagonal=False):
        # Compute metrics
        with torch.no_grad():
            # Calculate average similarity for positives
            pos_sim = 0
            neg_sim = 0
            n_pos = 0
            n_neg = 0
            
            for anchor_idx in batch_info.anchor_indices:
                if not diagonal:
                    # Get all positive and negative indices for this anchor
                    pos_indices = [pos_idx for a_idx, pos_idx in batch_info.positive_pairs 
                                  if a_idx == anchor_idx]
                    neg_indices = [neg_idx for a_idx, neg_idx in batch_info.negative_pairs 
                                  if a_idx == anchor_idx]
                else:
                    pos_indices = batch_info.anchor_positive_indexes[anchor_idx] 
                    neg_indices = batch_info.anchor_negative_indexes[anchor_idx]
                    
                
                # Add positive similarities
                for pos_idx in pos_indices:
                    pos_sim += sim_matrix[anchor_idx, pos_idx].item() if not diagonal else sim_matrix[pos_idx]
                    n_pos += 1
                    
                # Add negative similarities
                for neg_idx in neg_indices:
                    neg_sim += sim_matrix[anchor_idx, neg_idx].item() if not diagonal else sim_matrix[neg_idx]
                    n_neg += 1
            
            # Compute averages
            pos_sim = pos_sim / max(1, n_pos)
            neg_sim = neg_sim / max(1, n_neg)
            
            metrics = {
                'pos_similarity': pos_sim,
                'neg_similarity': neg_sim,
            }
        return metrics

    def _forward_matching(self, embeddings, batch_info):
        """Compute InfoNCE loss from embeddings and batch info
        
        Args:
            embeddings: Graph embeddings for all trees [n_total, hidden_dim]
            batch_info: BatchInfo object with anchor/positive/negative indices
        """
        # print("Num positive pairs:", sum(1 for _,_,flag in batch_info.pair_indices if flag))
        # print("First few pair indices:", batch_info.pair_indices[:5])
        if batch_info.strict_matching:
            # Extract pairs
            tree1_embeddings = embeddings[[item[0] for item in batch_info.pair_indices]]
            tree2_embeddings = embeddings[[item[1] for item in batch_info.pair_indices]]
            
            # Compute similarities between all pairs
            raw_sim_matrix = F.cosine_similarity(
                tree1_embeddings.unsqueeze(1),  # [n_anchors, 1, hidden_dim]
                tree2_embeddings.unsqueeze(0),         # [1, n_total, hidden_dim]
                dim=2
            ).diag()
            sim_matrix = raw_sim_matrix / self.temperature


            loss = 0
            n_valid_anchors = 0

            for anchor_idx in batch_info.anchor_indices:
                # Get all positive pairs for this anchor
                pos_index_pair_locations = batch_info.anchor_positive_indexes[anchor_idx] 
                
                #get the list of tuples of this anchors embedding indices in the matrix and each of those positive pairs
                

                neg_index_pair_locations = batch_info.anchor_negative_indexes[anchor_idx]

                neg_index_pair_idxs = [batch_info.pair_indices[p][0] for p in neg_index_pair_locations]


                #for each positive pair
                for p in pos_index_pair_locations:
                    pos_pair_idx = batch_info.pair_indices[p][0]
                    #target vector includes this pair label and all the negative pairs
                    target_vector = torch.zeros(1+len(neg_index_pair_idxs), device=self.device)
                    target_vector[0] = 1
                    similarity_mask = [pos_pair_idx] + neg_index_pair_idxs
                    anchor_similarities = sim_matrix[similarity_mask]

                    loss += F.cross_entropy(
                        anchor_similarities.unsqueeze(0),
                        torch.tensor([0], device=self.device)
                    )
                    n_valid_anchors += 1

            if n_valid_anchors > 0:
                loss = loss / n_valid_anchors

            metrics = self._get_metrics(batch_info, sim_matrix, diagonal=True)
        else:
            n_embeddings = embeddings.shape[0]
            sim_matrix = F.cosine_similarity(
                embeddings.unsqueeze(1),  # [n_embeddings, 1, hidden_dim]
                embeddings.unsqueeze(0),  # [1, n_embeddings, hidden_dim]
                dim=2
            )
            scaled_sim_matrix = sim_matrix / self.temperature
            
            loss = 0
            n_valid_anchors = 0
            for anchor_idx in batch_info.anchor_indices:
                pos_indices = [pos_idx for a_idx, pos_idx, _ in batch_info.pair_indices if a_idx == anchor_idx]
                
                if not pos_indices:
                    continue

                loss, n_valid_anchors = self._infonce_loss_update(loss, n_valid_anchors, scaled_sim_matrix, pos_indices, anchor_idx, n_embeddings)
            # Average loss across all anchors
            if n_valid_anchors > 0:
                loss = loss / n_valid_anchors

            


            metrics = self._get_metrics(batch_info, sim_matrix, diagonal=False)

        
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




