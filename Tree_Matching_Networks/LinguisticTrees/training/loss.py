# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

#only the TextLevel losses are used, and the InfoNCELoss can be used for pretraining with some modifications.
#training/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import TreeMatchingMetrics
from typing import Tuple, Dict, List
try:
    from ...models import TreeAggregator
except:
    from Tree_Matching_Networks.LinguisticTrees.models import TreeAggregator
# from scipy.stats import pearsonr, spearmanr



class BaseLoss(nn.Module):
    """Base class for loss functions"""
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device

    def _send_to_device(self, tensor):
        """Helper to send tensor to device"""
        return tensor.to(self.device, non_blocking=True)

    def _pearson_loss(self, pred, target):
        # Subtract means
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        
        # Compute numerator and denominator for Pearson correlation
        numerator = torch.sum(pred_centered * target_centered)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2)) * torch.sqrt(torch.sum(target_centered ** 2)) + 1e-8
        pearson_corr = numerator / denominator
        
        # If you want to maximize the correlation, you can minimize (1 - correlation)
        return 1 - pearson_corr

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
    def __init__(self, device, **kwargs):
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
    def __init__(self, device, temperature=0.07, **kwargs):
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

class TextLevelContrastiveLoss(BaseLoss):
    """Contrastive loss for text-level embeddings"""
    
    def __init__(self, device, temperature=0.07, aggregation='mean', positive_infonce_weight = True, inverse_infonce_weight = False, midpoint_infonce_weight = False, thresh_high = 0, thresh_low = -1, **kwargs):
        super().__init__(device)
        self.temperature = temperature
        self.aggregator = TreeAggregator(aggregation)
        self.positive_infonce_weight = positive_infonce_weight
        self.inverse_infonce_weight = inverse_infonce_weight
        self.midpoint_infonce_weight = midpoint_infonce_weight
        self.thresh_high = thresh_high
        self.thresh_low = thresh_low
    
    def forward(self, 
                embeddings: torch.Tensor, 
                batch_info: 'PairedGroupBatchInfo') -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute contrastive loss on text level
        
        Args:
            embeddings: Tree-level embeddings [n_trees, hidden_dim]
            batch_info: Batch information
            
        Returns:
            loss: Scalar loss value
            similarities: Similarity scores
            metrics: Dictionary of metrics
        """
        # 1. Aggregate tree embeddings into text embeddings
        text_embeddings = self.aggregator(embeddings, batch_info)
        
        # 2. Create text-level anchor-positive pairs
        n_groups = len(batch_info.group_indices)
        anchor_indices = list(range(0, n_groups*2, 2))  # Even indices
        pos_indices = list(range(1, n_groups*2, 2))     # Odd indices
        
        # 3. Compute similarity matrix
        sim_matrix = F.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_embeddings.unsqueeze(0),
            dim=2
        )
        dist_matrix = 1 - sim_matrix
        midpoint_matrix = 1 - torch.abs(sim_matrix)
        scaled_sim = sim_matrix / self.temperature
        scaled_dist = dist_matrix / self.temperature
        scaled_midpoint = midpoint_matrix / self.temperature
        
        # 4. Compute InfoNCE loss
        loss = 0
        n_valid_anchors = 0
        n_groups = len(batch_info.group_indices)

        for i, anchor_idx in enumerate(anchor_indices):
            # Current positive is the corresponding text in the pair
            pos_idx = pos_indices[i]

            group_label = batch_info.group_labels[i]
            if group_label > self.thresh_high:
                loss += self.positive_infonce_weight * F.cross_entropy(
                    scaled_sim[anchor_idx].unsqueeze(0),
                    torch.tensor([pos_idx], device=self.device)
                )
                n_valid_anchors += 1
            elif group_label > self.thresh_low:
                loss += self.midpoint_infonce_weight * F.cross_entropy(
                    scaled_midpoint[anchor_idx].unsqueeze(0),
                    torch.tensor([pos_idx], device=self.device)
                )
                n_valid_anchors += 1
            elif group_label <= self.thresh_low:
                # I think this is technically computing infonce on the distance matrix now
                loss += self.inverse_infonce_weight * F.cross_entropy(
                    scaled_dist[anchor_idx].unsqueeze(0),
                    torch.tensor([pos_idx], device=self.device)
                )
                n_valid_anchors += 1


        
            
            # # Use InfoNCE format (treat one positive as target, all others as negatives)
            # loss += F.cross_entropy(
            #     scaled_sim[anchor_idx].unsqueeze(0),
            #     torch.tensor([pos_idx], device=self.device)
            # )
            # n_valid_anchors += 1
        
        # 5. Compute metrics
        with torch.no_grad():
            metrics = self._compute_metrics(sim_matrix, dist_matrix, midpoint_matrix, anchor_indices, pos_indices, batch_info)
        
        return loss / max(1, n_valid_anchors), (sim_matrix, dist_matrix, midpoint_matrix), metrics
    
    def _compute_metrics(self, 
                         sim_matrix: torch.Tensor, 
                         dist_matrix: torch.Tensor,
                         midpoint_matrix: torch.Tensor,
                         anchor_indices: List[int], 
                         pos_indices: List[int],
                         batch_info) -> Dict:
        """Compute metrics for contrastive learning"""
        pos_sim = 0
        neg_sim = 0
        pos_dist = 0
        neg_dist = 0
        pos_mid = 0
        neg_mid = 0
        n_pos = 0
        n_neg = 0
        n_pos_dist = 0
        n_neg_dist = 0
        n_pos_mid = 0
        n_neg_mid = 0


        for i, anchor_idx in enumerate(anchor_indices):
            pos_idx = pos_indices[i]
            group_label = batch_info.group_labels[i]
            if group_label > self.thresh_high:
                # Positive similarity (with correct pair)
                pos_sim += sim_matrix[anchor_idx, pos_idx].item()
                n_pos += 1
            elif group_label > self.thresh_low:
                pos_mid += midpoint_matrix[anchor_idx, pos_idx].item()
                n_pos_mid += 1
            elif group_label <= self.thresh_low:
                pos_dist += dist_matrix[anchor_idx, pos_idx].item()
                n_pos_dist += 1
            else:
                continue
            
            # Negative similarities (with all other text embeddings)
            for j, other_idx in enumerate(pos_indices):
                if j != i:  # Skip the true positive
                    if group_label > self.thresh_high:
                        neg_sim += sim_matrix[anchor_idx, other_idx].item()
                        n_neg += 1
                    elif group_label > self.thresh_low:
                        neg_mid += midpoint_matrix[anchor_idx, other_idx].item()
                        n_neg_mid += 1
                    elif group_label <= self.thresh_low:
                        neg_dist += dist_matrix[anchor_idx, other_idx].item()
                        n_neg_dist += 1
                    else:
                        continue
        
        return {
            'pos_similarity': pos_sim / max(1, n_pos),
            'neg_similarity': neg_sim / max(1, n_neg),
            'pos_distance': pos_dist / max(1, n_pos_dist),
            'neg_distance': neg_dist / max(1, n_neg_dist),
            'pos_midpoint': pos_mid / max(1, n_pos_mid),
            'neg_midpoint': neg_mid / max(1, n_neg_mid),
            'similarity_gap': (pos_sim / max(1, n_pos)) - (neg_sim / max(1, n_neg)),
            'distance_gap': (pos_dist / max(1, n_pos_dist)) - (neg_dist / max(1, n_neg_dist)),
            'midpoint_gap': (pos_mid / max(1, n_pos_mid)) - (neg_mid / max(1, n_neg_mid))
        }


class TextLevelSimilarityLoss(BaseLoss):
    """Similarity regression loss for text-level embeddings"""
    
    def __init__(self, device, aggregation='mean', **kwargs):
        super().__init__(device)
        # self.similarity_loss = SimilarityLoss(device, margin=margin)
        self.aggregator = TreeAggregator(aggregation)
    
    def forward(self, 
                embeddings: torch.Tensor, 
                batch_info: 'PairedGroupBatchInfo') -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute similarity loss on text level
        
        Args:
            embeddings: Tree-level embeddings [n_trees, hidden_dim]
            batch_info: Batch information
            
        Returns:
            loss: Scalar loss value
            similarities: Similarity scores 
            metrics: Dictionary of metrics
        """
        # 1. Aggregate tree embeddings into text embeddings
        text_embeddings = self.aggregator(embeddings, batch_info)
        
        # 2. Extract text pairs and labels
        n_groups = len(batch_info.group_indices)
        text_a_embeddings = text_embeddings[0::2]  # Even indices
        text_b_embeddings = text_embeddings[1::2]  # Odd indices
        labels = torch.tensor(batch_info.group_labels, device=self.device)
        
        similarities = F.cosine_similarity(text_a_embeddings, text_b_embeddings, dim=1)
        loss = torch.nn.MSELoss()(similarities, labels)
        # 3. Compute similarity loss
        # loss, similarities, metrics = self.similarity_loss(
        #     text_a_embeddings, text_b_embeddings, labels
        # )
        with torch.no_grad():
            metrics = TreeMatchingMetrics.compute_task_metrics(
                similarities, labels, 'similarity'
            )
        return loss, similarities, metrics


class TextLevelEntailmentLoss(BaseLoss):
    """Classification loss for text-level entailment"""
    # Criterion version - bad 
    # def __init__(self, device, num_classes=3, aggregation='mean', classifier_input_dim=3584, classifier_hidden_dims=[512], **kwargs):
    #     super().__init__(device)
    #     self.aggregator = TreeAggregator(aggregation)
    #     layer_0 = nn.Linear(classifier_input_dim, classifier_hidden_dims[0])
    #     layers = [layer_0, nn.ReLU(), nn.Dropout(0.1)]
    #     for i in range(1, len(classifier_hidden_dims)):
    #         layers.append(nn.Linear(classifier_hidden_dims[i-1], classifier_hidden_dims[i]))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(0.1))
    #     layers.append(nn.Linear(classifier_hidden_dims[-1], num_classes))

    #     self.classifier = nn.Sequential(
    #         *layers
    #     ).to(device)
    #     self.criterion = nn.CrossEntropyLoss()

    def __init__(self, device, thresh_high, thresh_low, aggregation='mean', **kwargs):
        super().__init__(device)
        self.aggregator = TreeAggregator(aggregation)
        self.thresh_high = thresh_high
        self.thresh_low = thresh_low
        self.criterion = nn.MSELoss()
    
    def forward(self, 
                embeddings: torch.Tensor, 
                batch_info: 'PairedGroupBatchInfo') -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute entailment loss on text level
        
        Args:
            embeddings: Tree-level embeddings [n_trees, hidden_dim]
            batch_info: Batch information
            
        Returns:
            loss: Scalar loss value
            predictions: Predicted classes
            metrics: Dictionary of metrics
        """
        # 1. Aggregate tree embeddings into text embeddings
        text_embeddings = self.aggregator(embeddings, batch_info)
        
        # 2. Extract text pairs 
        n_groups = len(batch_info.group_indices)
        text_a_embeddings = text_embeddings[0::2]  # Even indices
        text_b_embeddings = text_embeddings[1::2]  # Odd indices
        
        # 3. Concatenate text embeddings for classification
        # pair_features = torch.cat([text_a_embeddings, text_b_embeddings], dim=1)
        sim_mat = F.cosine_similarity(text_a_embeddings, text_b_embeddings, dim=1)

        predictions = torch.where(sim_mat > self.thresh_high,
                                  torch.tensor(1, device=self.device),
                                  torch.where(sim_mat < self.thresh_low,
                                              torch.tensor(-1, device=self.device),
                                              torch.tensor(0, device=self.device)))
        
        # # 4. Convert labels (-1, 0, 1) to class indices (0, 1, 2)
        # labels = torch.tensor(
        #     [int(l + 1) for l in batch_info.group_labels], 
        #     device=self.device
        # ).long()
        target = torch.tensor(batch_info.group_labels, device=self.device).float()
        
        # 5. Compute classification logits and loss
        # logits = self.classifier(pair_features)
        loss = self.criterion(sim_mat, target)
        
        # 6. Get predictions and compute metrics
        # predictions = torch.argmax(logits, dim=1) - 1  # Convert back to (-1, 0, 1)
        
        with torch.no_grad():
            accuracy = (predictions == target.long()).float().mean().item()
            metrics = {
                'accuracy': accuracy,
            }
        
        return loss, predictions, metrics


class TextLevelBinaryLoss(BaseLoss):
    """Binary classification loss for text-level matching (e.g., patent matching)"""
    
    def __init__(self, device, threshold=0.5, aggregation='mean', temperature=0.2, **kwargs):
        super().__init__(device)
        self.aggregator = TreeAggregator(aggregation)
        self.threshold = threshold
        self.temperature = temperature
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, 
                embeddings: torch.Tensor, 
                batch_info: 'PairedGroupBatchInfo') -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute binary classification loss on text level
        
        Args:
            embeddings: Tree-level embeddings [n_trees, hidden_dim]
            batch_info: Batch information
            
        Returns:
            loss: Scalar loss value
            predictions: Binary predictions
            metrics: Dictionary of metrics
        """
        # 1. Aggregate tree embeddings into text embeddings
        text_embeddings = self.aggregator(embeddings, batch_info)
        
        # 2. Extract text pairs
        n_groups = len(batch_info.group_indices)
        text_a_embeddings = text_embeddings[0::2]  # Even indices
        text_b_embeddings = text_embeddings[1::2]  # Odd indices
        
        # 3. Compute similarity scores
        similarities = F.cosine_similarity(text_a_embeddings, text_b_embeddings)
        
        # 4. Normalize labels to binary values (0 or 1)
        binary_labels = torch.tensor(
            [(1 if l > self.threshold else 0) for l in batch_info.group_labels], 
            device=self.device
        ).float()
        
        # 5. Compute loss
        # Scale similarities from [-1,1] to logits
        scaled_similarities = similarities / self.temperature  # Scale factor for sharper logits
        loss = self.criterion(scaled_similarities, binary_labels)
        
        # 6. Get predictions and compute metrics
        predictions = (similarities > 0.0).float()  # Threshold at 0 for cosine similarity
        
        with torch.no_grad():
            metrics = {
                'accuracy': (predictions == binary_labels).float().mean().item(),
            }
        
        return loss, predictions, metrics

