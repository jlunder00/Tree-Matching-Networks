# GMN/tree_shape_positional_encoder.py
"""
Tree Shape-Based Positional Encoding with Dimension Partitioning

Each structural feature gets dedicated embedding dimensions for perfect orthogonality.
No signal dilution from averaging - features are concatenated instead.
"""

import torch
import torch.nn as nn
import math
from collections import defaultdict, deque


class TreeShapePositionalEncoder(nn.Module):
    """
    Positional encoding based on tree structural features using dimension partitioning.

    Each feature gets its own dedicated dimensions in the embedding space,
    ensuring perfect orthogonality and no interference between features.

    Available Features:
    - depth: Distance from root (0 for root, 1 for children, etc.)
    - num_siblings: Number of sibling nodes (same parent)
    - num_children: Number of direct children
    - num_grandparent_children: Number of parent's siblings (aunt/uncle nodes)
    - subtree_size: Total nodes in subtree rooted at this node
    - parent_num_children: How many children the parent has (context for this node)
    - distance_to_leaf: Depth of subtree below this node
    - nodes_at_level: Total nodes at this depth level across tree
    """

    # Default maximum values for each feature (for normalization/clamping)
    DEFAULT_MAX_VALUES = {
        'depth': 15,
        'num_siblings': 10,
        'num_children': 10,
        'num_grandparent_children': 10,
        'subtree_size': 40,
        'parent_num_children': 10,
        'distance_to_leaf': 10,
        'nodes_at_level': 20,
    }

    def __init__(self, embed_dim, max_values=None, features=None, learned=True):
        """
        Args:
            embed_dim: Total dimension of positional encoding (e.g., 768)
            max_values: Dict of maximum values for each feature (for clamping)
            features: List of feature names to use (None = use all available)
            learned: If True, use learned embeddings initialized with sinusoidal patterns
                    If False, compute sinusoidal patterns on-the-fly (no parameters)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.learned = learned

        # Use provided max values or defaults
        self.max_values = max_values or self.DEFAULT_MAX_VALUES.copy()

        # Select which features to use
        if features is None:
            self.features = list(self.DEFAULT_MAX_VALUES.keys())
        else:
            self.features = features
            # Ensure all requested features have max values
            for f in self.features:
                if f not in self.max_values:
                    self.max_values[f] = self.DEFAULT_MAX_VALUES.get(f, 20)

        self.num_features = len(self.features)

        # Partition dimensions among features
        base_dim = embed_dim // self.num_features
        remainder = embed_dim % self.num_features

        # Assign dimension ranges to each feature
        self.feature_dims = {}
        dim_offset = 0

        for i, feature_name in enumerate(self.features):
            # First 'remainder' features get one extra dimension
            dim_size = base_dim + (1 if i < remainder else 0)
            self.feature_dims[feature_name] = (dim_offset, dim_offset + dim_size)
            dim_offset += dim_size

        # Verify we used all dimensions
        assert dim_offset == embed_dim, f"Dimension mismatch: {dim_offset} != {embed_dim}"

        # CRITICAL FIX: Create learned embedding tables for each feature
        # Initialized with sinusoidal patterns, then made learnable
        if self.learned:
            self.feature_embeddings = nn.ModuleDict()

            for feature_name in self.features:
                max_val = self.max_values[feature_name]
                start_dim, end_dim = self.feature_dims[feature_name]
                feature_dim = end_dim - start_dim

                # Create sinusoidal initialization table
                sinusoidal_table = self._create_sinusoidal_table(max_val + 1, feature_dim)

                # Create learnable embedding initialized with sinusoidal patterns
                embedding = nn.Embedding(max_val + 1, feature_dim)
                embedding.weight.data = sinusoidal_table

                self.feature_embeddings[feature_name] = embedding

            print(f"TreeShapePositionalEncoder: Initialized with {self.num_features} LEARNED features")
            print(f"  Total positional encoding parameters: {sum(p.numel() for p in self.parameters()):,}")
        else:
            print(f"TreeShapePositionalEncoder: Using fixed sinusoidal encoding (no learnable params)")

    def forward(self, from_idx, to_idx, graph_idx, n_graphs):
        """
        Compute tree shape-based positional encodings.

        Args:
            from_idx: [n_edges] edge source indices
            to_idx: [n_edges] edge target indices
            graph_idx: [n_nodes] which graph each node belongs to
            n_graphs: int, number of graphs in batch

        Returns:
            pos_encodings: [n_nodes, embed_dim] with partitioned feature encodings
        """
        n_nodes = len(graph_idx)
        device = graph_idx.device

        # Compute all structural features
        feature_values = self._compute_tree_features(
            from_idx, to_idx, graph_idx, n_graphs
        )

        # Create full positional encoding tensor
        pos_encoding = torch.zeros(n_nodes, self.embed_dim, device=device)

        # Encode each feature in its dedicated dimension partition
        for feature_name in self.features:
            start_dim, end_dim = self.feature_dims[feature_name]
            feature_dim = end_dim - start_dim

            values = feature_values[feature_name]

            if self.learned:
                # Use LEARNED embeddings (lookup from embedding table)
                # Clamp values to valid range
                clamped_values = values.clamp(0, self.max_values[feature_name])
                feature_enc = self.feature_embeddings[feature_name](clamped_values)
            else:
                # Compute fixed sinusoidal encoding on-the-fly
                feature_enc = self._sinusoidal_encoding(
                    values,
                    feature_dim,
                    self.max_values[feature_name]
                )

            # Place in correct partition (concatenation)
            pos_encoding[:, start_dim:end_dim] = feature_enc

        return pos_encoding

    def _create_sinusoidal_table(self, num_positions, feature_dim):
        """
        Create a lookup table of sinusoidal positional encodings.

        This is used to initialize learned embeddings with sinusoidal patterns.

        Args:
            num_positions: Number of positions (e.g., max_value + 1)
            feature_dim: Dimension of encoding for this feature

        Returns:
            table: [num_positions, feature_dim] tensor of sinusoidal patterns
        """
        # Create position indices [0, 1, 2, ..., num_positions-1]
        position = torch.arange(num_positions, dtype=torch.float).unsqueeze(1)

        # Create frequency terms
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, dtype=torch.float) *
            -(math.log(10000.0) / max(feature_dim, 2))  # Avoid division by 0
        )

        # Create table
        pe = torch.zeros(num_positions, feature_dim)

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices (if feature_dim > 1)
        if feature_dim > 1:
            # Handle odd feature_dim (cos gets one less dimension)
            cos_dims = feature_dim // 2
            if cos_dims > 0:
                pe[:, 1::2] = torch.cos(position * div_term[:cos_dims])

        return pe

    def _sinusoidal_encoding(self, values, feature_dim, max_value):
        """
        Compute sinusoidal positional encoding for a specific feature.

        Args:
            values: [n_nodes] integer feature values
            feature_dim: number of dimensions allocated to this feature
            max_value: maximum expected value (for normalization)

        Returns:
            encodings: [n_nodes, feature_dim]
        """
        device = values.device

        # Normalize to [0, 1] range
        normalized = values.float() / max(max_value, 1.0)
        normalized = normalized.clamp(0.0, 1.0)  # Ensure valid range

        # Compute sinusoidal encoding
        position = normalized.unsqueeze(1)  # [n_nodes, 1]

        # Create frequency terms for this feature's dimensions
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, device=device).float() *
            -(math.log(10000.0) / max(feature_dim, 2))  # Avoid division by 0
        )

        pe = torch.zeros(len(values), feature_dim, device=device)

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices (if feature_dim > 1)
        if feature_dim > 1:
            # Handle odd feature_dim (cos gets one less dimension)
            cos_dims = feature_dim // 2
            if cos_dims > 0:
                pe[:, 1::2] = torch.cos(position * div_term[:cos_dims])

        return pe

    def _compute_tree_features(self, from_idx, to_idx, graph_idx, n_graphs):
        """
        Compute all structural features for each node.

        Returns:
            Dict mapping feature name to tensor of values [n_nodes]
        """
        n_nodes = len(graph_idx)
        device = graph_idx.device

        # Initialize all possible feature tensors
        features = {
            'depth': torch.zeros(n_nodes, dtype=torch.long, device=device),
            'num_siblings': torch.zeros(n_nodes, dtype=torch.long, device=device),
            'num_children': torch.zeros(n_nodes, dtype=torch.long, device=device),
            'num_grandparent_children': torch.zeros(n_nodes, dtype=torch.long, device=device),
            'subtree_size': torch.zeros(n_nodes, dtype=torch.long, device=device),
            'parent_num_children': torch.zeros(n_nodes, dtype=torch.long, device=device),
            'distance_to_leaf': torch.zeros(n_nodes, dtype=torch.long, device=device),
            'nodes_at_level': torch.zeros(n_nodes, dtype=torch.long, device=device),
        }

        # Process each graph separately
        for g in range(n_graphs):
            node_mask = (graph_idx == g)  # Boolean mask for this graph's nodes
            graph_nodes = torch.where(node_mask)[0].tolist()
            # edge_nodes = to_idx

            if not graph_nodes:
                continue

            # BUILD PER-GRAPH MAPS (CRITICAL FIX: prevents cross-graph contamination)
            # Filter edges using tensor operations to get only edges where BOTH endpoints
            # belong to this graph. This is much more efficient than iterating all edges.
            #
            # from_idx and to_idx contain POSITIONS in the batch tensor
            # graph_idx[position] tells us which graph that position belongs to
            # We want edges where graph_idx[from_pos] == g AND graph_idx[to_pos] == g
            edge_mask = (graph_idx[from_idx] == g) & (graph_idx[to_idx] == g)

            # Extract only this graph's edges
            graph_from_idx = from_idx[edge_mask]
            graph_to_idx = to_idx[edge_mask]

            # Convert to lists for building adjacency maps
            graph_from_list = graph_from_idx.tolist()
            graph_to_list = graph_to_idx.tolist()

            # Build adjacency structures using ONLY this graph's edges
            children_map = defaultdict(list)  # parent_position -> [child_positions]
            parent_map = {}  # child_position -> parent_position

            for parent, child in zip(graph_from_list, graph_to_list):
                children_map[parent].append(child)
                parent_map[child] = parent

            # Find root (node with no parent in this graph)
            roots = [n for n in graph_nodes
                    if n not in parent_map or parent_map[n] not in graph_nodes]
            root = roots[0] if roots else graph_nodes[0]

            # BFS to compute depths and build tree structure
            queue = deque([(root, 0, None)])  # (node, depth, parent)
            visited = {root}
            depth_map = {}

            while queue:
                node, depth, parent = queue.popleft()

                depth_map[node] = depth

                # Store depth (clamped to max)
                features['depth'][node] = min(depth, self.max_values.get('depth', 15))

                # Get children for this node
                node_children = [c for c in children_map[node] if c in graph_nodes]
                num_children = len(node_children)

                # Store number of children
                features['num_children'][node] = min(
                    num_children,
                    self.max_values.get('num_children', 10)
                )

                # Store parent's number of children (context)
                if parent is not None:
                    parent_children_count = len([c for c in children_map[parent]
                                                if c in graph_nodes])
                    features['parent_num_children'][node] = min(
                        parent_children_count,
                        self.max_values.get('parent_num_children', 10)
                    )

                # Store number of siblings
                if parent is not None:
                    siblings = [c for c in children_map[parent]
                               if c != node and c in graph_nodes]
                    features['num_siblings'][node] = min(
                        len(siblings),
                        self.max_values.get('num_siblings', 10)
                    )

                # Store number of grandparent's children (aunt/uncle nodes)
                if parent is not None and parent in parent_map:
                    grandparent = parent_map[parent]
                    if grandparent in graph_nodes:
                        aunts_uncles = [c for c in children_map[grandparent]
                                       if c != parent and c in graph_nodes]
                        features['num_grandparent_children'][node] = min(
                            len(aunts_uncles),
                            self.max_values.get('num_grandparent_children', 10)
                        )

                # Add children to queue
                for child in node_children:
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, depth + 1, node))

            # Compute nodes at each depth level
            depth_counts = defaultdict(int)
            for node in graph_nodes:
                d = depth_map.get(node, 0)
                depth_counts[d] += 1

            for node in graph_nodes:
                d = depth_map.get(node, 0)
                features['nodes_at_level'][node] = min(
                    depth_counts[d],
                    self.max_values.get('nodes_at_level', 20)
                )

            # Compute subtree sizes via DFS from each node
            for node in graph_nodes:
                subtree_sz = self._compute_subtree_size(node, children_map, graph_nodes)
                features['subtree_size'][node] = min(
                    subtree_sz,
                    self.max_values.get('subtree_size', 40)
                )

            # Compute distance to nearest leaf
            for node in graph_nodes:
                dist = self._distance_to_leaf(node, children_map, graph_nodes)
                features['distance_to_leaf'][node] = min(
                    dist,
                    self.max_values.get('distance_to_leaf', 10)
                )

        return features

    def _compute_subtree_size(self, root, children_map, valid_nodes):
        """Count nodes in subtree rooted at root."""
        count = 1  # Count self
        stack = [root]
        visited = {root}

        while stack:
            node = stack.pop()
            for child in children_map[node]:
                if child in valid_nodes and child not in visited:
                    visited.add(child)
                    count += 1
                    stack.append(child)

        return count

    def _distance_to_leaf(self, node, children_map, valid_nodes):
        """Compute distance from node to nearest leaf (DFS)."""
        children = [c for c in children_map[node] if c in valid_nodes]

        if not children:
            return 0  # This node is a leaf

        # Distance is 1 + min distance of children
        child_distances = [
            self._distance_to_leaf(c, children_map, valid_nodes)
            for c in children
        ]
        return 1 + min(child_distances)

    def get_feature_info(self):
        """Return information about feature partitioning (for debugging/analysis)."""
        info = {
            'embed_dim': self.embed_dim,
            'num_features': self.num_features,
            'features': self.features,
            'partitions': {}
        }

        for feature_name in self.features:
            start, end = self.feature_dims[feature_name]
            info['partitions'][feature_name] = {
                'dims': (start, end),
                'size': end - start,
                'max_value': self.max_values[feature_name]
            }

        return info

    def get_root_indices(self, from_idx, to_idx, graph_idx, n_graphs):
        """
        Identify the root node index for each graph in the batch.

        The root is defined as the node with no parent (not a target of any edge).
        In dependency trees, this is typically the main verb or sentence root.

        Args:
            from_idx: [n_edges] edge source indices (parents)
            to_idx: [n_edges] edge target indices (children)
            graph_idx: [n_nodes] which graph each node belongs to
            n_graphs: int, number of graphs in batch

        Returns:
            root_indices: [n_graphs] tensor of root node indices (global batch indices)
        """
        device = graph_idx.device
        root_indices = torch.zeros(n_graphs, dtype=torch.long, device=device)

        for g in range(n_graphs):
            node_mask = (graph_idx == g)
            graph_nodes = torch.where(node_mask)[0].tolist()

            if not graph_nodes:
                continue

            # Filter edges to only those within this graph
            edge_mask = (graph_idx[from_idx] == g) & (graph_idx[to_idx] == g)
            graph_to_idx = to_idx[edge_mask].tolist()

            # Build set of nodes that are children (have incoming edges)
            children_set = set(graph_to_idx)

            # Root is a node with no parent (not in children_set)
            roots = [n for n in graph_nodes if n not in children_set]
            root_indices[g] = roots[0] if roots else graph_nodes[0]

        return root_indices
