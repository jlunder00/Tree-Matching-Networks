# LinguisticTrees/tree_matching/dataset.py
from COMMON.src.dataset.base_dataset import BaseDataset
import json
import torch

class TreeMatchingDataset(BaseDataset):
    def __init__(self, data_path, length=None, cls=None):
        super().__init__()
        with open(data_path) as f:
            self.data = json.load(f)
        self.length = length or len(self.data['labels'])

    def get_pair(self, idx, cls):
        """Get a pair of graphs and their label"""
        graph_pair = self.data['graph_pairs'][idx]
        label = self.data['labels'][idx]
        
        graph1, graph2 = graph_pair
        
        # Convert lists back to tensors
        for graph in (graph1, graph2):
            for key in graph:
                if key != 'n_graphs':
                    graph[key] = torch.tensor(graph[key])
                
        return graph_pair, label







# # dataset.py
# import torch
# import numpy as np
# from typing import List, Tuple, Iterator
# import random
# from collections import namedtuple
# from TMN_DataGen import DependencyTree

# GraphData = namedtuple('GraphData', [
#     'from_idx',
#     'to_idx',
#     'node_features',
#     'edge_features',
#     'graph_idx',
#     'n_graphs'
# ])

# class EntailmentGraphDataset:
#     def __init__(self, sentence_pairs: List[Tuple[str, str]], 
#                  labels: List[str]):
#         """
#         Args:
#             sentence_pairs: List of (premise, hypothesis) sentence pairs
#             labels: List of entailment labels ('entails', 'contradicts', 'neutral')
#         """
#         self.parser = Parser.load('en')  # Load English model
#         self.trees = []
#         self.labels = []
#         
#         # Convert labels to numeric values
#         self.label_map = {
#             'entails': 1,
#             'contradicts': -1,
#             'neutral': 0
#         }
#         
#         for (premise, hypothesis), label in zip(sentence_pairs, labels):
#             # Parse sentences
#             premise_parse = self.parser.parse(premise)
#             hyp_parse = self.parser.parse(hypothesis)
#             
#             # Create trees
#             premise_tree = DependencyTree.from_diaparser_output(premise, premise_parse)
#             hyp_tree = DependencyTree.from_diaparser_output(hypothesis, hyp_parse)
#             
#             self.trees.append((premise_tree, hyp_tree))
#             self.labels.append(self.label_map[label])
#     
#     def pairs(self, batch_size: int) -> Iterator[Tuple[GraphData, np.ndarray]]:
#         """Generator for batches of graph pairs and labels"""
#         indices = list(range(len(self.trees)))
#         
#         while True:
#             random.shuffle(indices)
#             for i in range(0, len(indices), batch_size):
#                 batch_indices = indices[i:i+batch_size]
#                 
#                 batch_graphs = []
#                 batch_labels = []
#                 
#                 for idx in batch_indices:
#                     premise_tree, hyp_tree = self.trees[idx]
#                     
#                     # Convert trees to graph format
#                     premise_graph = premise_tree.to_graph_data()
#                     hyp_graph = hyp_tree.to_graph_data()
#                     
#                     batch_graphs.extend([premise_graph, hyp_graph])
#                     batch_labels.append(self.labels[idx])
#                 
#                 # Combine graphs into single GraphData object
#                 combined_graph = self._combine_graphs(batch_graphs)
#                 yield combined_graph, np.array(batch_labels)
#     
#     def _combine_graphs(self, graphs: List[Dict]) -> GraphData:
#         """Combine multiple graphs into a single GraphData object"""
#         # Implement graph combination logic here
#         # This should match the format expected by the GMN code
#         pass


