try:
    from .grouped_tree_dataset import GroupedTreeDataset
except:
    from grouped_tree_dataset import GroupedTreeDataset
import torch

config = {
    'word_embedding_model': 'bert-base-uncased',
    'use_gpu': True,
    'cache_embeddings': True,
    'embedding_cache_dir': '/home/jlunder/research/TMN_DataGen/embedding_cache3/',
    'do_not_store_word_embeddings': False,
    'is_runtime': True,
    'shard_size': 10000,
    'num_workers': 4
}

dataset = GroupedTreeDataset(data_path='/home/jlunder/research/data/wikiqs/dest3/test_output_shard/shard_000000.json', config=config)


train_loader = dataset.get_dataloader(
    batch_size=32,
    pos_pairs_per_anchor=2,
    neg_pairs_per_anchor=4,
    min_groups_per_batch=4,
    anchors_per_group=2
)

for graphs, batch_info in train_loader:
    
    print(batch_info)
    print(graphs)
