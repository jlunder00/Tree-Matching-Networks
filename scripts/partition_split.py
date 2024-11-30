# scripts/partition_split.py
import json
from pathlib import Path
import math

def split_partition_file(
    partition_file: Path,
    output_dir: Path, 
    target_size: int = 1000,  # Aim for 1000 pairs per shard
    prefix: str = None
):
    """Split large partition file into smaller shards"""
    with open(partition_file) as f:
        data = json.load(f)
    
    n_pairs = len(data['labels'])
    n_shards = math.ceil(n_pairs / target_size)
    
    prefix = prefix or partition_file.stem
    
    # Split into shards
    for i in range(n_shards):
        start_idx = i * target_size
        end_idx = min((i + 1) * target_size, n_pairs)
        
        shard_data = {
            'graph_pairs': data['graph_pairs'][start_idx:end_idx],
            'labels': data['labels'][start_idx:end_idx]
        }
        
        shard_file = output_dir / f"{prefix}_shard_{i:03d}.json"
        with open(shard_file, 'w') as f:
            json.dump(shard_data, f)
            
        # Save pair count
        count_file = output_dir / f"{prefix}_shard_{i:03d}_pair_count.json"
        with open(count_file, 'w') as f:
            json.dump({'n_pairs': end_idx - start_idx}, f)
