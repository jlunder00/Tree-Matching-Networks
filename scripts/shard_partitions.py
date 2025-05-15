# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

# scripts/shard_partitions.py
#old script, functionality has been subsumed by TMN_DataGen
import json
from pathlib import Path
import argparse
import logging
import math
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)

def shard_partition_file(json_path: Path, output_dir: Path, target_size: int):
    """Split a partition file into smaller shards"""
    try:
        # Load data
        with open(json_path) as f:
            logger.debug(f'Opening {f}')
            data = json.load(f)
            n_pairs = len(data['labels'])

        # Calculate number of shards
        n_shards = math.ceil(n_pairs / target_size)
        logger.debug(f"Splitting {json_path.name} into {n_shards} shards")
        
        # Create shards
        for i in range(n_shards):
            start_idx = i * target_size
            end_idx = min((i + 1) * target_size, n_pairs)
            
            shard_data = {
                'graph_pairs': data['graph_pairs'][start_idx:end_idx],
                'labels': data['labels'][start_idx:end_idx]
            }
            
            # Save shard with original partition number preserved
            partition_num = json_path.stem.split('_')[1]  # Extract number from 'part_X'
            shard_file = output_dir / f"part_{partition_num}_shard_{i:03d}.json"
            with open(shard_file, 'w') as f:
                json.dump(shard_data, f)
                
            # Save shard count file
            count_file = output_dir / f"part_{partition_num}_shard_{i:03d}_pair_count.json"
            with open(count_file, 'w') as f:
                json.dump({
                    'source_file': json_path.name,
                    'shard_index': i,
                    'n_pairs': end_idx - start_idx
                }, f)
                
        return json_path.name, n_shards
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return json_path.name, None

def is_partition_sharded(partition_path: Path, input_dir: Path, output_dir: Path, shard_size: int) -> bool:
    """Check if a partition is already sharded correctly"""
    try:
        partition_num = partition_path.stem.split('_')[1]
        count_file = input_dir / f"part_{partition_num}_pair_count.json"
        shard_file_pattern =f"part_{partition_num}_shard_[0-9][0-9][0-9].json"
        shard_files = list(output_dir.glob(shard_file_pattern))
        if output_dir / f"part_{partition_num}_shard_010.json" in shard_files:
            return True

        if not count_file.exists():
            return False  # Missing total pair count file
        
        # Load the total number of pairs from the count file
        with open(count_file) as f:
            total_pairs = json.load(f)['n_pairs']
        
        # Calculate the expected number of shards
        expected_shards = math.floor(total_pairs / shard_size)
         
        if len(shard_files) < expected_shards:
            return False
        
        # Check if all expected shard files and their count files exist
        # for i in range(expected_shards):
        #     shard_file = output_dir / f"part_{partition_num}_shard_{i:03d}.json"
        #     shard_count_file = output_dir / f"part_{partition_num}_shard_{i:03d}_pair_count.json"
        #     if not shard_file.exists() or not shard_count_file.exists():
        #         return False
        
        return True  # All shards and count files exist
    except Exception as e:
        logger.error(f"Error checking sharding status for {partition_path}: {e}")
        return False

def shard_directory(input_dir: str,
                   output_dir: str,
                   shard_size: int,
                   recursive: bool = False,
                   num_workers: int = None,
                   overwrite: bool = False):
    """Process all partition files in directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
        
    # Get partition files
    pattern = "**/part_[0-9]*.json" if recursive else "part_[0-9]*.json"
    partition_files = [
        f for f in input_dir.glob(pattern)
        if not f.name.endswith('_pair_count.json')
        and not '_shard_' in f.name
    ]

    if not partition_files:
        logger.info("No partition files found to shard")
        return

    logger.info(f"Found {len(partition_files)} partition files to shard")
    
    # Filter out already sharded partitions
    if not overwrite:
        partition_files = [
            f for f in partition_files
            if not is_partition_sharded(f, input_dir, output_dir, shard_size)
        ]
        logger.info(f"Skipping already sharded partitions. {len(partition_files)} files left to process.")

    # Create partial function with output dir
    process_func = partial(
        shard_partition_file,
        output_dir=output_dir,
        target_size=shard_size
    )
    
    # Use multiprocessing
    num_workers = num_workers or max(1, mp.cpu_count() - 1)
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, partition_files),
            total=len(partition_files),
            desc="Sharding files"
        ))

    # Summarize results
    successful = [(name, shards) for name, shards in results if shards is not None]
    failed = [name for name, shards in results if shards is None]
    
    total_shards = sum(shards for _, shards in successful)
    
    logger.info(f"\nSharding Summary:")
    logger.info(f"Successfully processed {len(successful)} partition files")
    logger.info(f"Created {total_shards} total shards")
    
    if failed:
        logger.warning(f"\nFailed to process {len(failed)} files:")
        for fname in failed:
            logger.warning(f"  {fname}")

def main():
    parser = argparse.ArgumentParser(description='Shard partition files into smaller chunks')
    parser.add_argument('input_dir', type=str,
                      help='Directory containing partition files')
    parser.add_argument('output_dir', type=str,
                      help='Output directory for shards')
    parser.add_argument('--shard-size', type=int, default=1000,
                      help='Target number of pairs per shard')
    parser.add_argument('--recursive', action='store_true',
                      help='Search directories recursively')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes')
    parser.add_argument('--overwrite', action='store_true',
                      help='Force re-sharding even if already sharded')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable debug logging')
                      
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        shard_directory(
            args.input_dir,
            args.output_dir, 
            args.shard_size,
            args.recursive,
            args.workers,
            args.overwrite
        )
        logger.info("Sharding complete")
    except Exception as e:
        logger.exception("Sharding failed:")
        raise

if __name__ == '__main__':
    main()



