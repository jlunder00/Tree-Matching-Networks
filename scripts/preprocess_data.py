# scripts/preprocess_data.py (renamed from count_pairs_per_file.py)
import json
from pathlib import Path
import argparse
import logging
import math
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)

def split_and_count_file(json_path: Path, target_size: int = 1000, output_dir: Path = None):
    """Split file into smaller shards and count pairs"""
    try:
        # Load data
        with open(json_path) as f:
            logger.debug(f'Opening {f}')
            data = json.load(f)
            logger.debug(f'Opened {f}')
            n_pairs = len(data['labels'])

        # Save count for original file
        count_path = json_path.parent / f"{json_path.stem}_pair_count.json"
        with open(count_path, 'w') as f:
            json.dump({
                'source_file': json_path.name,
                'n_pairs': n_pairs
            }, f)

        # Split into shards if requested
        if output_dir and target_size:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            n_shards = math.ceil(n_pairs / target_size)
            logger.debug(f"Splitting {json_path.name} into {n_shards} shards")
            
            for i in range(n_shards):
                start_idx = i * target_size
                end_idx = min((i + 1) * target_size, n_pairs)
                
                shard_data = {
                    'graph_pairs': data['graph_pairs'][start_idx:end_idx],
                    'labels': data['labels'][start_idx:end_idx]
                }
                
                # Save shard
                shard_file = output_dir / f"{json_path.stem}_shard_{i:03d}.json"
                with open(shard_file, 'w') as f:
                    json.dump(shard_data, f)
                    
                # Save shard count
                shard_count = output_dir / f"{json_path.stem}_shard_{i:03d}_pair_count.json"
                with open(shard_count, 'w') as f:
                    json.dump({
                        'source_file': json_path.name,
                        'shard_index': i,
                        'n_pairs': end_idx - start_idx
                    }, f)
                    
            return json_path.name, n_pairs, n_shards
            
        return json_path.name, n_pairs, 0

    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return json_path.name, None, 0

def process_directory(data_dir, 
                     recursive: bool = False,
                     num_workers: int = None,
                     shard_size: int = None,
                     output_dir: str = None,
                     overwrite: bool = False):
    """Process all data files in directory"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Directory not found: {data_dir}")
        
    # Get all json files
    pattern = "**/part_[0-9]*.json" if recursive else "part_[0-9]*.json"
    json_files = [f for f in data_dir.glob(pattern) if not f.name.endswith('_pair_count.json')]
    if not overwrite:
        # Filter out files that already have count files
        json_files = [f for f in json_files
                     if not (f.parent / f"{f.stem}_pair_count.json").exists()]
        if not json_files:
            logger.info("No new files to process")
            return

    logger.info(f"Found {len(json_files)} json files to process")
    
    # Create partial function with shard settings
    process_func = partial(
        split_and_count_file,
        target_size=shard_size,
        output_dir=output_dir
    )
    
    # Use multiprocessing
    num_workers = num_workers or max(1, mp.cpu_count() - 1)
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, json_files),
            total=len(json_files),
            desc="Processing files"
        ))

    # Summarize results
    successful = [(name, count, shards) for name, count, shards in results if count is not None]
    failed = [name for name, count, _ in results if count is None]
    
    total_pairs = sum(count for _, count, _ in successful)
    total_shards = sum(shards for _, _, shards in successful)
    
    logger.info(f"\nProcessing Summary:")
    logger.info(f"Successfully processed {len(successful)} files")
    logger.info(f"Total pairs: {total_pairs}")
    if total_shards > 0:
        logger.info(f"Created {total_shards} shards")
    
    if failed:
        logger.warning(f"\nFailed to process {len(failed)} files:")
        for fname in failed:
            logger.warning(f"  {fname}")

def main():
    parser = argparse.ArgumentParser(description='Process data files (count and optionally shard)')
    parser.add_argument('data_dir', type=str,
                      help='Directory containing json data files')
    parser.add_argument('--recursive', action='store_true',
                      help='Search directories recursively')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes')
    parser.add_argument('--shard-size', type=int,
                      help='Target number of pairs per shard (if sharding)')
    parser.add_argument('--output-dir', type=str,
                      help='Output directory for shards (if sharding)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite existing files')
                      
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        process_directory(
            args.data_dir,
            args.recursive,
            args.workers,
            shard_size=args.shard_size,
            output_dir=args.output_dir,
            overwrite=args.overwrite
        )
        logger.info("Processing complete")
    except Exception as e:
        logger.exception("Processing failed:")
        raise

if __name__ == '__main__':
    main()
