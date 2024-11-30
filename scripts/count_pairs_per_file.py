# scripts/count_pairs_per_file.py
import json
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import shutil

logger = logging.getLogger(__name__)

def count_pairs_in_file(json_path: Path):
    """Count pairs in a single file"""
    try:
        # Load and count
        with open(json_path) as f:
            logger.debug(f'Opening {json_path}')
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Handle JSON decoding errors
                raise ValueError(f"Invalid JSON in file: {json_path}")
            
            logger.debug(f'Opened {json_path}')
            n_pairs = len(data['labels'])
        
        # Save count
        count_path = json_path.parent / f"{json_path.stem}_pair_count.json"
        with open(count_path, 'w') as f:
            logger.debug(f'Saving {count_path}')
            json.dump({
                'source_file': json_path.name,
                'n_pairs': n_pairs
            }, f)
            logger.debug(f'Saved {count_path}')
        
        return json_path.name, n_pairs
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        # Move problematic file and its associated count file
        try:
            bad_files_dir = json_path.parent / "bad_files"
            bad_files_dir.mkdir(parents=True, exist_ok=True)
            
            # Move the problematic file
            bad_file_path = bad_files_dir / json_path.name
            shutil.move(str(json_path), str(bad_file_path))
            logger.warning(f"Moved bad file to: {bad_file_path}")
            
            # Check and move the count file if it exists
            count_path = json_path.parent / f"{json_path.stem}_pair_count.json"
            if count_path.exists():
                bad_count_path = bad_files_dir / count_path.name
                shutil.move(str(count_path), str(bad_count_path))
                logger.warning(f"Moved count file to: {bad_count_path}")
        except Exception as move_error:
            logger.error(f"Failed to move bad file {json_path} and its count file: {move_error}")
        
        return json_path.name, None


# def count_pairs_in_file(json_path: Path):
#     """Count pairs in a single file"""
#     try:
#         # Load and count
#         with open(json_path) as f:
#             logger.debug(f'opening {f}')
#             try:
#                 data = json.load(f)
#             except:
#                 #todo: add saving to file for bad json files and their count file if it exists

#             logger.debug(f'opened {f}')
#             n_pairs = len(data['labels'])
#             
#         # Save count
#         count_path = json_path.parent / f"{json_path.stem}_pair_count.json"
#         with open(count_path, 'w') as f:
#             logger.debug(f'saving {count_path}')
#             json.dump({
#                 'source_file': json_path.name,
#                 'n_pairs': n_pairs
#             }, f)
#             logger.debug(f'saved {count_path}')
#             
#         return json_path.name, n_pairs
#         
#     except Exception as e:
#         logger.error(f"Error processing {json_path}: {e}")
#         return json_path.name, None

def count_pairs_in_directory(data_dir, recursive: bool = False, num_workers: int = None, overwrite = False, shard=False):
    """Count pairs in all data files in directory using multiprocessing"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Directory not found: {data_dir}")
        
    # Get all json files
    if not shard:
        pattern = "**/part_[0-9]*.json" if recursive else "part_[0-9]*.json"
    else:
        pattern = "**/part_[0-9]*_shard_[0-9]*.json" if recursive else "part_[0-9]*_shard_[0-9]*.json"
    json_files = [f for f in data_dir.glob(pattern) if not f.name.endswith('_pair_count.json')]
    if not overwrite:
        # Filter out files that already have count files
        json_files = [f for f in json_files
                     if not (f.parent / f"{f.stem}_pair_count.json").exists()]
    if not json_files:
       logger.info("No new files to process")
       return
    
    logger.info(f"Found {len(json_files)} json files")
    
    # Use multiprocessing
    num_workers = num_workers or max(1, mp.cpu_count() - 1)
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(count_pairs_in_file, json_files),
            total=len(json_files),
            desc="Counting pairs"
        ))
    
    # Summarize results
    successful = [(name, count) for name, count in results if count is not None]
    failed = [name for name, count in results if count is None]
    
    logger.info(f"Successfully processed {len(successful)} files")
    if failed:
        logger.warning(f"Failed to process {len(failed)} files")
        for fname in failed:
            logger.warning(f"  {fname}")

def main():
    parser = argparse.ArgumentParser(description='Count pairs in data files')
    parser.add_argument('data_dir', type=str,
                      help='Directory containing json data files')
    parser.add_argument('--recursive', action='store_true',
                      help='Search directories recursively')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite existing count files')
    parser.add_argument('--shard', action='store_true',
                      help='using shards or not')
                      
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        count_pairs_in_directory(
            args.data_dir, 
            args.recursive,
            args.workers,
            overwrite=args.overwrite,
            shard = args.shard
        )
        logger.info("Processing complete")
    except Exception as e:
        logger.exception("Processing failed:")
        raise

if __name__ == '__main__':
    main()
