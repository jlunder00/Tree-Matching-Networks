# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# scripts/validate_data_files.py
# Old script used on old version of data to check validity, not relevant now
import json
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import shutil

logger = logging.getLogger(__name__)

def validate_file(json_path: Path):
    """Validate a single JSON file"""
    try:
        # Test JSON loading
        with open(json_path) as f:
            data = json.load(f)
            n_pairs = len(data['labels'])
            # Quick validation of expected structure
            if 'graph_pairs' not in data or len(data['graph_pairs']) != n_pairs:
                raise ValueError("Invalid data structure")
            
        return json_path.name, True, n_pairs, None
        
    except Exception as e:
        logger.error(f"Invalid file {json_path}: {e}")
        bad_files_dir = json_path.parent / "bad_files"
        bad_files_dir.mkdir(parents=True, exist_ok=True)
        
        # Move bad file
        bad_file_path = bad_files_dir / json_path.name
        shutil.move(str(json_path), str(bad_file_path))
        
        # Also move count file if it exists
        count_file = json_path.parent / f"{json_path.stem}_pair_count.json"
        if count_file.exists():
            bad_count_path = bad_files_dir / count_file.name
            shutil.move(str(count_file), str(bad_count_path))
            
        return json_path.name, False, 0, str(e)

def validate_directory(data_dir: str, recursive: bool = False, num_workers: int = None):
    """Validate all data files in directory"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Directory not found: {data_dir}")
    
    # Get all data files
    pattern = "**/part_*.json" if recursive else "part_*.json"
    json_files = [
        f for f in data_dir.glob(pattern) 
        if not f.name.endswith('_pair_count.json')
        and not f.parent.name == 'bad_files'  # Skip already marked bad files
    ]
    
    logger.info(f"Found {len(json_files)} files to validate")
    
    # Use maximum multiprocessing
    num_workers = num_workers or mp.cpu_count()
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(validate_file, json_files),
            total=len(json_files),
            desc="Validating files"
        ))
    
    # Summarize results
    valid_files = [(name, pairs) for name, valid, pairs, _ in results if valid]
    invalid_files = [(name, error) for name, valid, _, error in results if not valid]
    
    total_pairs = sum(pairs for _, pairs in valid_files)
    
    logger.info("\nValidation Summary:")
    logger.info(f"Total files: {len(json_files)}")
    logger.info(f"Valid files: {len(valid_files)}")
    logger.info(f"Invalid files: {len(invalid_files)}")
    logger.info(f"Total valid pairs: {total_pairs}")
    
    if invalid_files:
        logger.warning("\nInvalid files moved to bad_files/:")
        for name, error in invalid_files:
            logger.warning(f"  {name}: {error}")

def main():
    parser = argparse.ArgumentParser(description='Validate data files')
    parser.add_argument('data_dir', type=str,
                      help='Directory containing data files')
    parser.add_argument('--recursive', action='store_true',
                      help='Search directories recursively')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable debug logging')
                      
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        validate_directory(
            args.data_dir,
            args.recursive,
            args.workers
        )
        logger.info("Validation complete")
    except Exception as e:
        logger.exception("Validation failed:")
        raise

if __name__ == '__main__':
    main()
