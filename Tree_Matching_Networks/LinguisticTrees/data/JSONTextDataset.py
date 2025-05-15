import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Iterator
import os
from tqdm import tqdm

class JSONTextDataset(Dataset):
    """Dataset to extract text fields from JSON files for tokenizer training"""
    
    def __init__(self, data_files: List[Path]):
        """
        Args:
            data_files: List of Path objects pointing to JSON files
        """
        self.data_files = data_files
        self.file_lengths = self._count_samples()
        self.cumulative_lengths = self._get_cumulative_lengths()
        
    def _count_samples(self) -> List[int]:
        """Count samples in each file for efficient indexing"""
        lengths = []
        print("Counting samples in files...")
        for file_path in tqdm(self.data_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Count number of text entries in groups
                count = sum(1 for group in data.get('groups', []) if 'text' in group)
                lengths.append(count)
        return lengths
        
    def _get_cumulative_lengths(self) -> List[int]:
        """Calculate cumulative lengths for file indexing"""
        cum_lengths = [0]
        for length in self.file_lengths:
            cum_lengths.append(cum_lengths[-1] + length)
        return cum_lengths
    
    def _get_file_and_idx(self, idx: int):
        """Get file and local index from global index"""
        # Find which file this index belongs to
        file_idx = 0
        while file_idx < len(self.cumulative_lengths) - 1 and idx >= self.cumulative_lengths[file_idx + 1]:
            file_idx += 1
        
        # Calculate local index within the file
        local_idx = idx - self.cumulative_lengths[file_idx]
        return file_idx, local_idx
    
    def __len__(self) -> int:
        """Total number of text samples across all files"""
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx: int) -> str:
        """Get text sample at the given index"""
        file_idx, local_idx = self._get_file_and_idx(idx)
        file_path = self.data_files[file_idx]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Extract valid text samples
            text_samples = [group['text'] for group in data.get('groups', []) 
                           if 'text' in group]
            
            if local_idx >= len(text_samples):
                return ""  # Fallback for index errors
            
            return text_samples[local_idx]

class StreamingJSONTextDataset:
    """Memory-efficient streaming dataset that doesn't precompute indices"""
    
    def __init__(self, data_files: List[Path]):
        self.data_files = data_files
    
    def __iter__(self) -> Iterator[str]:
        """Iterate through all files and yield text fields"""
        for file_path in self.data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract and yield text from groups
                    for group in data.get('groups', []):
                        if 'text' in group and group['text']:
                            yield group['text']
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue


def gather_json_files(data_dirs: List[Path]) -> List[Path]:
    """Gather JSON files from the given directories"""
    data_files = []
    for data_dir in data_dirs:
        files = sorted(
            [f for f in data_dir.glob("part_*_shard_*.json") 
             if not f.name.endswith('_counts.json')],
            key=lambda x: (int(x.stem.split('_')[1]), 
                         int(x.stem.split('_shard_')[1]))
        )
        if not files:
            files = sorted(
                [f for f in data_dir.glob("part_*.json")
                 if not f.name.endswith('_counts.json')],
                key=lambda x: int(x.stem.split('_')[1])
            )
        data_files.extend(files)
    return data_files


def train_tokenizer_from_json_files(data_dirs: List[str], 
                                   vocab_size: int = 1200, 
                                   min_frequency: int = 5,
                                   batch_size: int = 1000,
                                   max_samples: int = None):
    """Train a tokenizer from JSON files containing text data"""
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace
    
    # Convert string paths to Path objects
    data_dirs = [Path(dir_path) for dir_path in data_dirs]
    
    # Gather files
    print(f"Gathering files from {len(data_dirs)} directories...")
    data_files = gather_json_files(data_dirs)
    print(f"Found {len(data_files)} files")
    
    # Create tokenizer components
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
    )
    
    # Create streaming dataset for memory efficiency
    dataset = StreamingJSONTextDataset(data_files)
    
    # Create a batched iterator for efficiency
    def batched_iterator():
        sample_count = 0
        batch = []
        
        print("Starting tokenizer training from text streams...")
        for text in tqdm(dataset):
            batch.append(text)
            sample_count += 1
            
            if len(batch) >= batch_size:
                for item in batch:
                    yield item
                batch = []
                
            # Optional early stopping
            if max_samples and sample_count >= max_samples:
                break
                
        # Yield remaining items
        for item in batch:
            yield item
    
    # Train tokenizer
    print(f"Training tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}")
    tokenizer.train_from_iterator(batched_iterator(), trainer=trainer)
    
    # Save the tokenizer
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/wordpiece-tokenizer.json")
    
    print("Tokenizer training complete")
    return tokenizer


# Example usage:
if __name__ == "__main__":
    # Replace with your actual data directories
    base_dir = "/home/jlunder/research/Tree-Matching-Networks/data/"
    train_dir = base_dir+'train/'
    test_dir = base_dir+'test/'
    dev_dir = base_dir+'dev/'
    base_dirs = [train_dir, test_dir, dev_dir]
    data_dirs = [b+dir for dir in ["amazonqa_multiple_converted_trf_sharded", "amazonqa_single_converted_trf_sharded"] for b in base_dirs]
    
    # Train tokenizer
    tokenizer = train_tokenizer_from_json_files(
        data_dirs=data_dirs,
        vocab_size=1200,  # Small vocabulary as required
        min_frequency=5,
        batch_size=1000,  # Process 1000 samples at a time
        max_samples=None  # Set to a number if you want to limit samples
    )
    
    # Convert to BERT format if needed
    from transformers import BertTokenizerFast
    
    os.makedirs("/home/jlunder/local_storage/tokenizers", exist_ok=True)
    bert_tokenizer = BertTokenizerFast(
        tokenizer_file="/home/jlunder/local_storage/tokenizers/wordpiece-tokenizer.json",
        do_lower_case=True,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    
    # Save in transformers format
    bert_tokenizer.save_pretrained("/home/jlunder/local_storage/tokenizers/tiny-bert-tokenizer")
