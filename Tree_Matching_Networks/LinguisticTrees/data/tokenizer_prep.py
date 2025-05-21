import json
import torch
from torch.utils.data import IterableDataset
from pathlib import Path
import os
from tqdm import tqdm
from typing import List, Iterator, Optional
import yaml

# Import your TreeDataConfig
from Tree_Matching_Networks.LinguisticTrees.configs.tree_data_config import TreeDataConfig

class StreamingTextDataset(IterableDataset):
    """Memory-efficient streaming dataset that handles multiple data formats"""
    
    def __init__(self, 
                 json_files: List[Path] = None, 
                 text_files: List[Path] = None, 
                 max_samples: Optional[int] = None,
                 delimiters: List[str] = None):
        self.json_files = json_files or []
        self.text_files = text_files or []
        self.max_samples = max_samples
        self.delimiters = delimiters or []
        
    def __iter__(self) -> Iterator[str]:
        """Iterate through all files and yield text"""
        sample_count = 0
        
        # Process JSON files first
        for file_path in self.json_files:
            if not os.path.exists(file_path):
                print(f"Warning: File does not exist: {file_path}")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract text from groups
                    for group in data.get('groups', []):
                        if 'text' in group and group['text']:
                            yield group['text']
                            sample_count += 1
                            
                            if self.max_samples and sample_count >= self.max_samples:
                                return
            except Exception as e:
                print(f"Error processing JSON file {file_path}: {e}")
                continue
        
        # Process text files next
        for file_path in self.text_files:
            if not os.path.exists(file_path):
                print(f"Warning: File does not exist: {file_path}")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        text = line.strip()
                        
                        # Remove delimiters if specified
                        if self.delimiters:
                            for delimiter in self.delimiters:
                                text = text.replace(delimiter, ' ')
                        
                        if text:
                            yield text
                            sample_count += 1
                            
                            if self.max_samples and sample_count >= self.max_samples:
                                return
            except Exception as e:
                print(f"Error processing text file {file_path}: {e}")
                continue


def get_dataset_roots_from_configs(configs):
    """Get all dataset root directories from multiple TreeDataConfig instances"""
    all_roots = []
    
    for config_dict in configs:
        # Create TreeDataConfig
        data_config = TreeDataConfig(
            data_root=config_dict.get('data_root', '/home/jlunder/research/Tree-Matching-Networks/data/processed_data'),
            dataset_specs=config_dict.get('dataset_specs', [config_dict.get('dataset_type', 'snli')]),
            task_type=config_dict.get('task_type', 'entailment'),
            use_sharded_train=config_dict.get('use_sharded_train', True),
            use_sharded_validate=config_dict.get('use_sharded_validate', True),
            use_sharded_test=config_dict.get('use_sharded_test', True),
            allow_cross_dataset_negatives=config_dict.get('allow_cross_dataset_negatives', True)
        )
        
        # Get dataset directories
        for split in ['train', 'dev', 'test']:
            paths = data_config.get_split_paths(split)
            all_roots.extend(paths)
    
    return all_roots


def gather_json_files(root_dirs):
    """Gather all JSON files from the given root directories using your globbing pattern"""
    data_files = []
    for data_dir in root_dirs:
        path = Path(data_dir)
        if not path.exists():
            print(f"Warning: Directory does not exist: {path}")
            continue
            
        # Try to find sharded files first
        files = sorted(
            [f for f in path.glob("part_*_shard_*.json") 
             if not f.name.endswith('_counts.json')],
            key=lambda x: (int(x.stem.split('_')[1]), 
                         int(x.stem.split('_shard_')[1]))
        )
        
        # If no sharded files, look for regular part files
        if not files:
            files = sorted(
                [f for f in path.glob("part_*.json")
                 if not f.name.endswith('_counts.json')],
                key=lambda x: int(x.stem.split('_')[1])
            )
            
        data_files.extend(files)
    
    return data_files


def gather_text_files(text_dirs, file_pattern="*.txt"):
    """Gather all text files from the given directories using the specified pattern"""
    text_files = []
    for text_dir in text_dirs:
        path = Path(text_dir)
        if not path.exists():
            print(f"Warning: Directory does not exist: {path}")
            continue
            
        # Find all text files matching the pattern
        files = [f for f in sorted(path.glob(file_pattern)) if f.is_file()]
        text_files.extend(files)
    
    return text_files


def train_tokenizer_from_multiple_sources(configs, 
                                         text_dirs=None,
                                         text_file_pattern="*.txt",
                                         delimiters=None,
                                         vocab_size=1200, 
                                         min_frequency=5,
                                         batch_size=1000,
                                         max_samples=None,
                                         tokenizer_save_path=None):
    """Train tokenizer using both TreeDataConfig JSON files and plain text files"""
    from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.normalizers import NFD, Lowercase, StripAccents
    
    # Get root directories from configs
    print("Getting dataset root directories from configs...")
    root_dirs = get_dataset_roots_from_configs(configs)
    print(f"Found {len(root_dirs)} root directories")
    
    # Get all JSON files using globbing
    print("Finding JSON files in root directories...")
    json_files = gather_json_files(root_dirs)
    print(f"Found {len(json_files)} JSON files")
    
    # Get all text files if specified
    text_files = []
    if text_dirs:
        print("Finding text files in specified directories...")
        text_files = gather_text_files(text_dirs, file_pattern=text_file_pattern)
        print(f"Found {len(text_files)} text files")
    
    if not json_files and not text_files:
        raise ValueError("No valid data files found")
    
    # Create tokenizer with proper BERT-style components
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    
    # Add a normalizer
    tokenizer.normalizer = normalizers.Sequence([
        NFD(),
        Lowercase(),
        StripAccents()
    ])
    
    # Add pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Add decoder
    tokenizer.decoder = decoders.WordPiece()
    
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
    )
    
    # Create streaming dataset
    dataset = StreamingTextDataset(
        json_files=json_files,
        text_files=text_files,
        max_samples=max_samples,
        delimiters=delimiters
    )
    
    # Create a batched iterator for efficiency
    def batched_iterator():
        batch = []
        
        print("Starting tokenizer training from text streams...")
        for text in tqdm(dataset):
            batch.append(text)
            
            if len(batch) >= batch_size:
                for item in batch:
                    yield item
                batch = []
                
        # Yield remaining items
        for item in batch:
            yield item
    
    # Train tokenizer
    print(f"Training tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}")
    tokenizer.train_from_iterator(batched_iterator(), trainer=trainer)
    
    # Save the tokenizer if path provided
    if tokenizer_save_path:
        os.makedirs(os.path.dirname(tokenizer_save_path), exist_ok=True)
        tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer saved to {tokenizer_save_path}")
    
    return tokenizer


def convert_to_transformers_tokenizer(json_path):
    """Convert the trained tokenizer to BertTokenizerFast format"""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast, BertTokenizerFast
    
    # Load the trained tokenizer
    tokenizer = Tokenizer.from_file(json_path)
    
    # Extract the vocabulary
    vocab_dict = tokenizer.get_vocab()
    
    # Sort vocabulary by token ids
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocab_list = [x[0] for x in sorted_vocab]
    
    # Create vocabulary files
    vocab_dir = os.path.dirname(json_path)
    vocab_file = os.path.join(vocab_dir, "vocab.txt")
    
    with open(vocab_file, "w", encoding="utf-8") as f:
        for token in vocab_list:
            f.write(f"{token}\n")
    
    # Create a PreTrainedTokenizerFast directly with all properties
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=json_path,
        vocab_file=vocab_file,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        model_max_length=512,
        padding_side="right",
        truncation_side="right",
    )
    
    # Save the tokenizer
    fast_tokenizer.save_pretrained(os.path.join(vocab_dir, "bert-tokenizer"))
    print(f"BertTokenizerFast saved to {os.path.join(vocab_dir, 'bert-tokenizer')}")
    return fast_tokenizer


# Example usage with multiple dataset configs and text files
if __name__ == "__main__":
    # Define configs for different datasets
    configs = [
        {
            # Config for snli
            "data_root": "/home/jlunder/research/Tree-Matching-Networks/data/processed_data",
            "dataset_specs": ["snli"],
            "task_type": "entailment",
            "use_sharded_train": True,
            "use_sharded_validate": True,
            "use_sharded_test": True
        },
        {
            # Config for amazonqa_multiple
            "data_root": "/home/jlunder/research/Tree-Matching-Networks/data/processed_data",
            "dataset_specs": ["amazonqa_multiple"],
            "task_type": "entailment",
            "use_sharded_train": True,
            "use_sharded_validate": True,
            "use_sharded_test": True
        },
        {
            # Config for amazonqa_single
            "data_root": "/home/jlunder/research/Tree-Matching-Networks/data/processed_data",
            "dataset_specs": ["amazonqa_single"],
            "task_type": "entailment",
            "use_sharded_train": True,
            "use_sharded_validate": True,
            "use_sharded_test": True
        },
        {
            "data_root": "/home/jlunder/research/Tree-Matching-Networks/data/processed_data",
            "dataset_specs": ["patentmatch_balanced", "semeval", "wikiqs"],
            "task_type": "entailment",
            "use_sharded_train": True,
            "use_sharded_validate": True,
            "use_sharded_test": True
        }
    ]
    
    # Define your text directories
    text_dirs = [
        "/home/jlunder/research/data/wikiqs/source/",
        "/home/jlunder/research/data/wikiqs/tmp/"
    ]
    
    # Define delimiters to remove
    delimiters = [" q:", " a:"]  # Your specific delimiters
    
    # Train tokenizer with both datasets
    tokenizer = train_tokenizer_from_multiple_sources(
        configs=configs,
        text_dirs=text_dirs,
        text_file_pattern="*",  # Adjust if your files have different extensions
        delimiters=delimiters,
        vocab_size=10000,
        min_frequency=2,
        batch_size=1000,
        max_samples=None,  # Set to a number if you want to limit samples
        tokenizer_save_path="/home/jlunder/local_storage/tokenizers/combined_tokenizer_10000_2.json"
    )
    
    # Convert to BertTokenizerFast format
    convert_to_transformers_tokenizer("/home/jlunder/local_storage/tokenizers/combined_tokenizer_10000_2.json")

# import json
# import torch
# from torch.utils.data import IterableDataset
# from pathlib import Path

# import os
# from tqdm import tqdm
# from typing import List, Iterator
# import yaml

# # Import your TreeDataConfig
# from Tree_Matching_Networks.LinguisticTrees.configs.tree_data_config import TreeDataConfig

# class StreamingTreeTextDataset(IterableDataset):
#     """Memory-efficient streaming dataset for tokenizer training from tree data"""
#     
#     def __init__(self, data_files: List[Path], max_samples=None):
#         self.data_files = data_files
#         self.max_samples = max_samples
#         
#     def __iter__(self) -> Iterator[str]:
#         """Iterate through all files and yield text fields from tree groups"""
#         sample_count = 0
#         
#         for file_path in self.data_files:
#             if not os.path.exists(file_path):
#                 print(f"Warning: File does not exist: {file_path}")
#                 continue
#                 
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#                     
#                     # Extract text from groups
#                     for group in data.get('groups', []):
#                         if 'text' in group and group['text']:
#                             yield group['text']
#                             sample_count += 1
#                             
#                             if self.max_samples and sample_count >= self.max_samples:
#                                 return
#             except Exception as e:
#                 print(f"Error processing file {file_path}: {e}")
#                 continue


# def get_dataset_roots_from_configs(configs):
#     """Get all dataset root directories from multiple TreeDataConfig instances"""
#     all_roots = []
#     
#     for config_dict in configs:
#         # Create TreeDataConfig
#         data_config = TreeDataConfig(
#             data_root=config_dict.get('data_root', '/home/jlunder/research/Tree-Matching-Networks/data/processed_data'),
#             dataset_specs=config_dict.get('dataset_specs', [config_dict.get('dataset_type', 'snli')]),
#             task_type=config_dict.get('task_type', 'entailment'),
#             use_sharded_train=config_dict.get('use_sharded_train', True),
#             use_sharded_validate=config_dict.get('use_sharded_validate', True),
#             use_sharded_test=config_dict.get('use_sharded_test', True),
#             allow_cross_dataset_negatives=config_dict.get('allow_cross_dataset_negatives', True)
#         )
#         
#         # Get dataset directories
#         for split in ['train', 'dev', 'test']:
#             paths = data_config.get_split_paths(split)
#             all_roots.extend(paths)
#     
#     return all_roots


# def gather_json_files(root_dirs):
#     """Gather all JSON files from the given root directories using your globbing pattern"""
#     data_files = []
#     for data_dir in root_dirs:
#         path = Path(data_dir)
#         if not path.exists():
#             print(f"Warning: Directory does not exist: {path}")
#             continue
#             
#         # Try to find sharded files first
#         files = sorted(
#             [f for f in path.glob("part_*_shard_*.json") 
#              if not f.name.endswith('_counts.json')],
#             key=lambda x: (int(x.stem.split('_')[1]), 
#                          int(x.stem.split('_shard_')[1]))
#         )
#         
#         # If no sharded files, look for regular part files
#         if not files:
#             files = sorted(
#                 [f for f in path.glob("part_*.json")
#                  if not f.name.endswith('_counts.json')],
#                 key=lambda x: int(x.stem.split('_')[1])
#             )
#             
#         data_files.extend(files)
#     
#     return data_files


# def train_tokenizer_from_config_roots(configs, 
#                                      vocab_size=1200, 
#                                      min_frequency=5,
#                                      batch_size=1000,
#                                      max_samples=None):
#     """Train tokenizer using TreeDataConfig to get root dirs and globbing to find files"""
#     from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
#     from tokenizers.models import WordPiece
#     from tokenizers.trainers import WordPieceTrainer
#     from tokenizers.pre_tokenizers import Whitespace
#     from tokenizers.normalizers import NFD, Lowercase, StripAccents
#     
#     # Get root directories from configs
#     print("Getting dataset root directories from configs...")
#     root_dirs = get_dataset_roots_from_configs(configs)
#     print(f"Found {len(root_dirs)} root directories")
#     
#     # Get all JSON files using globbing
#     print("Finding JSON files in root directories...")
#     data_files = gather_json_files(root_dirs)
#     print(f"Found {len(data_files)} JSON files")
#     
#     if not data_files:
#         raise ValueError("No valid data files found")
#     
#     # Create tokenizer with proper BERT-style components
#     tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
#     
#     # Add a normalizer
#     tokenizer.normalizer = normalizers.Sequence([
#         NFD(),
#         Lowercase(),
#         StripAccents()
#     ])
#     
#     # Add pre-tokenizer
#     tokenizer.pre_tokenizer = Whitespace()
#     
#     # Add decoder
#     tokenizer.decoder = decoders.WordPiece()
#     
#     trainer = WordPieceTrainer(
#         vocab_size=vocab_size,
#         min_frequency=min_frequency,
#         special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
#         show_progress=True,
#     )
#     
#     # Create streaming dataset
#     dataset = StreamingTreeTextDataset(data_files, max_samples=max_samples)
#     
#     # Create a batched iterator for efficiency
#     def batched_iterator():
#         batch = []
#         
#         print("Starting tokenizer training from text streams...")
#         for text in tqdm(dataset):
#             batch.append(text)
#             
#             if len(batch) >= batch_size:
#                 for item in batch:
#                     yield item
#                 batch = []
#                 
#         # Yield remaining items
#         for item in batch:
#             yield item
#     
#     # Train tokenizer
#     print(f"Training tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}")
#     tokenizer.train_from_iterator(batched_iterator(), trainer=trainer)
#     
#     # Save the tokenizer
#     os.makedirs("/home/jlunder/local_storage/tokenizers", exist_ok=True)
#     tokenizer.save("/home/jlunder/local_storage/tokenizers/tiny-bert-tokenizer_10000_2.json")
#     
#     print("Tokenizer training complete")
#     return tokenizer


# # Alternative approach to convert to BertTokenizerFast
# def convert_to_transformers_tokenizer(json_path):
#     """Convert the trained tokenizer to BertTokenizerFast format"""
#     from tokenizers import Tokenizer
#     from transformers import PreTrainedTokenizerFast, BertTokenizerFast
#     
#     # Load the trained tokenizer
#     tokenizer = Tokenizer.from_file(json_path)
#     
#     # Extract the vocabulary
#     vocab_dict = tokenizer.get_vocab()
#     
#     # Sort vocabulary by token ids
#     sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
#     vocab_list = [x[0] for x in sorted_vocab]
#     
#     # Create vocabulary files
#     vocab_dir = os.path.dirname(json_path)
#     vocab_file = os.path.join(vocab_dir, "vocab.txt")
#     
#     with open(vocab_file, "w", encoding="utf-8") as f:
#         for token in vocab_list:
#             f.write(f"{token}\n")
#     
#     # Create a PreTrainedTokenizerFast directly with all properties
#     fast_tokenizer = PreTrainedTokenizerFast(
#         tokenizer_file=json_path,
#         vocab_file=vocab_file,
#         do_lower_case=True,
#         unk_token="[UNK]",
#         sep_token="[SEP]",
#         pad_token="[PAD]",
#         cls_token="[CLS]",
#         mask_token="[MASK]",
#         model_max_length=512,
#         padding_side="right",
#         truncation_side="right",
#     )
#     
#     # Save the tokenizer
#     fast_tokenizer.save_pretrained(os.path.join(vocab_dir, "tiny-bert-tokenizer"))
#     print(f"BertTokenizerFast saved to {os.path.join(vocab_dir, 'tiny-bert-tokenizer')}")
#     return fast_tokenizer


# # Example usage with multiple dataset configs
# if __name__ == "__main__":
#     # Define configs for different datasets
#     configs = [
#         {
#             # Config for snli
#             "data_root": "/home/jlunder/research/Tree-Matching-Networks/data/processed_data",
#             "dataset_specs": ["snli"],
#             "task_type": "entailment",
#             "use_sharded_train": True,
#             "use_sharded_validate": True,
#             "use_sharded_test": True
#         },
#         {
#             # Config for amazonqa_multiple
#             "data_root": "/home/jlunder/research/Tree-Matching-Networks/data/processed_data",
#             "dataset_specs": ["amazonqa_multiple"],
#             "task_type": "entailment",
#             "use_sharded_train": True,
#             "use_sharded_validate": True,
#             "use_sharded_test": True
#         },
#         {
#             # Config for amazonqa_single
#             "data_root": "/home/jlunder/research/Tree-Matching-Networks/data/processed_data",
#             "dataset_specs": ["amazonqa_single"],
#             "task_type": "entailment",
#             "use_sharded_train": True,
#             "use_sharded_validate": True,
#             "use_sharded_test": True
#         },
#         {
#             "data_root": "/home/jlunder/research/Tree-Matching-Networks/data/processed_data",
#             "dataset_specs": ["patentmatch_balanced", "semeval", "wikiqs"],
#             "task_type": "entailment",
#             "use_sharded_train": True,
#             "use_sharded_validate": True,
#             "use_sharded_test": True
#         }
#     ]
#     
#     # Train tokenizer with multiple dataset configs
#     tokenizer = train_tokenizer_from_config_roots(
#         configs=configs,
#         vocab_size=10000,  # Small vocabulary as required
#         min_frequency=2,
#         batch_size=1000,  # Process 1000 samples at a time
#         max_samples=None  # Set to a number if you want to limit samples
#     )
#     
#     # Use the alternative approach to convert to BertTokenizerFast
#     convert_to_transformers_tokenizer("/home/jlunder/local_storage/tokenizers/tiny-bert-tokenizer_10000_2.json")
