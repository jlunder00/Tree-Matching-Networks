# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

from tokenizers import BertWordPieceTokenizer, WordPieceTrainer
import os

# Define parameters
vocab_size = 1200   # Small vocabulary size for resource constraints
min_frequency = 5   # Minimum frequency for a token to be included

# Initialize a tokenizer with BERT's configuration
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=True,
    lowercase=True,
)
trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

# Train from your corpus files
corpus_files = ["path/to/your/corpus.txt"]  # Replace with your data files
tokenizer.train(
    files=corpus_files,
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
)

# Save the tokenizer
os.makedirs("/home/jlunder/temp_storage/tokenizers/", exist_ok=True)
tokenizer.save_model("tokenizer_"+str(vocab_size)+"_"+str(min_frequency))
