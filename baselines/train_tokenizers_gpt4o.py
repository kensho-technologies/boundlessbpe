# since I have 12 jobs, just run each one single threaded
# keeping every core busy on my laptop
import os
os.environ["RAYON_NUM_THREADS"] = "3"  # e.g., internally use 1 thread per job

import multiprocessing
import json

from tokenizers import Tokenizer, Regex, decoders, trainers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from transformers import PreTrainedTokenizerFast

# this is a memory efficient way to load each process
def iter_jsonl_texts(path, limit):
    """Yield up to `limit` texts from a JSONL file and stop exactly at that line count."""
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break  # stop reading after hitting limit-th line
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip bad lines, but still count them toward the limit
            txt = obj.get("text")
            if txt:
                yield txt

def train_and_save_tokenizer(args):
    """Train a tokenizer and save it in Hugging Face format."""
    model_type, vocab_size, filepath, output_dir, limit = args

    ###########################
    # PRETOKENIZATION PATTERN
    ###########################

    # GPT4o regex
    # https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L101-L111
    # https://github.com/openai/tiktoken/blob/4560a889/tiktoken_ext/openai_public.py#L101-L114

    # This regex could be made more efficient. If I was the one working on this encoding, I would
    # have done a few other things differently too, e.g. I think you can allocate tokens more
    # efficiently across languages.
    GPT4O_SPLIT_PATTERN = "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    # GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # SUPER_BPE_SPLIT = r"""'?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+| (?=(\d{3})+(?!\d))"""
    # SHORT_PATTERN = r'\s+|\w+|[^\s\w]+'
    pretokenization_pattern = Regex(GPT4O_SPLIT_PATTERN)
    # pretokenization_pattern = Regex(GPT2_SPLIT_PATTERN)
    # pretokenization_pattern = Regex(SHORT_PATTERN)
    # pretokenization_pattern = Regex(SUPER_BPE_SPLIT)

    # set this to False to prevent this pre_tokenizer from using the GPT2 specific regexp for spliting on whitespace.
    # default: True
    use_regex = False

    # Whether to add a space to the first word if there isn’t already one
    # default: True
    # lets set this to False for now, since BoundlessBPE doesn't include it
    # to make it a more fair comparison
    add_prefix_space = False

    # we want to start with the single bytes in all cases
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    print("initial_alphabet:", len(initial_alphabet))
    
    # Define model & trainer
    if model_type == "bpe":
        model = BPE()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, 
                                      initial_alphabet=initial_alphabet,
                                      special_tokens=["<|endoftext|>"])
    elif model_type == "unigram":
        model = Unigram()
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size, 
                                          initial_alphabet=initial_alphabet,
                                          special_tokens=["<|endoftext|>"])
    elif model_type == "wordpiece":
        # any pretoken longer than max_input_chars_per_word turns into an [UNK]
        # default is 100, so set bigger
        model = WordPiece(max_input_chars_per_word = 5000, unk_token = "<unk>")
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, 
                                            initial_alphabet=initial_alphabet,
                                            special_tokens=["<|endoftext|>","<unk>"])
    else:
        raise ValueError("Invalid model type. Choose from 'bpe', 'unigram', or 'wordpiece'.")

    # Train tokenizer
    hf_tokenizer = Tokenizer(model)

    # Set custom pretokenization
    split = pre_tokenizers.Split(pattern=pretokenization_pattern, behavior="isolated")
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space, use_regex=use_regex)
    # TODO: Stéphan Tulkens (https://stephantul.github.io/)
    # suggested using a FixedLength to avoid max_input_chars_per_word problems here
    # pre_tokenizers.Sequence([split, FixedLength(100), byte_level])  
    hf_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([split, byte_level])

    # set trim_offsets to be the same value as add_prefix_space, for reversibility
    hf_tokenizer.post_processor = processors.ByteLevel(trim_offsets=add_prefix_space)

    hf_tokenizer.decoder = decoders.ByteLevel()

    # note: the rust code uses multiple cores in sections
    hf_tokenizer.train_from_iterator(iter_jsonl_texts(filepath,limit), trainer=trainer)


    # Wrap in PreTrainedTokenizerFast for HF compatibility
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=hf_tokenizer)
    
    # Define HF-compatible tokenizer directory
    hf_save_path = os.path.join(output_dir, f"{model_type}_{vocab_size}")
    os.makedirs(hf_save_path, exist_ok=True)
    
    # Save in Hugging Face format
    hf_tokenizer.save_pretrained(hf_save_path)
    
    print(f"Saved {model_type} tokenizer with vocab size {vocab_size} in {hf_save_path}")


if __name__ == "__main__":


    filepath = "../data/minipile.jsonl"


    limit = 170_721  # stop after this many docs

    # Tokenizer save path
    output_dir = "mini_w_space_gpt4o"
    os.makedirs(output_dir, exist_ok=True)

    # List of tasks for multiprocessing
    tasks = [(model_type, vocab_size, filepath, output_dir, limit) for model_type in ["bpe", "unigram", "wordpiece"] for vocab_size in [40960, 65536, 98304, 131072]]

    # Run in parallel using multiprocessing
    # don't set this too high as each train_from_iterator job will use multiple cores in training
    # this worked ok on my mac
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(train_and_save_tokenizer, tasks)

print("All tokenizers saved in Hugging Face format! ")

# script -q /dev/null python -u train_tokenizers_gpt4o.py 2>&1 | tee log_train_tokenizers_gpt4o.txt