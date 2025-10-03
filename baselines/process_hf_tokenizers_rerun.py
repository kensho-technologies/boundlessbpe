import sys
import os
import json
import time
from collections import Counter
from transformers import AutoTokenizer
from pathlib import Path
import multiprocessing
import regex as re

# Add parent directory to path if boundlessbpe package not found
try:
    from boundlessbpe.util import frombytes  #  bytes -> str
except ImportError:
    # If running from baselines/ directory, add parent directory to path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from boundlessbpe.util import frombytes  #  bytes -> str

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
gpt2pat = re.compile(GPT2_SPLIT_PATTERN)

# this is a memory efficient way to load each process
# skips the first skip documents
def iter_jsonl_texts_skipping(path, skip):
    """Yield up to `limit` texts from a JSONL file and stop exactly at that line count."""
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            # skip the initial lines used in training
            if idx < skip:
                continue
            yield json.loads(line)["text"]

def process_tokenizer(args):

    input_file, skip, tokenizer_dir, output_file, pretoken_file, single_token_file, tokenizer_name, vocab_size = args

    start_time = time.time()

    pid = os.getpid()
    print(f"PROCESS START: PID {pid}, {tokenizer_name}_{vocab_size}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if tokenizer_name == "wordpiece":
        # I'm not setting this to 5000 in the training code, so no need to manually set
        # Ok, so the mystery here is how we get [UNK] tokens even with 
        # byte level.  It turns out the wordpiece implementation has a constant
        # max_input_chars_per_word that defaults to 100, and any pretoken 
        # over that lenght becomes an [UNK], which throws an exception for me 
        # since I did add one
        # you can set it to a larger value as here, and then things work
        # print("max_input_chars_per_word:",tokenizer._tokenizer.model.max_input_chars_per_word )
        # I think we're going to have to add and <UNK> for this since there are 
        # still pretokens exceeing 5000 characters
        tokenizer.add_special_tokens({'unk_token': '<unk>'})
        tokenizer._tokenizer.model.max_input_chars_per_word = 1000000
        print("max_input_chars_per_word", tokenizer._tokenizer.model.max_input_chars_per_word)
        print("unk_token:", tokenizer.unk_token, tokenizer_name, vocab_size, "<unk>" in tokenizer.get_vocab())
        
    vocab = tokenizer.get_vocab()

    counter = Counter()
    pretoken_counter = Counter()
    single_tokens = 0
    total_tokens = 0
    total_pretokens = 0
    total_pretokens_in_vocab = 0
    for i, text in enumerate(iter_jsonl_texts_skipping(input_file, skip)):

        if i % 10000 == 0:
            print(i, tokenizer_name, vocab_size, time.time()-start_time)
            start_time = time.time()

        # die if it doesn't work
        # this returns a list of HF encoded strings
        tokens = tokenizer.tokenize(text, add_special_tokens=False)
        counter.update(tokens)

        total_tokens += len(tokens)

        # what fraction of tokens are single pretokens
        # lets compute the pretokens
        # these are strings, that we need to convert to HF encoded strings
        # str -> bytes -> HF encoded str
        pretokens = [frombytes(t.encode("utf-8")) for t in gpt2pat.findall(text)]
        total_pretokens += len(pretokens)
        pretoken_counter.update(pretokens)

        # count how of the pretokens are in the vocabulary, which should be single tokens
        # why does unigram have low values here
        pretokens_in_vocab = [p for p in pretokens if p in vocab]
        total_pretokens_in_vocab += len(pretokens_in_vocab)

        # get fast lookups
        pretokens = set(pretokens)

        # which were full pretokens
        # see how this compares to total_pretokens_in_vocab
        fullpretokens = [t for t in tokens if t in pretokens]
        single_tokens += len(fullpretokens)

    # so we don't need this conversion
    # note that since these are bytes objects we can't use json to store them
    # need to convert to encoded strings first
    # converted = { frombytes(k) : v for k,v in counter.items()}
    
    # Create the output file
    with open(output_file, "wt") as f:
        json.dump(counter, f, ensure_ascii=False)

    with open(pretoken_file, "wt") as f:
        json.dump(pretoken_counter, f, ensure_ascii=False)

    # also lets compute the weights of the pretokens that are also a token
    unique_tokens = set(counter.keys())

    pretoken_counts_in_token = sum([cnt for (pretok, cnt) in pretoken_counter.items() if pretok in unique_tokens])
    total_pretoken_counts = sum(pretoken_counter.values())

    # save these to a file too
    with open(single_token_file, "wt") as f:
        f.write(f"unique_tokens {len(counter)} tokenizer_name {tokenizer_name} vocab_size {vocab_size} single_tokens {single_tokens} total_tokens {total_tokens} total_pretokens {total_pretokens} pretokens_in_vocab {total_pretokens_in_vocab} pretoken_counts_in_token {pretoken_counts_in_token} total_pretoken_counts {total_pretoken_counts}\n")
        
    print(f"Successfully wrote {len(counter)} tokens to {tokenizer_name}_{vocab_size} single_tokens: {single_tokens} total_tokens: {total_tokens} total_pretokens: {total_pretokens} pretokens_in_vocab: {total_pretokens_in_vocab} pretoken_counts_in_token {pretoken_counts_in_token} total_pretoken_counts {total_pretoken_counts}")


if __name__ == "__main__":

    # things are scattered rather badly around this repo
    # these were created by ~/src/superwords-tokenization/train_tokenizers.py
    tokenizer_base = "../mini_w_space"
    output_dir = "baseline_token_counts"
    input_file = "../data/minipile.jsonl"
    skip = 170721

    os.makedirs(output_dir, exist_ok=True)

    # see what's in that directory
    subdirectories = sorted([
        item.name for item in Path(tokenizer_base).iterdir() 
        if item.is_dir()
    ])

    # these have the form bpe_40960
    pairs = [s.split("_") for s in subdirectories]
    # are they all pairs?
    assert all([len(p) == 2 for p in pairs])
    pairs = [(tok,int(vocab)) for (tok,vocab) in pairs]
    print(pairs)

    # collect our parameters
    params = []
    for tokenizer_name, vocab_size in pairs:

        # the bpe ran successfully so skip for now
        tokenizer_dir = os.path.join(tokenizer_base, f"{tokenizer_name}_{vocab_size}/")
        full_output_dir =   os.path.join(output_dir,  f"{tokenizer_name}_{vocab_size}")
        # make sure it exists
        os.makedirs(full_output_dir, exist_ok=True)
        output_file = os.path.join(full_output_dir, "token_frequencies.json")
        pretoken_file = os.path.join(full_output_dir, "pretoken_frequencies.json")
        single_token_file = os.path.join(full_output_dir, "single_token_pretokens.txt")
        params.append((input_file, skip, tokenizer_dir, output_file, pretoken_file, single_token_file, tokenizer_name, vocab_size))

    print(len(params))

    # process_tokenizer(input_file, skip, tokenizer_dir, output_file, tokenizer_name, vocab_size)
    # There are 12 jobs, so have each do roughly two
    with multiprocessing.Pool(processes=12) as pool:
        pool.map(process_tokenizer, params)

# why is the script necessary
# script -q /dev/null python -u process_hf_tokenizers_rerun.py 2>&1 | tee total_single_tokens_3.txt