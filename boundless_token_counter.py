import json
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
import argparse
import itertools
import sys
import time

# Import boundlessbpe modules
from boundlessbpe import FasterRegexInference
from boundlessbpe.util import frombytes  #  bytes -> str
  

# in this case we're just counting tokens on training or eval data
# so we don't expect to have any special tokens 
# we can call encode_ordinary_chunks directly below which gives us a list of bytes, 
# so don't need to set "<|endoftext|>"
def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from local path"""
    tokenizer = FasterRegexInference()
    tokenizer.load(tokenizer_path)
    # tokenizer.special_tokens = {"<|endoftext|>": vocab_size}
    return tokenizer


def tokenize_range(
    start_row: int,
    end_row: int,
    chunk_idx: int,
    output_dir: str,
    dataset_path: str,
    tokenizer_path: str,
    blowup: bool) -> None:
    """
    Tokenize a range of rows from the dataset file and write results directly to disk.
    
    Args:
        start_row: Starting row index to process (0-based)
        end_row: Ending row index to process (exclusive)
        chunk_idx: Index of this chunk for output file naming
        output_dir: Directory to write the output to
        dataset_path: Path to the JSONL dataset file
        tokenizer_path: Path to locally saved tokenizer
        blowup: Tokenizer blowup parameter
    """
    # Load tokenizer in each process (required for multiprocessing)
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Process directly from file without loading everything into memory
    token_counter = Counter() # bytes : int

    start_time = time.time()
    
    with open(dataset_path, "rt") as data:
        # Use itertools.islice to efficiently skip to start_row and limit to end_row
        # This avoids reading and discarding lines we don't need
        row_count = end_row - start_row
        lines = itertools.islice(data, start_row, end_row)
        
        # Process just the lines we need
        for i, row in enumerate(lines, start=start_row):
            if i % 100 == 0:
                # time status, and also how we're doing on unique vocab found
                print(f"status: chunk {chunk_idx}, processing {i-start_row} of {end_row - start_row} : {time.time() - start_time} {len(token_counter)}")

            # Parse JSON with error handling
            try:
                line = json.loads(row)
            except json.JSONDecodeError as e:
                error_msg = f"Error parsing JSON at line {i} in chunk {chunk_idx}: {e}"
                print(error_msg)
                raise ValueError(error_msg) from e

            # Extract text field with error handling
            try:
                text = line['text']
            except KeyError as e:
                error_msg = f"Missing 'text' field at line {i} in chunk {chunk_idx}"
                print(error_msg)
                raise ValueError(error_msg) from e

            # Tokenize with error handling - fail on any error
            try:
                # this gives us a list[bytes]
                # note that it doesn't support special tokens here
                # but that's ok because we're just counting tokens
                tokens = tokenizer.encode_ordinary_chunks(text, blowup=blowup)
                
                # tally these up
                token_counter.update(tokens)
            except Exception as e:
                error_msg = f"Error tokenizing text at line {i} in chunk {chunk_idx}: {e}"
                print(error_msg)
                # Fail the entire chunk - we want perfect results or nothing
                raise ValueError(error_msg) from e

    # Write results directly to disk inside the worker
    # don't bother sorting here
        
    try:
        # note that since these are bytes objects we can't use json to store them
        # need to convert to encoded strings first
        converted = { frombytes(k) : v for k,v in token_counter.items()}
        
        # Create the output file
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:05d}_{start_row}_{end_row}.json")
        with open(chunk_path, "wt", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False)
            
        print(f"Successfully wrote {len(token_counter)} tokens to {chunk_path}")
    except Exception as e:
        error_msg = f"Error writing output file for chunk {chunk_idx}: {e}"
        print(error_msg)
        raise  # Re-raise the exception to signal failure to the main process
        

def stream_and_tokenize_dataset(
    start_row: int = 170721,
    end_row: int = 1000000,  # Default None, but will be checked
    output_dir: str = "token_frequencies", 
    dataset_path: str = "data/minipile.jsonl",
    tokenizer_path: str = "./tokenizer",
    max_workers: int = None,
    blowup: bool = True) -> None: 
    """
    Process dataset by dividing it into equal chunks for worker processes.
    Each worker processes its chunk directly from disk, minimizing memory usage.
    """
    # Check that end_row was provided and validate range
    if end_row is None:
        raise ValueError("end_row must be provided")
    
    if start_row >= end_row:
        raise ValueError(f"start_row ({start_row}) must be less than end_row ({end_row})")
 
    if max_workers is None:
        # leave one for the head node
        max_workers = max(1,mp.cpu_count() - 1)

    print(f"Processing dataset from row {start_row} to {end_row} with {max_workers} workers...")
    print(f"Dataset path: {dataset_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    existing_chunks = [
        f for f in os.listdir(output_dir)
        if f.startswith("chunk_") and f.endswith(".json")
    ]
    if existing_chunks:
        print(f"Error: Found existing chunk files in {output_dir}:")
        for fname in existing_chunks:
            print(f"  {fname}")
        print("Aborting to avoid overwriting existing results.")
        sys.exit(1)
    
    print(f"Processing rows {start_row} to {end_row-1} ({end_row-start_row} rows total)")
    
    # Calculate chunk sizes - divide dataset evenly among workers
    total_rows = end_row - start_row
    max_workers = min(max_workers, total_rows)  # Don't use more workers than rows
    
    if max_workers <= 0:
        print("No rows to process or invalid worker count")
        return Counter()
    
    rows_per_worker = total_rows // max_workers
    extra_rows = total_rows % max_workers
    # make sure we did this right
    assert max_workers*rows_per_worker + extra_rows == total_rows
    assert extra_rows < max_workers
    
    print(f"Dividing into {max_workers} chunks of approximately {rows_per_worker} rows each")
    
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit one job per worker, each processing a chunk of the dataset
        current_row = start_row
        
        for chunk_idx in range(max_workers):
            # Calculate this worker's chunk range
            # Distribute extra rows evenly among first 'extra_rows' workers
            extra = 1 if chunk_idx < extra_rows else 0
            chunk_size = rows_per_worker + extra
            chunk_end = current_row + chunk_size
            
            print(f"Submitting chunk {chunk_idx}: rows {current_row} to {chunk_end-1}")
            future = executor.submit(
                tokenize_range,
                current_row,
                chunk_end,
                chunk_idx,
                output_dir,
                dataset_path,
                tokenizer_path,
                blowup,
            )
            futures.append(future)
            
            current_row = chunk_end
        
        # Wait for all futures to complete
        for i, future in enumerate(futures):
            try:
                future.result()  # Just wait for completion, results already written
            except Exception as e:
                print(f"Worker had an error in chunk {i}: {e}. Terminating process.")
                # Re-raise the exception to get a full stack trace
                raise
    
    print(f"\nProcessed dataset in {max_workers} chunks")

    print("\nMerging all chunk counters...")
    final_counter = Counter()
    for fname in sorted(os.listdir(output_dir)):
        if fname.startswith("chunk_") and fname.endswith(".json"):
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                chunk_dict = json.load(f)
                final_counter.update(chunk_dict)
#             os.remove(fpath)  # Optional: remove intermediate files

    final_path = os.path.join(output_dir, "token_frequencies.json")
    # these are already in our Huggingface encoding
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(dict(final_counter), f, indent=2, ensure_ascii=False)

    print(f"Final token frequencies saved to {final_path}")



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "tokenizer_name",
        type=str,
        help="Name of the tokenizer file"
    )

    parser.add_argument(
        "--tokenizer_base_dir",
        type=str,
        # on my mac:
        default="models/",
        help="Directory of tokenizer model files"
    )

    # TODO: how can it be the default
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. If not specified, use all CPU cores - 1."
    )
    
    parser.add_argument(
        "--start-row",
        type=int,
        default=170721,
        help="Starting row index for processing"
    )
    
    parser.add_argument(
        "--end-row",
        type=int,
        default=1000000,
        help="Ending row index for processing (exclusive)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ablation",
        help="Output directory"
    )
        
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/minipile.jsonl",
        help="Path to the JSONL dataset file"
    )
    
    return parser.parse_args()


if __name__ == "__main__":

    start_time = time.time()

    args = parse_arguments()

    # probably starts with pypyfasterhalfdirect_1000000_1000000000_131072_
    # which we can trim off to keep the directory names shorter
    tokenizer_path = os.path.join(args.tokenizer_base_dir, args.tokenizer_name)
    # still has the .model here
    assert not tokenizer_path.endswith(".model")
    tokenizer_path += ".model"

    # example: pypyfasterhalfdirect_1000000_1000000000_131072_0.9_1000_gpt4o_0_40960
    # go back to the args form
    tokenizer_name = args.tokenizer_name

    fields = tokenizer_name.split("_")
    assert len(fields) == 9, f"Tokenizer name format error: expected 9 fields, got {len(fields)}: {fields}"
    (_, _, _, _, _, _, _, blowup, vocab_size) = fields
    assert blowup in ("0", "1"), f"Expected blowup in ('0','1'), got {blowup}"
    assert vocab_size.isdigit(), f"vocab_size should be all digits, got {vocab_size}"

    blowup = bool(int(blowup))

    # for output can trim off the common parts                                 
    prefix =  "boundless_1000000_1000000000_131072_"
    if tokenizer_name.startswith(prefix):
        tokenizer_name = tokenizer_name.removeprefix(prefix)
    else:
        print("missing expected tokenizer name prefix: {tokenizer_name}")
    
    # just can use the tokenizer_name without the .model 
    output_dir = os.path.join(args.output_dir, tokenizer_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Tokenizer path: {tokenizer_path}")
    print(f"  Start row: {args.start_row}")
    print(f"  End row: {args.end_row}")
    print(f"  Workers: {args.workers}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output dir: {output_dir}")
    print(f"  Blowup: {blowup}")
    print(f"  Vocab Size: {vocab_size}\n")

    stream_and_tokenize_dataset(
        start_row=args.start_row,
        end_row=args.end_row,
        output_dir=output_dir,
        dataset_path=args.dataset,
        tokenizer_path=tokenizer_path,
        max_workers=args.workers,
        blowup=blowup
    )

    print("done", time.time() - start_time)