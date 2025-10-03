# BoundlessBPE

The repo contains the code to accompany the [BoundlessBPE paper](https://arxiv.org/abs/2504.00178).

This repository includes Python implementations of the BoundlessBPE tokenization algorithms, including:

- Training and Inference code for BoundlessBPE
- Code for regular BPE
- Code for the [PickyBPE](https://arxiv.org/abs/2409.04599) variant that includes deletions

Please report any issues or problems to the Craig Schmidt (email address in paper).

## Project Structure

```text
boundlessbpe/
├── boundlessbpe/                    # Core Python package
│   ├── __init__.py                  # Package exports (FasterRegexInference, PickyBPE, etc.)
│   ├── uniformbase.py               # Base tokenizer class and core BPE functions
│   ├── boundlessbpetrain.py         # BoundlessBPE training with supermerges & deletions
│   ├── boundlessbpeinference.py     # BoundlessBPE inference implementation
│   ├── bpetrain.py                  # Standard BPE training implementation
│   ├── pickybpe.py                  # PickyBPE with token deletion capabilities
│   ├── regexconstants.py            # Regex patterns (GPT2, GPT4O, ULTIMATE, etc.)
│   └── util.py                      # Utility functions for bytes/string conversion
├── models/                          # Pre-trained model files (.model format) used in the paper
│                                    # note that the models with a vocab size in the name do not currently have any special tokens
├── data/                            # Test datasets (minipile.jsonl - download separately)
├── baselines/                       # Baseline comparison scripts
│   ├── train_tokenizers_gpt4o.py    # Train HuggingFace tokenizers for comparison
│   └── process_hf_tokenizers_rerun.py # Corpus token counts for HuggingFace tokenizers
├── runboundlessbpetrain.py          # example of running BoundlessBPE training 
├── runboundlessinference.py         # example of running BoundlessBPE inference
├── runbpe.py                        # example of running regular BPE training
├── runpickybpe.py                   # example of running PickyBPE training
├── boundless_token_counter.py       # BoundlessBPE token counting code, which is another example of inference
├── run_boundless_token_counter.sh   # Shell script for launching BoundlessBPE token counting
└── requirements.txt                 # Python dependencies
```

## File Overview

### Core Package (`boundlessbpe/`)

- **`__init__.py`**: Package entry point, exports main tokenizer classes
- **`uniformbase.py`**: Base tokenizer class with core BPE merge logic, derived from minBPE
- **`boundlessbpetrain.py`**: BoundlessBPE training with supermerges and deletions (`FasterHalfDirectRegexTokenizer`)
- **`boundlessbpeinference.py`**: BoundlessBPE inference implementation (`FasterRegexInference`)
- **`bpetrain.py`**: Standard BPE training implementation (`OnePassRegexTokenizer`)
- **`pickybpe.py`**: "Picky" BPE with deletion capabilities (`PickyBPE`)
- **`regexconstants.py`**: Regex patterns for different tokenization schemes
- **`util.py`**: Helper functions for byte/string conversion and file I/O

### Training & Testing Scripts

- **`runboundlessbpetrain.py`**: Train FasterHalfDirectRegexTokenizer models
- **`runboundlessinference.py`**: Test inference with trained models
- **`runbpe.py`**: Train OnePassRegexTokenizer models
- **`runpickybpe.py`**: Train PickyBPE models (no regex pattern)

### Analysis & Utilities

- **`boundless_token_counter.py`**: Count the total tokens for a BounlessBPE tokenizer on the evaluation set
- **`run_boundless_token_counter.sh`**: runs of boundless_token_counter.py used in the paper

### Baselines (`baselines/`)

- **`train_tokenizers_gpt4o.py`**: Train HuggingFace BPE/Unigram/WordPiece tokenizers for comparison
- **`process_hf_tokenizers_rerun.py`**: Compute total token counts using HuggingFace tokenizers for benchmarking

## Algorithm Variants

This repository implements several BPE variants described in the BoundlessBPE paper:

1. **FasterHalfDirectRegexTokenizer**: Advanced BoundlessBPE training with supermerges and deletions
2. **FasterRegexInference**: Stand alone inference implementation for pre-trained models
3. **OnePassRegexTokenizer**: Standard BPE training algorithm, with some extra merge selection options
4. **PickyBPE**: BPE with token deletion capabilities (tau < 1.0 enables deletions). Can be used as regular BPE with tau > 1.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/boundlessbpe.git  # Update with actual URL
cd boundlessbpe

# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Inference

```python
from boundlessbpe import FasterRegexInference

# Create fast inference tokenizer
tokenizer = FasterRegexInference()

# Load your trained model
tokenizer.load("models/your_model.model")

# Tokenize text, ignoring any special tokens
# use encode if you have special tokens
text = "Hello, world! This is a test."
token_ids = tokenizer.encode_ordinary(text)
print(f"Tokens: {token_ids}")

# Decode back to text
decoded = tokenizer.decode(token_ids)
assert decoded == text

# Handle special tokens (call after training/loading)
tokenizer.register_special_tokens(["<|endoftext|>"])
tokens_with_special = tokenizer.encode("<|endoftext|>Hello world", allowed_special="all")
```

#### Training

```python
from boundlessbpe import FasterHalfDirectRegexTokenizer
from boundlessbpe.regexconstants import GPT4O_SPLIT_PATTERN

# PickyBPE deletion parameter for the IoS threshold, 
# set it to a value > 1.0 to disable deletions
tau = 0.9

# Create BoundlessBPE trainer
tokenizer = FasterHalfDirectRegexTokenizer(tau, pattern=GPT4O_SPLIT_PATTERN)

# Train on dataset
tokenizer.train(
    filepath="data/minipile.jsonl",
    outprefix="models/my_model",
    num_lines=100000,
    vocab_size=40960,
    recalc=1000  # How often to verify dynamic calculations
)

# Register special tokens after training
tokenizer.register_special_tokens({"<|endoftext|>": len(tokenizer.vocab)})

# Save the trained model
tokenizer.save("models/my_model.model")
```

#### Training Scripts

Here are some examples of how to invoke the training scripts. It is good to keep a logfile around to track progress. 

```bash
python -u runboundlessbpetrain.py logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate2_1.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate2_1.txt
python -u runboundlessbpetrain.py logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_gpt4o_1.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_gpt4o_1.txt
python -u runbpe.py 2>&1 | tee logfile_onepass_count_1M_40960_1.txt
python -u runpickybpe.py 2>&1 | tee logfile_pickbpe_0.9_1GB.txt
```

## Getting the data file

- Model files are included in `models/` directory
- Before running any of these scripts, you need to download the minipile dataset. The easiest way is to download it from the [Hugging Face dataset page](https://huggingface.co/datasets/JeanKaddour/minipile):

  ```bash
  # Create data directory
  mkdir -p data
  
  # Option 1: Use huggingface-hub (recommended)
  pip install huggingface-hub
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='JeanKaddour/minipile', filename='data/train.jsonl', local_dir='data', local_dir_use_symlinks=False)"
  mv data/data/train.jsonl data/minipile.jsonl
  
  # Option 2: Manual download
  # Visit https://huggingface.co/datasets/JeanKaddour/minipile/tree/main/data
  # Download train.jsonl and save as data/minipile.jsonl
  ```

## Training and inference speed

The training code is extremely slow. It takes about 5 days to train on 1GB of data.  The inference code is also quite slow. 
We're actively developing a much faster version of both.  You might want to wait until that is released to try training on more data.

## Requirements

- **Python**: 3.8+
- **Dependencies**:
  - Core: `regex`
  - Baselines (optional): `tokenizers`, `transformers`

## Acknowledgments

This project builds upon and extends [minBPE](https://github.com/karpathy/minbpe) by Andrej Karpathy, which is MIT licensed. Several base components, particularly in `uniformbase.py`, are derived from the original minBPE implementation, though substantially evolved and extended for the BoundlessBPE algorithm.

## License

Apache 2.0 License - see LICENSE file for details.

## Citation

If you use BoundlessBPE in your research, please cite:

```bibtex
@misc{schmidt2025boundlessbytepairencoding,
      title={Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier}, 
      author={Craig W. Schmidt and Varshini Reddy and Chris Tanner and Yuval Pinter},
      year={2025},
      eprint={2504.00178},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.00178}, 
}
```

This paper was presented at [COLM 2025](https://colmweb.org/).