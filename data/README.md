## Getting the data file

- Model files are included in `models/` directory
- Before running any of these scripts, you need to download the minipile dataset to the `data` directory. The easiest way is to download it from the [Hugging Face dataset page](https://huggingface.co/datasets/JeanKaddour/minipile):

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

## License

Apache 2.0 License - see LICENSE file for details.
