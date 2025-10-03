from boundlessbpe import OnePassRegexTokenizer
from boundlessbpe.regexconstants import *

filepath = "data/minipile.jsonl"
method = "count"
num_lines =  1000000 # 10000 # 100000
vocab_size = 131072 # 40960
outprefix = f"./models/onepass_{method}_{num_lines}_{vocab_size}"

tokenizer = OnePassRegexTokenizer(GPT4O_SPLIT_PATTERN)

tokenizer.train(filepath, outprefix, num_lines, vocab_size, method)
tokenizer.register_special_tokens({"<|endoftext|>": vocab_size})
print(tokenizer.encode("<|endoftext|>hello world", allowed_special="all"))

print("outprefix:", outprefix)

print("saving")
tokenizer.save(outprefix)

# make sure it loads
print("loading")
tokenizer2 = OnePassRegexTokenizer(GPT4O_SPLIT_PATTERN)
tokenizer2.load(outprefix + ".model")

print("done")
    
# python -u runbpe.py 2>&1 | tee logfile_onepass_count_1M_40960_1.txt