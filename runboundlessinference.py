from boundlessbpe import FasterRegexInference

import time

num_lines =  100000 # 100000
vocab_size = 131072 # 1000 # 131072 # 500 # 40960 # 256 + 50, 131072 ~ 128k
tau = 0.9 # deletion threshold 
blowup = False

# inputprefix = "./models/pypyfastersuperrollbackfixupdate1GB_1000000_131072_0.9"
# inputprefix = "./models/pypyfastersuperhalfdirect_1000000_131072_0.9_131072"
inputname = "./models/boundless_1000000_1000000000_131072_0.9_1000_ultimate_0_40960.model"
# inputname = "./models/boundless_1000000_1000000000_131072_0.9_1000_ultimate_1_40960.model"

print("inputname:", inputname)

# make sure it loads
print("loading")
tokenizer = FasterRegexInference()
tokenizer.load(inputname)

# print("debug:  Kent", b' Kent' in tokenizer.deletions)
# print("debug: ĠKent", 'ĠKent' in tokenizer.deletions)

# print("debug:", (b' of', b' the') in tokenizer.supermerges)
# print("debug:", b' signific' in tokenizer.deletions)

# this handles special tokens correctly
# print(tokenizer.encode("<|endoftext|>hello world", allowed_special="all"))
# print()

# print(tokenizer.encode("King of the hill"))
# print()
# print()

# # if you just want to see the tokens rather than the int values do this
# print(tokenizer.encode_ordinary_chunks("Government of the people, by the people, for the people, shall not perish from the earth", blowup=blowup, verbose=True))
# print()
# print()

# # extercise our deletion code
# "to be, is to feel boundless wonder."
"tail of the bobcat is short"
print(tokenizer.encode_ordinary_chunks("tail of the bobcat, is short", blowup=blowup, verbose=True))
print()

start_time = time.time()
# print(tokenizer.encode_ordinary_chunks(" Kentucky Kent Ken Kentuck", blowup=blowup, verbose=True))
# print()
# print(time.time() - start_time)

# print(tokenizer.encode_ordinary_chunks("To be or not to be", blowup=blowup, verbose=True))

# print(tokenizer.encode_ordinary_chunks("significant other", blowup=blowup, verbose=True))

# print(tokenizer.encode_ordinary_chunks("Government of the people, by the people, for the people.", blowup=blowup, verbose=True))

# print(tokenizer.encode_ordinary_chunks("King of the Hill", blowup=blowup, verbose=True))

# print(tokenizer.encode_ordinary_chunks("Tip of the hat", blowup=blowup, verbose=True))

# print(tokenizer.encode_ordinary_chunks("Hello, this is a test of the tokenizer!", blowup=blowup, verbose=True))

# print(tokenizer.encode_ordinary_chunks("between, ween", blowup=blowup, verbose=True))

print("done")
    
# python -u run4fasterinference.py 2>&1 | tee best_pair_pypyfastersuper_count_100k_good.txt
