# note: I'm rerunning the 3-6 examples below 
# I did these on a C7-12 instance, and memory is running low
# I'm restarting them a C7-24 so if the smaller instance dies 
# we haven't lost all that time

import sys

from boundlessbpe import FasterHalfDirectRegexTokenizer
from boundlessbpe.regexconstants import *

filepath = "data/minipile.jsonl"

# automatically put the parameters from the log file 
def main(logfilename):

    (logfile, halfdirect, num_lines, max_bytes, vocab_size, tau, recalc, patname, blowup) = logfilename.split("_")

    assert logfile == "logfile"
    assert halfdirect == "halfdirectrerun"

    num_lines =  int(num_lines) # 1000000 # 100000   # now stop due to max_bytes of 1GB
    print("num_lines:", num_lines)
    max_bytes =  int(max_bytes) # 100000000 # 1000000000 # stop after 1GB
    print("max_bytes:", max_bytes)
    vocab_size = int(vocab_size) # 2500 # 131072 # 1000 # 131072 # 500 # 40960 # 256 + 50, 131072 ~ 128k
    print("vocab_size:", vocab_size)
    tau = float(tau) # 0.9 # deletion threshold
    print("tau:", tau)
    recalc = int(recalc) 
    print("recalc:", recalc)
    assert recalc >= 1

    print("patname:",patname)
    # break off the .txt
    blowup, txt = blowup.split(".")
    # should be a 0 or 1
    blowup = bool(int(blowup))
    print("blowup:", blowup)
    assert txt == "txt"

    print("tau:", tau)  # TODO: have two values for each
    outprefix = f"./models/boundless_{num_lines}_{max_bytes}_{vocab_size}_{tau}_{recalc}_{patname}_{int(blowup)}"
    print("outprefix:", outprefix)

    # which pattern to use
    if patname == "ultimate":
        tokenizer = FasterHalfDirectRegexTokenizer(tau, ULTIMATE_PATTERN_V1)
    if patname == "ultimate2":
        tokenizer = FasterHalfDirectRegexTokenizer(tau, ULTIMATE_PATTERN)
    elif patname == "gpt2": 
        tokenizer = FasterHalfDirectRegexTokenizer(tau, GPT2_SPLIT_PATTERN)
    elif patname == "gpt4":
        tokenizer = FasterHalfDirectRegexTokenizer(tau, GPT4_SPLIT_PATTERN)
    elif patname == "gpt4o":
        tokenizer = FasterHalfDirectRegexTokenizer(tau, GPT4O_SPLIT_PATTERN)
    else:
        assert False, "bad patname:" + patname

    tokenizer.train(filepath, outprefix, num_lines, vocab_size, recalc, blowup, max_bytes)
    tokenizer.register_special_tokens({"<|endoftext|>": vocab_size})
    # print(tokenizer.encode("<|endoftext|>hello world", allowed_special="all"))

    print("saving")
    tokenizer.save(outprefix)

    # make sure it loads
    print("loading")
    tokenizer2 = FasterHalfDirectRegexTokenizer(tau)
    tokenizer2.load(outprefix + ".model")

    print("done")
        

if __name__ == "__main__":
    
    logfilename = sys.argv[1]
    main(logfilename)


# 1GB, old v1 patterns
# python -u runboundlessbpetrain logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate_1.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate_1.txt
# python -u runboundlessbpetrain logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate_0.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate_0.txt
# python -u runboundlessbpetrain logfile_halfdirectrerun_1000000_1000000000_131072_1.1_1000_ultimate_1.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_1.1_1000_ultimate_1.txt

# 1GB, updated v2 pattern
# python -u runboundlessbpetrain logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate2_1.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate2_1.txt
# python -u runboundlessbpetrain logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate2_0.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_ultimate2_0.txt
# python -u runboundlessbpetrain logfile_halfdirectrerun_1000000_1000000000_131072_1.1_1000_ultimate2_1.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_1.1_1000_ultimate2_1.txt

# 1GB, gpt4o pattern
# python -u runboundlessbpetrain logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_gpt4o_1.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_gpt4o_1.txt
# python -u runboundlessbpetrain logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_gpt4o_0.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_0.9_1000_gpt4o_0.txt
# python -u runboundlessbpetrain logfile_halfdirectrerun_1000000_1000000000_131072_1.1_1000_gpt4o_1.txt 2>&1 | tee logfile_halfdirectrerun_1000000_1000000000_131072_1.1_1000_gpt4o_1.txt
