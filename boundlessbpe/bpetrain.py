"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
import time, json
from typing import Optional, Tuple

from .util import frombytes, verify_dicts

from .uniformbase import UniformTokenizer, merge, min_merge, get_stats_simple
from .regexconstants import *


class OnePassRegexTokenizer(UniformTokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = FULL_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        
        # TODO: these need more testing
        self.special_tokens = {}
        self.inverse_special_tokens = {}

        # two parallel lists of equal dimension
        # the text_chunks are split into single bytes initially 
        # and merged over time
        # the corresponding text_counts are used for adjusting counts 
        # in pair_counts.  These are static over time
        self.text_chunks : Optional[list[list[bytes]]] = None
        self.text_counts : Optional[list[int]] = None

        # count of each pair of adjacent tokens in self.text_chunks
        self.pair_counts : Optional[dict[tuple[bytes,bytes], int]] = None
        
        # counts of each individual tokens in self.text_chunks
        self.single_counts : Optional[dict[bytes, int]] = None

        # should alays be the sum of self.single_counts
        self.corpus_token_count : Optional[int] = None

        # should always be the sum of the squares of the self.single_counts
        self.corpus_square_count : Optional[int] = None

        # iteratively merge the most common pairs to create new tokens
        # the value of merges records the order index
        # with merges we want to take the merge with the smallest index first
        self.merges : Optional[dict[tuple[bytes,bytes],int]] = None  # (bytes, bytes) -> int
        self.vocab : Optional[dict[bytes,int]] = None # bytes -> int

        # keep the single bytes around for computing the total number of single bytes
        self.single_bytes : Optional[set[bytes]] = None 

    # work at a chunk level with counts
    def pretokenize(self, filepath : str, num_lines : int) -> None:

        # should have already set pattern
        assert len(self.pattern) > 0

        start_time = time.time()

       # get counts of pre-tokenized chunks for each document
        chunk_tally : dict[str, int] = {}

        with open(filepath) as f:
            for i in range(num_lines):
                if i % 10000 == 0:
                    print("document", i, time.time() - start_time)
                line = f.readline()
                text = json.loads(line.rstrip())["text"]

                # split the text up into text chunks
                for chunk in re.findall(self.compiled_pattern, text):
                    chunk_tally[chunk] = chunk_tally.get(chunk, 0) + 1

        # lets make parallel list of chunks and counts before we split up the chunks
        # sort descending by count for neatness, TODO: can take out later
        cnt_chk = sorted([(cnt,chk) for chk,cnt in chunk_tally.items()], reverse=True)

        # write out a report
        # with open("text_chunk_report.txt", "wt") as out:
        #     for i, (cnt, chk) in enumerate(cnt_chk):
        #         out.write(str(i) + " " + str(cnt) + " " + frombytes(chk.encode("utf-8")) + "\n")

        print("number of pre-toknization chunks:", len(cnt_chk))
        print("top 10 pre-tokenization chunks:")
        for j in range(10):
            print(j, cnt_chk[j])
        print()

        # store in tokenizer
        self.text_counts, self.text_chunks = zip(*cnt_chk)
        
        # convert string to bytes, and split into single bytes to get started
        self.text_chunks = [[bytes([b]) for b in ch.encode("utf-8")] for ch in self.text_chunks]
        
        # should always stay in parallel
        assert len(self.text_chunks) == len(self.text_counts)

        total_counts = sum(self.text_counts)

        print("pretokenization time:",time.time() - start_time, len(self.text_counts), total_counts)


    # all the setup we only do before the first pass
    def initial_setup(self, filepath : str, num_lines : int) -> None:

       # set up self.text_counts and self.text_chunks
        self.pretokenize(filepath, num_lines)

        # only set these up once, keep going on the second round
        # iteratively merge the most common pairs to create new tokens
        # the value of merges records the order index
        # with merges we want to take the merge with the smallest index first
        self.merges = {} # (bytes, bytes) -> int

        # these bytes never appear in valid utf-8
        # see https://aclanthology.org/2024.emnlp-main.649/ Appendix C
        not_in_utf8 = set([192,193] + list(range(245,256)))
        # get the list of good bytes
        vocab = [bytes([idx]) for idx in range(256) if idx not in not_in_utf8]
        # they won't all line up now, due to skipping some
        # so give them consecutive indices
        self.vocab = {v : idx for idx, v in enumerate(vocab)} # bytes -> int

        # keep the single bytes around for computing the total number of single bytes
        self.single_bytes = set(self.vocab.keys())

    def print_tokenization(self):

        print()
        print("final tokenization:")
        for tokens, cnt in zip(self.text_chunks, self.text_counts):

            # I'm curious about ones with a single byte in them
            minlen = min([len(tok) for tok in tokens])
            output = " ".join([frombytes(tok) for tok in tokens])

            print("!", minlen, len(tokens), output, cnt)

    #############################################################################

    # return the pair counts    
    def _calc_pair_counts(self):

        pair_counts : dict[Tuple[bytes,bytes], int] = {}
        for tokens, cnt in zip(self.text_chunks, self.text_counts):
            # passing in stats will update it in place, adding up counts, 
            # incrementing each occurence by cnt
            # be sure to use get_stats_simple here to handle overlapping tokens correctly
            get_stats_simple(tokens, pair_counts, cnt)

        return pair_counts

    # get initial pair_counts, which we'll keep updating
    def initial_pair_counts(self):
    
        self.pair_counts = self._calc_pair_counts()

    # double check our dynamic single counts are being updated correctly
    def verify_pair_counts(self):
        
        from_scratch = self._calc_pair_counts()
        verify_dicts(from_scratch, self.pair_counts)

    # compute single counts from scratch, and return them
    def _calc_single_counts(self, verbose=False) -> dict[bytes, int]:

        single_counts : dict[bytes, int] = {}
        for tokens, cnt in zip(self.text_chunks, self.text_counts):
            for tok in tokens:
                single_counts[tok] = single_counts.get(tok, 0) + cnt

        # also add entries for any single byte token that 
        # didn't have any counts
        added = 0 
        for sb in self.single_bytes:
            if sb not in single_counts:
                single_counts[sb] = 0
                added += 1

        if verbose and added > 0:
            print("added missing single byte tokens:", added)

        # from_scratch doesn't include the 0 count 
        # tokens that were previously 
        # in the tokenization
        # they will appear in the merge rules
        added_merge = 0
        for left, right in self.merges:
            if left not in single_counts:
                single_counts[left] = 0
                added_merge += 1 
            if right not in single_counts:
                single_counts[right] = 0            
                added_merge += 1
                
        if verbose and added_merge > 0:
            print("added extra merge rule tokens:", added_merge)

        return single_counts

    # set up the single token counts, which also update over time
    def initial_single_counts(self):
        self.single_counts = self._calc_single_counts(verbose=True)

    def _calc_token_counts(self) -> Tuple[int,int]:

        # this must already be set up
        assert self.single_counts is not None

        corpus_token_count = sum(self.single_counts.values())
        corpus_square_count = sum([cnt*cnt for cnt in self.single_counts.values()])

        return corpus_token_count, corpus_square_count
    
    def initial_token_counts(self):

        self.corpus_token_count, self.corpus_square_count = self._calc_token_counts()

    def verify_token_counts(self):

        corpus_token_count, corpus_square_count = self._calc_token_counts()

        assert self.corpus_token_count == corpus_token_count
        assert self.corpus_square_count == corpus_square_count

    # double check our dynamic single counts are being updated correctly
    def verify_single_counts(self) -> None:
        from_scratch = self._calc_single_counts()
        verify_dicts(from_scratch,  self.single_counts)

    # just compute and return the loss after the merge
    def calc_l2_loss(self, pair : Tuple[bytes, bytes]) -> float:
        a, b = pair

        c_a = self.single_counts[a]
        c_b = self.single_counts[b]
        c_ab = self.pair_counts[pair]

        m = len(self.single_counts)
        n = self.corpus_token_count
        t = self.corpus_square_count

        if a != b:
            numerator = t + c_ab*c_ab - (c_a*c_a - (c_a - c_ab)*(c_a - c_ab)) - (c_b*c_b - (c_b-c_ab)*(c_b-c_ab))        
        else:
            numerator = t - c_a*c_a + (c_a - 2*c_ab)*(c_a - 2*c_ab) + c_ab*c_ab

        denominator = (m + 1)*(n - c_ab)*(n - c_ab)
    
        loss = numerator/denominator - 1/((m + 1)*(m + 1))

        # should never go negative
        assert loss >= 0.0

        return loss
    
    def calc_median_single_count(self):
        # was np.median(np.array(list(self.single_counts.values())))
    
        # Get the values from the dictionary
        values = list(self.single_counts.values())

        # Raise an exception if the list is empty
        if not values:
            raise ValueError("No values available to compute median single count.")

        # Sort the values to compute the median
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Compute the median manually
        if n % 2 == 1:
            median_value = sorted_values[n // 2]
        else:
            median_value = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0

        return median_value
    
   # just compute and return the loss after the merge
   # from target t, which is the median
   # only allow when < 0, so have an improvement
    def calc_l1_loss(self, pair : Tuple[bytes, bytes], t : Optional[float] = None) -> float:

        # if don't pass it in (for speed) then compute it
        if t is None:
            t = self.calc_median_single_count()

        a, b = pair

        c_a = self.single_counts[a]
        c_b = self.single_counts[b]
        c_ab = self.pair_counts[pair]

        # \Delta L &=  |c_a-t| + |c_b - t|  \\ 
        #    &-\left( |c_a - c_{ab} - t| + |c_b - c_{ab} - t| + |c_{ab}-t| \right)

        # want it below zero for an improvement
        loss =  ( abs(c_a-c_ab-t) + abs(c_b-c_ab-t) + abs(c_ab-t) ) - (abs(c_a-t) + abs(c_b-t))

        return loss


    # verify it all
    def verify_state(self) -> None:

        self.verify_pair_counts()
        self.verify_single_counts()
        self.verify_token_counts()


    # compute our initial state from self.text_counts 
    def initial_counts(self) -> None:

        self.initial_pair_counts()
        self.initial_single_counts()
        self.initial_token_counts()

    
    # what is the overall loss of our vocab
    def calc_mse(self) -> float:

        scale = float(1.0/self.corpus_token_count)

        m = len(self.single_counts)
        pbar = 1.0/m

        return sum([(scale*cnt-pbar)**2 for cnt in self.single_counts.values()])/m
        
    # compute the mean absolute deviation from the median
    def calc_mad(self) -> float:
        # Get the values from the dictionary
        values = list(self.single_counts.values())

        # Raise an exception if the list is empty
        if not values:
            raise ValueError("No values available to compute mean absolute deviation.")

        # Sort the values to compute the median
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Compute the median manually
        if n % 2 == 1:
            median_value = sorted_values[n // 2]
        else:
            median_value = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0

        # Calculate the mean absolute deviation from the median
        total_abs_deviation = sum(abs(x - median_value) for x in values)
        mad = total_abs_deviation / n
        
        return mad

    # a specialized version 
    def choose_max_cnt_pair(self) -> Optional[Tuple[bytes, bytes]]:
        
        max_cnt = -1
        max_cnt_pair = None
                
        # find the k pairs with the highest max_count
        for pair, cnt in self.pair_counts.items():

            # break ties deterministically
            if (cnt > max_cnt) or (cnt == max_cnt and pair < max_cnt_pair):
                max_cnt = cnt 
                max_cnt_pair = pair

        return max_cnt_pair

    # a specialized version 
    def choose_min_l2_pair(self) -> Optional[Tuple[bytes, bytes]]:
        
        # we really want a max heap here, so we'll flip signs
        min_loss = 1e100
        min_loss_pair = None

        # find the k pairs with the smallest min_loss
        for pair, cnt in self.pair_counts.items():

            # compute the losses from scratch
            loss = self.calc_l2_loss(pair)

            if (loss < min_loss) or (loss == min_loss and pair < min_loss_pair):
                min_loss = loss
                min_loss_pair = pair

        return min_loss_pair
    
    # a specialized version 
    def choose_min_l1_pair(self) -> Optional[Tuple[bytes, bytes]]:
        
        # we really want a max heap here, so we'll flip signs
        min_loss = 1e100
        min_loss_pair = None

        # compute the current median value, which will be the target value
        # that way we don't have to do it for each pair
        median = self.calc_median_single_count()

        # find the k pairs with the smallest min_loss
        for pair, cnt in self.pair_counts.items():

            # compute the losses from scratch
            # only an improvement if negative
            # is just a delta from the previous mad 
            loss = self.calc_l1_loss(pair, median)

            # TODO: do we want to limit to loss < 0.0
            if loss < 0.0:

                if (loss < min_loss) or (loss == min_loss and pair < min_loss_pair):
                    min_loss = loss
                    min_loss_pair = pair

        # this is None if all losses are non-negative
        return min_loss_pair
    
    def choose_best_pair(self, method : str):
        
        if method ==  "count":
            best_pair = self.choose_max_cnt_pair()
        elif method == "l1":
            best_pair = self.choose_min_l1_pair()
        else:
            assert method == "l2"
            best_pair = self.choose_min_l2_pair()

        assert best_pair not in self.merges  # should be a new merge

        # This is None if L1 loss is positive
        return best_pair


    def merge_and_update(self, max_pair : Tuple[bytes,bytes]) -> Tuple[int,int]:

        # unpack once 
        first, second = max_pair

        # the combined changes over all text_chunks
        # this way we can track the number of changing values from this merge
        # and only update those values in 
        overall_change = {}

        # how many of the merged count were actually created
        # need this to be accurate even with overlapping merges
        total_merge_cnt = 0
        whole_word_increase = 0

        for tokens, cnt in zip(self.text_chunks, self.text_counts):

            # this is usually False, so is a big speedup
            if (first in tokens) and (second in tokens):

                # just find the deltas in the current text_chunk
                # handling overlapping tokens correctly
                # handling overlapping stuff correctly
                local_delta = get_stats_simple(tokens, counts=None, multiplier=-cnt)

                # as another short cut, we must have found our adjacent pair 
                # or there is no change no other computation needed on this iteration
                if max_pair not in local_delta:
                    continue

                before_len = len(tokens)

                # then merge each chunk independently
                # replace all occurrences of pair in tokens with the combined
                # the ending number may be less than the number of pairs
                # this is the unweighted number of merges
                newtokens, merge_cnt = merge(tokens, max_pair)
                # use the slice notation so the change persists outside of the iteration
                tokens[:] = newtokens

                # this tokens appears cnt times in the pre-tokenization                 
                total_merge_cnt += merge_cnt*cnt

                # and finally, increment local_delta for each ending pair
                get_stats_simple(tokens, counts=local_delta, multiplier=cnt)

                # copy over the local ones to the overall change
                # ignore the ones that cancelled out, which should be many 
                for pair, delta in local_delta.items():
                    if delta != 0:
                        overall_change[pair] = overall_change.get(pair, 0) + delta

                        if overall_change[pair] == 0:
                            del overall_change[pair]

                if len(tokens) == 1 and (before_len > 1):
                    whole_word_increase += cnt

        for pair, delta in overall_change.items():
            if delta != 0:
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + delta

                # delete when we have a 0 entry
                if self.pair_counts[pair] == 0:
                    del self.pair_counts[pair]

        # should no longer have max_pair
        assert max_pair not in self.pair_counts

        # need to handle the case where first == second differently
        if first != second:
            self.corpus_square_count -= self.single_counts[first]*self.single_counts[first]
            self.corpus_square_count -= self.single_counts[second]*self.single_counts[second]
        else:
            # don't double deduct when same
            self.corpus_square_count -= self.single_counts[first]*self.single_counts[first]

        # add the squares of the updated new values c_i - c_{ab}
        if first != second:
            self.corpus_square_count += (self.single_counts[first]-total_merge_cnt)*(self.single_counts[first]-total_merge_cnt)
            self.corpus_square_count += (self.single_counts[second]-total_merge_cnt)*(self.single_counts[second]-total_merge_cnt)
        else:
            # note we decrase single count double here
            self.corpus_square_count += (self.single_counts[second]-2*total_merge_cnt)*(self.single_counts[second]-2*total_merge_cnt)
        # plus the new token
        self.corpus_square_count += total_merge_cnt*total_merge_cnt

        # now also should update the 3 single counts that changed because of the merge
        merged_pair = first + second
        assert merged_pair not in self.single_counts
        self.single_counts[merged_pair] = total_merge_cnt
        # and decrease the count of first and second by the same amount
        self.single_counts[first] -= total_merge_cnt
        self.single_counts[second] -= total_merge_cnt

        # since we add this to one, and subtract from two
        self.corpus_token_count -= total_merge_cnt

        # the single counts decreased by total_merge_cnt
        return total_merge_cnt, whole_word_increase


    # file_path : the location of the MiniPile *.jsonl file
    # num_lines : how many documents to read from MiniPile, out of 1M
    # vocab_size : when to stop
    # method : "count", "l2", "l1"
    # verbose : should we print output?
    def train(self, 
              filepath : str, 
              outprefix : str,
              num_lines : int, 
              vocab_size: int, 
              method : str,

              verbose:bool = True) -> None:
        
        assert method in ["count", "l2", "l1"]

        assert vocab_size >= 256
        num_merges = vocab_size - 256  # leave room for single bytes

        start_overall = time.time()

        # do the setup we only want do to once
        self.initial_setup(filepath, num_lines)

        # get an initial count of this, which we update
        whole_words = sum([cnt for tc,cnt in zip(self.text_chunks, self.text_counts) if len(tc) == 1])

        # timing splits
        total_merge = 0.0
        total_max_value = 0.0
        total_verify = 0.0

        # set up our initial counts
        start_ic = time.time()
        self.initial_counts()

        # must have set up the two previous first
        total_ic = time.time() - start_ic
        
        # time each iteration between every iterations
        start_time = time.time()

        if verbose:

            header = ["*", "i", "n", "vocab", "method", \
                    "left_tok", "right_tok", "c_ab", "c_a", "c_b", "l2", "l1", "time", \
                    "pair_cnt", "single_byte_cnt",  "zero_tokens", "whole_words", "corpus_token_count"]
            
            print("\t".join(header))

        for i in range(num_merges):

            # we have to stop when everything is merged into a single token
            # hence there are no pairs with a count
            if len(self.pair_counts) == 0:
                print("only single element chunks", i)
                break

            start_max = time.time()
            best_pair = self.choose_best_pair(method)
            total_max_value += (time.time() - start_max)
            
            # should break out of while loop if didn't find anything
            # because of positive loss in L1 
            if best_pair is None:
                print("best is None")
                break

            left, right = best_pair

            # the c_ab and best_loss we used in choose_best_pair 
            # to select best_pair may have changed by now
            # due to previous merges in batch, so recompute
            c_ab = self.pair_counts[best_pair]

            # make sure we get a non-negative value
            best_l2_loss = self.calc_l2_loss(best_pair)
            best_l1_loss = self.calc_l1_loss(best_pair)

            # look up the corresponding single values to for output
            c_a = self.single_counts[left]
            c_b = self.single_counts[right]

            # do the merges while updating pair_counts as appropriate
            start_merge = time.time()
            total_merge_cnt, whole_word_increase = self.merge_and_update(best_pair)
            total_merge += (time.time() - start_merge)

            # did we merge what we expected to?
            assert c_ab == total_merge_cnt

            whole_words += whole_word_increase

            # create new token
            merged_tok = left + right
            assert merged_tok not in self.vocab

            # save the merge and vocab with next available index
            self.merges[best_pair] = len(self.merges)
            self.vocab[merged_tok] = len(self.vocab)  # bigger than merges due to single bytes

            # should stay in sync with self.single_counts
            if len(self.vocab) != len(self.single_counts):
                print("vocab sizes out of sync:", len(self.vocab), len(self.single_counts))
            assert len(self.vocab) == len(self.single_counts)

            # we print this at the end, so compute even if not verbose
            single_byte_cnt = sum([self.single_counts.get(tok,0) for tok in self.single_bytes])

            zero_tokens = len([cnt for cnt in self.single_counts.values() if cnt == 0])

            # prints
            if verbose:

                left_tok = frombytes(left)
                right_tok = frombytes(right)

                output = ["*", i+1, num_merges, len(self.vocab), method, \
                        left_tok, right_tok, c_ab, c_a, c_b, f"{best_l2_loss:.5e}", f"{best_l1_loss:.5e}", round(time.time()-start_time,5), \
                        len(self.pair_counts), single_byte_cnt, zero_tokens, whole_words, self.corpus_token_count]
                        
                print("\t".join([str(x) for x in output]))
                # restart another batch
                start_time = time.time()

            # verify dynamic stuff from scratch every so often
            # TODO: magic number here
            if i % 1000 == 0 and i > 0:
                verify_start = time.time()
                self.verify_state()
                verify_time = time.time() - verify_start
                total_verify += verify_time

            # save intermediate output
            if len(self.vocab) % 8192 == 0:

                overall_time = time.time() - start_overall

                print(":num_merges:", num_merges)
                print(":len(vocab):", len(self.vocab))
                print(":method:", method)
                print()
                print(":pair_counts:", len(self.pair_counts)), 
                print(":single_byte_cnt:", single_byte_cnt)
                print(":zero_tokens:", zero_tokens)
                print(":whole_words:", whole_words)
                print(":corpus_token_count:", self.corpus_token_count)
                print(":mse:", self.calc_mse())
                print(":mad:", self.calc_mad())
                print()
                print(":training time breakdown")
                print(":total_initalize_counts:", total_ic)
                print(":total_max_value:", total_max_value)
                print(":total_merge:", total_merge)
                print(":total_verify:", total_verify)
                print(":overall_time:", overall_time)
                print(":missing:", overall_time - total_ic - total_max_value - total_merge - total_verify)

                outfile = outprefix + "_" + str(len(self.vocab))
                print(":outprefix:", outfile)
                self.save(outfile)


        # write our vocab with counts
        print()
        print("vocab:")
        for tok, index in self.vocab.items():
            print("+", index, frombytes(tok), self.single_counts.get(tok, 0))
        print()

        self.print_tokenization()

        # compute the inverse vocabulary at the end for decoding
        # TODO: test decoding
        self.inv_vocab = { v : k for (k,v) in self.vocab.items()}  # int -> bytes

        overall_time = time.time() - start_overall

        print(":num_merges:", num_merges)
        print(":len(vocab):", len(self.vocab))
        print(":method:", method)
        print()
        print(":pair_counts:", len(self.pair_counts)), 
        print(":single_byte_cnt:", single_byte_cnt)
        print(":zero_tokens:", zero_tokens)
        print(":whole_words:", whole_words)
        print(":corpus_token_count:", self.corpus_token_count)
        print(":mse:", self.calc_mse())
        print(":mad:", self.calc_mad())
        print()
        print(":training time breakdown")
        print(":total_initalize_counts:", total_ic)
        print(":total_max_value:", total_max_value)
        print(":total_merge:", total_merge)
        print(":total_verify:", total_verify)
        print(":overall_time:", overall_time)
        print(":missing:", overall_time - total_ic - total_max_value - total_merge - total_verify)
        

    # TODO: test this
    def register_special_tokens(self, special_tokens : list[str]):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids : list[int]) -> str:
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.inv_vocab:
                part_bytes.append(self.inv_vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        # won't always decode to a string
        # TODO: return a bytes instead?
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    # TODO: turn off verbose at some point
    def _encode_chunk(self, text_bytes:bytes, verbose=True) -> list[int]:
        # given bytes, return the token ids
        tokens = [bytes([b]) for b in text_bytes] # list of single byte bytes objects to get started
        # stop if we get down to a single token
        while len(tokens) >= 2:
            # get the merge with the min index to do next
            pair = min_merge(tokens, self.merges)
            if pair is None:
                break # nothing else can be merged anymore
            tokens, _ = merge(tokens, pair)

        if verbose:
            print(tokens)

        # look up the token ids before returning 
        return [self.vocab[tok] for tok in tokens]

    def encode_ordinary(self, text:str) -> list[int]:
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text:str, allowed_special:str="none_raise") -> list[int]:
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
