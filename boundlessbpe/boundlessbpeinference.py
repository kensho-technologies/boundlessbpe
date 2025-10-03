"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

- this will be regular BPE with Picky BPE deletions

"""

from typing import Optional, Tuple
# note: need the fancy regex here for character classes
import regex as re
import json
import time

from .util import frombytes, frombytespair, _read_sorted_dict_intkey, blow_up

from .uniformbase import merge, _read_sorted_dict
from .regexconstants import *

# TODO: put a limit on token length of 16 here for whitespace?


# verify indices are correct
def verify_indicies(words_state_merges, 
                    words_state_deletions, 
                    superwords_state_merges, 
                    superwords_state_deletions) -> None:

    indices = sorted(list(words_state_merges.keys()) + \
                        list(words_state_deletions.keys()) + \
                        list(superwords_state_merges.keys()) + \
                        list(superwords_state_deletions.keys()))

    # should be an ordered list of indices
    for i, ind in enumerate(indices):

        # about to die so dump them
        if i != ind:

            print("debug 3:", i, ind)
            print("word merges:")
            for idx, pair in words_state_merges.items():
                print(idx, frombytespair(pair))
            print()
            print("word deletions:")
            for idx, tok in words_state_deletions.items():
                print(idx, frombytes(tok))
            print()
            print("superword merges:")
            for idx, pair in superwords_state_merges.items():
                print(idx, frombytespair(pair))
            print()
            print("superword deletions:")
            for idx, tok in superwords_state_deletions.items():
                print(idx, frombytes(tok))
            print()

        assert i == ind


class FasterRegexInference():

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """

        self.pattern = ULTIMATE_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

        self.text_chunks : Optional[list[list[bytes]]] = None

        # iteratively merge the most common pairs to create new tokens
        # the value of merges records the order index
        # with merges we want to take the merge with the smallest index first

        # flip these around for index lookup
        # note: this is stored backwards from the training code
        # which is dict[int, tuple[bytes,bytes]]
        self.merges : Optional[dict[tuple[bytes,bytes], int]] = {}       # (bytes, bytes) -> int 
        self.supermerges : Optional[dict[tuple[bytes,bytes], int]] = {}  # (bytes, bytes) -> int 
        self.deletions : Optional[dict[bytes,int]] = {}                  # bytes -> int 

        # TODO: these need more testing
        self.special_tokens = {}          # str -> int
        self.inverse_special_tokens = {}  # int -> str

        # and then it gets converted after training to a dict, 
        # once we're done with deletions
        self.vocab : Optional[dict[bytes,int]] = None
        self.inv_vocab : Optional[dict[int,bytes]] = None

        # the set of vocab where we know we can leave a pretoken as a single token
        self.fast_vocab : Optional[set[bytes]] = None

        # TODO: hardcoded, pass this in
        # must have at least one letter, then can also have spaces, underscores or apostrophes
        # use the fancier string based regex now
        self.merge_pattern = re.compile(IMPROVED_MERGE_PATTERN) # ORIGINAL_MERGE_PATTERN) #  

    def can_merge(self, left: bytes, right: bytes) -> bool:
            # decode both sides; if decoding fails, we treat it as “cannot merge”
            try:
                left_str = left.decode('utf-8')
                right_str = right.decode('utf-8')
            except UnicodeDecodeError:
                return False

            # same logic as before, but on the decoded strings
            return bool(self.merge_pattern.match(left_str)) and bool(self.merge_pattern.match(right_str))

    def could_merge(self, tok: bytes) -> bool:
        try:
            tok_str = tok.decode('utf-8')
        except UnicodeDecodeError:
            return False

        return bool(self.merge_pattern.match(tok_str))

    def can_merge_bytes(self, left: bytes, right: bytes) -> bool:
            # same logic as before, but on the decoded strings
            return bool(self.merge_pattern.match(left)) and bool(self.merge_pattern.match(right))

    def could_merge_bytes(self, tok: bytes) -> bool:
        return bool(self.merge_pattern.match(tok))



    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        with open(model_file, 'rt', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "word v1"

            # read the pattern
            self.pattern = json.loads(f.readline().strip())
            self.compiled_pattern = re.compile(self.pattern)

            self.vocab = _read_sorted_dict(f, ispair=False, isstr=False)
            self.inv_vocab = { v:k for (k,v) in self.vocab.items()}        

            self.special_tokens = _read_sorted_dict(f, ispair=False, isstr=True)  # these are strings, why?
            print("special_tokens:", self.special_tokens)
            self.inv_special_tokens = { v : k for (k,v) in self.special_tokens.items()}

            # flip these around for inference
            words_state_merges  = _read_sorted_dict_intkey(f, ispair=True, isstr=False)
            words_state_deletions  = _read_sorted_dict_intkey(f, ispair=False, isstr=False)
            superwords_state_merges  = _read_sorted_dict_intkey(f, ispair=True, isstr=False)
            superwords_state_deletions  = _read_sorted_dict_intkey(f, ispair=False, isstr=False)

            # make sure indices are consistent across these four
            verify_indicies(words_state_merges, 
                            words_state_deletions, 
                            superwords_state_merges, 
                            superwords_state_deletions)  

            # for now, I'm not supporting superwords_state_deletions
            if len(superwords_state_deletions) > 0:
                print("not supporting superwords_state_deletions yet", len(superwords_state_deletions))
                assert len(superwords_state_deletions) == 0

            self.merges      = { v : k for (k,v) in words_state_merges.items() }  # (bytes, bytes) -> int 
            self.supermerges = { v : k for (k,v) in superwords_state_merges.items() }  # (bytes, bytes) -> int 
            self.deletions   = { v : k for (k,v) in words_state_deletions.items() }    # bytes -> int 


    #################################################

    # only choose merges with an index > limit
    def min_merge(self, limit : int) -> Optional[Tuple[Tuple[bytes,bytes],int]]:
        """
        Given a list of bytes object tokens, 
        return the merge or deletion with the lowest index
        or None if none are found
        """
        min_pair = None             # returns None if nothing to merge 
        min_ind = 1e9  # initialize to a value larger than any index

        for tokens in self.text_chunks:

            for pair in zip(tokens, tokens[1:]): # iterate consecutive elements
                if pair in self.merges and self.merges[pair] > limit and self.merges[pair] < min_ind:
                    min_ind = self.merges[pair]
                    min_pair = pair

        return min_pair, min_ind
    
    def min_deletion(self, limit : int) -> Optional[Tuple[bytes,int]]:
        """
        Given a list of bytes object tokens, 
        return the merge or deletion with the lowest index
        or None if none are found
        """
        min_tok = None  # returns None if nothing to merge 
        min_ind = 1e9   # initialize to a value larger than any index

        for tokens in self.text_chunks:

            # and now check if there is a deletion with a lower count
            for tok in tokens:
                if tok in self.deletions and self.deletions[tok] > limit and self.deletions[tok] < min_ind:
                    min_ind = self.deletions[tok]
                    min_tok = tok

        return min_tok, min_ind

    def min_supermerge(self, limit : int) -> Optional[Tuple[Tuple[bytes,bytes],int]]:
        """
        Given a list of bytes object tokens, 
        return the merge or deletion with the lowest index
        or None if none are found
        """
        min_pair = None             # returns None if nothing to merge 
        min_ind = 1e9  # initialize to a value larger than any index

        # over pairs of chunks here
        for pair in zip(self.text_chunks, self.text_chunks[1:]): # iterate consecutive elements
            if len(pair[0]) == 1 and len(pair[1]) == 1:
                # now look up those elements
                derefpair = (pair[0][0], pair[1][0])
                if derefpair in self.supermerges and self.supermerges[derefpair] > limit and self.supermerges[derefpair] < min_ind:
                    min_ind = self.supermerges[derefpair]
                    min_pair = derefpair

        return min_pair, min_ind


    # what's next between a merge, a supermerge, or a deletion
    def overall_min(self, limit : int, verbose : bool):

        # keep track of the best as we go
        best = None
        best_cnt = 1e9
        best_kind = ""

        min_merge_pair, min_merge_cnt = self.min_merge(limit)
        if min_merge_cnt < best_cnt:
            best = min_merge_pair
            best_cnt = min_merge_cnt
            best_kind = "merge"

        min_deletion_tok, min_deletion_cnt = self.min_deletion(limit)
        if min_deletion_cnt < best_cnt: 
            best = min_deletion_tok
            best_cnt = min_deletion_cnt
            best_kind = "deletion"

        min_supermerge_pair, min_supermerge_cnt = self.min_supermerge(limit)
        if min_supermerge_cnt < best_cnt:
            best = min_supermerge_pair
            best_cnt = min_supermerge_cnt
            best_kind = "supermerge"

        if verbose and best is not None:
            print("* " + best_kind + ":", best, best_cnt, ) 

        return best, best_cnt, best_kind


    def supermerge(self, pair : Tuple[bytes,bytes]) -> Tuple[list[bytes],int]:
        """
        In the list of tokens, replace all consecutive occurrences
        of pair (t1,t2) with the combined token t1+t2
        Example: tokens=[b'a', b'b', b'c', b'a', b'b'], 
        pair=(b'a', b'b') -> [b'ab', b'c', b'ab']
        will have max_count merges found, 
        unless there are the pair elements are runs of the same
        """
        i = 0
        merged = pair[0] + pair[1]
        while i < len(self.text_chunks)-1:
            
            # if not at the very last position AND the pair matches, replace it
            if len(self.text_chunks[i]) == 1 and self.text_chunks[i][0] == pair[0] and len(self.text_chunks[i+1]) == 1 and self.text_chunks[i+1][0] == pair[1]:
                self.text_chunks[i][0] = merged
                del self.text_chunks[i+1]
            
            i += 1
        
    
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
    

    # convert list of list of bytes 
    # to a list of list of encoded text
    def text_chunk_encoded(self):    
        return [[frombytes(tok) for tok in ch] for ch in self.text_chunks]

    # needed for deletion
    def get_replacement_parts(self, bad_token : bytes, blowup : bool) -> list[bytes]:

        # we can either blow it up into bytes, or find the actual pair that created it
        if not blowup:

            parts = None

            # replace each bad_token with the previous pair
            # TODO: update if we allow more than one merge rule for a token
            # do we just go with the first one found in that case?
            # note: in the inference code the pair is the key,
            # not the value like in the training
            for (left, right) in self.merges:
                if left + right == bad_token:
                    parts = [left,right]
                    break
            assert parts is not None, "couldn't find merge for " + str(bad_token)

        else:
            # else blow it up to bytes
            parts = [bytes([b]) for b in bad_token]   

        return parts       


    # take a document and setup out self.text_chunks
    # where a document is stored as a list of chunks
    # returns a list of tokens, rather than the ids
    # since that can be useful sometimes
    # if blowup is True, then a deleted token is blown up into single bytes
    # as in our paper,  if False, then split into the pair that created it
    # as in the original paper.  This was used in the ablations
    def encode_ordinary_chunks(self, text : str, blowup : bool = True, verbose : bool = False) -> list[bytes]:

        if verbose:
            print("text:", len(text), text )

        # get the pre-tokenized chunks
        self.text_chunks = re.findall(self.compiled_pattern, text)

        # convert string to bytes, and split into single bytes to get started
        self.text_chunks = [[bytes([b]) for b in ch.encode("utf-8")] for ch in self.text_chunks]

        best = None
        best_ind = 1e9
        best_kind = ""  # drop into loop
        prev_ind = -1

        while True:

            if verbose:
                print("text_chunks:",self.text_chunk_encoded())

            best, best_ind, best_kind = self.overall_min(prev_ind, verbose)
            assert best_ind > prev_ind
            # ignore any lower rules going forward
            # so we don't get into a loop with deletions
            prev_ind = best_ind

            # nothing left to do
            if best is None:
                break

            if best_kind == "merge":

                first, second = best

                for tokens in self.text_chunks:

                    # this is usually False, so is a big speedup
                    possible = False

                    # does `first` appear next to `second` in the list?
                    # I'm hoping this is faster for long lists of tokens
                    # since only need one pass throught the list
                    for i in range(len(tokens) - 1):  # Iterate up to the second-to-last element
                        if tokens[i] == first and tokens[i + 1] == second:
                            possible = True
                            break

                    if possible:
                        newtokens, _ = merge(tokens, best)
                        tokens[:] = newtokens

            elif best_kind == "deletion": 

                # get the replacement values for best just once
                parts = self.get_replacement_parts(best, blowup)

                # TODO: does this stay?
                for tokens in self.text_chunks:
                    if best in tokens:
                        # ignore the deletion count
                        newtokens, _ = blow_up(tokens, best, parts)
                        tokens[:] = newtokens

            else:
                assert best_kind == "supermerge"
                self.supermerge(best)

        # flatten our list of chunks into a single list of tokens
        tokens = []
        for chunk in self.text_chunks:
            tokens.extend(chunk)

        if verbose:
            print("tokens:", [frombytes(t) for t in tokens])

        # reset self.text_chunks to None
        # to prevent accidental access
        self.text_chunks = None

        return tokens

    # finally do the integer lookup
    def encode_ordinary(self, text:str, blowup:bool=True) -> list[int]:
        """Encoding that ignores any special tokens."""

        return [self.vocab[tok] for tok in self.encode_ordinary_chunks(text, blowup)]

    def encode(self, text:str, allowed_special:str="none_raise", blowup:bool=True) -> list[int]:
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
            return self.encode_ordinary(text, blowup)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # this can have empty strings, if a special_pattern is at the start
        special_chunks = [ch for ch in special_chunks if len(ch) > 0]

        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part, blowup))
        return ids


    def register_special_tokens(self, special_tokens : list[str]):
        # special_tokens is a list of str
        # example: {"<|endoftext|>": 100257}

        # call this *after* training
        if self.vocab is None or len(self.vocab) == 0:
            print("please call register_special_tokens after training")
        assert self.vocab is not None and (len(self.vocab) > 0)

        self.special_tokens = {}
        for tok in special_tokens:
            self.special_tokens[tok] = len(self.vocab) + len(self.special_tokens)

        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()} 