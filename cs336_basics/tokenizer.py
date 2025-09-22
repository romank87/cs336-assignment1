from __future__ import annotations

import ast
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import regex as re


class Tokenizer:
    """
    Byte-Pair Encoding (BPE)-style Tokenizer interface.

    Constructed from a vocabulary and merge rules, with optional special tokens.
    """

    def __init__(
            self,
            vocab: Dict[int, bytes],
            merges: List[Tuple[bytes, bytes]],
            special_tokens: Optional[List[str]] = None,
    ) -> None:
        self.vocab = vocab
        self.invert_vocab = {v: k for k, v in vocab.items()}

        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        vocab = {}
        with open(vocab_filepath, "r") as vocab_file:
            for line in vocab_file.readlines():
                num, val = ast.literal_eval(line)
                vocab[num] = val

        merges = []
        with open(merges_filepath, "r") as merges_file:
            for line in merges_file.readlines():
                tup = ast.literal_eval(line)
                merges.append(tup)

        return Tokenizer(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        exp = "(" + "|".join([re.escape(token) for token in sorted(self.special_tokens, reverse=True)]) + ")"

        docs = [text]
        if self.special_tokens:
            docs = re.split(exp, text)
        result = []

        for doc in docs:
            if not doc:
                continue
            if doc in self.special_tokens:
                # print("Adding special token:", doc)
                result.append(doc.encode('utf-8'))
                continue

            pre_tokens = re.findall(self.pat, doc)
            for pre_token in pre_tokens:
                tokens = [bytes([b]) for b in pre_token.encode("utf-8", errors="ignore")]
                pairs = {p for p in zip(tokens[:-1], tokens[1:])}

                for merge in self.merges:
                    if len(tokens) == 1:
                        break
                    if merge not in pairs:
                        continue
                    i = 0
                    new_tokens = []
                    while i < len(tokens):
                        curr_token = tokens[i]
                        next_token = tokens[i + 1] if i + 1 < len(tokens) else None

                        if (curr_token, next_token) == merge:
                            new_tokens.append(curr_token + next_token)
                            i += 2
                        else:
                            new_tokens.append(curr_token)
                            i += 1

                    tokens = new_tokens
                    pairs = {p for p in zip(tokens[:-1], tokens[1:])}

                result.extend(tokens)

        return [self.invert_vocab[token] for token in result]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            for token_id in self.encode(it):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        ret = b"".join([self.vocab.get(i, b'') for i in ids])
        return ret.decode("utf-8", errors="replace")


if __name__ == "__main__":
    dir = "/Users/roman/dev/cs336/assignment1-basics/output/"

    tokenizer = Tokenizer.from_files(vocab_filepath=f"{dir}/owt_train_vocab.txt",
                                     merges_filepath=f"{dir}/owt_train_merges.txt",
                                     special_tokens=["<|endoftext|>"])
