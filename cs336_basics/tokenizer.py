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
        self.merges_index = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []

        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

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

        # print("Encoding text of length", len(text))
        docs = [text]
        if self.special_tokens:
            docs = re.split(exp, text)
        result = []
        # print("Split done, num docs:", len(docs))
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

                while True:
                    min_idx = min((self.merges_index[pair] for pair in pairs if pair in self.merges_index),
                                  default=len(self.merges_index))
                    if min_idx >= len(self.merges_index):
                        break
                    merge = self.merges[min_idx]

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
        """
        A robust, streaming version of encode that handles chunk boundaries.
        It expects an iterable of strings (e.g., a file handle) and
        yields token IDs.
        """

        special_token_regex = re.compile(
            "(" + "|".join([re.escape(token) for token in sorted(self.special_tokens, reverse=True)]) + ")"
        )

        buffer = ""
        for chunk in iterable:  # chunk is (e.g.) one line from the file
            buffer += chunk

            parts = special_token_regex.split(buffer)

            for i in range(len(parts) - 1):
                part = parts[i]
                if not part:
                    continue

                for token_id in self.encode(part):
                    yield token_id

            buffer = parts[-1]

        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        ret = b"".join([self.vocab.get(i, b'') for i in ids])
        return ret.decode("utf-8", errors="replace")


if __name__ == "__main__":
    dir = "/Users/roman/dev/cs336/assignment1-basics/output/"

    tokenizer = Tokenizer.from_files(vocab_filepath=f"{dir}/owt_train_vocab.txt",
                                     merges_filepath=f"{dir}/owt_train_merges.txt",
                                     special_tokens=["<|endoftext|>"])
