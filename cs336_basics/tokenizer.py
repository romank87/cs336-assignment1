from __future__ import annotations
import os
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
    def from_dir(cls, dir: str, special_tokens: Optional[List[str]] = None, ) -> "Tokenizer":
        vocab_path = os.path.join(dir, "vocab.txt")
        merges_path = os.path.join(dir, "merges.txt")
        return cls.from_files(vocab_path, merges_path, special_tokens=special_tokens)

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
    import argparse
    from pathlib import Path
    from tqdm import tqdm

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="Tokenize datasets using a trained BPE tokenizer")
    parser.add_argument("--tokenizer-dir", type=str, default=str(script_dir / "../tokenizer/tiny_stories"), help="Directory containing vocab.txt and merges.txt")
    parser.add_argument("--dataset", type=str, default=str(script_dir / "../data/TinyStoriesV2-GPT4-train.txt"), help="Path to input dataset file")
    parser.add_argument("--output-dir", type=str, default=str(script_dir / "../tokenized"), help="Directory to save tokenized output")

    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer_dir}")
    tokenizer = Tokenizer.from_dir(str(args.tokenizer_dir), special_tokens=["<|endoftext|>"])

    # Create output filename based on tokenizer and dataset names
    tokenizer_name = args.tokenizer_dir.name
    dataset_name = args.dataset_path.stem  # filename without extension
    output_filename = f"tokenized-{tokenizer_name}-{dataset_name}.bin"
    output_path = args.output_dir / output_filename

    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tokenizing {args.dataset_path}")
    print(f"Output: {output_path}")

    # Tokenize
    with open(args.dataset_path, "r", encoding="utf-8") as f_in:
        with open(output_path, "wb") as f_out:
            for token in tqdm(tokenizer.encode_iterable(f_in), desc="Tokenizing"):
                f_out.write(token.to_bytes(2, 'little', signed=False))

    print(f"Done! Output saved to {output_path}")
