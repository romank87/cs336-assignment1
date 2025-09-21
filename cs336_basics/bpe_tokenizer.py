import heapq
import os
import pickle
import re
from collections import defaultdict
from multiprocessing import Process
from pathlib import Path

import regex as re

import os
from typing import BinaryIO

from networkx.classes import neighbors


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def re_split(chunk, special_tokens):
    return re.split("|".join([re.escape(token) for token in special_tokens]), chunk)


pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def generate_pre_tokens(doc: str):
    return re.findall(pat, doc)


def proc_func(index, chunk, special_tokens):
    docs = re_split(chunk, special_tokens)

    pre_tokens = defaultdict(int)
    for doc in docs:
        lst = generate_pre_tokens(doc)
        for pre_token in lst:
            key = tuple(bytes([val]) for val in pre_token.encode("utf-8", errors="ignore"))
            pre_tokens[key] += 1

    with open(f"/tmp/tokenizer_tmp_{index}.pkl", "wb") as fd:
        pickle.dump(pre_tokens, fd)


def init_pre_tokens(input_path, special_tokens):
    num_processes = os.cpu_count()
    # num_processes=1
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_processes, split_special_token=b"<|endoftext|>")

        processes = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            p = Process(target=proc_func, args=(i, chunk, special_tokens))
            p.start()
            processes.append(p)

        pre_tokens = defaultdict(int)
        for index, p in enumerate(processes):
            p.join()
            with open(f"/tmp/tokenizer_tmp_{index}.pkl", "rb") as fd:
                for k, v in pickle.load(fd).items():
                    pre_tokens[k] += v
            print(f"Process f{index} done")

        return pre_tokens


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    from types import SimpleNamespace as sn

    vocab = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    pre_tokens_dict = init_pre_tokens(input_path, special_tokens)

    def _invert_token(token: bytes) -> tuple[int, ...]:
        """Produce a tuple that inverts lexicographic order for use in the heap."""
        return tuple(-b for b in token) + (-len(token),)

    def _heap_item(count: int, pair: tuple[bytes, bytes], version: int):
        key = (_invert_token(pair[0]), _invert_token(pair[1]))
        return (-count, key, version, pair)

    def build_pair_index():
        pair_index = defaultdict(lambda: sn(pre_tokens=set(), count=0, ver=0))
        for pre_token, count in pre_tokens_dict.items():
            for i in range(1, len(pre_token)):
                pair = tuple((pre_token[i - 1], pre_token[i]))
                pair_index[pair].pre_tokens.add(pre_token)
                pair_index[pair].count += count

        heap = [_heap_item(struct.count, pair, 0) for pair, struct in pair_index.items()]
        heapq.heapify(heap)

        return pair_index, heap

    pair_index, heap = build_pair_index()
    print(f"Pair index size: {len(pair_index)}")

    merges = []

    def run_single_merge_with_index():
        if len(vocab) % 1000 == 0:
            print(f"vocab size: {len(vocab)}. Pair index size: {len(pair_index)}")

        # pair, struct = max(pair_index.items(), key=lambda x: x[1].count)
        # maxval = struct.count
        # candidates = [(p, st) for p, st in pair_index.items() if st.count == maxval]
        # pair, struct = max(candidates)


        neg_count, _, ver, pair = heapq.heappop(heap)
        while pair not in pair_index or pair_index[pair].count != -neg_count or pair_index[pair].ver != ver:
            neg_count, _, ver, pair = heapq.heappop(heap)

        struct = pair_index[pair]

        merges.append(pair)
        vocab[len(vocab)] = b''.join(pair)

        pairs = set()
        for pre_token in list(struct.pre_tokens):
            if pre_token not in pre_tokens_dict:
                continue

            local_count = pre_tokens_dict[pre_token]
            original_pre_token = pre_token

            # for given pre-token, decrease count of all pairs in index
            for i in range(1, len(original_pre_token)):
                p = tuple((original_pre_token[i - 1], original_pre_token[i]))
                pairs.add(p)
                pair_index[p].count -= local_count
                pair_index[p].pre_tokens.discard(original_pre_token)

            # now merge within one pre_token
            acc = [original_pre_token[0]]
            for i in range(1, len(original_pre_token)):
                if pair[1] == original_pre_token[i] and pair[0] == acc[-1]:
                    acc.pop()
                    acc.append(pair[0] + pair[1])
                else:
                    acc.append(original_pre_token[i])

            new_pre_token = tuple(acc)
            pre_tokens_dict.pop(original_pre_token)
            pre_tokens_dict[new_pre_token] = local_count
            pre_token = new_pre_token

            for i in range(1, len(pre_token)):
                p = tuple((pre_token[i - 1], pre_token[i]))
                pairs.add(p)
                pair_index[p].count += local_count
                pair_index[p].pre_tokens.add(pre_token)

        struct.pre_tokens.clear()

        for p in (pairs - {pair}):
            pair_index[p].ver = len(vocab)
            heapq.heappush(heap, _heap_item(pair_index[p].count, p, len(vocab)))

        pair_index.pop(pair)

    def run_single_merge():
        if len(vocab) % 100 == 0:
            print(f"Building vocab: {len(vocab)}/{vocab_size}")
        merge_candidates = defaultdict(int)
        best_count = 0
        for pre_token, count in pre_tokens_dict.items():
            for i in range(len(pre_token) - 1):
                merge = (pre_token[i], pre_token[i + 1])
                merge_candidates[merge] += count
                best_count = max(best_count, merge_candidates[merge])

        values = [merge for merge, count in merge_candidates.items() if count == best_count]
        best_merge = max(values)

        # Update merges and vocab
        merges.append(best_merge)
        vocab[len(vocab)] = b''.join(best_merge)

    while len(vocab) < vocab_size:
        run_single_merge_with_index()
        # run_single_merge()
        # pre_tokens_dict = update_pre_tokens(pre_tokens_dict, merges[-1])

    return vocab, merges


def update_pre_tokens(pre_tokens_dict, best_merge):
    new_pre_tokens = {}
    for pre_token, count in list(pre_tokens_dict.items()):
        new_pre_token = pre_token

        pairs = zip(pre_token[:-1], pre_token[1:])
        if best_merge in pairs:
            acc = [pre_token[0]]
            for i in range(1, len(pre_token)):
                if best_merge[1] == pre_token[i] and best_merge[0] == acc[-1]:
                    acc.pop()
                    acc.append(best_merge[0] + best_merge[1])
                else:
                    acc.append(pre_token[i])
            new_pre_token = tuple(acc)

        assert new_pre_token not in new_pre_tokens
        new_pre_tokens[new_pre_token] = count
    return new_pre_tokens


if __name__ == "__main__":
    input_path = Path(__file__).parent / "../data/TinyStoriesV2-GPT4-train.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    print("Done!")
    # print("Vocab:")
    # for k, v in vocab.items():
    #     print(f"{k}: {v}")
    # print("\nMerges:")
    # for merge in merges:
    #     print(merge)
