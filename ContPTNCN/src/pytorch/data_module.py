"""
Dataloader for WikiText2 dataset.

version: 1.0
author: Hitesh Vaidya
"""

# import libraries
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Iterable
import numpy as np

# torch libraries
import torch
import torch.nn as nn


def _tokenize(line: str) -> list[str]:
    """Basic English tokenizer: lowercase + split on whitespace/punctuation."""
    import re
    return re.findall(r"[a-z]+|[0-9]+|[^\w\s]", line.lower())


def build_vocab(train_lines: list[str], max_tokens: int = 10_000) -> dict:
    """Build a {token: index} vocab from training lines."""
    from collections import Counter
    counter: Counter = Counter()
    for line in train_lines:
        counter.update(_tokenize(line))
    specials = ['<unk>', '<eos>']
    # most common tokens up to max_tokens (excluding specials)
    most_common = [tok for tok, _ in counter.most_common(max_tokens - len(specials))]
    tokens = specials + most_common
    return {tok: idx for idx, tok in enumerate(tokens)}


def encode(lines: Iterable[str], vocab: dict) -> list[int]:
    """Convert lines of text to a flat list of token ids."""
    unk_id = vocab['<unk>']
    eos_id = vocab['<eos>']
    ids: list[int] = []
    for line in lines:
        ids += [vocab.get(tok, unk_id) for tok in _tokenize(line)]
        ids.append(eos_id)
    return ids


def load_wikitext2():
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_lines = ds["train"]["text"]
    val_lines   = ds["validation"]["text"]
    test_lines  = ds["test"]["text"]
    vocab     = build_vocab(train_lines)
    train_ids = encode(train_lines, vocab)
    val_ids   = encode(val_lines,   vocab)
    test_ids  = encode(test_lines,  vocab)
    return train_ids, val_ids, test_ids, vocab


def load_ptb(data_dir: str = "../../data/ptb_char"):
    """
    Load the pre-encoded character-level PTB dataset.
    Each line in the files is a comma-separated sequence of integer token ids.
    Returns flat token id lists and a vocab dict {idx: char} built from vocab.txt.
    """
    import os
    def _read_ids(path: str) -> list[int]:
        ids: list[int] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids += [int(x) for x in line.split(",")]
        return ids

    train_ids = _read_ids(os.path.join(data_dir, "trainX.txt"))
    val_ids   = _read_ids(os.path.join(data_dir, "validX.txt"))
    test_ids  = _read_ids(os.path.join(data_dir, "testX.txt"))

    # Build vocab dict from vocab.txt for size reporting
    vocab_path = os.path.join(data_dir, "vocab.txt")
    with open(vocab_path) as f:
        chars = [line.rstrip("\n") for line in f]
    vocab = {i: ch for i, ch in enumerate(chars)}

    return train_ids, val_ids, test_ids, vocab

@dataclass
class SequenceBatch:
    tokens: torch.Tensor    # (batch_size, seq_len) - input token ids
    labels: torch.Tensor    # (batch_size, seq_len) - next-token targets (shifted by 1)
    mask:   torch.Tensor    # (batch_size, seq_len) - 1 for valid, 0 for padding


class WikiTextLoader:
    """
    Splits a flat token list into (batch_size, seq_len) chunks.

    Example with batch_size=2, seq_len=3, data=[1,2,3,4,5,6,7,8]:
        Stream 0: [1,2,3,4]
        Stream 1: [5,6,7,8]
        Batch 0 - tokens=[[1,2,3],[5,6,7]]
        Batch 1 - tokens=[[2,3,4],[6,7,8]] <- targets for batch 0 
    """ 
    def __init__(self, 
                token_ids: list[int], 
                batch_size: int, 
                seq_len: int, 
                shuffle:bool = False    # Shuffle only for training
                ) -> None:
        """
        Initialize the WikiTextLoader.
        Args:
            token_ids: List of token indices.
            batch_size: Number of sequences per batch.
            seq_len: Length of each sequence.
            shuffle: Whether to shuffle the data.
        Returns:
            Iterator of SequenceBatch objects.
        """
        self.seq_len = seq_len
        self.batch_size = batch_size
        self._shuffle = shuffle

        # Trim so total length divides evenly into batch_size streams
        total = (len(token_ids) // batch_size) * batch_size
        data = np.array(token_ids[:total], dtype=np.int64)
        # Reshape into (batch_size, stream_len)
        self._data = data.reshape(batch_size, -1)   # (batch_size, stream_len)
        self._shuffle = shuffle
        
    def __iter__(self) -> Iterator[SequenceBatch]:
        """
        Iterate over the data in batches.
        Args:
            None
        Returns:
            Iterator of SequenceBatch objects.
        """
        stream_len = self._data.shape[1]
        # Last position cannot be used as a target, so we go to stream_len-1
        positions = list(range(0, stream_len-1, self.seq_len))
        if self._shuffle:
            np.random.shuffle(positions)

        for pos in positions:
            end = min(pos + self.seq_len, stream_len - 1)
            tokens = torch.from_numpy(self._data[:, pos:end].copy())       # (B, T)
            labels = torch.from_numpy(self._data[:, pos+1:end+1].copy())   # (B, T) shifted by 1
            mask   = torch.ones(tokens.shape[0], 1)                        # (B, 1) all valid
            yield SequenceBatch(tokens=tokens, labels=labels, mask=mask)