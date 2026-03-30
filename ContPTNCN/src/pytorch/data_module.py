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


def load_wikitext2():
    # Lazy imports so that data_module can be imported without torchtext installed
    from torchtext.datasets import WikiText2
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

    tokenizer = get_tokenizer("basic_english")

    def _build_vocab(train_lines: list[str]):
        def _yield(lines):
            for line in lines:
                yield tokenizer(line)
        vocab = build_vocab_from_iterator(
            _yield(train_lines),
            specials=['<unk>', '<eos>'],
            max_tokens=10_000
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def _encode(data_iter: Iterable[str], vocab) -> list[int]:
        tokens = []
        for line in data_iter:
            tokens += vocab(tokenizer(line)) + [vocab['<eos>']]
        return tokens

    train_iter, val_iter, test_iter = WikiText2()
    train_lines = list(train_iter)   # buffer so we can iterate twice
    vocab = _build_vocab(train_lines)
    train_ids = _encode(train_lines, vocab)
    val_ids   = _encode(val_iter, vocab)
    test_ids  = _encode(test_iter, vocab)
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