from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


class Vocab:
    def __init__(self, filename: str | Path) -> None:
        tokens: list[str] = []
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if parts:
                    tokens.append(parts[0])
        self._idx_to_token = tokens
        self._token_to_idx = {token: idx for idx, token in enumerate(tokens)}

    @property
    def size(self) -> int:
        return len(self._idx_to_token)

    def idx_to_token(self, idx: int) -> str:
        return self._idx_to_token[idx]


@dataclass
class SequenceBatch:
    tokens: torch.Tensor
    mask: torch.Tensor


class PaddedSequenceLoader:
    """
    Loads the repo's PTB-char files, where each line is already tokenized as a
    comma-separated integer sequence. Mini-batches are dynamically padded with -1.
    """

    def __init__(self, filename: str | Path, batch_size: int, *, shuffle: bool = True) -> None:
        self.filename = Path(filename)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequences = self._read_sequences()

    def _read_sequences(self) -> list[list[int]]:
        sequences: list[list[int]] = []
        with open(self.filename, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                sequences.append([int(token) for token in stripped.split(",")])
        return sequences

    def __len__(self) -> int:
        return math.ceil(len(self.sequences) / self.batch_size)

    def __iter__(self) -> Iterator[SequenceBatch]:
        indices = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_sequences = [self.sequences[idx] for idx in batch_indices]
            max_len = max(len(sequence) for sequence in batch_sequences)
            batch = np.full((len(batch_sequences), max_len), -1, dtype=np.int64)
            for row, sequence in enumerate(batch_sequences):
                batch[row, : len(sequence)] = sequence

            tokens = torch.from_numpy(batch)
            mask = (tokens >= 0).float()
            yield SequenceBatch(tokens=tokens, mask=mask)
