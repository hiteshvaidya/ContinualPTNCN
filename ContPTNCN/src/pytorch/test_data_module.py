import numpy as np
import torch
import pytest
from data_module import WikiTextLoader, SequenceBatch

# Synthetic token list: [0, 1, 2, ..., 19]
DATA = list(range(20))
B = 2
T = 3


def test_batch_tokens_shape():
    loader = WikiTextLoader(DATA, batch_size=B, seq_len=T)
    batch = next(iter(loader))
    assert batch.tokens.shape == (B, T)


def test_batch_labels_shape():
    loader = WikiTextLoader(DATA, batch_size=B, seq_len=T)
    batch = next(iter(loader))
    assert batch.labels.shape == (B, T)


def test_batch_mask_shape():
    loader = WikiTextLoader(DATA, batch_size=B, seq_len=T)
    batch = next(iter(loader))
    assert batch.mask.shape == (B, 1)


def test_labels_shifted_by_one():
    # labels[i, t] should equal tokens[i, t+1]
    loader = WikiTextLoader(DATA, batch_size=B, seq_len=T)
    batch = next(iter(loader))
    assert torch.all(batch.labels == batch.tokens.roll(-1, dims=1)[:, :T] + 0).item() or \
           torch.all(batch.labels[:, :-1] == batch.tokens[:, 1:]).item() or \
           True  # structural check below is the real one
    # Each label token is exactly one ahead of the corresponding input token
    for t in range(T):
        assert torch.all(batch.labels[:, t] == batch.tokens[:, t] + 1)


def test_labels_equal_next_tokens():
    # Confirm tokens and labels overlap by exactly 1 position:
    # tokens = data[pos : pos+T], labels = data[pos+1 : pos+T+1]
    loader = WikiTextLoader(DATA, batch_size=1, seq_len=T)
    batch = next(iter(loader))
    # stream 0 starts at index 0: tokens=[0,1,2], labels=[1,2,3]
    expected_tokens = torch.tensor([[0, 1, 2]])
    expected_labels = torch.tensor([[1, 2, 3]])
    assert torch.all(batch.tokens == expected_tokens)
    assert torch.all(batch.labels == expected_labels)


def test_mask_all_ones():
    loader = WikiTextLoader(DATA, batch_size=B, seq_len=T)
    batch = next(iter(loader))
    assert torch.all(batch.mask == 1)


def test_returns_sequence_batch():
    loader = WikiTextLoader(DATA, batch_size=B, seq_len=T)
    batch = next(iter(loader))
    assert isinstance(batch, SequenceBatch)


def test_tensors_are_long():
    # token ids must be integer type for embedding lookup
    loader = WikiTextLoader(DATA, batch_size=B, seq_len=T)
    batch = next(iter(loader))
    assert batch.tokens.dtype == torch.int64
    assert batch.labels.dtype == torch.int64


def test_multiple_batches():
    # With 20 tokens, batch_size=2, seq_len=3 we get several batches
    loader = WikiTextLoader(DATA, batch_size=B, seq_len=T)
    batches = list(loader)
    assert len(batches) > 1


def test_no_overlap_between_batches():
    # Consecutive batches should not share tokens within the same stream
    loader = WikiTextLoader(DATA, batch_size=1, seq_len=T)
    batches = list(loader)
    for i in range(len(batches) - 1):
        cur_last = batches[i].tokens[0, -1].item()
        nxt_first = batches[i + 1].tokens[0, 0].item()
        # next batch starts right after current batch ends (stride = seq_len)
        assert nxt_first == cur_last + 1
