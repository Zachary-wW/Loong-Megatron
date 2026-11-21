import pytest
import torch

from megatron.core.rerun_state_machine import ChunkDataIterator


def test_chunk_data_iterator_splits_variable_chunk_table_and_preserves_metadata():
    metadata = [2, 3]
    batch = {
        "tokens": torch.arange(5).unsqueeze(0),
        "attention_mask": torch.ones(1, 1, 5),
        "chunkpipe_component_sizes": metadata,
    }
    iterator = ChunkDataIterator(2, iter([batch]), chunk_sizes=[2, 3])

    first = next(iterator)
    second = next(iterator)

    assert first["tokens"].tolist() == [[0, 1]]
    assert second["tokens"].tolist() == [[2, 3, 4]]
    assert first["attention_mask"].shape == (1, 1, 2)
    assert second["attention_mask"].shape == (1, 1, 3)
    assert first["chunkpipe_component_sizes"] is metadata
    assert second["chunkpipe_component_sizes"] is metadata


def test_chunk_data_iterator_rejects_chunk_size_length_mismatch():
    with pytest.raises(AssertionError, match="chunk_sizes length"):
        ChunkDataIterator(2, iter([{}]), chunk_sizes=[2])


def test_chunk_data_iterator_rejects_sequence_length_mismatch():
    batch = {"tokens": torch.zeros(1, 4, dtype=torch.long)}
    iterator = ChunkDataIterator(2, iter([batch]), chunk_sizes=[2, 3])

    with pytest.raises(ValueError, match="dim size 4 != sum\\(chunk_sizes\\) 5"):
        next(iterator)


def test_chunk_data_iterator_requires_next_before_get_next_chunk():
    iterator = ChunkDataIterator(2, iter([{}]), chunk_sizes=[2, 3])

    with pytest.raises(AssertionError, match="should call next before get_next_chunk"):
        iterator.get_next_chunk()
