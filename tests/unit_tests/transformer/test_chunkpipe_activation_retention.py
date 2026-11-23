from types import SimpleNamespace

import pytest

from megatron.core.transformer.attention import Attention


def _attention(config, training=True, num_chunks_per_seq=4):
    return SimpleNamespace(
        config=SimpleNamespace(**config),
        training=training,
        num_chunks_per_seq=num_chunks_per_seq,
    )


def test_chunkpipe_retention_requires_chunkpipe():
    attention = _attention({"enable_chunkpipe": False})
    with pytest.raises(RuntimeError, match="Chunkpipe"):
        Attention.is_enable_grad_chunkpipe(attention)


@pytest.mark.parametrize(
    ("config", "training", "expected"),
    [
        (
            {
                "enable_chunkpipe": True,
                "chunkpipe_forward": False,
                "keep_activations_chunks": 0,
            },
            True,
            True,
        ),
        (
            {
                "enable_chunkpipe": True,
                "chunkpipe_forward": True,
                "sft_chunkpipe_mode": False,
                "chunk_num_per_seq": 4,
                "chunkpipe_forward_microbatch": 2,
                "keep_activations_chunks": 1,
            },
            False,
            False,
        ),
        (
            {
                "enable_chunkpipe": True,
                "chunkpipe_forward": True,
                "sft_chunkpipe_mode": False,
                "chunk_num_per_seq": 4,
                "chunkpipe_forward_microbatch": 2,
                "keep_activations_chunks": 1,
            },
            True,
            False,
        ),
        (
            {
                "enable_chunkpipe": True,
                "chunkpipe_forward": True,
                "sft_chunkpipe_mode": False,
                "chunk_num_per_seq": 4,
                "chunkpipe_forward_microbatch": 3,
                "keep_activations_chunks": 1,
            },
            True,
            True,
        ),
        (
            {
                "enable_chunkpipe": True,
                "chunkpipe_forward": True,
                "sft_chunkpipe_mode": True,
                "chunkpipe_chunk_idx_in_group": 1,
                "chunkpipe_current_group_size": 4,
                "keep_activations_chunks": 1,
            },
            True,
            False,
        ),
        (
            {
                "enable_chunkpipe": True,
                "chunkpipe_forward": True,
                "sft_chunkpipe_mode": True,
                "chunkpipe_chunk_idx_in_group": 3,
                "chunkpipe_current_group_size": 4,
                "keep_activations_chunks": 1,
            },
            True,
            True,
        ),
    ],
)
def test_chunkpipe_retention_boundary(config, training, expected):
    attention = _attention(config, training=training)
    assert Attention.is_enable_grad_chunkpipe(attention) is expected
