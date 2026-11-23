import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


def test_rotary_inv_freq_accepts_explicit_dtype():
    rope = RotaryEmbedding(
        kv_channels=8,
        rotary_percent=1.0,
        use_cpu_initialization=True,
        dtype=torch.bfloat16,
    )

    assert rope.inv_freq.dtype == torch.bfloat16


def test_rotary_inv_freq_defaults_to_float32():
    rope = RotaryEmbedding(
        kv_channels=8,
        rotary_percent=1.0,
        use_cpu_initialization=True,
    )

    assert rope.inv_freq.dtype == torch.float32
