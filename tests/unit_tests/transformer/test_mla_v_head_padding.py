# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Component checks for MLA value-head padding to QK width."""

from types import SimpleNamespace

import torch

from megatron.core.transformer.multi_latent_attention import (
    _prepare_mla_core_attention_value,
    _trim_mla_core_attention_output,
)


def _attention_config(*, padding_v_to_qk_dim=False):
    return SimpleNamespace(
        config=SimpleNamespace(
            experimental_attention_variant=None,
            padding_v_to_qk_dim=padding_v_to_qk_dim,
        )
    )


def test_pads_value_to_query_width_and_trims_output():
    query = torch.zeros(2, 1, 3, 8)
    value = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)

    padded, need_pad, original_dim, padded_dim = _prepare_mla_core_attention_value(
        _attention_config(padding_v_to_qk_dim=True),
        query,
        value,
        packed_seq_params=None,
    )

    assert need_pad is True
    assert original_dim == 4
    assert padded_dim == 8
    assert padded.shape == (2, 1, 3, 8)
    assert torch.equal(padded[..., :4], value)
    assert torch.count_nonzero(padded[..., 4:]) == 0

    trimmed = _trim_mla_core_attention_output(padded, need_pad, original_dim, padded_dim)
    assert trimmed.shape == value.shape
    assert torch.equal(trimmed, value)


def test_thd_packed_path_requests_padding_without_explicit_config():
    query = torch.zeros(4, 1, 2, 8)
    value = torch.ones(4, 1, 2, 4)
    packed_seq_params = SimpleNamespace(qkv_format="thd")

    padded, need_pad, original_dim, padded_dim = _prepare_mla_core_attention_value(
        _attention_config(padding_v_to_qk_dim=False),
        query,
        value,
        packed_seq_params=packed_seq_params,
    )

    assert need_pad is True
    assert original_dim == 4
    assert padded_dim == 8
    assert padded.shape == (4, 1, 2, 8)
    assert torch.equal(padded[..., :4], value)
    assert torch.count_nonzero(padded[..., 4:]) == 0


def test_equal_width_or_non_thd_path_is_unchanged():
    query = torch.zeros(2, 1, 3, 8)
    value = torch.ones(2, 1, 3, 8)

    padded, need_pad, original_dim, padded_dim = _prepare_mla_core_attention_value(
        _attention_config(padding_v_to_qk_dim=True),
        query,
        value,
        packed_seq_params=None,
    )

    assert need_pad is False
    assert original_dim == padded_dim == 8
    assert padded is value
    assert torch.equal(
        _trim_mla_core_attention_output(padded, need_pad, original_dim, padded_dim),
        value,
    )
