# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.extensions.transformer_engine import _has_padding_attention_mask


def test_detects_standard_padding_mask():
    padding_mask = torch.tensor([[[[False, False], [True, True]]]])

    assert _has_padding_attention_mask(padding_mask)


def test_ignores_empty_and_nonstandard_masks():
    empty_mask = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
    nonstandard_mask = torch.ones((1, 2, 2, 2), dtype=torch.bool)

    assert not _has_padding_attention_mask(None)
    assert not _has_padding_attention_mask(empty_mask)
    assert not _has_padding_attention_mask(nonstandard_mask)


def test_detects_padding_mask_in_sequence():
    padding_mask = torch.tensor([[[[False, True]]]])

    assert _has_padding_attention_mask((None, padding_mask))
