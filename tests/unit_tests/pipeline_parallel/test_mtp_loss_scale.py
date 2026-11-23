# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel import schedules
from megatron.core.pipeline_parallel.schedules import _get_mtp_loss_scale
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler


def test_mtp_grad_scale_func_config():
    """Test that mtp_grad_scale_func config defaults to None and can be set."""
    config = ModelParallelConfig()
    assert config.mtp_grad_scale_func is None

    scale_fn = lambda: torch.tensor(0.5)
    config = ModelParallelConfig(mtp_grad_scale_func=scale_fn)
    assert config.mtp_grad_scale_func is scale_fn
    assert config.mtp_grad_scale_func().item() == 0.5


def test_mtp_loss_scale_selection():
    """Test MTP loss scale selection and device normalization."""

    device = (
        torch.device('cuda', torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    # Case 1: mtp_grad_scale_func takes priority
    config = ModelParallelConfig(
        mtp_grad_scale_func=lambda: torch.tensor([0.25], device='cpu'),
        grad_scale_func=lambda x: x * 2.0,
    )
    loss_scale = _get_mtp_loss_scale(config, device)
    assert loss_scale.item() == 0.25
    assert loss_scale.device == device

    # Case 2: Falls back to grad_scale_func
    config = ModelParallelConfig(grad_scale_func=lambda x: x * 3.0)
    loss_scale = _get_mtp_loss_scale(config, device)
    assert loss_scale.item() == 3.0
    assert loss_scale.device == device

    # Case 3: Falls back to ones
    config = ModelParallelConfig()
    loss_scale = _get_mtp_loss_scale(config, device)
    assert loss_scale.item() == 1.0
    assert loss_scale.device == device


@pytest.mark.parametrize(
    "config,scale_func_name",
    [
        (
            ModelParallelConfig(mtp_grad_scale_func=lambda: torch.tensor([0.25, 0.5])),
            "mtp_grad_scale_func",
        ),
        (
            ModelParallelConfig(grad_scale_func=lambda _: torch.tensor([0.25, 0.5])),
            "grad_scale_func",
        ),
    ],
)
def test_mtp_loss_scale_rejects_non_scalar_scale(config, scale_func_name):
    """Test that MTP loss scaling rejects per-token or per-sample scale values."""

    device = (
        torch.device('cuda', torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    with pytest.raises(ValueError, match=f"{scale_func_name} must return a scalar"):
        _get_mtp_loss_scale(config, device)


def _chunkpipe_config(**overrides):
    values = {
        "calculate_per_token_loss": False,
        "mtp_num_layers": 1,
        "mtp_grad_scale_func": lambda: torch.tensor(3.0),
        "enable_chunkpipe": True,
        "sft_chunkpipe_mode": True,
        "chunkpipe_step_num_groups": 4,
        "num_moe_experts": None,
        "timers": None,
        "experimental_attention_variant_loss_scale_func": None,
        "experimental_attention_variant": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_mtp_loss_scale(config, monkeypatch, dp_size=8, num_microbatches=2):
    monkeypatch.setattr(
        schedules.parallel_state, "get_data_parallel_world_size", lambda: dp_size
    )
    MTPLossAutoScaler.main_loss_backward_scale = None
    schedules.forward_step_calc_loss(
        model=None,
        output_tensor=torch.tensor(1.0),
        loss_func=None,
        config=config,
        vp_stage=None,
        collect_non_loss_data=False,
        num_microbatches=num_microbatches,
        forward_data_store=[],
        cp_group_size=1,
        is_last_stage=True,
    )
    return MTPLossAutoScaler.main_loss_backward_scale


def test_sft_chunkpipe_mtp_loss_scale_uses_dp_and_group_count(monkeypatch):
    scale = _run_mtp_loss_scale(_chunkpipe_config(), monkeypatch)

    torch.testing.assert_close(scale, torch.tensor(6.0))
    MTPLossAutoScaler.main_loss_backward_scale = None


def test_sft_chunkpipe_mtp_loss_scale_requires_positive_group_count(monkeypatch):
    with pytest.raises(AssertionError, match="chunkpipe_step_num_groups > 0"):
        _run_mtp_loss_scale(
            _chunkpipe_config(chunkpipe_step_num_groups=0), monkeypatch
        )
    MTPLossAutoScaler.main_loss_backward_scale = None


def test_non_sft_chunkpipe_mtp_loss_scale_uses_chunk_denominator(monkeypatch):
    config = _chunkpipe_config(
        sft_chunkpipe_mode=False,
        chunk_num_per_seq=4,
    )

    scale = _run_mtp_loss_scale(config, monkeypatch, num_microbatches=8)

    torch.testing.assert_close(scale, torch.tensor(1.5))
    MTPLossAutoScaler.main_loss_backward_scale = None
