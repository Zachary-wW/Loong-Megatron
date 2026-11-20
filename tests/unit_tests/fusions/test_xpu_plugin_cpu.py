from __future__ import annotations

import importlib.util
import inspect
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.embeddings import rope_utils

MODULE_PATH = Path(__file__).resolve().parents[3] / "xpu_plugin" / "mock_megatron.py"


@pytest.fixture(scope="module")
def mock_megatron():
    spec = importlib.util.spec_from_file_location("test_xpu_plugin_mock_megatron", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch_xmlir = None
    return module


def test_layer_norm_matches_torch(mock_megatron):
    torch.manual_seed(0)
    layer_norm = mock_megatron.MockMixedFusedLayerNorm(8, eps=1e-5)
    sample = torch.randn(2, 3, 8)
    expected = torch.nn.functional.layer_norm(
        sample, (8,), layer_norm.weight, layer_norm.bias, layer_norm.eps
    )

    torch.testing.assert_close(layer_norm(sample), expected)
    assert layer_norm.weight.sequence_parallel is False
    assert layer_norm.bias.sequence_parallel is False


def test_weighted_bias_swiglu_clamp_matches_reference(mock_megatron):
    torch.manual_seed(0)
    input_tensor = torch.randn(2, 6)
    bias = torch.randn(6)
    weights = torch.randn(2, 1)
    clamp_value = 0.5

    actual = mock_megatron.mock_weighted_bias_swiglu_impl(
        input_tensor, bias, weights, clamp_value=clamp_value
    )
    gate, linear = torch.chunk((input_tensor + bias).float(), 2, dim=-1)
    expected = (
        torch.nn.functional.silu(gate.clamp(max=clamp_value))
        * linear.clamp(min=-clamp_value, max=clamp_value)
    ).to(input_tensor.dtype) * weights

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "rotary_interleaved": True,
            "mla_rotary_interleaved": True,
            "mscale": 1.25,
            "inverse": True,
            "mla_output_remove_interleaving": True,
        },
    ],
    ids=("default", "mla_yarn_inverse"),
)
def test_apply_rotary_pos_emb_matches_core(mock_megatron, kwargs):
    torch.manual_seed(0)
    t = torch.randn(2, 2, 2, 4)
    freqs = torch.randn(2, 1, 1, 4)

    actual = mock_megatron.mock_apply_rotary_pos_emb_bshd(t, freqs, **kwargs)
    expected = rope_utils._apply_rotary_pos_emb_bshd(t, freqs, **kwargs)

    torch.testing.assert_close(actual, expected)


def test_apply_rotary_signature_matches_core(mock_megatron):
    plugin_parameters = inspect.signature(mock_megatron.mock_apply_rotary_pos_emb_bshd).parameters
    core_parameters = inspect.signature(rope_utils._apply_rotary_pos_emb_bshd).parameters

    assert set(core_parameters).issubset(plugin_parameters)


def test_kernel_availability_matches_legacy_contract(mock_megatron):
    softmax = SimpleNamespace(scaled_masked_softmax_fusion=True, input_in_float16=True)
    is_kernel_available = types.MethodType(mock_megatron.mock_is_kernel_available, softmax)

    assert is_kernel_available(None, 1, 1, 1, 16384)
    assert not is_kernel_available(None, 1, 1, 1, 16385)


def test_removed_non_migration_symbols_are_not_restored(mock_megatron):
    assert not hasattr(mock_megatron, "mock_lce_backward")
    assert not hasattr(mock_megatron, "mock_bias_swiglu_impl")
    assert not hasattr(mock_megatron, "mock_unpermute")
