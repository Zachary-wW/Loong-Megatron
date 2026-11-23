# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEFusedMLP,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


pytestmark = pytest.mark.skipif(
    not HAVE_TE or TEFusedMLP is None or not torch.cuda.is_available(),
    reason="mlp_act fallback parity requires Transformer Engine and CUDA",
)


def _config(*, mlp_act: bool) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=256,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        use_cpu_initialization=True,
        bf16=True,
    )
    if mlp_act:
        kwargs.update(
            recompute_granularity="selective",
            recompute_modules=["mlp_act"],
        )
    return TransformerConfig(**kwargs)


def _submodules() -> MLPSubmodules:
    return MLPSubmodules(
        linear_fc1=TELayerNormColumnParallelLinear,
        linear_fc2=TERowParallelLinear,
    )


def _fill_deterministic(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.linspace(
                -0.25 - 0.01 * index,
                0.25 + 0.01 * index,
                parameter.numel(),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.copy_(values.reshape_as(parameter))


def test_mlp_act_fallback_matches_eager_output_and_gradients():
    """The unfused mlp_act path must match the ordinary MLP on real CUDA kernels."""

    Utils.initialize_model_parallel(1, 1)
    try:
        torch.cuda.reset_peak_memory_stats()
        eager = MLP(_config(mlp_act=False), _submodules()).cuda()
        fallback = TEFusedMLP(_config(mlp_act=True), _submodules()).cuda()
        _fill_deterministic(eager)
        fallback.load_state_dict(eager.state_dict())

        assert eager.activation_recompute is False
        assert fallback.activation_recompute is True
        assert getattr(fallback, "_fused_impl", None) is None

        eager_input = torch.linspace(
            -1.0, 1.0, 64 * 2 * eager.config.hidden_size, device="cuda"
        ).reshape(64, 2, eager.config.hidden_size)
        fallback_input = eager_input.clone()
        eager_input.requires_grad_(True)
        fallback_input.requires_grad_(True)

        eager_output, eager_bias = eager(eager_input)
        fallback_output, fallback_bias = fallback(fallback_input)

        torch.testing.assert_close(fallback_output, eager_output, rtol=0.0, atol=0.0)
        assert eager_bias is None
        assert fallback_bias is None

        eager_output.float().square().mean().backward()
        fallback_output.float().square().mean().backward()

        torch.testing.assert_close(fallback_input.grad, eager_input.grad, rtol=0.0, atol=0.0)
        eager_parameters = dict(eager.named_parameters())
        fallback_parameters = dict(fallback.named_parameters())
        assert eager_parameters.keys() == fallback_parameters.keys()
        for name, eager_parameter in eager_parameters.items():
            fallback_parameter = fallback_parameters[name]
            assert eager_parameter.grad is not None
            assert fallback_parameter.grad is not None
            torch.testing.assert_close(
                fallback_parameter.grad,
                eager_parameter.grad,
                rtol=0.0,
                atol=0.0,
                msg=lambda message: f"{name}: {message}",
            )
        print(
            "peak_allocated_mib="
            f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} "
            "peak_reserved_mib="
            f"{torch.cuda.max_memory_reserved() / 1024**2:.1f}"
        )
    finally:
        Utils.destroy_model_parallel()
