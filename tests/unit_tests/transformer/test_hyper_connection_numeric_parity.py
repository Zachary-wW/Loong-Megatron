# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import pytest

from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.hyper_connection import HyperConnectionModule


def _dsv4_config(legacy_numeric_parity, **overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        csa_compress_ratios=[0],
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=24,
        enable_mhc_connections=True,
        mhc_num_residual_streams=2,
        mhc_sinkhorn_iterations=2,
        mhc_init_gating_factor=0.25,
        dsv4_legacy_numeric_parity=legacy_numeric_parity,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


def test_dsv4_legacy_numeric_parity_defaults_off():
    config = _dsv4_config(False)

    assert config.dsv4_legacy_numeric_parity is False


def test_dsv4_legacy_numeric_parity_requires_dsv4():
    with pytest.raises(ValueError, match="requires experimental_attention_variant"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            dsv4_legacy_numeric_parity=True,
        )


def test_dsv4_legacy_numeric_parity_rejects_fused_mhc():
    with pytest.raises(ValueError, match="not supported with use_fused_mhc"):
        _dsv4_config(legacy_numeric_parity=True, use_fused_mhc=True)


@pytest.mark.parametrize("legacy_numeric_parity", [False, True])
def test_mhc_mapping_dtype_is_controlled_by_numeric_parity(legacy_numeric_parity):
    """The opt-in path computes mHC mappings in BF16 while storage stays FP32."""
    config = _dsv4_config(legacy_numeric_parity)
    module = HyperConnectionModule(config, layer_number=1)
    x = torch.randn(2, 3, module.n * module.hidden_size, dtype=torch.bfloat16)
    output_size = module.n * module.n + 2 * module.n
    seen = {}

    def fake_proj_rms(x_2d, weight, _eps):
        seen["activation_dtype"] = x_2d.dtype
        seen["weight_dtype"] = weight.dtype
        seen["storage_dtype"] = module.mapping_proj.weight.dtype
        compute_dtype = x_2d.dtype
        proj = torch.linspace(
            -1, 1, x_2d.shape[0] * output_size, dtype=compute_dtype
        ).reshape(x_2d.shape[0], output_size)
        r = torch.linspace(
            0.5, 1.5, x_2d.shape[0], dtype=compute_dtype
        ).reshape(-1, 1)
        return proj, r

    module._proj_rms_op = fake_proj_rms
    proj, r = module._projection_and_get_norm(x)

    expected_compute_dtype = torch.bfloat16 if legacy_numeric_parity else torch.float32
    assert seen["activation_dtype"] == expected_compute_dtype
    assert seen["weight_dtype"] == expected_compute_dtype
    assert seen["storage_dtype"] == torch.float32
    assert module.mapping_proj.weight.dtype == torch.float32
    assert module.alpha_pre.dtype == torch.float32
    assert module.alpha_post.dtype == torch.float32
    assert module.alpha_res.dtype == torch.float32
    assert module.bias.dtype == torch.float32

    compute_h = HyperConnectionModule._compute_h.__wrapped__
    h_pre, h_post, h_res = compute_h(module, proj, r)
    if legacy_numeric_parity:
        alpha = torch.cat(
            [
                module.alpha_pre.to(proj.dtype).expand(module.n),
                module.alpha_post.to(proj.dtype).expand(module.n),
                module.alpha_res.to(proj.dtype).expand(module.n * module.n),
            ],
            dim=-1,
        )
        expected = r * proj * alpha + module.bias.to(proj.dtype)
    else:
        alpha = torch.cat(
            [
                module.alpha_pre.expand(module.n),
                module.alpha_post.expand(module.n),
                module.alpha_res.expand(module.n * module.n),
            ],
            dim=-1,
        )
        expected = r * proj * alpha + module.bias

    torch.testing.assert_close(
        h_pre, expected[..., : module.n].sigmoid() + module.compute_h_eps
    )
    torch.testing.assert_close(
        h_post, expected[..., module.n : 2 * module.n].sigmoid() * 2
    )
    torch.testing.assert_close(h_res, expected[..., 2 * module.n :])
    assert h_pre.dtype == expected_compute_dtype
    assert h_post.dtype == expected_compute_dtype
    assert h_res.dtype == expected_compute_dtype
