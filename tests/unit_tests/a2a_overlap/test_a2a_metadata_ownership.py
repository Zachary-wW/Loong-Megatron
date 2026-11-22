# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.models.gpt.fine_grained_callables as fine_grained_callables


class _FakeMoELayer:
    pass


@pytest.mark.parametrize("backend", ["deepep", "hybridep", "ncclep"])
def test_a2a_metadata_holds_detached_probabilities(monkeypatch, backend):
    monkeypatch.setattr(fine_grained_callables, "MoELayer", _FakeMoELayer)

    metadata = SimpleNamespace(
        get_number_of_tokens_per_expert=lambda: torch.ones(2),
    )
    captured = {}
    token_dispatcher = SimpleNamespace(
        metadata=metadata,
        get_number_of_tokens_per_expert=lambda: torch.ones(2),
    )

    def dispatch(local_tokens, probs):
        captured["dispatch_probs"] = metadata.token_probs
        return local_tokens + 1, probs + 1

    def routed_experts_compute(dispatched_tokens, dispatched_probs):
        captured["expert_probs"] = metadata.dispatched_probs
        return dispatched_tokens + 2, None

    mlp = _FakeMoELayer()
    mlp.token_dispatcher = token_dispatcher
    mlp.use_shared_expert = False
    mlp.shared_expert_overlap = False
    mlp.dispatch = dispatch
    mlp.routed_experts_compute = routed_experts_compute

    layer = SimpleNamespace(
        config=SimpleNamespace(
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend=backend,
            moe_ncclep_zero_copy=False,
            moe_enable_echo=False,
            enable_moe_mem_monitor=False,
            recompute_granularity=None,
            recompute_modules=None,
            fp8=None,
            fp4=None,
        ),
        mlp=mlp,
        a2a_overlap_attn_recompute=False,
        a2a_overlap_post_attn_recompute=False,
        a2a_overlap_mlp_recompute=False,
        recompute_pre_mlp_layernorm=False,
        _forward_mlp=lambda *args, **kwargs: None,
        init_backward_dw_wrapper=lambda: None,
        backward_dw_wrapper=None,
    )
    node = SimpleNamespace(
        layer_state=SimpleNamespace(),
        detach=lambda tensor: tensor.detach().requires_grad_(True),
    )

    forward_funcs, _ = fine_grained_callables.build_transformer_layer_callables(layer)
    dispatch_forward, moe_forward = forward_funcs[1:3]
    local_tokens = torch.ones(4, 2, requires_grad=True)
    probs = torch.full((4, 2), 0.5, requires_grad=True)
    detached_probs = probs.detach().requires_grad_(True)

    dispatched_tokens = dispatch_forward(node, local_tokens, detached_probs)
    assert captured["dispatch_probs"] is detached_probs
    assert captured["dispatch_probs"].grad_fn is None

    moe_forward(node, dispatched_tokens)
    assert captured["expert_probs"] is node.layer_state.dispatched_probs
    assert captured["expert_probs"].grad_fn is None
    torch.testing.assert_close(captured["expert_probs"], detached_probs + 1)
