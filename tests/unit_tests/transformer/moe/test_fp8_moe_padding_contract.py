# Copyright (c) 2026, Baidu, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.experts import TEGroupedMLP


class _RecordingGroupedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.use_bias = False

    def will_execute_quantized(self, is_context_quantized=True):
        return True

    def forward(self, hidden_states, tokens_per_expert):
        self.calls.append((tuple(hidden_states.shape), list(tokens_per_expert)))
        return hidden_states, None


def _build_module(*, router_padding=False, flex_backend=None):
    from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module._with_fused_impl = False
    module._use_grouped_tensor = False
    module.offload_expert_fc1 = False
    module.offload_moe_act = False
    module.activation_recompute = False
    module.activation_func = F.relu
    module.linear_fc1 = _RecordingGroupedLinear()
    module.linear_fc2 = _RecordingGroupedLinear()
    module.num_local_experts = 2
    module.quantization_padding = Fp8Padding(module.num_local_experts)
    module.quantization_unpadding = Fp8Unpadding(module.num_local_experts)
    module.config = SimpleNamespace(
        fp8="hybrid",
        fp4=False,
        fp8_recipe=None,
        fp4_recipe=None,
        moe_received_token_capacity=None,
        moe_router_padding_for_quantization=router_padding,
        moe_token_dispatcher_type="flex" if flex_backend is not None else "alltoall",
        moe_flex_dispatcher_backend=flex_backend,
        moe_use_grouped_tensor=False,
        moe_apply_probs_on_input=False,
        delay_offload_until_cuda_graph=False,
        gated_linear_unit=False,
        use_te_activation_func=False,
        bias_activation_fusion=False,
        use_fused_weighted_squared_relu=False,
        activation_func=F.relu,
        activation_func_fp8_input_store=False,
        activation_func_clamp_value=None,
        activation_func_tanh_clamp_scale=None,
        activation_func_tanh_clamp_scale_linear=None,
        glu_linear_offset=0.0,
        moe_mlp_glu_interleave_size=None,
        moe_router_topk=1,
    )
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantized_grouped_mlp_pads_and_unpads_real_fp8_segments():
    module = _build_module()
    hidden = torch.arange(6 * 4, device="cuda", dtype=torch.float32).reshape(6, 4)
    probs = torch.ones(6, device="cuda")
    tokens_per_expert = torch.tensor([2, 4], device="cuda")

    output, output_bias = module(hidden, tokens_per_expert, probs)

    assert output.shape == hidden.shape
    assert output_bias is None
    torch.testing.assert_close(output, hidden)
    assert module.linear_fc1.calls == [((32, 4), [16, 16])]
    assert module.linear_fc2.calls == [((32, 4), [16, 16])]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("router_padding,flex_backend", [(True, None), (False, "hybridep")])
def test_already_padded_routing_skips_second_fp8_padding(router_padding, flex_backend):
    module = _build_module(router_padding=router_padding, flex_backend=flex_backend)
    hidden = torch.arange(6 * 4, device="cuda", dtype=torch.float32).reshape(6, 4)
    probs = torch.ones(6, device="cuda")
    tokens_per_expert = torch.tensor([2, 4], device="cuda")

    output, _ = module(hidden, tokens_per_expert, probs)

    torch.testing.assert_close(output, hidden)
    assert module.linear_fc1.calls == [((6, 4), [2, 4])]
    assert module.linear_fc2.calls == [((6, 4), [2, 4])]
