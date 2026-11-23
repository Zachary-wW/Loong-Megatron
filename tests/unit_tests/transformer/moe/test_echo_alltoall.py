# Copyright (c) 2026, Baidu, Inc. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_echo_alltoall_forward_backward_and_weight_lifecycle():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
    )
    try:
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_ffn_hidden_size=128,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_enable_echo=True,
            moe_num_echo_experts=2,
            moe_echo_algorithm="greedy",
            moe_echo_expert_dispatcher_type="alltoall",
            moe_echo_expert_dispatch_overlap=False,
            gradient_accumulation_fusion=True,
            expert_model_parallel_size=2,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=False,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=config.num_moe_experts, moe_grouped_gemm=True
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        layer = MoELayer(config, submodules).cuda()

        hidden = torch.randn(
            32,
            2,
            config.hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        rerouting_map, rerouted_probs, expert_offloading_map = layer._echo_route(hidden)
        assert rerouting_map.shape[0] == hidden.numel() // config.hidden_size
        assert rerouted_probs.shape == rerouting_map.shape
        assert expert_offloading_map.shape[0] == config.num_moe_experts
        num_tokens = hidden.numel() // config.hidden_size
        assert rerouting_map.sum().item() <= num_tokens * config.moe_router_topk
        assert expert_offloading_map.any()

        output, bias = layer(hidden)
        assert output.shape == hidden.shape
        assert bias is None
        assert torch.isfinite(output).all()
        output.sum().backward()
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()

        for module_name in ("fc1", "fc2"):
            for index in layer.home_expert_indices:
                weights = layer.experts.get_expert_weights(module_name, [index])
                assert weights[0] is not None
                assert weights[0].numel() > 0
            for index in layer.echo_expert_indices:
                weights = layer.experts.get_expert_weights(module_name, [index])
                assert weights[0] is not None
                assert weights[0].numel() > 0
    finally:
        Utils.destroy_model_parallel()
