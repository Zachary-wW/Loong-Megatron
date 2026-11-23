import pytest

from megatron.core.transformer import TransformerConfig


def _echo_config(**overrides):
    values = {
        "num_layers": 1,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_enable_echo": True,
        "gradient_accumulation_fusion": True,
        "bf16": True,
        "moe_grouped_gemm": True,
        "expert_model_parallel_size": 2,
        "moe_num_echo_experts": 2,
        "moe_echo_expert_dispatch_overlap": True,
        "moe_echo_expert_dispatcher_type": "hybridep",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_echo_config_accepts_supported_topology():
    config = _echo_config()
    assert config.moe_enable_echo
    assert config.moe_num_echo_experts == 2
    assert config.moe_echo_expert_dispatch_overlap


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion"),
        ({"moe_num_echo_experts": None}, "must be set"),
        ({"moe_num_echo_experts": 0}, "must be positive"),
        ({"expert_model_parallel_size": 3, "moe_num_echo_experts": 2}, "must be divisible"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        ({"moe_echo_expert_dispatcher_type": "alltoall"}, "dispatch_overlap requires"),
    ],
)
def test_echo_config_rejects_invalid_combinations(overrides, match):
    with pytest.raises(ValueError, match=match):
        _echo_config(**overrides)
