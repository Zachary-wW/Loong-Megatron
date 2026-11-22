"""CPU component checks for the XPU MLA five-tuple QKV contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import megatron.core.transformer.multi_latent_attention as core_mla
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig


WORKTREE_ROOT = Path(__file__).resolve().parents[3]
XPU_MLA_PATH = (
    WORKTREE_ROOT / "xpu_plugin" / "multi_latent_attention" / "multi_latent_attention.py"
)


class _CPUProjection(torch.nn.Module):
    """Minimal non-TE projection for the CPU adapter contract."""

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config=None,
        init_method=None,
        bias=True,
        gather_output=False,
        skip_bias_add=False,
        is_expert=False,
        tp_comm_buffer_name=None,
        tp_group=None,
        name=None,
        skip_weight_param_allocation=False,
        input_is_parallel=False,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)
        self.skip_bias_add = skip_bias_add

    def forward(self, x, *args, **kwargs):
        output = self.linear(x)
        return (output, self.linear.bias) if self.skip_bias_add else (output, None)


class _StubCoreAttention(torch.nn.Module):
    """Constructor-only core attention; QKV generation does not call it."""

    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_size_per_attention_head_v = kwargs.get("v_channels")

    def forward(self, *args, **kwargs):
        raise AssertionError("core attention must not run in the QKV component test")


class _CPURotaryEmbedding(torch.nn.Module):
    """Deterministic CPU RoPE source; production RoPE remains unchanged."""

    def __init__(
        self,
        kv_channels,
        rotary_percent=1.0,
        rotary_base=10000,
        cp_group=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = kv_channels
        self.inv_freq = 1.0 / (
            rotary_base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

    def get_rotary_seq_len(
        self,
        inference_context,
        transformer,
        transformer_input,
        transformer_config,
        packed_seq_params=None,
        *,
        inference_params=None,
    ):
        return transformer_input.size(0)

    def forward(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        positions = torch.arange(max_seq_len, dtype=torch.float32) + offset
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[:, None, None, :]


@pytest.fixture(scope="module")
def xpu_mla_module():
    spec = importlib.util.spec_from_file_location("test_xpu_mla_adapter", XPU_MLA_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cpu_rotary_embedding(monkeypatch):
    monkeypatch.setattr(core_mla, "RotaryEmbedding", _CPURotaryEmbedding)


@pytest.fixture
def one_rank_gloo(tmp_path):
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'store'}",
            rank=0,
            world_size=1,
        )
    if dist.get_world_size() != 1:
        pytest.skip("XPU MLA CPU component test requires one rank")
    try:
        yield dist.group.WORLD
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _build_attention(
    xpu_mla_module,
    process_group,
    *,
    q_lora_rank,
    padding_v_to_qk_dim,
):
    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        perform_initialization=True,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=8,
        qk_head_dim=16,
        v_head_dim=16,
        qk_pos_emb_head_dim=8,
        rope_type="rope",
        rotary_base=10000,
        original_max_position_embeddings=16,
        qk_layernorm=False,
        padding_v_to_qk_dim=padding_v_to_qk_dim,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    submodules = MLASelfAttentionSubmodules(
        linear_proj=_CPUProjection,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
        linear_q_proj=_CPUProjection,
        linear_q_down_proj=ColumnParallelLinear,
        linear_q_up_proj=_CPUProjection,
        linear_kv_down_proj=ColumnParallelLinear,
        linear_kv_up_proj=_CPUProjection,
        core_attention=_StubCoreAttention,
    )
    pg_collection = ProcessGroupCollection(tp=process_group, cp=process_group)
    return xpu_mla_module.MLASelfAttention(
        config,
        submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
    )


@pytest.mark.parametrize("padding_v_to_qk_dim", [False, True], ids=("v16", "v24"))
@pytest.mark.parametrize("q_lora_rank", [8, None], ids=("q_lora", "q_direct"))
def test_five_tuple_qkv_forward_backward(
    xpu_mla_module,
    one_rank_gloo,
    q_lora_rank,
    padding_v_to_qk_dim,
):
    attention = _build_attention(
        xpu_mla_module,
        one_rank_gloo,
        q_lora_rank=q_lora_rank,
        padding_v_to_qk_dim=padding_v_to_qk_dim,
    )
    hidden_states = torch.randn(4, 2, 32, requires_grad=True)

    outputs = attention.get_query_key_value_tensors(hidden_states)

    assert len(outputs) == 5
    query, key, value, q_compressed, kv_compressed = outputs
    expected_value_width = 24 if padding_v_to_qk_dim else 16
    assert query.shape == (4, 2, 4, 24)
    assert key.shape == (4, 2, 4, 24)
    assert value.shape == (4, 2, 4, expected_value_width)
    assert kv_compressed.shape == (4, 2, 8)
    if q_lora_rank is None:
        assert q_compressed is hidden_states
    else:
        assert q_compressed.shape == (4, 2, q_lora_rank)

    for tensor in outputs:
        assert tensor is not None
        assert torch.isfinite(tensor).all()

    loss = sum(tensor.float().square().mean() for tensor in outputs)
    loss.backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert attention.linear_kv_down_proj.weight.grad is not None
    assert torch.isfinite(attention.linear_kv_down_proj.weight.grad).all()
    q_projection = (
        attention.linear_q_proj if q_lora_rank is None else attention.linear_q_up_proj
    )
    assert q_projection.linear.weight.grad is not None
    assert torch.isfinite(q_projection.linear.weight.grad).all()
    assert attention.linear_kv_up_proj.linear.weight.grad is not None
    assert torch.isfinite(attention.linear_kv_up_proj.linear.weight.grad).all()
