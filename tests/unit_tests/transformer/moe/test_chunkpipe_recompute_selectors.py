from types import SimpleNamespace

import torch

from megatron.core.transformer import transformer_layer as transformer_layer_module
from megatron.core.transformer.moe import moe_layer as moe_layer_module
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer


def test_routed_experts_selector_replays_route_dispatch_pipeline(monkeypatch):
    events = []
    config = SimpleNamespace(
        moe_enable_echo=False,
        fp8=False,
        fp4=False,
    )
    layer = SimpleNamespace(
        training=True,
        attn_tp_group=SimpleNamespace(size=lambda: 1),
        config=config,
        moe_layer_recompute=False,
        routed_experts_recompute=True,
        fwd_execution_map=["route", "expert_compute", "postprocess"],
        select_token_dispatcher=lambda: None,
    )

    def shared_experts_compute(_hidden_states):
        events.append("shared_experts")
        return torch.ones(1)

    def route(_hidden_states, *_args, **_kwargs):
        events.append("route")
        return torch.ones(2, 1), torch.ones(2, 1)

    def preprocess(hidden_states, probs, _routing_map, _padding_mask):
        events.append("preprocess")
        return hidden_states, probs

    def dispatch(hidden_states, probs):
        events.append("dispatch")
        return hidden_states, probs

    def routed_experts_compute(hidden_states, _probs):
        events.append("routed_experts_compute")
        return hidden_states, None

    def combine(hidden_states):
        events.append("combine")
        return hidden_states

    def postprocess(hidden_states, _shared_expert_output):
        events.append("postprocess")
        return hidden_states

    layer.shared_experts_compute = shared_experts_compute
    layer.route = route
    layer.preprocess = preprocess
    layer.dispatch = dispatch
    layer.routed_experts_compute = routed_experts_compute
    layer.combine = combine
    layer.postprocess = postprocess

    def fake_checkpoint(function, _distribute, *args):
        events.append("checkpoint")
        return function(*args)

    monkeypatch.setattr(
        moe_layer_module.tensor_parallel, "checkpoint", fake_checkpoint
    )

    output, bias = MoELayer.forward(layer, torch.ones(2, 2))

    assert bias is None
    torch.testing.assert_close(output, torch.ones(2, 2))
    assert events == [
        "shared_experts",
        "checkpoint",
        "route",
        "preprocess",
        "dispatch",
        "routed_experts_compute",
        "combine",
        "postprocess",
    ]


def test_pre_mlp_selector_uses_checkpointed_attention_path():
    events = []
    layer = SimpleNamespace(
        training=True,
        recompute_pre_mlp=True,
        layer_number=1,
        config=SimpleNamespace(moe_echo_expert_dispatch_overlap=False),
    )

    def checkpoint_pre_mlp(*_args, **_kwargs):
        events.append("checkpoint_pre_mlp")
        return torch.ones(2, 2), None

    def forward_attention(*_args, **_kwargs):
        events.append("full_attention")
        return torch.zeros(2, 2), None

    def forward_mlp(hidden_states, *_args, **_kwargs):
        events.append("mlp")
        return hidden_states

    layer._checkpoint_pre_mlp_forward = checkpoint_pre_mlp
    layer._forward_attention = forward_attention
    layer._forward_mlp = forward_mlp

    output, context = TransformerLayer.forward(layer, hidden_states=torch.ones(2, 2))

    assert context is None
    torch.testing.assert_close(output, torch.ones(2, 2))
    assert events == ["checkpoint_pre_mlp", "mlp"]


def test_checkpoint_pre_mlp_wraps_complete_attention_path(monkeypatch):
    events = []
    layer = SimpleNamespace(
        config=SimpleNamespace(fp8=False, fp4=False),
        pg_collection=SimpleNamespace(tp=None),
    )

    def forward_attention(hidden_states, **_kwargs):
        events.append("full_attention")
        return hidden_states + 1, "context"

    def fake_checkpoint(function, _distribute, *args):
        events.append("checkpoint")
        return function(*args)

    layer._forward_attention = forward_attention
    monkeypatch.setattr(
        transformer_layer_module.tensor_parallel,
        "checkpoint",
        fake_checkpoint,
    )

    output, context = TransformerLayer._checkpoint_pre_mlp_forward(
        layer,
        torch.ones(2, 2),
    )

    assert events == ["checkpoint", "full_attention"]
    assert context == "context"
    torch.testing.assert_close(output, torch.ones(2, 2) + 1)
