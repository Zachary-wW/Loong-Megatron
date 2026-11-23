from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.transformer_block as transformer_block_module
from megatron.core.transformer.transformer_block import TransformerBlock


class _CaptureLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.kwargs = None

    def forward(self, hidden_states, **kwargs):
        self.kwargs = kwargs
        return hidden_states, None


def _make_block(monkeypatch, recompute_granularity=None):
    monkeypatch.setattr(
        transformer_block_module, "get_transformer_layer_offset", lambda *_args: 0
    )
    monkeypatch.setattr(transformer_block_module, "get_pg_rank", lambda _group: 0)
    config = SimpleNamespace(
        enable_mhc_connections=False,
        sequence_parallel=False,
        fp8=None,
        fp4=None,
        enable_chunkpipe=False,
        recompute_granularity=recompute_granularity,
        recompute_num_layers=1,
        custom_pipeline_recompute_layers=None,
        cpu_offloading=False,
        mtp_num_layers=None,
        inference_fuse_tp_communication=False,
        hidden_size=4,
        layernorm_epsilon=1e-5,
        mhc_recompute_layer_num=None,
    )
    layer = _CaptureLayer()
    block = object.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = config
    block.pg_collection = SimpleNamespace(pp=None)
    block.vp_stage = None
    block.pp_rank = 0
    block.pre_process = True
    block.post_process = True
    block.post_layer_norm = True
    block.training = True
    block.mhc_recompute_enabled = False
    block.layers = torch.nn.ModuleList([layer])
    block.offload_context = nullcontext()
    block.input_tensor = None
    block.final_layernorm = None
    return block, layer


def test_transformer_block_forwards_extra_kwargs(monkeypatch):
    block, layer = _make_block(monkeypatch)

    output = block.forward(
        hidden_states=torch.ones(2, 1, 4),
        attention_mask=None,
        custom_metadata={"source": "caller"},
    )

    assert output.shape == (2, 1, 4)
    assert layer.kwargs["custom_metadata"] == {"source": "caller"}


def test_transformer_block_forwards_extra_kwargs_to_checkpointed_forward(monkeypatch):
    block, _ = _make_block(monkeypatch, recompute_granularity="full")
    captured = {}

    def fake_checkpointed_forward(_block, **kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    monkeypatch.setattr(
        transformer_block_module, "checkpointed_forward", fake_checkpointed_forward
    )

    output = block.forward(
        hidden_states=torch.ones(2, 1, 4),
        attention_mask=None,
        custom_metadata={"source": "checkpoint"},
    )

    assert output.shape == (2, 1, 4)
    assert captured["custom_metadata"] == {"source": "checkpoint"}


def test_transformer_block_rejects_duplicate_mhc_manager_kwarg(monkeypatch):
    block, _ = _make_block(monkeypatch)
    block.mhc_recompute_enabled = True
    block.config.mhc_recompute_layer_num = 1
    monkeypatch.setattr(
        transformer_block_module,
        "build_mhc_recompute_layer_plan",
        lambda **_: ([SimpleNamespace(is_last_layer_in_recompute_block=False)], [True]),
    )

    with pytest.raises(TypeError, match="multiple values"):
        block.forward(
            hidden_states=torch.ones(2, 1, 4),
            attention_mask=None,
            mhc_recompute_manager="caller-owned",
        )


def test_transformer_block_chunkpipe_recompute_retains_trailing_chunk(monkeypatch):
    block, _ = _make_block(monkeypatch)
    block.config.enable_chunkpipe = True
    block.config.chunk_num_per_seq = 4
    block.config.keep_activations_chunks = 1
    checkpointed_microbatches = []

    def fake_checkpointed_forward(_block, **kwargs):
        checkpointed_microbatches.append(block.config.chunkpipe_forward_microbatch)
        return kwargs["hidden_states"]

    monkeypatch.setattr(
        transformer_block_module, "checkpointed_forward", fake_checkpointed_forward
    )

    for microbatch in range(4):
        block.config.chunkpipe_forward_microbatch = microbatch
        block.forward(hidden_states=torch.ones(2, 1, 4), attention_mask=None)

    assert checkpointed_microbatches == [0, 1, 2]
