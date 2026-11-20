from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe import moe_layer as moe_layer_module
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


class _DummyDispatcher(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def dispatch_postprocess(self, hidden_states, probs):
        return hidden_states, torch.tensor([1, 2, 3]), probs


class _DummyRouter:
    def set_layer_number(self, layer_number):
        pass


def _build_experts(*args, **kwargs):
    return torch.nn.Module()


def _build_router(*args, **kwargs):
    return _DummyRouter()


@pytest.fixture(autouse=True)
def _reset_monitor_state(monkeypatch):
    monkeypatch.setattr(moe_utils, "CUDA_MONITOR_UTIL", {})
    monkeypatch.setattr(moe_utils, "MEM_MONITOR_FORCE_PRINT_TOKEN_THRESHOLD", 1_000_000_000)


def _set_rank(monkeypatch, rank):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *args, **kwargs: rank)


def _build_layer(monkeypatch, monitor_dir):
    monkeypatch.setattr(moe_layer_module, "MoEAllGatherTokenDispatcher", _DummyDispatcher)
    monkeypatch.setattr(moe_layer_module.utils, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(moe_layer_module.utils, "get_pg_rank", lambda group: 0)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=1,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type="allgather",
        moe_ffn_hidden_size=16,
        use_cpu_initialization=True,
        add_bias_linear=False,
        moe_mem_monitor_log=str(monitor_dir),
    )
    return MoELayer(
        config,
        MoESubmodules(experts=_build_experts, router=_build_router),
        pg_collection=SimpleNamespace(tp=object(), pp=object(), ep=object()),
    )


def test_transformer_config_preserves_fp32_and_monitor_options():
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2)

    assert config.use_fp32_dtype_for_param_pattern is None
    assert config.enable_moe_mem_monitor is False
    assert config.print_moe_mem_monitor_interval == 1000
    assert config.moe_mem_monitor_log is None
    assert config.moe_mem_monitor_force_print_token_threshold == 100_000_000


def test_disabled_monitor_does_not_initialize_or_touch_cuda(monkeypatch, tmp_path):
    initialize_calls = []
    monkeypatch.setattr(
        moe_layer_module,
        "initialize_cuda_monitoring",
        lambda **kwargs: initialize_calls.append(kwargs),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled monitor must not call CUDA or write files")

    monkeypatch.setattr(moe_layer_module, "monitor_max_memory_usage", fail_if_called)
    monkeypatch.setattr(moe_layer_module, "monitor_max_dispatcher_tokens", fail_if_called)
    monkeypatch.setattr(moe_layer_module, "write_monitor_data_to_file", fail_if_called)

    layer = _build_layer(monkeypatch, tmp_path)
    hidden_states = torch.ones(3, 8)
    probs = torch.ones(3, 1)
    dispatched_input, tokens_per_expert, permuted_probs = (
        layer._dispatch_postprocess_for_experts(hidden_states, probs)
    )

    assert initialize_calls == []
    assert torch.equal(dispatched_input, hidden_states)
    assert torch.equal(tokens_per_expert, torch.tensor([1, 2, 3]))
    assert torch.equal(permuted_probs, probs)
    assert list(tmp_path.iterdir()) == []


def test_initialize_cuda_monitoring_is_idempotent(monkeypatch):
    _set_rank(monkeypatch, 3)

    moe_utils.initialize_cuda_monitoring(
        tp_rank=1, pp_rank=2, ep_rank=3, mem_monitor_force_print_token_threshold=100
    )
    moe_utils.CUDA_MONITOR_UTIL[3]["max_dispatcher_tokens"] = 7
    moe_utils.initialize_cuda_monitoring(
        tp_rank=9, pp_rank=8, ep_rank=7, mem_monitor_force_print_token_threshold=5
    )

    assert moe_utils.CUDA_MONITOR_UTIL[3] == {
        "tp_rank": 1,
        "pp_rank": 2,
        "ep_rank": 3,
        "max_dispatcher_tokens": 7,
        "index": 0,
    }
    assert moe_utils.MEM_MONITOR_FORCE_PRINT_TOKEN_THRESHOLD == 100


def test_monitor_max_dispatcher_tokens_tracks_peak(monkeypatch):
    _set_rank(monkeypatch, 1)
    moe_utils.initialize_cuda_monitoring(tp_rank=0, pp_rank=0, ep_rank=1)

    moe_utils.monitor_max_dispatcher_tokens(torch.tensor([1, 2, 3]))
    moe_utils.monitor_max_dispatcher_tokens(torch.tensor([1]))
    moe_utils.monitor_max_dispatcher_tokens(torch.tensor([4, 5]))

    assert moe_utils.CUDA_MONITOR_UTIL[1]["max_dispatcher_tokens"] == 9


def test_monitor_file_writes_follow_interval(monkeypatch, tmp_path):
    _set_rank(monkeypatch, 0)
    moe_utils.initialize_cuda_monitoring(
        tp_rank=0, pp_rank=0, ep_rank=0, mem_monitor_force_print_token_threshold=100
    )
    data = moe_utils.CUDA_MONITOR_UTIL[0]
    data.update(max_mem_reserved=10, max_mem_allocated=8, max_dispatcher_tokens=6)

    moe_utils.write_monitor_data_to_file(str(tmp_path), print_mem_monitor_interval=3)
    moe_utils.write_monitor_data_to_file(str(tmp_path), print_mem_monitor_interval=3)
    assert not (tmp_path / "0").exists()

    moe_utils.write_monitor_data_to_file(str(tmp_path), print_mem_monitor_interval=3)
    assert (tmp_path / "0").read_text(encoding="utf-8").startswith(
        "rank: 0|t: 0|p: 0|e: 0"
    )

    moe_utils.write_monitor_data_to_file(str(tmp_path), print_mem_monitor_interval=3)
    aggregates = list(tmp_path.glob("monitor_data*"))
    assert len(aggregates) == 1
    assert aggregates[0].read_bytes().startswith(b"rank: 0|")


def test_monitor_file_force_prints_above_token_threshold(monkeypatch, tmp_path, capsys):
    _set_rank(monkeypatch, 0)
    moe_utils.initialize_cuda_monitoring(
        tp_rank=0, pp_rank=0, ep_rank=0, mem_monitor_force_print_token_threshold=5
    )
    data = moe_utils.CUDA_MONITOR_UTIL[0]
    data.update(max_mem_reserved=10, max_mem_allocated=8, max_dispatcher_tokens=6)

    moe_utils.write_monitor_data_to_file(str(tmp_path), print_mem_monitor_interval=10)
    assert not (tmp_path / "0").exists()

    moe_utils.write_monitor_data_to_file(str(tmp_path), print_mem_monitor_interval=10)
    assert (tmp_path / "0").exists()
    assert "exceeds threshold" in capsys.readouterr().out


def test_rank_file_is_written_with_atomic_replace(monkeypatch, tmp_path):
    _set_rank(monkeypatch, 2)
    moe_utils.initialize_cuda_monitoring(
        tp_rank=1, pp_rank=1, ep_rank=1, mem_monitor_force_print_token_threshold=100
    )
    data = moe_utils.CUDA_MONITOR_UTIL[2]
    data.update(max_mem_reserved=10, max_mem_allocated=8, max_dispatcher_tokens=6)

    real_replace = moe_utils.os.replace
    replacements = []

    def record_replace(source, destination):
        source_path = Path(source)
        assert source_path.name == "2_temp"
        assert source_path.read_text(encoding="utf-8").startswith("rank: 2|")
        replacements.append((source_path, Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(moe_utils.os, "replace", record_replace)

    moe_utils.write_monitor_data_to_file(str(tmp_path), print_mem_monitor_interval=1)

    assert replacements == [(tmp_path / "2_temp", tmp_path / "2")]
    assert (tmp_path / "2").exists()
    assert not (tmp_path / "2_temp").exists()
