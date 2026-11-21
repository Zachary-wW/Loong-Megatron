from argparse import ArgumentParser

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer.module import (
    apply_fp32_param_patterns,
    convert_module_to_dtype_except_fp32_marked,
)
from megatron.training.arguments import add_megatron_arguments


def test_rerun_data_iterator_iter_returns_self():
    wrapped = RerunDataIterator(iter([{"value": 1}, {"value": 2}]))

    assert iter(wrapped) is wrapped
    assert next(wrapped) == {"value": 1}

    wrapped.advance()
    assert next(wrapped) == {"value": 2}


def test_legacy_keep_in_float32_marker_survives_dtype_conversion():
    module = torch.nn.Module()
    module.matrix = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    module.legacy_marker = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    module.legacy_marker._keep_in_float32 = True

    converted = convert_module_to_dtype_except_fp32_marked(module, torch.float16)

    assert converted.matrix.dtype == torch.float16
    assert converted.legacy_marker.dtype == torch.float32


def test_fp32_param_patterns_select_parameters_and_buffers_by_name():
    module = torch.nn.Module()
    module.experts = torch.nn.Module()
    module.experts.expert_bias = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
    module.router = torch.nn.Module()
    module.router.register_buffer("routing_bias", torch.ones(2, dtype=torch.float16))
    module.output_layer = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))

    apply_fp32_param_patterns(module, ["expert_bias", "router"])

    assert module.experts.expert_bias.dtype == torch.float32
    assert module.experts.expert_bias.keep_in_fp32 is True
    assert module.router.routing_bias.dtype == torch.float32
    assert module.router.routing_bias.keep_in_fp32 is True
    assert module.output_layer.dtype == torch.float16
    assert not getattr(module.output_layer, "keep_in_fp32", False)


def test_packed_seq_params_carries_cpu_offsets():
    cpu_offsets = torch.tensor([0, 3, 7], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_cpu=cpu_offsets,
    )

    assert params.cu_seqlens_cpu is cpu_offsets
    assert params.cu_seqlens_cpu.device.type == "cpu"


def test_bindpcie_argument_contract():
    parser = ArgumentParser()
    add_megatron_arguments(parser)

    assert parser.parse_args([]).bindpcie is False
    assert parser.parse_args(["--bindpcie"]).bindpcie is True


def test_context_parallel_src_rank_uses_first_global_rank(monkeypatch):
    monkeypatch.setattr(parallel_state, "_CONTEXT_PARALLEL_GLOBAL_RANKS", [5, 7, 3])
    assert parallel_state.get_context_parallel_src_rank() == 5

    monkeypatch.setattr(parallel_state, "_CONTEXT_PARALLEL_GLOBAL_RANKS", None)
    with pytest.raises(AssertionError, match="context parallel group is not initialized"):
        parallel_state.get_context_parallel_src_rank()
