# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from contextlib import nullcontext

import pytest
import torch

from megatron.core.models.common import fine_grained_callables as common_callables
from megatron.core.models.common.fine_grained_callables import build_layer_callables
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    DummyNode,
    DummyState,
    build_data,
    compare_captures,
    deterministic_mode,
    get_test_config,
    get_valid_flex_dispatcher_backend,
    get_valid_token_dispatcher_types,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils


def run_model_ref_with_capture(model, input_tensors, iterations):
    """
    Runs the model in reference mode and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each iteration.
        iterations: Number of iterations to run the model.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """

    output_tensors = []
    for i in range(iterations):
        output = model(input_tensors[i].clone())[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def run_model_submodules_with_capture(model, input_tensors, microbatches):
    """
    Runs the model with all-to-all overlap optimization and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each microbatch.
        microbatches: Number of microbatches to process.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """

    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()

    output_tensors = []
    # get callables
    callables, dw = build_layer_callables(model)
    attn, dispatch, moe, combine, post_process = callables
    assert post_process is None
    dummy_model = DummyState()
    dummy_model.decoder = DummyState()
    dummy_model.decoder.final_layernorm = None
    for i in range(microbatches):
        # build mock func/state
        node = DummyNode()
        node.is_mtp = False
        node.chunk_state.model = dummy_model

        # attn fwd
        local_tokens, probs = attn(node, input_tensors[i])

        # dispatch fwd
        dispatched_tokens = dispatch(node, local_tokens, probs)

        # moe fwd
        expert_output = moe(node, dispatched_tokens)

        # combine fwd
        hidden_states = combine(node, expert_output)

        # loss
        output_tensors.append(hidden_states)
        hidden_states.backward(torch.ones_like(hidden_states))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def test_mtp_pre_dispatch_applies_hybrid_empty_decoder_final_norm(monkeypatch):
    """Covers the HybridModel empty-decoder MTP pre-dispatch final_norm path."""

    from megatron.core.models.hybrid.hybrid_model import HybridModel

    def inner_pre_dispatch(_node, hidden_states):
        return hidden_states

    def unused_forward(*_args, **_kwargs):
        raise AssertionError("only MTP pre-dispatch should run in this test")

    def fake_build_layer_callables(_layer):
        return (
            [inner_pre_dispatch, unused_forward, unused_forward, unused_forward, None],
            {"pre_dispatch_computation": object()},
        )

    class FakeMTPConfig:
        sequence_parallel = False

    class FakeMTPLayer:
        config = FakeMTPConfig()
        eh_proj = object()
        mtp_model_layer = object()

        def _get_embeddings(
            self, input_ids, position_ids, embedding, hidden_states, packed_seq_params, padding_mask
        ):
            return input_ids, position_ids, padding_mask, None, hidden_states

        def _concat_embeddings(self, hidden_states, decoder_input):
            return hidden_states

        def _postprocess(self, hidden_states):
            return hidden_states

    monkeypatch.setattr(common_callables, "build_layer_callables", fake_build_layer_callables)
    monkeypatch.setattr(common_callables, "get_layer_moe_metadata", lambda _layer: (True, 1))
    monkeypatch.setattr(common_callables, "get_mtp_layer_offset", lambda _config, _vp_stage: 0)

    model = HybridModel.__new__(HybridModel)
    torch.nn.Module.__init__(model)
    model.decoder = DummyState()
    model.decoder.layers = []
    model.decoder.final_norm = lambda hidden_states: hidden_states + 4.0
    model.embedding = object()
    model.vp_stage = None

    node = DummyNode()
    node.chunk_state = DummyState()
    node.chunk_state.model = model
    node.chunk_state.context = None
    node.chunk_state.packed_seq_params = None
    node.is_first_layer = True

    hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2).requires_grad_()
    expected = hidden_states + 4.0
    forward_funcs, _ = common_callables.build_mtp_layer_callables(FakeMTPLayer())

    output = forward_funcs[0](node, hidden_states)

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(node.chunk_state.mtp_hidden_states[0], expected)


def test_echo_fine_grained_callables_combine_once(monkeypatch):
    """ECHO uses the split callables and hands one preprocessed value to combine."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda _self, _stream: None)

    class Metadata:
        token_probs = None
        dispatched_probs = None

        @staticmethod
        def get_number_of_tokens_per_expert():
            return torch.tensor([1])

    class FakeTokenDispatcher:
        metadata = Metadata()

    class EchoMLP(MoELayer):
        num_local_experts = 1
        use_shared_expert = False
        shared_expert_overlap = False
        token_dispatcher = FakeTokenDispatcher()

        def __init__(self):
            self.calls = []

        def shared_experts_compute(self, hidden_states):
            return None

        def _echo_route(self, hidden_states):
            self.calls.append("route")
            routing_map = torch.ones(hidden_states.shape[0], 2, dtype=torch.bool)
            probs = torch.ones(hidden_states.shape[0], 2)
            return routing_map, probs, torch.ones(2, 2, dtype=torch.bool)

        def preprocess(self, hidden_states, probs, routing_map):
            self.calls.append("preprocess")
            return hidden_states, probs

        def dispatch(self, hidden_states, probs):
            self.calls.append("dispatch")
            return hidden_states, probs

        def _prepare_echo_expert_dispatch(self, expert_offloading_map):
            self.calls.append("prepare_expert_dispatch")
            return object(), object()

        def echo_routed_experts_compute(
            self,
            dispatched_input,
            dispatched_probs,
            fc1_expert_dispatch_metadata,
            fc2_expert_dispatch_metadata,
        ):
            self.calls.append("expert_compute")
            return dispatched_input + 1

        def combine(self, output):
            self.calls.append("combine")
            return output + 1

        def postprocess(self, output, shared_expert_output):
            self.calls.append("postprocess")
            return output + 1

    class FakeConfig:
        moe_token_dispatcher_type = "flex"
        moe_flex_dispatcher_backend = "hybridep"
        moe_enable_echo = True
        moe_ncclep_zero_copy = False
        bias_dropout_fusion = False

    class FakeLayer:
        config = FakeConfig()
        mlp = EchoMLP()
        offload_mlp_norm = False
        recompute_pre_mlp_layernorm = False
        hidden_dropout = 0.0
        training = True

        @staticmethod
        def _forward_attention(hidden_states, **_kwargs):
            return hidden_states + 1, None

        @staticmethod
        def pre_mlp_layernorm(hidden_states):
            return hidden_states

        @staticmethod
        def init_backward_dw_wrapper():
            return None

        backward_dw_wrapper = staticmethod(lambda: None)

        @staticmethod
        def bias_dropout_add_exec_handler():
            return nullcontext()

        @staticmethod
        def mlp_bda(_training, _bias_dropout_fusion):
            return lambda mlp_output_with_bias, residual, _dropout: (
                mlp_output_with_bias[0] + residual
            )

    layer = FakeLayer()
    forward_funcs, _ = common_callables.build_transformer_layer_callables(layer)
    pre_dispatch, dispatch, moe, combine, post_process = forward_funcs
    assert post_process is None

    node = DummyNode()
    node.layer_state = DummyState()
    node.chunk_state = DummyState()
    node.chunk_state.model = DummyState()
    node.chunk_state.model.decoder = DummyState()
    node.chunk_state.model.decoder.final_layernorm = None
    node.is_mtp = False
    node.is_last_layer = False

    hidden_states = torch.zeros(2, 1, 2)
    local_tokens, probs = pre_dispatch(node, hidden_states)
    dispatched_tokens = dispatch(node, local_tokens, probs)
    expert_output = moe(node, dispatched_tokens)
    output = combine(node, expert_output)

    assert layer.mlp.calls.count("route") == 1
    assert layer.mlp.calls.count("combine") == 1
    assert layer.mlp.calls.count("postprocess") == 1
    torch.testing.assert_close(output, torch.full_like(output, 5.0))


class TestTransformerLayerSubmoduleCallables:
    """
    Test class for transformer layer submodule callables.

    This class contains tests to verify that the transformer layer submodule callables
    provide the same results as the reference implementation.
    """

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    @pytest.mark.parametrize("permute_fusion", [True, False])
    def test_1f1b_overlap(self, dispatcher_type, grouped_gemm, permute_fusion):
        """
        Tests the 1-forward-1-backward overlap optimization.

        This test verifies that the all-to-all overlap optimization produces
        the same results as the reference implementation.
        """

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            expert_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )
        qk_layernorm = True
        extra_kwargs = {
            "moe_token_dispatcher_type": dispatcher_type,
            "moe_permute_fusion": permute_fusion,
            "qk_layernorm": qk_layernorm,
        }
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = get_valid_flex_dispatcher_backend()
        config = get_test_config(extra_kwargs=extra_kwargs, moe_grouped_gemm=grouped_gemm)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_submodules = get_gpt_layer_with_transformer_engine_submodules(
                num_experts=8,
                moe_grouped_gemm=grouped_gemm,
                qk_layernorm=qk_layernorm,
                multi_latent_attention=True,
            )
            model = TransformerLayer(config, transformer_layer_submodules)

            params = reset_model(model)
            input_tensors = [build_data() for _ in range(microbatches)]

            capture_ref = run_model_ref_with_capture(model, input_tensors, microbatches)
            reset_model(model, params)
            capture_callables = run_model_submodules_with_capture(
                model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_callables, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
            Utils.destroy_model_parallel()
