# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Mixture-of-experts layer implementations and echo expert orchestration."""

from contextlib import nullcontext
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import torch

from megatron.core.transformer.moe.moe_utils import GLOBAL_MOE_ROUTING_TRACKER

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    get_default_pg_collection,
    initialize_cuda_monitoring,
    monitor_max_memory_usage,
    monitor_max_dispatcher_tokens,
    write_monitor_data_to_file,
)
from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.moe.fused_a2a import set_hybrid_ep_buffer_idx
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.offloading_planner import gen_offloading_plan, gen_random_offloading_plan
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEElasticExpertDispatcher,
    MoESyncFreeElasticExpertDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ):
        """Initialize base MoE layer state and process groups."""
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        if self.config.moe_enable_echo:
            self.num_local_total_experts = (
                self.config.num_moe_experts + self.config.moe_num_echo_experts
            ) // ep_size
        else:
            self.num_local_total_experts = self.config.num_moe_experts // ep_size
        self.num_home_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_total_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_total_experts)
        ]
        if self.config.moe_enable_echo:
            assert all(
                map(
                    lambda x: x < self.config.num_moe_experts + self.config.moe_num_echo_experts,
                    self.local_expert_indices,
                )
            )
        else:
            assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class FlushPendingGradAccum(torch.autograd.Function):
    """Identity in forward.

    In backward, submits deferred main_grad.add_() to moe_a2a_stream.
    This Function is inserted into the autograd graph just before dispatch_preprocess
    in the overlap path.  Its backward is called by the autograd engine AFTER
    token_dispatch_bwd and dispatch_preprocess_bwd have already been submitted to
    moe_a2a_stream, so appending add_() here puts it at the end of the stream queue —
    after all backward A2A ops — with no extra synchronisation required.
    """

    @staticmethod
    def forward(ctx, x):
        """Return the input unchanged in the forward pass."""
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """Flush pending expert weight gradients during backward."""
        # MoELayer is a module-level name in this same file (moe_layer.py).
        # Python resolves it via the function's __globals__ at call time — no
        # local import is needed.  A local `from .moe_layer import MoELayer`
        # inside an autograd backward can trigger Python's import lock in
        # PyTorch's C++ autograd threads, causing pybind11 exception-state
        # inconsistency that shows up as aten::detach RecordFunction warnings.
        pending = getattr(MoELayer, 'pending_expert_wgrads', None)
        if pending:
            from megatron.core.transformer.moe.fused_a2a import _a2a_log
            _a2a_log("FlushPendingGrad BWD: entering a2a_stream")
            with torch.cuda.stream(MoELayer.moe_a2a_stream):
                # grad_combine_event was recorded on moe_a2a_stream after the last
                # combine_with_unpermute A2A.  Waiting here is a safety net; FIFO
                # already guarantees ordering since all A2A ops were submitted first.
                MoELayer.moe_a2a_stream.wait_event(MoELayer.grad_combine_event)
                # wgrad_home is accumulated to main_grad on the DEFAULT stream by
                # gradient-accumulation fusion (inside the home expert GEMM backward).
                # wgrad_echo (below) runs on moe_a2a_stream.  Without an explicit
                # dependency, both add_() calls can overlap on the GPU and corrupt
                # main_grad.  wait_stream serialises them: all default-stream ops
                # (including wgrad_home add) must complete before wgrad_echo add.
                _a2a_log("FlushPendingGrad BWD: wait_stream(default) BEGIN")
                MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                _a2a_log("FlushPendingGrad BWD: add_() BEGIN")
                for weight, wgrad in pending:
                    weight.main_grad.add_(wgrad)
                _a2a_log("FlushPendingGrad BWD: add_() DONE")
            pending.clear()
        return grad_output


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )
        self.routed_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "routed_experts" in config.recompute_modules
        )

        # Initialize router
        self.router = TopKRouter(
            config=self.config,
            pg_collection=pg_collection,
            layer_number=layer_number,
            is_mtp_layer=self.is_mtp_layer,
        )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_total_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_total_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_total_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        if config.moe_enable_echo:
            if config.moe_echo_expert_dispatcher_type == "hybridep":
                self.expert_dispatcher = MoESyncFreeElasticExpertDispatcher(
                    config=self.config, pg_collection=pg_collection
                )
            elif config.moe_echo_expert_dispatcher_type == "alltoall":
                self.expert_dispatcher = MoEElasticExpertDispatcher(
                    config=self.config, pg_collection=pg_collection
                )
            else:
                raise ValueError(f"Unsupported expert dispatcher type: {config.moe_echo_expert_dispatcher_type}")
            num_echo_local_experts = self.config.moe_num_echo_experts // self.ep_group.size()
            echo_config = dataclasses.replace(
                self.config, gradient_accumulation_fusion=False, moe_enable_echo=False
            )
            self.experts = build_module(
                self.submodules.experts,
                num_echo_local_experts + self.num_home_experts,
                config=echo_config,
                pg_collection=pg_collection,
            )
            self.echo_expert_indices = list(
                range(self.num_home_experts, num_echo_local_experts + self.num_home_experts)
            )
            self.home_expert_indices = list(range(self.num_home_experts))
            self.experts.free_expert_parameters(self.echo_expert_indices)
        else:
            self.experts = build_module(
                self.submodules.experts,
                self.num_home_experts,
                self.config,
                pg_collection=pg_collection,
            )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, pg_collection=pg_collection
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

        if config.enable_moe_mem_monitor:
            initialize_cuda_monitoring(
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                pp_rank=parallel_state.get_pipeline_model_parallel_rank(),
                ep_rank=parallel_state.get_expert_model_parallel_rank(),
                mem_monitor_force_print_token_threshold=config.moe_mem_monitor_force_print_token_threshold
            )

    def router_and_preprocess(
        self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ):
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.
        """
        residual = hidden_states
        probs, routing_map = self.router(hidden_states, input_ids=input_ids)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, probs, metadata
        )
        return hidden_states, probs, metadata

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor, metadata):
        """Dispatches tokens to assigned expert ranks via communication.
        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs, metadata)

    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts."""
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8:
                    shared_expert_output = te_checkpoint(
                        self.shared_experts,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        self.shared_experts, False, hidden_states
                    )
            else:
                shared_expert_output = self.shared_experts(hidden_states)

        return shared_expert_output

    def pre_routed_experts_compute(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        metadata: torch.Tensor,
    ):
        """Pre-processing before expert computation."""
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs, metadata)
        )

        return dispatched_input, tokens_per_expert, permuted_probs


    def routed_experts_compute(
        self,
        dispatched_input: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Computes the output of the routed experts on the dispatched tokens."""

        if self.config.enable_moe_mem_monitor:
            monitor_max_memory_usage()
            monitor_max_dispatcher_tokens(tokens_per_expert)
            if self.config.moe_mem_monitor_log is not None:
                write_monitor_data_to_file(self.config.moe_mem_monitor_log, self.config.print_moe_mem_monitor_interval)

        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        return expert_output, mlp_bias        
    def post_routed_experts_compute(
        self,
        expert_output: torch.Tensor,
        metadata: torch.Tensor,
    ):
        """Post-processing after expert computation."""
        output = self.token_dispatcher.combine_preprocess(expert_output, metadata)
        return output


    def combine(self, output: torch.Tensor, metadata: torch.Tensor):
        """Combines expert outputs via communication.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication).
        """
        output = self.token_dispatcher.token_combine(output, metadata)
        return output

    def post_combine(self, output: torch.Tensor, metadata: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Post-processes combined output and adds shared expert output.

        This method applies post-processing to the combined expert outputs and
        adds the output from the shared expert if it exists.
        """
        output = self.token_dispatcher.combine_postprocess(output, metadata)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        When echo mode is enabled (moe_enable_echo=True), the forward pass uses an alternative
        implementation that offloads overflow tokens to echo experts for better load balancing.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.
            input_ids (torch.Tensor, optional): The input IDs tensor. Shape [seq_length, bsz].
                Only used for hash-based MoE routing. Defaults to None.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if (
            self.training
            and self.attn_tp_group.size() > 1
            and not self.config.sequence_parallel
            and self.config.experimental_attention_variant != "dsv4_hybrid"
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # MoE forward: route -> dispatch -> compute -> combine -> post-combine
        def custom_forward(hidden_states):
            shared_expert_output = self.shared_experts_compute(hidden_states)
            hidden_states, probs, residual = self.router_and_preprocess(
                hidden_states, input_ids=input_ids
            )
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            dispatched_input, tokens_per_expert, permuted_probs = self.pre_routed_experts_compute(
                dispatched_input, probs, metadata)
            expert_output, mlp_bias = self.routed_experts_compute(dispatched_input, tokens_per_expert, permuted_probs)
            output = self.post_routed_experts_compute(expert_output, metadata)
            output = self.combine(output, metadata)
            output = self.post_combine(output, metadata, shared_expert_output)
            return output, mlp_bias

        # Use echo forward if echo mode is enabled
        if self.config.moe_enable_echo:
            return self.echo_forward(hidden_states)

        # TODO: moe layer recompute without router

        def custom_forward_exclude_shared_experts(hidden_states):
            hidden_states, probs, residual = self.router_and_preprocess(
                hidden_states, input_ids=input_ids
            )
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            dispatched_input, tokens_per_expert, permuted_probs = self.pre_routed_experts_compute(
                dispatched_input, probs, metadata)
            expert_output, mlp_bias = self.routed_experts_compute(dispatched_input, tokens_per_expert, permuted_probs)
            output = self.post_routed_experts_compute(expert_output, metadata)
            output = self.token_dispatcher.token_combine(output)
            output = self.token_dispatcher.combine_postprocess(output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        elif self.routed_experts_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward_exclude_shared_experts,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(
                   custom_forward_exclude_shared_experts, 
                   False, 
                   hidden_states,
                )
            if self.use_shared_expert and not self.shared_expert_overlap:
                output = output + self.shared_experts_compute(hidden_states)

        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
