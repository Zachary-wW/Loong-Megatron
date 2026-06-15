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
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection
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
        self.router = TopKRouter(config=self.config, pg_collection=pg_collection)

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

        if config.moe_echo_expert_dispatch_overlap and not hasattr(MoELayer, 'moe_a2a_stream'):
            # Single stream for all token + expert-weight A2A ops.  Serialising every
            # collective on one stream eliminates concurrent IB/RDMA usage between the
            # token-dispatch buffer and the expert-weight-dispatch buffer, which caused
            # hangs in multi-node (EP > node size) scenarios.
            MoELayer.moe_a2a_stream = torch.cuda.Stream()
            MoELayer.fc1_expert_dispatch_event = torch.cuda.Event()
            MoELayer.fc2_expert_dispatch_event = torch.cuda.Event()
            MoELayer.token_dispatch_event = torch.cuda.Event()
            MoELayer.token_combine_event = torch.cuda.Event()
            # grad_combine_event: recorded on moe_a2a_stream after each combine_with_unpermute
            # backward A2A.  FlushPendingGradAccum waits on it before issuing add_() so the
            # GPU executes add_() only after all A2A ops have completed.
            MoELayer.grad_combine_event = torch.cuda.Event()
            # dispatch_bwd_event / combine_bwd_event: recorded on moe_a2a_stream after each
            # backward A2A.  default_stream.wait_event() on these provides GPU-side ordering
            # so subsequent backward ops on default_stream (expert GEMM bwd, router bwd) see
            # completed A2A results.  wait_event does NOT block the CPU, avoiding the circular
            # deadlock that wait_stream caused with synchronous NCCL on other ranks.
            MoELayer.dispatch_bwd_event = torch.cuda.Event()
            MoELayer.combine_bwd_event = torch.cuda.Event()
            # pending_expert_wgrads: (weight, wgrad) pairs collected in
            # HybridEPExpertDispatch.backward; consumed by FlushPendingGradAccum.backward
            # which fires after token_dispatch_bwd is in the moe_a2a_stream queue.
            MoELayer.pending_expert_wgrads = []

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.
        """
        probs, routing_map = self.router(hidden_states)
        metadata = self.token_dispatcher.preprocess(routing_map)
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

    def echo_forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer with echo experts.

        This implements the echo expert logic where overflow tokens are offloaded
        to spare/echo experts for better load balancing.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        residual = hidden_states
        router = self.router

        # Step 1: Routing and offloading planning
        with torch.cuda.nvtx.range("router"):
            probs, routing_map = router(hidden_states)

        with torch.cuda.nvtx.range("rerouting"):
            tokens_per_expert_current_ep_rank = routing_map.sum(dim=0)
            if self.config.moe_echo_expert_dispatch_overlap and hasattr(MoELayer, 'moe_a2a_stream'):
                from megatron.core.transformer.moe.fused_a2a import _a2a_log
                _a2a_log("FWD: wait_stream(a2a) before allgather BEGIN")
                torch.cuda.current_stream().wait_stream(MoELayer.moe_a2a_stream)
                _a2a_log("FWD: wait_stream(a2a) before allgather DONE")
            _a2a_log("FWD: NCCL allgather BEGIN") if self.config.moe_echo_expert_dispatch_overlap else None
            tokens_per_expert_per_ep_rank = gather_from_sequence_parallel_region(
                tokens_per_expert_current_ep_rank, group=self.ep_group
            ).reshape(self.ep_group.size(), self.config.num_moe_experts)
            if self.config.moe_echo_expert_dispatch_overlap:
                _a2a_log("FWD: NCCL allgather DONE")

            # Generate offloading plan to redistribute tokens to echo experts
            if self.config.moe_echo_enable_random_offloading:
                rerouting_map, rerouted_probs, expert_offloading_map = gen_random_offloading_plan(
                    routing_map,
                    probs,
                    tokens_per_expert_per_ep_rank,
                    self.ep_group.rank(),
                    ep=self.ep_group.size(),
                    spare_expert_per_ep_rank=self.config.moe_num_echo_experts // self.ep_group.size(),
                )
            else:
                num_spare_experts_per_ep_rank = self.config.moe_num_echo_experts // self.ep_group.size()
                if self.config.moe_echo_algorithm == "greedy":
                    if num_spare_experts_per_ep_rank == 1:
                        assignment_algorithm = "approx_bin_packing"
                    else:
                        assignment_algorithm = "one_shot_greedy"
                else:
                    assignment_algorithm = "sinkhorn"
                rerouting_map, rerouted_probs, expert_offloading_map = gen_offloading_plan(
                    routing_map,
                    probs,
                    tokens_per_expert_per_ep_rank,
                    self.ep_group.rank(),
                    num_ep_ranks=self.ep_group.size(),
                    num_spare_experts_per_ep_rank=num_spare_experts_per_ep_rank,
                    assignment_algorithm=assignment_algorithm,
                )
            if self.config.moe_echo_dump_dir is not None:
                GLOBAL_MOE_ROUTING_TRACKER.set_rank_info(self.ep_group)
                num_offloaded_experts_per_rank = expert_offloading_map.reshape(self.ep_group.size(), -1).sum(dim=-1)
                GLOBAL_MOE_ROUTING_TRACKER.add_data(
                    self.layer_number,
                    "num_offloaded_experts_per_rank",
                    num_offloaded_experts_per_rank,
                )
                rerouted_tokens_per_rank = rerouting_map.sum(dim=0).reshape(self.ep_group.size(), -1).sum(dim=-1)
                torch.distributed.all_reduce(
                    rerouted_tokens_per_rank,
                    group=self.ep_group,
                    op=torch.distributed.ReduceOp.SUM,
                )
                GLOBAL_MOE_ROUTING_TRACKER.add_data(
                    self.layer_number,
                    "rerouted_tokens_per_rank",
                    rerouted_tokens_per_rank,
                )

            if self.config.moe_echo_log_file is not None:
                self._echo_log(
                    tokens_per_expert_per_ep_rank,
                    rerouting_map,
                    expert_offloading_map,
                )
                GLOBAL_MOE_ROUTING_TRACKER.add_data(
                    self.layer_number,
                    "tokens_per_expert_per_ep_rank",
                    tokens_per_expert_per_ep_rank,
                )
                tokens_per_rank = tokens_per_expert_per_ep_rank.reshape(
                    self.ep_group.size(),
                    self.ep_group.size(),
                    -1,
                ).sum(dim=[0, 2])
                GLOBAL_MOE_ROUTING_TRACKER.add_data(
                    self.layer_number, "tokens_per_rank", tokens_per_rank
                )
        # Step 2: Expert weight dispatch for echo experts
        if self.config.moe_echo_expert_dispatch_overlap:
            # Overlap mode: expert dispatch is launched *inside* forward_with_dispatch_overlap,
            # after token A2A completes, so it overlaps only with home GEMM — not with
            # token A2A (which would cause two concurrent A2As contending for the NIC).
            if self.config.moe_echo_recompute_expert_dispatch:
                raise ValueError(
                    "moe_echo_expert_dispatch_overlap and moe_echo_recompute_expert_dispatch "
                    "cannot be enabled simultaneously."
                )
            # CPU-side preprocess only; actual GPU dispatch is deferred.
            fc1_expert_dispatch_metadata = self.expert_dispatcher.preprocess(expert_offloading_map)
            fc2_expert_dispatch_metadata = self.expert_dispatcher.preprocess(expert_offloading_map)
            fc1_expert_dispatch_metadata.buffer_idx = 0
            fc2_expert_dispatch_metadata.buffer_idx = 1
        else:
            # Original synchronous dispatch path
            # Create checkpoints for gradient computation
            fc1_expert_checkpoint = tensor_parallel.CheckpointWithoutOutput(only_calculate_input_grad=True)
            fc2_expert_checkpoint = tensor_parallel.CheckpointWithoutOutput(only_calculate_input_grad=True)

            # TODO: share the same preprocess operation but use different metadata object
            fc1_expert_dispatch_metadata = self.expert_dispatcher.preprocess(expert_offloading_map)
            fc2_expert_dispatch_metadata = self.expert_dispatcher.preprocess(expert_offloading_map)
            fc1_expert_dispatch_metadata.buffer_idx = 0
            fc2_expert_dispatch_metadata.buffer_idx = 1

            with torch.cuda.nvtx.range("expert_dispatch"):
                fc1_expert_weights = self.experts.get_expert_weights(
                    "fc1", self.home_expert_indices
                )
                if self.config.moe_echo_recompute_expert_dispatch:
                    dispatched_fc1_weights = fc1_expert_checkpoint.checkpoint(
                        partial(
                            self.expert_dispatcher.expert_dispatch,
                            fc1_expert_dispatch_metadata,
                        ),
                        *fc1_expert_weights,
                    )
                else:
                    dispatched_fc1_weights = self.expert_dispatcher.expert_dispatch(
                        fc1_expert_dispatch_metadata,
                        *fc1_expert_weights,
                    )
                self.experts.set_expert_weights(
                    "fc1",
                    dispatched_fc1_weights,
                    self.echo_expert_indices,
                )

                fc2_expert_weights = self.experts.get_expert_weights(
                    "fc2", self.home_expert_indices
                )
                if self.config.moe_echo_recompute_expert_dispatch:
                    dispatched_fc2_weights = fc2_expert_checkpoint.checkpoint(
                        partial(
                            self.expert_dispatcher.expert_dispatch,
                            fc2_expert_dispatch_metadata,
                        ),
                        *fc2_expert_weights,
                    )
                else:
                    dispatched_fc2_weights = self.expert_dispatcher.expert_dispatch(
                        fc2_expert_dispatch_metadata,
                        *fc2_expert_weights,
                    )
                self.experts.set_expert_weights(
                    "fc2",
                    dispatched_fc2_weights,
                    self.echo_expert_indices,
                )

        # Step 3: Token dispatch preprocess (on moe_a2a_stream to serialise all A2A on one stream)
        set_hybrid_ep_buffer_idx(self.layer_number)
        if self.config.moe_echo_expert_dispatch_overlap:
            from megatron.core.transformer.moe.fused_a2a import _a2a_log
            _a2a_log("FWD Step3: entering a2a_stream for preprocess")
            with torch.cuda.stream(MoELayer.moe_a2a_stream):
                _a2a_log("FWD Step3: wait_stream(default) BEGIN")
                MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                _a2a_log("FWD Step3: preprocess BEGIN")
                with torch.cuda.nvtx.range("token_dispatch_preprocess"):
                    metadata = self.token_dispatcher.preprocess(rerouting_map)
                    hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
                        hidden_states, rerouted_probs, metadata
                    )
                _a2a_log("FWD Step3: preprocess DONE")
        else:
            with torch.cuda.nvtx.range("token_dispatch_preprocess"):
                metadata = self.token_dispatcher.preprocess(rerouting_map)
                hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
                    hidden_states, rerouted_probs, metadata
                )
        # Step 4: Token dispatch
        def dispatch_and_compute(hidden_states, probs, metadata):
            """Dispatch tokens, run experts, and combine outputs."""
            if self.config.moe_echo_expert_dispatch_overlap:
                from megatron.core.transformer.moe.fused_a2a import _a2a_log
                with torch.cuda.stream(MoELayer.moe_a2a_stream):
                    # dispatch_preprocess already ran on moe_a2a_stream; wait_stream ensures
                    # any default-stream work submitted between Step 3 and here is also visible.
                    _a2a_log("FWD Step4: wait_stream(default) BEGIN")
                    MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                    _a2a_log("FWD Step4: wait_stream(default) DONE, token_dispatch BEGIN")
                    with torch.cuda.nvtx.range("token_dispatch"):
                        dispatched_input, probs = self.token_dispatcher.token_dispatch(
                            hidden_states, probs, metadata
                        )
                        dispatched_input, tokens_per_expert, permuted_probs = (
                            self.token_dispatcher.dispatch_postprocess(dispatched_input, probs, metadata)
                        )
                    _a2a_log("FWD Step4: token_dispatch DONE")
                    MoELayer.token_dispatch_event.record()
                # Let default stream use dispatched tokens for GEMM only after A2A completes.
                torch.cuda.default_stream().wait_event(MoELayer.token_dispatch_event)
            else:
                with torch.cuda.nvtx.range("token_dispatch"):
                    dispatched_input, probs = self.token_dispatcher.token_dispatch(
                        hidden_states, probs, metadata
                    )
                    dispatched_input, tokens_per_expert, permuted_probs = (
                        self.token_dispatcher.dispatch_postprocess(dispatched_input, probs, metadata)
                    )

            # Step 5: Expert computation
            with torch.cuda.nvtx.range("expert_compute"):
                if self.config.moe_echo_expert_dispatch_overlap:
                    # Overlap path: dispatch is launched inside forward_with_dispatch_overlap
                    # *after* token A2A (here), so expert weight A2A overlaps with home GEMM
                    # only — not with token A2A.
                    expert_output, mlp_bias = self.experts.forward_with_dispatch_overlap(
                        dispatched_input,
                        tokens_per_expert,
                        permuted_probs,
                        len(self.home_expert_indices),
                        self.home_expert_indices,
                        self.echo_expert_indices,
                        self.expert_dispatcher,
                        fc1_expert_dispatch_metadata,
                        fc2_expert_dispatch_metadata,
                        MoELayer.moe_a2a_stream,
                        MoELayer.fc1_expert_dispatch_event,
                        MoELayer.fc2_expert_dispatch_event,
                    )
                else:
                    expert_output, mlp_bias = self.experts(
                        dispatched_input, tokens_per_expert, permuted_probs
                    )

            if self.config.moe_echo_expert_dispatch_overlap:
                # Step 6: Token combine (on moe_a2a_stream, after all expert GEMM on default stream)
                with torch.cuda.stream(MoELayer.moe_a2a_stream):
                    # Wait for all expert GEMM on default stream to finish before combine A2A.
                    MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                    with torch.cuda.nvtx.range("token_combine"):
                        output = self.token_dispatcher.combine_preprocess(expert_output, metadata)
                        output = self.token_dispatcher.token_combine(output, metadata)
                        output = self.token_dispatcher.combine_postprocess(output, metadata)
                    MoELayer.token_combine_event.record()
                # Default stream waits for combine A2A to complete before returning output.
                torch.cuda.default_stream().wait_event(MoELayer.token_combine_event)
            else:
                with torch.cuda.nvtx.range("token_combine"):
                    # Step 6: Token combine
                    output = self.token_dispatcher.combine_preprocess(expert_output, metadata)
                    output = self.token_dispatcher.token_combine(output, metadata)
                    output = self.token_dispatcher.combine_postprocess(output, metadata)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(
                partial(dispatch_and_compute, metadata=metadata), False, hidden_states, probs
            )
        else:
            output, mlp_bias = dispatch_and_compute(hidden_states, probs, metadata)

        # Register for gradient computation (only in synchronous dispatch path)
        if self.config.moe_echo_recompute_expert_dispatch:
            fc1_expert_checkpoint.discard_output_and_register_recompute(output)
            fc2_expert_checkpoint.discard_output_and_register_recompute(output)

        # Handle shared expert if configured
        if self.use_shared_expert and not self.shared_expert_overlap:
            shared_expert_output = self.shared_experts(residual)
            output = output + shared_expert_output

        return output, mlp_bias

    def _echo_log(
        self,
        tokens_per_expert_per_ep_rank: torch.Tensor,
        rerouting_map: torch.Tensor,
        expert_offloading_map: torch.Tensor,
    ):
        """Log echo expert stats for the current step and layer.

        All EP ranks must call this together (collective ops inside).
        Only EP rank 0 writes the result to file.
        """
        from megatron.training.global_vars import get_args
        args = get_args()
        iteration = args.curr_iteration

        # Step filter — consistent across all ranks, no collective ops yet
        if self.config.moe_echo_log_steps is not None:
            log_steps = {int(s) for s in self.config.moe_echo_log_steps.split(',')}
            if iteration not in log_steps:
                return

        # Layer filter — consistent across all ranks
        if self.config.moe_echo_log_layers is not None:
            log_layers = {int(s) for s in self.config.moe_echo_log_layers.split(',')}
            if self.layer_number not in log_layers:
                return

        # ---- Collective operation: all EP ranks must reach here together ----
        # Each rank has its local rerouting_map; sum and all_reduce to get global counts
        ep_size = self.ep_group.size()
        num_experts = self.config.num_moe_experts
        num_home_experts_per_rank = num_experts // ep_size
        num_echo_slots = expert_offloading_map.shape[1] if expert_offloading_map.ndim == 2 else 1
        num_echo_slots_per_rank = max(1, num_echo_slots // ep_size)
        # rerouting_map is the postprocessed tensor from gen_offloading_plan, whose column layout
        # interleaves home experts and echo slots per EP rank:
        #   [home_r0..., echo_r0..., home_r1..., echo_r1..., ...]
        # Each rank occupies `section` consecutive columns.
        section = num_home_experts_per_rank + num_echo_slots_per_rank

        local_after = rerouting_map.sum(dim=0).float().to(tokens_per_expert_per_ep_rank.device)
        torch.distributed.all_reduce(local_after, group=self.ep_group)
        after_counts = local_after.long().cpu()

        # ---- Only rank 0 formats and writes from here ----
        if self.ep_group.rank() != 0:
            return

        tokens_per_expert = tokens_per_expert_per_ep_rank.sum(dim=0).cpu()

        rank_loads_before = [
            tokens_per_expert[r * num_home_experts_per_rank:(r + 1) * num_home_experts_per_rank].sum().item()
            for r in range(ep_size)
        ]
        total_tokens = sum(rank_loads_before)
        avg_rank_load = total_tokens // ep_size
        expert_hot_threshold = avg_rank_load // num_home_experts_per_rank

        rank_loads_after = []
        for r in range(ep_size):
            # postprocessed layout: each rank occupies `section` consecutive columns
            start = r * section
            load = after_counts[start:start + section].sum().item()
            rank_loads_after.append(load)

        def imbalance_pct(loads):
            avg = sum(loads) / len(loads)
            return (max(loads) - avg) / avg * 100 if avg > 0 else 0.0

        imb_before = imbalance_pct(rank_loads_before)
        imb_after = imbalance_pct(rank_loads_after)
        total_rerouted = sum(
            after_counts[r * section + num_home_experts_per_rank:r * section + section].sum().item()
            for r in range(ep_size)
        )

        SEP = "=" * 72
        lines = [
            SEP,
            f"[Echo Expert Stats] L{self.layer_number} | step={iteration} | "
            f"EP={ep_size} ranks | {num_echo_slots_per_rank} echo slots/rank",
            SEP,
            f"[1/4] BEFORE echo routing  "
            f"(rank-hot threshold={avg_rank_load} tokens, "
            f"expert-hot threshold={expert_hot_threshold} tokens)",
            f"  EP rank  load (tot)   per-expert load  (* = hot)",
            f"  {'-' * 64}",
        ]
        for r in range(ep_size):
            load = rank_loads_before[r]
            hot_tag = " HOT" if load > avg_rank_load else "    "
            start = r * num_home_experts_per_rank
            parts = []
            for i in range(num_home_experts_per_rank):
                eidx = start + i
                cnt = tokens_per_expert[eidx].item()
                star = "*" if cnt > expert_hot_threshold else " "
                parts.append(f"E{eidx}{star}: {cnt}")
            lines.append(f"  rank {r}  {load:>9}{hot_tag}  |  {'  '.join(parts)}")
        lines += [
            f"  {'-' * 64}",
            f"  avg={avg_rank_load}  max={max(rank_loads_before)}  "
            f"min={min(rank_loads_before)}  imbalance={imb_before:.1f}%",
            "",
            "[2/4] Echo cloning plan",
        ]
        offload_map_cpu = expert_offloading_map.cpu()
        plan_count = 0
        planned_echo_slots = set()
        for home_local_idx in range(offload_map_cpu.shape[0]):
            for echo_slot_local in range(offload_map_cpu.shape[1]):
                if not offload_map_cpu[home_local_idx, echo_slot_local]:
                    continue
                home_rank = home_local_idx // num_home_experts_per_rank
                home_global_idx = home_rank * num_home_experts_per_rank + (home_local_idx % num_home_experts_per_rank)
                home_load = tokens_per_expert[home_global_idx].item()
                spillover = max(0, home_load - expert_hot_threshold)
                echo_rank = echo_slot_local // num_echo_slots_per_rank if num_echo_slots_per_rank > 0 else 0
                echo_local_in_rank = echo_slot_local % max(num_echo_slots_per_rank, 1)
                # fix: use postprocessed interleaved layout to find the correct column
                echo_col = echo_rank * section + num_home_experts_per_rank + echo_local_in_rank
                rerouted = after_counts[echo_col].item() if echo_col < len(after_counts) else 0
                planned_echo_slots.add(echo_slot_local)
                plan_count += 1
                lines.append(
                    f"  E{home_global_idx} (rank {home_rank}, local {home_local_idx % num_home_experts_per_rank}) "
                    f"| load={home_load} | spillover={spillover}  =>  "
                    f"echo slot {echo_slot_local} (rank {echo_rank}, local {echo_local_in_rank}) "
                    f"=>  {rerouted} tokens rerouted"
                )
        # Also show echo slots activated by depth_first_allocation (not in expert_offloading_map)
        num_echo_slots_total = offload_map_cpu.shape[1] if offload_map_cpu.ndim == 2 else 1
        for echo_slot_local in range(num_echo_slots_total):
            if echo_slot_local in planned_echo_slots:
                continue
            echo_rank = echo_slot_local // num_echo_slots_per_rank if num_echo_slots_per_rank > 0 else 0
            echo_local_in_rank = echo_slot_local % max(num_echo_slots_per_rank, 1)
            echo_col = echo_rank * section + num_home_experts_per_rank + echo_local_in_rank
            rerouted = after_counts[echo_col].item() if echo_col < len(after_counts) else 0
            if rerouted > 0:
                plan_count += 1
                lines.append(
                    f"  [unplanned/DFS] echo slot {echo_slot_local} "
                    f"(rank {echo_rank}, local {echo_local_in_rank}) "
                    f"=>  {rerouted} tokens rerouted"
                )
        if plan_count == 0:
            lines.append("  (no cloning this step)")
        lines += ["", "[3/4] AFTER echo routing  (projected)",
                  "  EP rank  load (tot)  Δ  per-expert load (home + echo slots)",
                  f"  {'-' * 72}"]
        for r in range(ep_size):
            load_after = rank_loads_after[r]
            delta = load_after - rank_loads_before[r]
            delta_str = f"+{delta}" if delta >= 0 else str(delta)
            # fix: rank r's home experts start at col r*section in the postprocessed layout
            rank_start = r * section
            parts = [f"E{r * num_home_experts_per_rank + i}: {after_counts[rank_start + i].item()}"
                     for i in range(num_home_experts_per_rank)]
            for slot in range(num_echo_slots_per_rank):
                col = rank_start + num_home_experts_per_rank + slot
                if col < len(after_counts) and after_counts[col].item() > 0:
                    parts.append(f"Echo{slot}: {after_counts[col].item()}")
            lines.append(f"  rank {r}  {load_after:>9} {delta_str:>7}  |  {'  '.join(parts)}")
        lines += [
            f"  {'-' * 72}",
            f"  avg={avg_rank_load}  max={max(rank_loads_after)}  "
            f"min={min(rank_loads_after)}  imbalance={imb_after:.1f}%",
            "",
            "[4/4] Load-balance improvement",
            f"  imbalance  before: {imb_before:.1f}%  ->  after: {imb_after:.1f}%  "
            f"({imb_after - imb_before:+.1f} pp)",
            f"  max rank load      before: {max(rank_loads_before)}  ->  "
            f"after: {max(rank_loads_after)}  ({max(rank_loads_after) - max(rank_loads_before):+d} tokens)",
            f"  tokens rerouted: {total_rerouted} / {total_tokens} "
            f"({total_rerouted / total_tokens * 100:.1f}%)" if total_tokens > 0 else
            f"  tokens rerouted: {total_rerouted} / {total_tokens}",
            SEP,
            "",
        ]
        with open(self.config.moe_echo_log_file, "a") as f:
            f.write("\n".join(lines))

    def forward(self, hidden_states: torch.Tensor):
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

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # MoE forward: route -> dispatch -> compute -> combine -> post-combine
        def custom_forward(hidden_states):
            shared_expert_output = self.shared_experts_compute(hidden_states)
            hidden_states, probs, metadata = self.router_and_preprocess(hidden_states)
            dispatched_input, probs = self.dispatch(hidden_states, probs, metadata)
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
            hidden_states, probs, metadata = self.router_and_preprocess(hidden_states)
            dispatched_input, probs = self.dispatch(hidden_states, probs, metadata)
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
