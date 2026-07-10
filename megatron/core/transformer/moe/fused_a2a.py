# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

"""Fused all-to-all helpers for DeepEP and HybridEP dispatch paths."""

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

from typing import Optional
import torch
from torch._subclasses.fake_tensor import DispatchCacheInfo

# Lazy reference to MoELayer — resolved at first use to avoid a circular import
# at module load time (moe_layer imports fused_a2a, fused_a2a would import moe_layer).
# Placed at module level so it is never imported inside an autograd backward, which
# can trigger Python's import lock in PyTorch's C++ autograd threads and cause
# pybind11 exception-state inconsistencies.
_MoELayer = None


def _get_moe_layer_cls():
    """Return the lazily imported MoELayer class."""
    global _MoELayer
    if _MoELayer is None:
        from megatron.core.transformer.moe import moe_layer as _ml  # noqa: PLC0415
        _MoELayer = _ml.MoELayer
    return _MoELayer

_buffer = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


try:
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None

# ── Debug: trace every RDMA op to identify hang point ──────────────────────────
import sys as _sys
_a2a_op_counter = 0

_hybrid_ep_buffers = [None, None]
_hybrid_ep_buffer_idx = 0
_hybrid_ep_buf_events = [None, None]
_expert_dispatch_buf_events = [None, None]


def set_hybrid_ep_buffer_idx(layer_number: int) -> None:
    """Select which of the two buffers to use for the current layer.

    Using 2 buffers (mod 2) tolerates up to 1-layer rank drift in the backward
    pass, where no NCCL collective limits inter-rank divergence.
    """
    global _hybrid_ep_buffer_idx
    _hybrid_ep_buffer_idx = layer_number % 2


def get_hybrid_ep_buffer():
    """Return the active HybridEPBuffer for the current layer."""
    buf = _hybrid_ep_buffers[_hybrid_ep_buffer_idx]
    if buf is None:
        buf = _hybrid_ep_buffer  # fallback to legacy single buffer
    return buf


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
) -> None:
    '''
    Initialize the HybridEP buffer, including buffer allocation and metadata
    initialization.

    If a runtime dispatch/combine requires a larger buffer than the one
    initialized, the buffer will be reallocated at runtime,
    incuring extra run-time overhead.

    Args:
        group (torch.distributed.ProcessGroup):
            Process group for HybridEP all-to-all communication.
        hidden_dim (int):
            Hidden dimension of the input tensor.
        seq_len (int):
            Maximum sequence length of the input tensor.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (int):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (int):
            Number of SMs used by the combine API.
        fp8_dispatch (bool):
            Whether to use FP8 communication during the dispatch phase.
    '''
    global _hybrid_ep_buffer, _hybrid_ep_buffers
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
    )
    # Initialize both double-buffer slots with the same parameters.
    _hybrid_ep_buffers[0] = _hybrid_ep_buffer
    _hybrid_ep_buffers[1] = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
    )


class HybridEPDispatch(torch.autograd.Function):
    '''
    Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(
        ctx,
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        '''
        Forward pass of fused dispatch of the HybridEP backend
        '''
        if _hybrid_ep_buffer is None:
            seq_len, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 token dispatch
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                seq_len,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                fp8_dispatch,
            )
        buffer = get_hybrid_ep_buffer()
        # Defaultly, the output token_per_expert and num_dispatched_tokens_tensor
        # will be put on the CPU to avoid the potential sync in combine/backward pass,
        # but if we provide the num_dispatched_tokens and num_permuted_tokens on CPU,
        # we do not need to the D2H here.
        non_blocking = num_permuted_tokens is not None
        # Process the dispatch
        # ── Per-buffer event sync: ensure this buffer's previous combine has completed
        # on GPU so that DeepEP's host-side readiness flag is up-to-date. ──
        buf_event = _hybrid_ep_buf_events[_hybrid_ep_buffer_idx]
        if buf_event is not None:
            buf_event.synchronize()
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
            use_fp8=False,
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.buffer_idx = _hybrid_ep_buffer_idx
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
        """Backward pass of fused dispatch of the HybridEP backend."""
        handle = ctx.handle
        buffer = (
            _hybrid_ep_buffers[ctx.buffer_idx]
            if _hybrid_ep_buffers[ctx.buffer_idx] is not None
            else _hybrid_ep_buffer
        )
        MoELayer = _get_moe_layer_cls()
        if hasattr(MoELayer, 'moe_a2a_stream'):
            with torch.cuda.stream(MoELayer.moe_a2a_stream):
                MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                combined_hidden, combined_probs = buffer.combine_with_unpermute(
                    hidden=grad_x,
                    probs=grad_probs,
                    handle=handle,
                    pad_multiple=ctx.pad_multiple,
                )
                # Record event: this buffer's combine is done on moe_a2a_stream.
                if _hybrid_ep_buf_events[ctx.buffer_idx] is None:
                    _hybrid_ep_buf_events[ctx.buffer_idx] = torch.cuda.Event()
                _hybrid_ep_buf_events[ctx.buffer_idx].record()
                MoELayer.dispatch_bwd_event.record()
            # GPU-side wait: default stream waits for moe_a2a_stream to finish the A2A
            # before consuming combined_hidden in subsequent backward ops (e.g. router
            # backward AllReduce). Using wait_event (not wait_stream) keeps the CPU
            # non-blocking, avoiding the circular CPU deadlock with synchronous NCCL on
            # other ranks that wait_stream caused previously.
            torch.cuda.default_stream().wait_event(MoELayer.dispatch_bwd_event)
        else:
            combined_hidden, combined_probs = buffer.combine_with_unpermute(
                hidden=grad_x,
                probs=grad_probs,
                handle=handle,
                pad_multiple=ctx.pad_multiple,
            )
            # Record event even on non-stream path
            if _hybrid_ep_buf_events[ctx.buffer_idx] is None:
                _hybrid_ep_buf_events[ctx.buffer_idx] = torch.cuda.Event()
            _hybrid_ep_buf_events[ctx.buffer_idx].record()
        return combined_hidden, None, combined_probs, None, None, None, None, None, None, None


class HybridEPCombine(torch.autograd.Function):
    '''
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(
        ctx, x, handle, num_permuted_tokens=None, pad_multiple=None
    ):
        '''
        Forward pass of fused combine of the HybridEP backend
        '''
        buffer = get_hybrid_ep_buffer()
        combined_hidden, _ = buffer.combine_with_unpermute(
            hidden=x,
            handle=handle,
            pad_multiple=pad_multiple,
        )
        # Record event on current stream marking this buffer's combine completion.
        # The next dispatch_with_permute on the same buffer will event.synchronize()
        # to ensure the host-side readiness flag is fresh.
        if _hybrid_ep_buf_events[_hybrid_ep_buffer_idx] is None:
            _hybrid_ep_buf_events[_hybrid_ep_buffer_idx] = torch.cuda.Event()
        _hybrid_ep_buf_events[_hybrid_ep_buffer_idx].record()
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        ctx.buffer_idx = _hybrid_ep_buffer_idx
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        '''
        Backward pass of fused combine of the HybridEP backend
        '''
        handle = ctx.handle
        buffer = (
            _hybrid_ep_buffers[ctx.buffer_idx]
            if _hybrid_ep_buffers[ctx.buffer_idx] is not None
            else _hybrid_ep_buffer
        )
        MoELayer = _get_moe_layer_cls()
        if hasattr(MoELayer, 'moe_a2a_stream'):
            with torch.cuda.stream(MoELayer.moe_a2a_stream):
                MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                # Per-buffer event sync: ensure previous combine on this buffer finished
                buf_event = _hybrid_ep_buf_events[ctx.buffer_idx]
                if buf_event is not None:
                    buf_event.synchronize()
                dispatched_hidden, _, _, _, _ = buffer.dispatch_with_permute(
                    hidden=grad_x,
                    scaling_factor=None,
                    handle=handle,
                    pad_multiple=ctx.pad_multiple,
                    num_permuted_tokens=ctx.num_permuted_tokens,
                )
                MoELayer.combine_bwd_event.record()
            # GPU-side wait: default stream waits for moe_a2a_stream to finish the A2A
            # before consuming dispatched_hidden in subsequent expert GEMM backward.
            # Using wait_event (not wait_stream) keeps the CPU non-blocking.
            torch.cuda.default_stream().wait_event(MoELayer.combine_bwd_event)
        else:
            buf_event = _hybrid_ep_buf_events[ctx.buffer_idx]
            if buf_event is not None:
                buf_event.synchronize()
            dispatched_hidden, _, _, _, _ = buffer.dispatch_with_permute(
                hidden=grad_x,
                scaling_factor=None,
                handle=handle,
                pad_multiple=ctx.pad_multiple,
                num_permuted_tokens=ctx.num_permuted_tokens,
            )
        return dispatched_hidden, None, None, None, None

try:
    from transformer_engine.pytorch.tensor import QuantizedTensor
except ImportError:
    HAVE_TE_QUANTIZED_TENSOR = False
else:
    HAVE_TE_QUANTIZED_TENSOR = True

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
import transformer_engine_torch as tex


class HybridEPExpertDispatch(torch.autograd.Function):
    '''
    Fused dispatch operation for expert dispatch using the HybridEP backend
    '''
    # Two buffers: [0]=FC1, [1]=FC2.  Separate buffers avoid host-flag collision
    # when backward does two consecutive combine_with_unpermute calls.
    expert_dispatch_buffers = [None, None]

    @staticmethod
    def preprocess(routing_map, weight_chunk_size, *expert_weights):
        '''
        Preprocess expert weights before dispatch: extract raw data, stack into a
        contiguous weight_tensor, and expand routing_map to match chunked shape.

        Separated from forward so callers can run this on the default stream (where
        it overlaps with preceding computation rather than competing with dispatch
        communication for HBM bandwidth).

        Returns a dict with keys:
            weight_tensor, scale_tensor, routing_map_expanded,
            weight_shape, num_chunks_per_weight,
            fp8_dispatch, quantized_tensor_class, blockwise_is_2d_scaled
        '''
        num_total_experts = routing_map.shape[1]
        num_local_home_experts = len(expert_weights)
        weight_list = []
        scale_list = []
        weight_shape = expert_weights[0].shape
        fp8_dispatch = False
        quantized_tensor_class = None
        blockwise_is_2d_scaled = False
        for weight in expert_weights:
            if HAVE_TE_QUANTIZED_TENSOR and isinstance(weight, QuantizedTensor):
                quantized_tensor_class = weight.__class__
                row_weight, col_weight = weight.get_data_tensors()
                metadata = weight.get_metadata()
                # MXFP8: uint8 E8M0, view converts 4 bytes -> 1 float32
                row_scale = metadata['rowwise_scale_inv'].view(torch.float32).ravel()
                col_scale = metadata['columnwise_scale_inv'].view(torch.float32).ravel()
                weight_list.extend([row_weight.ravel(), col_weight.ravel()])
                scale_list.extend([row_scale.ravel(), col_scale.ravel()])
                fp8_dispatch = True
            else:
                weight_list.append(weight.ravel())

        # Chunk the weight for hybridep to dispatch a small piece each time
        weight_tensor = torch.stack(weight_list, dim=0).reshape(num_local_home_experts, -1)
        num_chunks_per_weight = weight_tensor.shape[1] // weight_chunk_size
        weight_tensor = weight_tensor.reshape(num_local_home_experts * num_chunks_per_weight, weight_chunk_size)

        if fp8_dispatch:
            scale_tensor = torch.stack(scale_list, dim=0)
            scale_tensor = scale_tensor.reshape(num_local_home_experts * num_chunks_per_weight, -1)
        else:
            scale_tensor = None
        routing_map_expanded = (
            routing_map.reshape(num_local_home_experts, 1, num_total_experts)
            .expand(-1, num_chunks_per_weight, -1)
            .reshape(num_local_home_experts * num_chunks_per_weight, num_total_experts)
        ).contiguous()

        return dict(
            weight_tensor=weight_tensor,
            scale_tensor=scale_tensor,
            routing_map_expanded=routing_map_expanded,
            meta=dict(
                weight_shape=weight_shape,
                num_chunks_per_weight=num_chunks_per_weight,
                fp8_dispatch=fp8_dispatch,
                quantized_tensor_class=quantized_tensor_class,
                blockwise_is_2d_scaled=blockwise_is_2d_scaled,
            ),
        )

    @staticmethod
    def forward(
        ctx,
        weight_tensor,
        routing_map_expanded,
        scale_tensor,
        preprocess_meta,
        group,
        handle,
        num_local_echo_experts,
        num_sms_dispatch_api,
        num_sms_combine_api,
        num_dispatched_weights,
        *expert_weights,
    ):
        """
        Forward pass of fused dispatch of the HybridEP backend.

        weight_tensor / routing_map_expanded / scale_tensor: tensor outputs of
            HybridEPExpertDispatch.preprocess(), passed as explicit tensor args so
            autograd can track device placement correctly.
        preprocess_meta: non-tensor metadata dict from preprocess()
            (weight_shape, num_chunks_per_weight, fp8_dispatch, etc.).
        expert_weights: original home expert weight tensors, kept as autograd inputs
            so backward can accumulate gradients to their main_grad.
        """
        weight_shape          = preprocess_meta['weight_shape']
        num_chunks_per_weight = preprocess_meta['num_chunks_per_weight']
        fp8_dispatch          = preprocess_meta['fp8_dispatch']
        quantized_tensor_class = preprocess_meta['quantized_tensor_class']
        blockwise_is_2d_scaled = preprocess_meta['blockwise_is_2d_scaled']
        buffer_idx            = preprocess_meta.get('buffer_idx', 0)
        routing_map           = routing_map_expanded

        num_local_home_experts = len(expert_weights)
        ctx.weight_shape = weight_shape
        ctx.num_chunks_per_weight = num_chunks_per_weight
        ctx.num_local_echo_experts = num_local_echo_experts
        ctx.num_local_home_experts = num_local_home_experts
        ctx.buffer_idx = buffer_idx

        # Dispatch the data and scales with hybridep
        ## Initialize the buffer for hybridep
        seq_len = routing_map.shape[0]
        if HybridEPExpertDispatch.expert_dispatch_buffers[buffer_idx] is None:
            seq_len, hidden_dim = weight_tensor.shape
            HybridEPExpertDispatch.expert_dispatch_buffers[buffer_idx] = HybridEPBuffer(
                group=group,
                hidden_dim=hidden_dim,
                max_num_of_tokens_per_rank=seq_len,
                num_local_experts=num_local_echo_experts,
                use_fp8=fp8_dispatch,
                num_sms_dispatch_api=num_sms_dispatch_api,
                num_sms_combine_api=num_sms_combine_api,
            )
        buffer = HybridEPExpertDispatch.expert_dispatch_buffers[buffer_idx]
        non_blocking = num_dispatched_weights is not None
        if fp8_dispatch:
            assert scale_tensor.dtype == torch.float32
            assert weight_tensor.shape[1] // scale_tensor.shape[1] == 128
        # Per-buffer event sync: wait for THIS buffer's last backward combine to
        # finish on GPU before entering dispatch's host busy-poll.
        global _expert_dispatch_buf_events
        buf_event = _expert_dispatch_buf_events[buffer_idx]
        if buf_event is not None:
            buf_event.synchronize()
        if handle is None:
            # Process the dispatch
            (
                dispatched_weight,
                _,
                dispatched_scaling_factor,
                tokens_per_expert,
                handle,
            ) = buffer.dispatch_with_permute(
                hidden=weight_tensor,
                routing_map=routing_map,
                probs=None,
                scaling_factor=scale_tensor,
                pad_multiple=None,
                num_permuted_tokens=num_dispatched_weights * num_chunks_per_weight,
                non_blocking=non_blocking,
            )
        else:
            (
                dispatched_weight,
                _,
                dispatched_scaling_factor,
                tokens_per_expert,
                handle
            ) = buffer.dispatch_with_permute(
                hidden=weight_tensor,
                scaling_factor=scale_tensor,
                handle=handle,
                pad_multiple=None,
                num_permuted_tokens=num_dispatched_weights * num_chunks_per_weight,
            )


        ctx.handle = handle

        # Wrap the data into quantized tensor
        if fp8_dispatch:
            dispatched_raw_weight = dispatched_weight.chunk(num_dispatched_weights, dim=0)
            dispatched_raw_scale = dispatched_scaling_factor.chunk(num_dispatched_weights, dim=0)
            dispatched_weight_list = []
            for i in range(num_dispatched_weights):
                row_weight, col_weight = dispatched_raw_weight[i].chunk(2, dim=0)
                row_scale, col_scale = dispatched_raw_scale[i].chunk(2, dim=0)
                if quantized_tensor_class is MXFP8Tensor:
                    weight_tensor = MXFP8Tensor(
                        weight_shape,
                        torch.bfloat16,
                        rowwise_data=row_weight.reshape(weight_shape),
                        rowwise_scale_inv=row_scale.view(torch.uint8).reshape(weight_shape[0], -1),
                        columnwise_data=col_weight.reshape(weight_shape),
                        columnwise_scale_inv=col_scale.view(torch.uint8).reshape(-1, weight_shape[1]),
                        fp8_dtype=tex.DType.kFloat8E4M3,
                        quantizer=None,
                    )
                elif quantized_tensor_class is Float8BlockwiseQTensor:
                    if blockwise_is_2d_scaled:
                        # Reverse the 1D expansion back to 2D block scales.
                        # For weight (M, K):
                        # Reverse the expanded rowwise and columnwise block scales.
                        # Reverse: ravel -> reshape(M, K//B) or reshape(K, M//B) -> [::B, :]
                        _B = 128
                        M, N = weight_shape  # M=out_features, N=K=in_features
                        row_scale_2d = row_scale.reshape(M, N // _B)[::_B, :].contiguous()  # (M//B, K//B)
                        col_scale_2d = col_scale.reshape(N, M // _B)[::_B, :].contiguous()  # (K//B, M//B)
                        weight_tensor = Float8BlockwiseQTensor(
                            weight_shape,
                            torch.bfloat16,
                            rowwise_data=row_weight.reshape(weight_shape),           # (M, K)
                            rowwise_scale_inv=row_scale_2d,                          # (M//B, K//B)
                            columnwise_data=col_weight.reshape(N, M),                # (K, M) transposed
                            columnwise_scale_inv=col_scale_2d,                       # (K//B, M//B)
                            fp8_dtype=tex.DType.kFloat8E4M3,
                            quantizer=None,
                            is_2D_scaled=True,
                        )
                    else:
                        weight_tensor = Float8BlockwiseQTensor(
                            weight_shape,
                            torch.bfloat16,
                            rowwise_data=row_weight.reshape(weight_shape),
                            rowwise_scale_inv=row_scale.reshape(weight_shape[0], -1),
                            columnwise_data=col_weight.reshape(weight_shape),
                            columnwise_scale_inv=col_scale.reshape(-1, weight_shape[1]),
                            fp8_dtype=tex.DType.kFloat8E4M3,
                            quantizer=None,
                            is_2D_scaled=False,
                        )
                dispatched_weight_list.append(weight_tensor)
        else:
            dispatched_weight_list = [
                weight.reshape(weight_shape)
                for weight in dispatched_weight.chunk(num_dispatched_weights, dim=0)
            ]

        ctx.handle = handle
        ctx.fp8_dispatch = fp8_dispatch
        ctx.blockwise_is_2d_scaled = blockwise_is_2d_scaled
        ctx.num_local_echo_experts = num_local_echo_experts
        ctx.num_local_home_experts = num_local_home_experts
        ctx.expert_weights = expert_weights
        return (*dispatched_weight_list, handle)
    
    @staticmethod
    def backward(ctx, *grad_expert_weights_and_handle):
        '''
        Backward pass of fused dispatch of the HybridEP backend
        '''
        buffer_idx = ctx.buffer_idx
        # Last element is grad for handle (None), rest are grad for expert weights
        grad_expert_weights = grad_expert_weights_and_handle[:-1]
        # TODO: dispatch and accmualte the gradient of the expert weights with fp32
        num_chunks_per_weight = ctx.num_chunks_per_weight
        weight_shape = ctx.weight_shape
        if ctx.fp8_dispatch:
            ctx.handle[-2].hidden_dim //= 2
        # chunk the grad_expert_weights into pieces
        expert_grad_tensor = torch.stack(grad_expert_weights, dim=0).reshape(
            ctx.num_local_echo_experts * num_chunks_per_weight, -1
        )

        MoELayer = _get_moe_layer_cls()
        buffer = HybridEPExpertDispatch.expert_dispatch_buffers[buffer_idx]
        # No event.synchronize() in backward: each FC uses its own buffer (FC1=buf[0],
        # FC2=buf[1]), so back-to-back combines don't collide on host flags.
        # event.synchronize() is only needed in FORWARD (before dispatch) to ensure
        # the corresponding buffer's previous backward combine finished on GPU.
        if hasattr(MoELayer, 'moe_a2a_stream'):
            with torch.cuda.stream(MoELayer.moe_a2a_stream):
                MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                combined_expert_grad, _ = buffer.combine_with_unpermute(
                    hidden=expert_grad_tensor,
                    probs=None,
                    handle=ctx.handle,
                    pad_multiple=None,
                )
                # Record event: THIS buffer's combine is done on moe_a2a_stream.
                global _expert_dispatch_buf_events
                if _expert_dispatch_buf_events[buffer_idx] is None:
                    _expert_dispatch_buf_events[buffer_idx] = torch.cuda.Event()
                _expert_dispatch_buf_events[buffer_idx].record()
                # Record grad_combine_event on moe_a2a_stream right after A2A.
                MoELayer.grad_combine_event.record()
            # No default_stream.wait_stream here: same reason as HybridEPDispatch.backward.
            # grad_combine_event is recorded on moe_a2a_stream and consumed by
            # FlushPendingGradAccum.backward which runs on moe_a2a_stream anyway.
        else:
            combined_expert_grad, _ = buffer.combine_with_unpermute(
                hidden=expert_grad_tensor,
                probs=None,
                handle=ctx.handle,
                pad_multiple=None,
            )
            # Record event even on non-stream path
            if _expert_dispatch_buf_events[buffer_idx] is None:
                _expert_dispatch_buf_events[buffer_idx] = torch.cuda.Event()
            _expert_dispatch_buf_events[buffer_idx].record()
        # Extract grad for each expert
        weight_grad_list = [
            weight_grad.reshape(weight_shape)
            for weight_grad in combined_expert_grad.chunk(ctx.num_local_home_experts, dim=0)
        ]

        # Note: with gradient accumulation fusion, home expert backward sets
        # grad_added_to_main_grad to True, so the DDP backward hook will not
        # accumulate the returned gradient into main_grad.
        # Here we manually accumulate the expert grad from echo experts into main_grad of home experts.
        #
        # Deferred add_() via FlushPendingGradAccum:
        # This backward runs on moe_a2a_stream (the forward was called inside
        # `with torch.cuda.stream(a2a_stream)`).  We do NOT call add_() here to avoid
        # inserting pure-compute kernels between consecutive backward A2A ops on
        # moe_a2a_stream.  Instead we:
        #   1. Record grad_combine_event on moe_a2a_stream (right after the A2A above).
        #   2. Stash (weight, wgrad) pairs in MoELayer.pending_expert_wgrads.
        # FlushPendingGradAccum.backward (inserted in moe_layer.py before dispatch_preprocess)
        # fires after token_dispatch_bwd and dispatch_preprocess_bwd are submitted to
        # moe_a2a_stream, so appending add_() there puts it at the end of the queue —
        # after all backward A2A ops.
        #
        # _get_moe_layer_cls() is called at most once per process (result is cached in
        # _MoELayer).  It must NOT be a bare `from ... import` inside backward because
        # that can acquire Python's import lock in PyTorch's C++ autograd threads and
        # cause pybind11 exception-state inconsistencies (aten::detach RecordFunction
        # warnings).
        dummy_grad_list = []
        if hasattr(MoELayer, 'moe_a2a_stream'):
            # Set Python flag immediately so DDP backward hook does not double-accumulate.
            for weight in ctx.expert_weights:
                assert weight.main_grad is not None, "weight has no main_grad"
                weight.grad_added_to_main_grad = True
                dummy_grad_list.append(None)
            # Stash for deferred add_().
            for weight, wgrad in zip(ctx.expert_weights, weight_grad_list):
                MoELayer.pending_expert_wgrads.append((weight, wgrad))
        else:
            for i, (weight, wgrad) in enumerate(zip(ctx.expert_weights, weight_grad_list)):
                assert weight.main_grad is not None, f"weight {i} has no main_grad"
                weight.main_grad.add_(wgrad)
                weight.grad_added_to_main_grad = True
                dummy_grad_list.append(None)

        return None, None, None, None, None, None, None, None, None, None, *dummy_grad_list

if HAVE_HYBRIDEP:

    def hybrid_ep_expert_dispatch(
        expert_weights,
        routing_map,
        group, 
        num_local_experts,
        num_of_experts,
        num_sms_dispatch_api,
        num_dispatched_weights,
    ):
        """
        """
        pass

    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        '''
        Perform fused dispatch for "permute + dispatch a2a + permute" using the
        HybridEP backend.

        Args:
            x (torch.Tensor):
                Input hidden states to dispatch.
            routing_map (torch.Tensor):
                Map indicating which expert each token is routed to.
            probs (torch.Tensor):
                Routing probabilities for each token-expert pair.
            group (torch.distributed.ProcessGroup):
                Process group used for communication.
            num_local_experts (int):
                Number of local experts.
            num_sms_dispatch_api (int):
                Number of SMs used by the dispatch API.
            num_sms_combine_api (int):
                Number of SMs used by the combine API.
            num_permuted_tokens (int):
                Number of tokens after permute. HybridEP uses this to allocate buffers.
                If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                Alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        '''
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_permuted_tokens,
            pad_multiple,
        )

    def hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple):
        '''
        Perform fused combine operation for unpermute + combine a2a + unpermute
        using the HybridEP backend

        args:
            x (torch.Tensor):
                Input hidden states to combine
            handle (EventHandle):
                Communication handle from dispatch operation
            num_permuted_tokens (int): The number of tokens before unpermute. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                The alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        '''
        return HybridEPCombine.apply(
            x, handle, num_permuted_tokens, pad_multiple
        )

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None