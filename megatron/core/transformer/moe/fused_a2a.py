# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

from typing import Optional

from megatron.core.utils import internal_api

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

import torch

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


# HybridEP dispatch/combine kernels use 64-token chunks for their public APIs.
HYBRIDEP_TOKEN_ALIGNMENT = 64


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    num_tokens: int,
    num_local_experts: int,
    num_sms_dispatch_api: Optional[int] = None,
    num_sms_combine_api: Optional[int] = None,
    num_blocks_permute: Optional[int] = None,
    num_blocks_unpermute: Optional[int] = None,
    fp8_dispatch: bool = False,
    num_sms_preprocessing_api: Optional[int] = None,
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
        num_tokens (int):
            Maximum token count of the input tensor.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (Optional[int]):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (Optional[int]):
            Number of SMs used by the combine API.
        num_blocks_permute (Optional[int]):
            Number of blocks used by the permute part.
        num_blocks_unpermute (Optional[int]):
            Number of blocks used by the unpermute part.
        fp8_dispatch (bool):
            Whether to use FP8 communication during the dispatch phase.
        num_sms_preprocessing_api (Optional[int]):
            Number of SMs used by the preprocessing (metadata scan) kernel.
    '''
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    global _hybrid_ep_buffer
    kwargs = {}
    if num_sms_dispatch_api is not None:
        kwargs['num_sms_dispatch_api'] = num_sms_dispatch_api
    if num_sms_combine_api is not None:
        kwargs['num_sms_combine_api'] = num_sms_combine_api
    if num_blocks_permute is not None:
        kwargs['num_blocks_permute'] = num_blocks_permute
    if num_blocks_unpermute is not None:
        kwargs['num_blocks_unpermute'] = num_blocks_unpermute
    if num_sms_preprocessing_api is not None:
        kwargs['num_sms_preprocessing_api'] = num_sms_preprocessing_api
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=num_tokens,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        **kwargs,
    )


def reset_hybrid_ep_buffer():
    '''
    Reset the HybridEP buffer
    '''
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = None


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
        num_sms_dispatch_api=None,
        num_sms_combine_api=None,
        num_blocks_permute=None,
        num_blocks_unpermute=None,
        fused=False,
        num_permuted_tokens=None,
        pad_multiple=None,
        num_sms_preprocessing_api=108,
    ):
        '''
        Forward pass of fused dispatch of the HybridEP backend
        '''
        if fused or num_blocks_permute is not None or num_blocks_unpermute is not None:
            import inspect
            import warnings

            sig = inspect.signature(HybridEPBuffer.dispatch_with_permute)
            if 'fuse_permute_dispatch' not in sig.parameters:
                warnings.warn(
                    "Current DeepEP version does not support fused permute dispatch or "
                    "num_blocks_permute/num_blocks_unpermute. Falling back to unfused "
                    "HybridEP dispatch.",
                    UserWarning,
                    stacklevel=2,
                )
                fused = False
                num_blocks_permute = None
                num_blocks_unpermute = None

        if _hybrid_ep_buffer is None:
            num_tokens, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 dispatch
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                num_tokens,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                num_blocks_permute,
                num_blocks_unpermute,
                fp8_dispatch,
                num_sms_preprocessing_api,
            )
        # If we provide the num_permuted_tokens, we do not need to use sync to
        # wait for the data in pinned memory ready
        non_blocking = num_permuted_tokens is not None
        # Process the dispatch
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
            **({"fuse_permute_dispatch": fused} if fused else {}),
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.fused = fused
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
        '''
        Backward pass of fused dispatch of the HybridEP backend
        '''
        handle = ctx.handle
        combined_hidden, combined_probs = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=grad_x,
            probs=grad_probs,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            **({"fuse_unpermute_combine": ctx.fused} if ctx.fused else {}),
        )
        return (
            combined_hidden,
            None,
            combined_probs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@internal_api
class HybridEPCombine(torch.autograd.Function):
    '''
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None, fused=False):
        '''
        Forward pass of fused combine of the HybridEP backend
        '''
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x,
            handle=handle,
            pad_multiple=pad_multiple,
            **({"fuse_unpermute_combine": fused} if fused else {}),
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        ctx.fused = fused
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        '''
        Backward pass of fused combine of the HybridEP backend
        '''
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
            **({"fuse_permute_dispatch": ctx.fused} if ctx.fused else {}),
        )
        return dispatched_hidden, None, None, None, None


if HAVE_HYBRIDEP:

    @internal_api
    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=None,
        num_sms_combine_api=None,
        num_blocks_permute=None,
        num_blocks_unpermute=None,
        fused=False,
        num_permuted_tokens=None,
        pad_multiple=None,
        num_sms_preprocessing_api=108,
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
            num_sms_dispatch_api (Optional[int]):
                Number of SMs used by the dispatch API.
            num_sms_combine_api (Optional[int]):
                Number of SMs used by the combine API.
            num_blocks_permute (Optional[int]):
                Number of blocks used by the permute part.
            num_blocks_unpermute (Optional[int]):
                Number of blocks used by the unpermute part.
            num_permuted_tokens (int):
                Number of tokens after permute. HybridEP uses this to allocate buffers.
                If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                Alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
            num_sms_preprocessing_api (int):
                Number of SMs used by the preprocessing (metadata scan) kernel.
        '''
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_blocks_permute,
            num_blocks_unpermute,
            fused,
            num_permuted_tokens,
            pad_multiple,
            num_sms_preprocessing_api,
        )

    @internal_api
    def hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple, fused=False):
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
        return HybridEPCombine.apply(x, handle, num_permuted_tokens, pad_multiple, fused)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None


try:
    from transformer_engine.pytorch import ep as te_ep

    HAVE_TE_EP = True
except ImportError:
    HAVE_TE_EP = False


def ensure_nccl_ep_bootstrapped(
    ep_group,
    num_experts,
    max_tokens_per_rank,
    recv_capacity_per_rank,
    hidden_dim,
    num_sms=0,
    zero_copy=False,
):
    """Initialize the process-wide NCCL EP context once. Idempotent.

    Collective on ``ep_group``: TE's ``ep_bootstrap`` issues a barrier and borrows the
    group's NCCL communicator, so every rank must call this with identical arguments
    before the first dispatch. Reuses TransformerEngine's own one-time flag, so repeated
    calls (e.g. once per MoE layer) are no-ops.

    Args:
        ep_group (torch.distributed.ProcessGroup): The expert-parallel process group.
        num_experts (int): Total experts across ``ep_group`` (global, not per-rank).
        max_tokens_per_rank (int): Upper bound on local input tokens per forward. Must be
            even (NCCL EP requires ``num_tokens_per_rank * inner_dim % 4 == 0``).
        recv_capacity_per_rank (int): Per-rank receive-buffer capacity in tokens. Must be
            ``>= max_tokens_per_rank``; runtime overflow hard-traps (no soft drop).
        hidden_dim (int): Token hidden size.
        num_sms (int): SM cap passed to TE as ``max_num_sms`` (0 lets TE/NCCL choose).
    """
    if not HAVE_TE_EP:
        raise RuntimeError(
            "transformer_engine.pytorch.ep is unavailable. The 'ncclep' flex dispatcher backend "
            "requires a TransformerEngine build with NCCL EP support (NVTE_BUILD_WITH_NCCL_EP=1)."
        )
    if te_ep._BOOTSTRAPPED:  # reuse TE's own one-time guard; no parallel state to drift
        return
    te_ep.ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=hidden_dim,
        max_num_sms=num_sms,
        zero_copy=zero_copy,
    )


def nccl_ep_finalize():
    """Tear down the NCCL EP context. Idempotent; safe when never bootstrapped.

    Releases the borrowed NCCL communicator and must run before the process group is
    destroyed.
    """
    if HAVE_TE_EP:
        te_ep.ep_finalize()


if HAVE_TE_EP:

    def alloc_ep_symm_buffer(shape, dtype, ep_group):
        """Allocate one persistent NCCL symm-mem buffer (per-buffer collective rendezvous). mcore's
        zero-copy buffers are all persistent and non-pool; the symm mem-pool is used only by TE for
        the per-call recv buffers it recycles."""
        return te_ep.symm_mem_alloc(shape, dtype, ep_group)

    def new_nccl_ep_buffer(
        top_k,
        max_tokens_per_rank,
        recv_capacity_per_rank,
        hidden_dim,
        num_local_experts,
        alignment=0,
    ):
        """Build a fresh TE EpBuffer for one dispatch/combine pair.

        The buffer owns handle_mem (the routing table dispatch writes and combine reads); a new one
        is built per dispatch and dropped after combine. Payload symm buffers are not owned here —
        they are caller-supplied to dispatch/combine or allocated on the fly by TE.
        """
        return te_ep.EpBuffer(
            top_k=top_k,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=hidden_dim,
            num_local_experts=num_local_experts,
            alignment=alignment,
        )

    def nccl_ep_dispatch(
        buffer, tokens, topk_idx, topk_weights, recv_tokens=None, recv_topk_weights=None
    ):
        """Autograd-aware prepare + dispatch via TransformerEngine NCCL EP.

        Args:
            buffer (te_ep.EpBuffer): The TE EP buffer for this dispatch.
            tokens (torch.Tensor): Local input tokens ``[num_local_tokens, hidden]``
                (leading dims flattened by TE), ``payload_dtype``.
            topk_idx (torch.Tensor): ``int64`` ``[num_local_tokens, top_k]`` global expert
                ids per token.
            topk_weights (torch.Tensor): ``float32`` ``[num_local_tokens, top_k]`` weights.
            recv_tokens, recv_topk_weights (torch.Tensor, optional): caller-owned symm dispatch
                recv buffers (fp8 zero-copy). Left None, TE allocates them (bf16 zero-copy: symm
                mem-pool; normal: plain).

        Returns:
            tuple: ``(recv_tokens, tokens_per_expert, dispatched_probs)``:
              * ``recv_tokens``: packed received tokens ``[recv_capacity_per_rank, hidden]``,
                grouped by local expert (no separate compaction step).
              * ``tokens_per_expert``: ``int32`` ``[num_local_experts]`` device tensor of
                received counts per local expert (feeds grouped GEMM as group sizes;
                alignment-padded, == actual when ``alignment=0``).
              * ``dispatched_probs``: ``float32`` ``[recv_capacity_per_rank]`` per-slot
                weights; apply them in the expert MLP (combine is called unweighted).

            ``tokens_per_expert`` is non-differentiable.
        """
        recv_tokens, dispatched_probs, tokens_per_expert = te_ep.ep_dispatch(
            buffer,
            tokens,
            topk_idx,
            topk_weights,
            recv_tokens=recv_tokens,
            recv_topk_weights=recv_topk_weights,
        )
        return recv_tokens, tokens_per_expert, dispatched_probs

    def nccl_ep_combine(buffer, expert_out, num_local_tokens=None, grad_out=None):
        """Autograd-aware combine via TransformerEngine NCCL EP (no scatter step).

        Args:
            buffer (te_ep.EpBuffer): The TE EP buffer for this combine.
            expert_out (torch.Tensor): Expert outputs ``[recv_capacity_per_rank, hidden]``,
                already weighted.
            num_local_tokens (int): Rows of the result (local token count for this
                forward). When None, TE uses ``buffer.max_tokens_per_rank``.
            grad_out (torch.Tensor, optional): caller-owned symm buffer the backward scatters the
                expert_out grad into (zero-copy). Left None, TE allocates it (bf16: symm mem-pool;
                normal: plain).

        Returns:
            torch.Tensor: ``[num_local_tokens, hidden]`` combined output, in local token
            order.
        """
        return te_ep.ep_combine(
            buffer, expert_out, num_local_tokens=num_local_tokens, grad_out=grad_out
        )

else:
    alloc_ep_symm_buffer = None
    new_nccl_ep_buffer = None
    nccl_ep_dispatch = None
    nccl_ep_combine = None


# ============================================================================
# ECHO hotspot expert cloning (migrated from AIAK, M-31): expert-weight
# dispatch over the HybridEP backend. Sends home expert weight chunks to the
# ranks whose echo experts replicate them, and accumulates the echo-side
# weight gradients back into the home experts' main_grad in backward.
# ============================================================================

_MoELayer = None


def _get_moe_layer_cls():
    """Return the lazily imported MoELayer class.

    Lazy on purpose: moe_layer imports fused_a2a, so a module-level import
    would be circular. Resolved at most once per process and cached; never
    import inside an autograd backward (Python import lock in C++ autograd
    threads triggers pybind11 exception-state inconsistencies).
    """
    global _MoELayer
    if _MoELayer is None:
        from megatron.core.transformer.moe import moe_layer as _ml  # noqa: PLC0415

        _MoELayer = _ml.MoELayer
    return _MoELayer


try:
    from transformer_engine.pytorch.tensor import QuantizedTensor

    HAVE_TE_QUANTIZED_TENSOR = True
except ImportError:  # pragma: no cover - TE not installed
    QuantizedTensor = None
    HAVE_TE_QUANTIZED_TENSOR = False

try:
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
    import transformer_engine_torch as tex
except ImportError:  # pragma: no cover - TE not installed
    Float8BlockwiseQTensor = None
    MXFP8Tensor = None
    tex = None

if HAVE_HYBRIDEP:

    _expert_dispatch_buf_events = [None, None]

    class HybridEPExpertDispatch(torch.autograd.Function):
        """Fused expert-WEIGHT dispatch over the HybridEP backend (ECHO).

        Two class-level buffers: [0]=FC1, [1]=FC2. Separate buffers avoid
        host-flag collision when backward issues two consecutive
        combine_with_unpermute calls.
        """

        expert_dispatch_buffers = [None, None]

        @staticmethod
        def preprocess(routing_map, weight_chunk_size, *expert_weights):
            """Stack home expert weights into chunked rows ready for dispatch.

            Runs on the default stream (overlapping preceding computation)
            rather than competing with dispatch communication for HBM
            bandwidth. Returns a dict with keys: weight_tensor, scale_tensor,
            routing_map_expanded, meta.
            """
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

            # Chunk the weight so hybridep dispatches a small piece each time
            weight_tensor = torch.stack(weight_list, dim=0).reshape(num_local_home_experts, -1)
            num_chunks_per_weight = weight_tensor.shape[1] // weight_chunk_size
            weight_tensor = weight_tensor.reshape(
                num_local_home_experts * num_chunks_per_weight, weight_chunk_size
            )

            if fp8_dispatch:
                scale_tensor = torch.stack(scale_list, dim=0)
                scale_tensor = scale_tensor.reshape(
                    num_local_home_experts * num_chunks_per_weight, -1
                )
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
            """Dispatch chunked expert weights via HybridEP A2A.

            weight_tensor / routing_map_expanded / scale_tensor: outputs of
                preprocess(), explicit tensor args so autograd tracks placement.
            preprocess_meta: non-tensor metadata dict from preprocess().
            expert_weights: original home weights, kept as autograd inputs so
                backward can accumulate gradients into their main_grad.
            """
            weight_shape = preprocess_meta['weight_shape']
            num_chunks_per_weight = preprocess_meta['num_chunks_per_weight']
            fp8_dispatch = preprocess_meta['fp8_dispatch']
            quantized_tensor_class = preprocess_meta['quantized_tensor_class']
            blockwise_is_2d_scaled = preprocess_meta['blockwise_is_2d_scaled']
            buffer_idx = preprocess_meta.get('buffer_idx', 0)
            routing_map = routing_map_expanded

            num_local_home_experts = len(expert_weights)
            ctx.weight_shape = weight_shape
            ctx.num_chunks_per_weight = num_chunks_per_weight
            ctx.num_local_echo_experts = num_local_echo_experts
            ctx.num_local_home_experts = num_local_home_experts
            ctx.buffer_idx = buffer_idx

            # Allocate the per-FC expert-dispatch buffer on first use.
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
            # Wait for THIS buffer's last backward combine to finish on GPU
            # before entering dispatch's host busy-poll.
            buf_event = _expert_dispatch_buf_events[buffer_idx]
            if buf_event is not None:
                buf_event.synchronize()
            if handle is None:
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
                    handle,
                ) = buffer.dispatch_with_permute(
                    hidden=weight_tensor,
                    scaling_factor=scale_tensor,
                    handle=handle,
                    pad_multiple=None,
                    num_permuted_tokens=num_dispatched_weights * num_chunks_per_weight,
                )

            ctx.handle = handle

            # Wrap the dispatched rows back into quantized tensors
            if fp8_dispatch:
                dispatched_raw_weight = dispatched_weight.chunk(num_dispatched_weights, dim=0)
                dispatched_raw_scale = dispatched_scaling_factor.chunk(
                    num_dispatched_weights, dim=0
                )
                dispatched_weight_list = []
                for i in range(num_dispatched_weights):
                    row_weight, col_weight = dispatched_raw_weight[i].chunk(2, dim=0)
                    row_scale, col_scale = dispatched_raw_scale[i].chunk(2, dim=0)
                    if quantized_tensor_class is MXFP8Tensor:
                        quant_weight = MXFP8Tensor(
                            weight_shape,
                            torch.bfloat16,
                            rowwise_data=row_weight.reshape(weight_shape),
                            rowwise_scale_inv=row_scale.view(torch.uint8).reshape(
                                weight_shape[0], -1
                            ),
                            columnwise_data=col_weight.reshape(weight_shape),
                            columnwise_scale_inv=col_scale.view(torch.uint8).reshape(
                                -1, weight_shape[1]
                            ),
                            fp8_dtype=tex.DType.kFloat8E4M3,
                            quantizer=None,
                        )
                    elif quantized_tensor_class is Float8BlockwiseQTensor:
                        if blockwise_is_2d_scaled:
                            # Reverse the 1D expansion back to 2D block scales.
                            _B = 128
                            M, N = weight_shape  # M=out_features, N=K=in_features
                            row_scale_2d = row_scale.reshape(M, N // _B)[::_B, :].contiguous()
                            col_scale_2d = col_scale.reshape(N, M // _B)[::_B, :].contiguous()
                            quant_weight = Float8BlockwiseQTensor(
                                weight_shape,
                                torch.bfloat16,
                                rowwise_data=row_weight.reshape(weight_shape),
                                rowwise_scale_inv=row_scale_2d,
                                columnwise_data=col_weight.reshape(N, M),
                                columnwise_scale_inv=col_scale_2d,
                                fp8_dtype=tex.DType.kFloat8E4M3,
                                quantizer=None,
                                is_2D_scaled=True,
                            )
                        else:
                            quant_weight = Float8BlockwiseQTensor(
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
                    else:
                        raise RuntimeError(
                            f"Unsupported quantized tensor class: {quantized_tensor_class}"
                        )
                    dispatched_weight_list.append(quant_weight)
            else:
                dispatched_weight_list = [
                    weight.reshape(weight_shape)
                    for weight in dispatched_weight.chunk(num_dispatched_weights, dim=0)
                ]

            ctx.fp8_dispatch = fp8_dispatch
            ctx.blockwise_is_2d_scaled = blockwise_is_2d_scaled
            ctx.expert_weights = expert_weights
            return (*dispatched_weight_list, handle)

        @staticmethod
        def backward(ctx, *grad_expert_weights_and_handle):
            """Combine expert weight gradients and add them to home main_grad."""
            buffer_idx = ctx.buffer_idx
            # Last element is grad for handle (None), rest are grads for weights
            grad_expert_weights = grad_expert_weights_and_handle[:-1]
            num_chunks_per_weight = ctx.num_chunks_per_weight
            weight_shape = ctx.weight_shape
            if ctx.fp8_dispatch:
                ctx.handle[-2].hidden_dim //= 2
            expert_grad_tensor = torch.stack(grad_expert_weights, dim=0).reshape(
                ctx.num_local_echo_experts * num_chunks_per_weight, -1
            )

            MoELayer = _get_moe_layer_cls()
            buffer = HybridEPExpertDispatch.expert_dispatch_buffers[buffer_idx]
            # No event.synchronize() in backward: each FC uses its own buffer
            # (FC1=buf[0], FC2=buf[1]), so back-to-back combines don't collide
            # on host flags. event.synchronize() is only needed in FORWARD.
            if hasattr(MoELayer, 'moe_a2a_stream'):
                with torch.cuda.stream(MoELayer.moe_a2a_stream):
                    MoELayer.moe_a2a_stream.wait_stream(torch.cuda.default_stream())
                    combined_expert_grad, _ = buffer.combine_with_unpermute(
                        hidden=expert_grad_tensor,
                        probs=None,
                        handle=ctx.handle,
                        pad_multiple=None,
                    )
                    global _expert_dispatch_buf_events
                    if _expert_dispatch_buf_events[buffer_idx] is None:
                        _expert_dispatch_buf_events[buffer_idx] = torch.cuda.Event()
                    _expert_dispatch_buf_events[buffer_idx].record()
            else:
                combined_expert_grad, _ = buffer.combine_with_unpermute(
                    hidden=expert_grad_tensor,
                    probs=None,
                    handle=ctx.handle,
                    pad_multiple=None,
                )
                if _expert_dispatch_buf_events[buffer_idx] is None:
                    _expert_dispatch_buf_events[buffer_idx] = torch.cuda.Event()
                _expert_dispatch_buf_events[buffer_idx].record()
            weight_grad_list = [
                weight_grad.reshape(weight_shape)
                for weight_grad in combined_expert_grad.chunk(ctx.num_local_home_experts, dim=0)
            ]

            # With gradient accumulation fusion, home expert backward sets
            # grad_added_to_main_grad=True so the DDP hook skips accumulation;
            # echo-side gradients are accumulated into the home experts'
            # main_grad here (the echo experts are weight clones, not
            # separately-registered DDP parameters).
            dummy_grad_list = []
            if hasattr(MoELayer, 'moe_a2a_stream'):
                # Stream path defers the add_() to FlushPendingGradAccum
                # (overhead mode only); set the flag now so the DDP backward
                # hook does not double-accumulate.
                for weight in ctx.expert_weights:
                    assert weight.main_grad is not None, "weight has no main_grad"
                    weight.grad_added_to_main_grad = True
                    dummy_grad_list.append(None)
                for weight, wgrad in zip(ctx.expert_weights, weight_grad_list):
                    MoELayer.pending_expert_wgrads.append((weight, wgrad))
            else:
                for i, (weight, wgrad) in enumerate(zip(ctx.expert_weights, weight_grad_list)):
                    assert weight.main_grad is not None, f"weight {i} has no main_grad"
                    weight.main_grad.add_(wgrad)
                    weight.grad_added_to_main_grad = True
                    dummy_grad_list.append(None)

            return None, None, None, None, None, None, None, None, None, None, *dummy_grad_list

else:
    _expert_dispatch_buf_events = [None, None]
