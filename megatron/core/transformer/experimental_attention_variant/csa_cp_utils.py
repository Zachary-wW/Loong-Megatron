# Copyright (c) 2025, Baidu, Inc. All rights reserved.
# Adapted from NVIDIA Megatron-LM DSv4 THD context-parallel utilities.
"""Utilities for DeepSeek-V4 THD context-parallel path.

This module provides CP row mapping, boundary exchange, compressor-input layout,
and indexer top-k metadata. It is used by AIAK-Training-Omni's DSv4 CSA module.

Key functions:
- exchange_cp_boundary_hidden: P2P boundary exchange between adjacent CP ranks
- prepare_cp_compressor_input: Build fixed-capacity compressor input from boundary + local hidden
- apply_thd_cp_local_rope_fused/unfused: Apply contiguous CP local RoPE
- compute_cp_indexer_topk: Compute global top-k on all-gathered compressed K
"""

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist

# Try importing fused RoPE kernel
try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace

    _FUSED_ROPE_AVAILABLE = True
except ImportError:
    fused_mla_rope_inplace = None
    _FUSED_ROPE_AVAILABLE = False

# Try importing CuTeDSL layout kernels (optional, for performance)
try:
    from megatron.core.transformer.experimental_attention_variant import csa_cp_layout_kernels

    _LAYOUT_KERNELS_AVAILABLE = True
except ImportError:
    csa_cp_layout_kernels = None
    _LAYOUT_KERNELS_AVAILABLE = False

# Try importing fused indexer topk
try:
    from megatron.core.transformer.experimental_attention_variant.dsa_kernels import indexer_topk

    _FUSED_INDEXER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    indexer_topk = None
    _FUSED_INDEXER_AVAILABLE = False

# Try importing unfused RoPE utility
try:
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
except ImportError:
    _apply_rotary_pos_emb_bshd = None


# =============================================================================
# RoPE Wrappers
# =============================================================================


def _thd_cp_position_ids(
    cu_seqlens_padded: torch.Tensor, global_start: int, local_rows: int
) -> torch.Tensor:
    """Map a consecutive CP row interval to positions within packed sequences.

    For contiguous CP partition, each rank holds rows [global_start, global_start + local_rows).
    This function computes the within-sequence position for each row.
    """
    global_rows = torch.arange(
        int(global_start),
        int(global_start) + int(local_rows),
        dtype=cu_seqlens_padded.dtype,
        device=cu_seqlens_padded.device,
    )
    sequence_ids = torch.bucketize(
        global_rows, cu_seqlens_padded[1:], out_int32=True, right=True
    ).clamp_max(cu_seqlens_padded.shape[0] - 2)
    sequence_starts = cu_seqlens_padded[sequence_ids]
    sequence_ends = cu_seqlens_padded[sequence_ids + 1]
    valid_rows = (global_rows >= sequence_starts) & (global_rows < sequence_ends)
    return torch.where(valid_rows, global_rows - sequence_starts, 0)


def apply_thd_cp_local_rope_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply fused non-interleaved RoPE to local THD CP rows.

    Uses MCore's fused_mla_rope_inplace with position_ids computed from
    the contiguous CP partition.
    """
    if not _FUSED_ROPE_AVAILABLE:
        raise RuntimeError(
            "Fused MLA RoPE not available. Install megatron.core.fusions."
        )
    position_ids = _thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0])

    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    if inverse:
        rope_input = rope_input.clone()
    output = fused_mla_rope_inplace(
        rope_input,
        cos,
        sin,
        nope_dim,
        pos_dim,
        cu_seqlens_q=cu_seqlens_padded,
        inverse=inverse,
        remove_interleaving=True,
        position_ids=position_ids,
    )
    if squeezed_batch:
        return output.unsqueeze(1)
    if squeezed_head:
        return output.squeeze(1)
    return output


def apply_thd_cp_local_rope_unfused(
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    config,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply unfused RoPE to a consecutive interval of packed CP rows."""
    if _apply_rotary_pos_emb_bshd is None:
        raise RuntimeError("Cannot import _apply_rotary_pos_emb_bshd from MCore.")
    position_ids = _thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0])
    freqs = torch.index_select(rotary_pos_emb, 0, position_ids.long())

    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    content, rotary = torch.split(rope_input, [nope_dim, pos_dim], dim=-1)
    rotary = _apply_rotary_pos_emb_bshd(
        rotary,
        freqs,
        rotary_interleaved=config.rotary_interleaved,
        mscale=1.0,
        multi_latent_attention=True,
        inverse=inverse,
        mla_output_remove_interleaving=True,
    )
    output = torch.cat((content, rotary), dim=-1)
    if squeezed_batch:
        return output.unsqueeze(1)
    if squeezed_head:
        return output.squeeze(1)
    return output


# =============================================================================
# Boundary Hidden Exchange
# =============================================================================


class _LeftBoundaryExchange(torch.autograd.Function):
    """Exchange fixed left-boundary windows and scatter gradients back to senders.

    In contiguous CP partition, rank r holds global rows [r*L, (r+1)*L).
    Each rank needs the last d_window rows from rank r-1 for sliding window
    attention and compressor overlap.
    """

    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, d_window: int, cp_group: dist.ProcessGroup
    ):
        """Receive fixed left-boundary hidden rows needed by this CP rank."""
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        if tensor.shape[0] < d_window:
            raise RuntimeError(
                "DSv4 CP boundary exchange requires local rows >= D_window: "
                f"local_rows={tensor.shape[0]}, D_window={d_window}."
            )
        boundary = tensor.new_zeros((d_window,) + tuple(tensor.shape[1:]))

        ops = []
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    boundary,
                    dist.get_global_rank(cp_group, cp_rank - 1),
                    cp_group,
                )
            )
        if cp_rank + 1 < cp_size:
            send_tail = tensor[-d_window:].contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_tail,
                    dist.get_global_rank(cp_group, cp_rank + 1),
                    cp_group,
                )
            )
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        """Send boundary gradients back to ranks that own those hidden rows."""
        cp_group = ctx.cp_group
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        d_window = ctx.d_window
        grad_input = grad_boundary.new_zeros(ctx.input_shape)

        ops = []
        if cp_rank > 0:
            send_grad = grad_boundary.contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_grad,
                    dist.get_global_rank(cp_group, cp_rank - 1),
                    cp_group,
                )
            )
        if cp_rank + 1 < cp_size:
            recv_grad = grad_boundary.new_empty(grad_boundary.shape)
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_grad,
                    dist.get_global_rank(cp_group, cp_rank + 1),
                    cp_group,
                )
            )
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if cp_rank + 1 < cp_size:
            grad_input[-d_window:] = recv_grad
        return grad_input, None, None


def exchange_cp_boundary_hidden(
    hidden_states: torch.Tensor,
    compress_ratio: int,
    csa_window_size: int,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Exchange hidden-state rows immediately left of this rank's token block.

    Args:
        hidden_states: Local hidden states, shape (local_rows, ...).
        compress_ratio: Per-layer compression ratio (0, 4, or 128).
        csa_window_size: Sliding window size for CSA.
        cp_group: Context parallel process group.

    Returns:
        boundary_hidden: Shape (d_window, ...), left boundary from rank-1.
    """
    d_comp = 8 if compress_ratio == 4 else compress_ratio if compress_ratio > 1 else 0
    d_window = max(int(csa_window_size), d_comp)
    hidden_flat = hidden_states.view(hidden_states.shape[0], -1)
    boundary_hidden = _LeftBoundaryExchange.apply(hidden_flat, d_window, cp_group)
    return boundary_hidden.reshape((d_window,) + tuple(hidden_states.shape[1:]))


# =============================================================================
# Compressor Input Preparation
# =============================================================================


def _prepare_cp_compressor_input_torch(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback for compressor input compaction.

    Copies visible compression groups from boundary+local hidden into a
    fixed-capacity compact buffer.

    Returns:
        hidden_compact: Shape (c_cap * ratio, hidden_dim), compacted hidden rows.
        compressed_group_ids: Shape (c_cap,), per-group sequence-local compress ID.
    """
    l_local = hidden_local.shape[0]
    d_window = boundary_hidden.shape[0]
    hidden_dim = hidden_local.shape[1] if hidden_local.ndim > 1 else 1
    compact_len = c_cap * ratio
    range_start = global_start
    range_end = global_start + l_local
    first_range_group_start = range_start - d_comp

    device = hidden_local.device
    dtype = hidden_local.dtype

    hidden_compact = torch.zeros(
        (compact_len,) + tuple(hidden_local.shape[1:]), dtype=dtype, device=device
    )
    compressed_group_ids = torch.full(
        (c_cap,), -1, dtype=torch.int32, device=device
    )

    n_seq = cu_seqlens.shape[0] - 1
    running_tokens = 0

    for seq in range(n_seq):
        seq_start = int(cu_seqlens[seq].item())
        seq_end = int(cu_seqlens[seq + 1].item())
        local_seq_end = min(seq_end, range_end)

        if seq_start >= local_seq_end or range_start >= local_seq_end:
            continue

        first_visible_numer = max(0, first_range_group_start - seq_start)
        first_visible_group = (first_visible_numer + ratio - 1) // ratio
        stop_visible_group = (local_seq_end - seq_start) // ratio
        visible_group_count = max(0, stop_visible_group - first_visible_group)
        visible_token_count = visible_group_count * ratio

        for token_idx in range(visible_token_count):
            compact_row = running_tokens + token_idx
            if compact_row >= compact_len:
                break
            comp_id = first_visible_group + token_idx // ratio
            token_in_group = token_idx % ratio
            src_global = seq_start + comp_id * ratio + token_in_group

            if src_global < range_start:
                # From boundary
                src_row = src_global - (range_start - d_window)
                if 0 <= src_row < d_window:
                    hidden_compact[compact_row] = boundary_hidden[src_row]
            else:
                # From local
                src_row = src_global - range_start
                if 0 <= src_row < l_local:
                    hidden_compact[compact_row] = hidden_local[src_row]

            if token_idx % ratio == 0:
                compressed_group_ids[compact_row // ratio] = comp_id

        running_tokens += visible_token_count

    return hidden_compact, compressed_group_ids


def prepare_cp_compressor_input(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    cp_size: int,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-capacity compressor input for this rank's token block.

    Args:
        hidden_local: Local hidden states, shape (l_local, hidden_dim).
        boundary_hidden: Left boundary from rank-1, shape (d_window, hidden_dim).
        cu_seqlens: Cumulative sequence lengths, shape (n_seq+1,).
        cu_seqlens_compressed: Cumulative compressed lengths, shape (n_seq+1,).
        global_start: First global row index for this rank.
        cp_size: Number of CP ranks.
        ratio: Compression ratio (4 or 128).

    Returns:
        hidden_compact: Fixed-capacity compressor input (c_cap * ratio, hidden_dim).
        compressed_group_ids: Per-group sequence-local compress ID (c_cap,).
        seq_to_rank_row: Map from sequence-major compressed rows to rank-major
            all-gather rows (total_compressed_rows,).
    """
    cp_size = int(cp_size)
    ratio = int(ratio)
    d_comp = 8 if ratio == 4 else ratio
    global_start = int(global_start)
    l_local = hidden_local.shape[0]
    group_alignment = 32 // math.gcd(32, ratio)
    c_cap = max(1, (l_local + d_comp) // ratio)
    c_cap = ((c_cap + group_alignment - 1) // group_alignment) * group_alignment

    # Use CuTeDSL kernel if available, else PyTorch fallback
    if _LAYOUT_KERNELS_AVAILABLE and hidden_local.is_cuda:
        hidden_compact, compressed_group_ids = (
            csa_cp_layout_kernels.CompressorInputCompact.apply(
                hidden_local, boundary_hidden, cu_seqlens,
                global_start, ratio, d_comp, c_cap
            )
        )
    else:
        hidden_compact, compressed_group_ids = _prepare_cp_compressor_input_torch(
            hidden_local, boundary_hidden, cu_seqlens,
            global_start, ratio, d_comp, c_cap
        )

    # Build seq_to_rank_row mapping: sequence-major compressed row -> rank-major row
    seq_major_rows = (l_local * cp_size) // ratio
    n_seq = cu_seqlens.shape[0] - 1
    logical_rows = torch.arange(
        seq_major_rows, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    seq_ids = torch.bucketize(
        logical_rows, cu_seqlens_compressed[1:], out_int32=True, right=True
    ).clamp_max(n_seq - 1)
    comp_ids = logical_rows - cu_seqlens_compressed[seq_ids]
    group_last_rows = cu_seqlens[seq_ids] + (comp_ids + 1) * ratio - 1
    owner_ranks = torch.div(
        group_last_rows, l_local, rounding_mode="floor"
    ).clamp_(0, cp_size - 1)

    rank_starts = (
        torch.arange(cp_size, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        * l_local
    )
    first_seq_ids = torch.bucketize(
        rank_starts, cu_seqlens[1:], out_int32=True, right=True
    ).clamp_max(n_seq - 1)
    first_comp_ids = torch.div(
        (rank_starts - d_comp - cu_seqlens[first_seq_ids]).clamp_min_(0) + ratio - 1,
        ratio,
        rounding_mode="floor",
    )
    first_logical_rows = cu_seqlens_compressed[first_seq_ids] + first_comp_ids
    rank_slots = logical_rows - first_logical_rows[owner_ranks]
    rank_rows = owner_ranks * c_cap + rank_slots
    seq_to_rank_row = torch.where(
        logical_rows < cu_seqlens_compressed[-1], rank_rows, -1
    ).to(torch.int32)

    return hidden_compact, compressed_group_ids, seq_to_rank_row


# =============================================================================
# Indexer Top-K
# =============================================================================


@torch.compile
def _build_cp_indexer_layout(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    local_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the indexer's packed local-Q/full-K metadata.

    Each real Q segment is the intersection of a sequence with this rank's
    row interval [global_start, global_start + local_rows). K keeps the
    sequence's full compressed segment.
    """
    global_end = global_start + local_rows
    zero = torch.zeros((1,), dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    local_starts = cu_seqlens_q[:-1].clamp_min(global_start)
    local_ends = cu_seqlens_q[1:].clamp_max(global_end)
    q_lens = (local_ends - local_starts).clamp_min(0)
    q_prefix = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
    padding_q = (global_end - cu_seqlens_q[-1].clamp_min(global_start)).clamp_min(0)
    cu_q_topk = torch.cat((zero, q_prefix, (q_prefix[-1] + padding_q).view(1)))
    cu_k_topk = torch.cat((cu_seqlens_compressed, cu_seqlens_compressed[-1:]))
    q_causal_offsets = torch.cat(
        (torch.where(q_lens > 0, local_starts - cu_seqlens_q[:-1], 0), zero)
    )
    return cu_q_topk, cu_k_topk, q_causal_offsets


def compute_cp_indexer_topk(
    q_indexer_local: torch.Tensor,
    weights_indexer_local: torch.Tensor,
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    ratio: int,
    topk_width: int,
    indexer_softmax_scale: float,
    max_seqlen_q: int,
    use_fused: bool,
) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Compute per-query top-k compressed positions from global compressed K.

    Args:
        q_indexer_local: Local Q indexer embeddings, shape (l_local, n_heads, head_dim).
        weights_indexer_local: Per-head weights, shape (l_local, n_heads).
        k_indexer_seq_major: Global compressed K in sequence-major order.
        cu_seqlens_q: Original sequence boundaries (global).
        cu_seqlens_compressed: Compressed sequence boundaries (global).
        global_start: First global row on this CP rank.
        ratio: Compression ratio.
        topk_width: Number of top-k compressed positions per query.
        indexer_softmax_scale: Softmax temperature scale.
        max_seqlen_q: Maximum sequence length (for causal mask).
        use_fused: Whether to use fused indexer kernel.

    Returns:
        topk: Per-query compressed position ids, shape (l_local, topk_width).
        indexer_layout: Tuple of (cu_q_topk, cu_k_topk, q_causal_offsets).
    """
    topk_width = int(topk_width)
    if topk_width == 0 or k_indexer_seq_major.shape[0] == 0:
        return None, None
    max_seqlen_kv = int(max_seqlen_q) // int(ratio)
    if max_seqlen_kv == 0:
        return None, None

    global_start = int(global_start)
    l_local = q_indexer_local.shape[0]

    cu_q_topk, cu_k_topk, q_causal_offsets = _build_cp_indexer_layout(
        cu_seqlens_q, cu_seqlens_compressed, global_start, l_local
    )

    if not use_fused or not _FUSED_INDEXER_AVAILABLE:
        # Unfused PyTorch fallback
        global_rows = torch.arange(
            global_start, global_start + l_local,
            dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device,
        )
        sequence_ids = torch.bucketize(
            global_rows, cu_seqlens_q[1:], out_int32=True, right=True
        ).clamp_max(cu_seqlens_q.shape[0] - 2)
        positions = global_rows - cu_seqlens_q[sequence_ids]
        visible_k = torch.minimum(
            torch.div(positions + 1, int(ratio), rounding_mode="floor"),
            cu_seqlens_compressed[sequence_ids + 1]
            - cu_seqlens_compressed[sequence_ids],
        ).clamp_min(0)
        valid_q = (global_rows >= cu_seqlens_q[sequence_ids]) & (
            global_rows < cu_seqlens_q[sequence_ids + 1]
        )

        k_rows = torch.arange(
            k_indexer_seq_major.shape[0],
            dtype=cu_seqlens_compressed.dtype,
            device=cu_seqlens_compressed.device,
        )
        k_sequence_ids = torch.bucketize(
            k_rows, cu_seqlens_compressed[1:], out_int32=True, right=True
        ).clamp_max(cu_seqlens_compressed.shape[0] - 2)
        k_positions = k_rows - cu_seqlens_compressed[k_sequence_ids]

        output = torch.full(
            (l_local, topk_width), -1, dtype=torch.int32,
            device=q_indexer_local.device,
        )
        selected_width = min(topk_width, k_indexer_seq_major.shape[0])
        for start in range(0, l_local, 128):
            end = min(start + 128, l_local)
            scores = torch.einsum(
                "rhd,kd->rhk",
                q_indexer_local[start:end].float(),
                k_indexer_seq_major.float(),
            )
            scores = (
                torch.relu(scores)
                * weights_indexer_local[start:end].float().unsqueeze(-1)
            )
            scores = scores.sum(dim=1) * float(indexer_softmax_scale)
            valid_k = (
                (k_sequence_ids.unsqueeze(0) == sequence_ids[start:end].unsqueeze(1))
                & (k_positions.unsqueeze(0) < visible_k[start:end].unsqueeze(1))
                & valid_q[start:end].unsqueeze(1)
            )
            scores = scores.masked_fill(~valid_k, float("-inf"))
            values, rows = torch.topk(scores, selected_width, dim=-1)
            local_rows_out = k_positions[rows].to(torch.int32)
            output[start:end, :selected_width] = torch.where(
                torch.isfinite(values), local_rows_out, -1
            )
        return output, (cu_q_topk, cu_k_topk, q_causal_offsets)

    # Fused path
    topk, _ = indexer_topk(
        q_indexer_local,
        k_indexer_seq_major,
        weights_indexer_local,
        topk=topk_width,
        ratio=ratio,
        indexer_softmax_scale=indexer_softmax_scale,
        cu_seqlens_q=cu_q_topk,
        cu_seqlens_kv=cu_k_topk,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_kv=int(max_seqlen_kv),
        q_causal_offsets=q_causal_offsets,
    )
    return topk, (cu_q_topk, cu_k_topk, q_causal_offsets)


# =============================================================================
# Attention Index Building
# =============================================================================


def build_attention_indices(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    compressed_topk: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
    seq_to_rank_row: Optional[torch.Tensor] = None,
    for_indexer_loss: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build final sparse-attention physical indices.

    Maps logical window positions and compressed top-k positions to physical
    row indices in the concatenated kv_full buffer:
        kv_full = cat(boundary_kv[d_window], kv_local[l_local], compressed_kv[...])

    Uses CuTeDSL kernel if available, otherwise falls back to PyTorch.

    Returns:
        topk_idxs: int32 (l_local, window_size + compressed_width)
        topk_length: int32 (l_local,) or None
        indexer_rank_major: int32 (l_local, compressed_width) or None
    """
    if _LAYOUT_KERNELS_AVAILABLE and cu_seqlens.is_cuda:
        return csa_cp_layout_kernels.build_attention_indices(
            cu_seqlens=cu_seqlens,
            global_start=global_start,
            l_local=l_local,
            d_window=d_window,
            window_size=window_size,
            ratio=ratio,
            compressed_width=compressed_width,
            compressed_topk=compressed_topk,
            cu_seqlens_compressed=cu_seqlens_compressed,
            seq_to_rank_row=seq_to_rank_row,
            for_indexer_loss=for_indexer_loss,
        )

    # PyTorch fallback
    return _build_attention_indices_torch(
        cu_seqlens=cu_seqlens,
        global_start=global_start,
        l_local=l_local,
        d_window=d_window,
        window_size=window_size,
        ratio=ratio,
        compressed_width=compressed_width,
        compressed_topk=compressed_topk,
        cu_seqlens_compressed=cu_seqlens_compressed,
        seq_to_rank_row=seq_to_rank_row,
        for_indexer_loss=for_indexer_loss,
    )


def _build_attention_indices_torch(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    compressed_topk: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
    seq_to_rank_row: Optional[torch.Tensor] = None,
    for_indexer_loss: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Pure PyTorch fallback for building attention indices."""
    device = cu_seqlens.device
    total_width = window_size + compressed_width
    compressed_base = d_window + l_local

    topk_idxs = torch.full(
        (l_local, total_width), -1, dtype=torch.int32, device=device
    )
    topk_length = torch.zeros(l_local, dtype=torch.int32, device=device)
    indexer_rank_major = None
    if for_indexer_loss:
        indexer_rank_major = torch.full(
            (l_local, compressed_width), -1, dtype=torch.int32, device=device
        )

    n_seq = cu_seqlens.shape[0] - 1
    global_rows = torch.arange(
        global_start, global_start + l_local, dtype=torch.int32, device=device
    )
    seq_ids = torch.bucketize(
        global_rows, cu_seqlens[1:], out_int32=True, right=True
    ).clamp_max(n_seq - 1)
    seq_starts = cu_seqlens[seq_ids]
    seq_ends = cu_seqlens[seq_ids + 1]
    valid = (global_rows >= seq_starts) & (global_rows < seq_ends)

    for row in range(l_local):
        if not valid[row]:
            # Padding row: write a dummy index
            if total_width > 0:
                topk_idxs[row, 0] = 0
                topk_length[row] = 1
            continue

        global_q = global_start + row
        seq_start = int(seq_starts[row].item())

        # Window indices
        win_start = max(global_q - window_size + 1, seq_start)
        win_count = global_q - win_start + 1
        write_col = 0

        if not for_indexer_loss:
            # Normal mode: window first, then compressed
            for w in range(min(win_count, window_size)):
                pos = win_start + w
                if pos < global_start:
                    topk_idxs[row, write_col] = pos - (global_start - d_window)
                else:
                    topk_idxs[row, write_col] = d_window + pos - global_start
                write_col += 1

            # Compressed indices
            if compressed_topk is not None and ratio > 1 and compressed_width > 0:
                seq_comp_start = int(cu_seqlens_compressed[seq_ids[row]].item())
                seq_comp_len = int(
                    cu_seqlens_compressed[seq_ids[row] + 1].item()
                ) - seq_comp_start
                for c in range(compressed_width):
                    comp_id = int(compressed_topk[row, c].item())
                    if comp_id >= 0 and comp_id < seq_comp_len:
                        seq_major_id = seq_comp_start + comp_id
                        if seq_major_id < seq_to_rank_row.shape[0]:
                            rank_row = int(seq_to_rank_row[seq_major_id].item())
                            if rank_row >= 0:
                                topk_idxs[row, write_col] = compressed_base + rank_row
                                write_col += 1
            topk_length[row] = write_col
        else:
            # Indexer loss mode: compressed first, window second
            seq_comp_start = int(cu_seqlens_compressed[seq_ids[row]].item())
            seq_comp_len = int(
                cu_seqlens_compressed[seq_ids[row] + 1].item()
            ) - seq_comp_start
            for c in range(compressed_width):
                comp_id = int(compressed_topk[row, c].item())
                if comp_id >= 0 and comp_id < seq_comp_len:
                    seq_major_id = seq_comp_start + comp_id
                    if seq_major_id < seq_to_rank_row.shape[0]:
                        rank_row = int(seq_to_rank_row[seq_major_id].item())
                        if rank_row >= 0:
                            topk_idxs[row, write_col] = compressed_base + rank_row
                            indexer_rank_major[row, c] = rank_row
                write_col += 1
            for w in range(min(win_count, window_size)):
                pos = win_start + w
                if pos < global_start:
                    topk_idxs[row, write_col] = pos - (global_start - d_window)
                else:
                    topk_idxs[row, write_col] = d_window + pos - global_start
                write_col += 1

    if for_indexer_loss:
        return topk_idxs, None, indexer_rank_major
    return topk_idxs, topk_length, None

