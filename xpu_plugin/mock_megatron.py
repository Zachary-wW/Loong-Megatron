"""
Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
This module provides megatron-core-xpu plugin.
"""

import sys
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import LayerNorm as TorchLayerNorm

try:
    import torch_xmlir
except Exception:
    torch_xmlir = None

try:
    import xspeedgate_ops
except Exception:
    xspeedgate_ops = None


class MockMixedFusedLayerNorm(TorchLayerNorm):
    """
    MockMixedFusedLayerNorm
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        no_persist_layer_norm=True,
        sequence_parallel=False,
        apply_layernorm_1p=False,
    ):
        """
        init
        """
        super().__init__(normalized_shape, eps=eps)
        self.sequence_parallel = sequence_parallel
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)
        setattr(self.bias, "sequence_parallel", self.sequence_parallel)


class MockWeightedSwiGLUFunction(torch.autograd.Function):
    """Weighted SwiGLU with bias support for XPU.

    Computes: swiglu(input + bias) * weights

    Optimizations over the original WeightedSwiGLUFunction:
    1. Uses xmlir's fused swiglu_forward/swiglu_backward custom ops
    2. Saves swiglu output in forward to avoid recomputation in backward
    3. Supports bias != None (original raises NotImplementedError)
    """

    @staticmethod
    def forward(ctx, input, bias, weights, fp8_input_store):
        """
        forward
        """
        assert not fp8_input_store, "fp8_input_store is not supported on XPU"

        if bias is not None:
            input = input + bias

        # Ensure xmlir custom ops are registered
        from torch_xmlir.nn.swiglu import SwiGLUFunction  # noqa: F401

        # Use xmlir's fused swiglu_forward directly (no autograd overhead)
        output_shape = list(input.shape)
        output_shape[-1] = input.shape[-1] // 2
        output = input.new_empty(output_shape)

        if input.numel() > 0:
            torch.ops.custom_ops.swiglu_forward(input, -1, True, out=output)

        ctx.save_for_backward(input, weights, output)
        ctx.has_bias = bias is not None
        ctx.ori_input_dtype = input.dtype

        return (output * weights).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        backward
        """
        input, weights, output = ctx.saved_tensors

        # d_weights: sum(swiglu_output * grad_output) over hidden dim
        # Compute in weights' precision for accuracy (matches original weighted_swiglu_back)
        grad_weights = torch.sum(
            output * grad_output.to(output.dtype), dim=-1, keepdim=True
        ).to(weights.dtype)

        # d_input via swiglu backward
        # Chain rule: d_input = swiglu_backward(grad_output * weights, input)
        # Compute grad_swiglu in higher precision, then cast for swiglu_backward
        grad_swiglu = (grad_output.to(weights.dtype) * weights).to(input.dtype)

        d_input = input.new_empty(input.shape)
        if input.numel() > 0:
            torch.ops.custom_ops.swiglu_backward(
                input, grad_swiglu, -1, True, dx=d_input
            )

        # d_bias = d_input (same as bias_swiglu_impl)
        if ctx.has_bias:
            return d_input, d_input, grad_weights, None
        else:
            return d_input, None, grad_weights, None


def mock_weighted_bias_swiglu_impl(
    input,
    bias,
    weights,
    fp8_input_store=False,
    clamp_value=None,
    gate_clamp_scale: Optional[float] = None,
    linear_clamp_scale: Optional[float] = None,
):
    """Token-wise-weighted bias swiglu fusion for XPU.

    Computes: swiglu(input + bias) * weights

    Uses xmlir's optimized SwiGLU custom ops for XPU performance.
    Unlike the original implementation, this also supports bias != None.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    assert not fp8_input_store
    input = input.view(-1, ori_shape[-1])

    # Reshape weights to be broadcastable with 2D input
    # weights may be [s, b, 1] when input is 3D, need [s*b, 1]
    if weights.dim() > 2:
        weights = weights.view(-1, weights.shape[-1])

    if clamp_value is not None or gate_clamp_scale is not None:
        from megatron.core.fusions.fused_bias_swiglu import (
            clamped_weighted_swiglu,
            weighted_situ_glu,
        )

        assert gate_clamp_scale is not None or linear_clamp_scale is None
        assert gate_clamp_scale is None or clamp_value is None
        fused_input = input if bias is None else input + bias
        if gate_clamp_scale is not None:
            output = weighted_situ_glu(
                fused_input, weights, gate_clamp_scale, linear_clamp_scale
            )
        else:
            output = clamped_weighted_swiglu(fused_input, weights, clamp_value)
        return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)

    output = MockWeightedSwiGLUFunction.apply(input, bias, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)

class RotaryPositionalEmbeddingWithFreqFunction(torch.autograd.Function):
    """
    RotaryPositionalEmbeddingWithFreqFunction
    """

    @staticmethod
    # @nvtx.annotate("RotaryPositionalEmbeddingFunction forward", color="skyblue")
    def forward(ctx, t, freqs):
        """
        forward
        """
        output = t.new_empty(t.shape)
        torch.ops.custom_ops.rotary_pos_emb(t, freqs, out=output)
        ctx.freqs = freqs
        return output

    @staticmethod
    # @nvtx.annotate("RotaryPositionalEmbeddingFunction backward", color="violet")
    def backward(ctx, grad_output):
        """
        backward
        """
        freqs_ = ctx.freqs
        grad_t = grad_output.new_empty(grad_output.shape)
        torch.ops.custom_ops.rotary_pos_emb_backward(grad_output, freqs_, out=grad_t)
        return grad_t, None


def mock_apply_rotary_pos_emb_bshd(
    t: Tensor, freqs: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    from megatron.core.models.common.embeddings.rotary_pos_embedding import _rotate_half

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    if torch_xmlir is not None:
        t = RotaryPositionalEmbeddingWithFreqFunction.apply(t, freqs)
    else:
        t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


def read_metadata_dist_without_DFS(tracker_filename):
    """
    read_metadata_dist_without_DFS
    """
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    from megatron.training import print_rank_0

    iteration = 0
    release = False
    if torch.distributed.get_rank() == 0:
        with open(tracker_filename, "r") as f:
            metastring = f.read().strip()
            try:
                iteration = int(metastring)
            except ValueError:
                release = metastring == "release"
                if not release:
                    print_rank_0(
                        "ERROR: Invalid metadata file {}. Exiting".format(tracker_filename)
                    )
                    sys.exit()
        assert iteration > 0 or release, "error parsing metadata file {}".format(tracker_filename)

        iters_cuda = torch.cuda.LongTensor([iteration])
        torch.distributed.broadcast(iters_cuda, 0)
        release_cuda = torch.cuda.ByteTensor([release])
        torch.distributed.broadcast(release_cuda, 0)
    else:
        iters_cuda = torch.cuda.LongTensor([0])
        torch.distributed.broadcast(iters_cuda, 0)
        release_cuda = torch.cuda.ByteTensor([False])
        torch.distributed.broadcast(release_cuda, 0)
        iteration = iters_cuda[0].item()
        release = bool(release_cuda[0].item())
    return iteration, release


def mock_is_kernel_available(self, mask, b, np, sq, sk):
    """
    mock_is_kernel_available
    """
    if (
        self.scaled_masked_softmax_fusion  # user want to fuse
        and self.input_in_float16  # input must be fp16
        and sk <= 16384  # sk must be <= 16384
    ):
        return True
    return False


def mock_gather_along_first_dim(
    input_, group=None, output_split_sizes=None, use_global_buffer=False
):
    """
    mock_gather_along_first_dim
    """
    from megatron.core.parallel_state import (
        get_global_memory_buffer,
        get_tensor_model_parallel_group,
    )

    if group is None:
        group = get_tensor_model_parallel_group()
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_

    dim_size = list(input_.size())

    if input_.numel() == 0:
        dummy_input = torch.zeros(1, *input_.shape[1:], dtype=input_.dtype, device=input_.device)

        adjusted_split_sizes = [s if s > 0 else 1 for s in output_split_sizes]

        adjusted_dim_size = [sum(adjusted_split_sizes)] + dim_size[1:]
        if use_global_buffer:
            adjusted_output = get_global_memory_buffer().get_tensor(
                adjusted_dim_size, input_.dtype, "mpu"
            )
        else:
            adjusted_output = torch.empty(
                adjusted_dim_size, dtype=input_.dtype, device=input_.device
            )

        output_tensor_list = list(torch.split(adjusted_output, adjusted_split_sizes, dim=0))

        torch.distributed.all_gather(output_tensor_list, dummy_input.contiguous(), group=group)

        final_dim_size = [sum(output_split_sizes)] + dim_size[1:]
        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(final_dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(final_dim_size, dtype=input_.dtype, device=input_.device)

        start_idx = 0
        for i, s in enumerate(output_split_sizes):
            if s > 0:
                output[start_idx : start_idx + s] = output_tensor_list[i][:s]
                start_idx += s
        return output

    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size
        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed._all_gather_base(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_.contiguous(), group=group)

    return output


def mock_reduce_scatter_along_first_dim(
    input_, group=None, input_split_sizes=None, use_global_buffer=False
):
    """
    mock_reduce_scatter_along_first_dim
    """
    from megatron.core.parallel_state import (
        get_global_memory_buffer,
        get_tensor_model_parallel_group,
    )

    """Reduce-scatter the input tensor across model parallel group.

    Args:
        input_ (torch.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
    """
    if group is None:
        group = get_tensor_model_parallel_group()
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)
    else:
        rank = torch.distributed.get_rank(group)
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))
        if sum(input_split_sizes) == 0:
            if use_global_buffer:
                output = get_global_memory_buffer().get_tensor(
                    input_tensor_list[rank].shape, input_.dtype, "mpu"
                )
            else:
                output = torch.empty_like(input_tensor_list[rank])
            torch.distributed.reduce_scatter(output, input_tensor_list, group=group)
        else:
            original_chunk_sizes = [t.size(0) for t in input_tensor_list]

            world_size = torch.distributed.get_world_size(group=group)
            local_max = max(original_chunk_sizes) if original_chunk_sizes else 0
            # global_max = torch.tensor([local_max], dtype=torch.long, device=input_.device)
            # torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=group)
            max_chunk_size = local_max  # global_max.item()

            padded_tensor_list = []
            for i, tensor in enumerate(input_tensor_list):
                if tensor.size(0) < max_chunk_size:
                    pad_size = max_chunk_size - tensor.size(0)
                    pad_shape = (pad_size,) + tensor.shape[1:]
                    pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                    padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
                else:
                    padded_tensor = tensor
                padded_tensor_list.append(padded_tensor)

            if use_global_buffer:
                temp_output = get_global_memory_buffer().get_tensor(
                    (max_chunk_size,) + input_.shape[1:], input_.dtype, "mpu"
                )
            else:
                temp_output = torch.empty(
                    (max_chunk_size,) + input_.shape[1:], dtype=input_.dtype, device=input_.device
                )
            torch.distributed.reduce_scatter(temp_output, padded_tensor_list, group=group)

            # actual_output_size = original_chunk_sizes[rank] if rank < len(original_chunk_sizes) else 0
            output = temp_output[: original_chunk_sizes[rank]]
            output = output.contiguous()
    return output


def mock_thd_get_partitioned_indices(cu_seqlens, total_tokens, world_size, rank):
    """
    mock_thd_get_partitioned_indices
    """
    assert xspeedgate_ops is not None
    return torch.ops.xspeedgate_ops.thd_get_partitioned_indices(cu_seqlens, total_tokens, world_size, rank)
