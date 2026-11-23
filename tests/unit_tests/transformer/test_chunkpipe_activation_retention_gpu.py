# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
from tests.unit_tests.test_utilities import Utils


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ChunkPipe activation retention requires CUDA"
)


class _RetentionProbeLayer(BaseTransformerLayer, torch.nn.Module):
    """A deterministic layer with a large saved activation for memory accounting."""

    def __init__(self, config, layer_number=1, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.layer_number = layer_number
        self.up = torch.nn.Linear(config.hidden_size, 8192, bias=False)
        self.down = torch.nn.Linear(8192, config.hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        intermediate = torch.nn.functional.silu(self.up(hidden_states))
        return self.down(intermediate), None


def _config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=8192,
        use_cpu_initialization=True,
        enable_chunkpipe=True,
        chunksize=64,
        chunk_num_per_seq=4,
        max_chunk_size=64,
        micro_batch_size=4,
        keep_activations_chunks=0,
    )


def _make_inputs(device: torch.device) -> list[torch.Tensor]:
    return [
        torch.linspace(-1.0, 1.0, 64 * 4 * 256, device=device).reshape(64, 4, 256)
        for _ in range(4)
    ]


def _fill_deterministic(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            values = torch.linspace(
                -0.01,
                0.01,
                parameter.numel(),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.copy_(values.reshape_as(parameter))


def _run_chunks(block: TransformerBlock, keep_activations_chunks: int):
    config = block.config
    config.keep_activations_chunks = keep_activations_chunks
    block.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = [tensor.requires_grad_() for tensor in _make_inputs(torch.device("cuda"))]
    outputs = []
    for microbatch, hidden_states in enumerate(inputs):
        config.chunkpipe_forward_microbatch = microbatch
        outputs.append(block(hidden_states=hidden_states, attention_mask=None))
    torch.cuda.synchronize()

    allocated_after_forward = torch.cuda.memory_allocated()
    loss = sum(output.float().square().mean() for output in outputs)
    loss.backward()
    torch.cuda.synchronize()

    return SimpleNamespace(
        allocated_after_forward=allocated_after_forward,
        peak_allocated=torch.cuda.max_memory_allocated(),
        outputs=[output.detach().clone() for output in outputs],
        input_grads=[hidden_states.grad.detach().clone() for hidden_states in inputs],
        parameter_grads=[parameter.grad.detach().clone() for parameter in block.parameters()],
    )


def test_chunkpipe_retention_keeps_only_trailing_chunk_activations_on_gpu():
    """The policy must trade recompute for memory without changing the math."""

    Utils.initialize_model_parallel(1, 1)
    try:
        block = TransformerBlock(
            _config(),
            ModuleSpec(module=_RetentionProbeLayer),
            pre_process=True,
            post_process=False,
            post_layer_norm=False,
        ).cuda()
        _fill_deterministic(block)
        block.train()

        no_retention = _run_chunks(block, keep_activations_chunks=0)
        trailing_retention = _run_chunks(block, keep_activations_chunks=1)

        saved_mib = (
            trailing_retention.allocated_after_forward
            - no_retention.allocated_after_forward
        ) / 1024**2
        print(
            "forward_allocated_mib "
            f"no_retention={no_retention.allocated_after_forward / 1024**2:.1f} "
            f"trailing_retention={trailing_retention.allocated_after_forward / 1024**2:.1f} "
            f"saved={saved_mib:.1f}; "
            "peak_allocated_mib "
            f"no_retention={no_retention.peak_allocated / 1024**2:.1f} "
            f"trailing_retention={trailing_retention.peak_allocated / 1024**2:.1f}"
        )

        assert saved_mib > 8.0, (
            "keeping the trailing chunk should retain a real activation; "
            f"observed only {saved_mib:.1f} MiB"
        )

        for expected, actual in zip(no_retention.outputs, trailing_retention.outputs):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        for expected, actual in zip(no_retention.input_grads, trailing_retention.input_grads):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
        for expected, actual in zip(
            no_retention.parameter_grads, trailing_retention.parameter_grads
        ):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
    finally:
        Utils.destroy_model_parallel()
