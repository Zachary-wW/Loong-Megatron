# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2 or not torch.cuda.is_available(),
    reason="ChunkPipe MTP retention probe requires at least two CUDA ranks",
)


def _config() -> TransformerConfig:
    return TransformerConfig(
        mtp_num_layers=1,
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=256,
        use_cpu_initialization=True,
        enable_chunkpipe=True,
        chunksize=16,
        chunk_num_per_seq=4,
        max_chunk_size=16,
        micro_batch_size=2,
        keep_activations_chunks=1,
        mtp_loss_scaling_factor=0.1,
        chunkpipe_forward=True,
    )


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


def test_chunkpipe_mtp_retention_is_rank_invariant():
    """Both ranks must retain the same trailing MTP chunk and finish a real forward."""

    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _config()
        spec = get_gpt_layer_local_spec()
        mtp_spec = get_gpt_mtp_block_spec(
            config=config,
            spec=spec,
            use_transformer_engine=False,
        )
        block = MultiTokenPredictionBlock(config=config, spec=mtp_spec).cuda()
        block.train()
        _fill_deterministic(block)
        for module in block.modules():
            if hasattr(module, "init_chunk_key_value_cache"):
                module.init_chunk_key_value_cache()

        checkpointed_microbatches = []
        original_checkpointed_forward = block.layers[0]._checkpointed_forward

        def recording_checkpointed_forward(**kwargs):
            checkpointed_microbatches.append(config.chunkpipe_forward_microbatch)
            return original_checkpointed_forward(**kwargs)

        block.layers[0]._checkpointed_forward = recording_checkpointed_forward

        output_checksums = []
        for microbatch in range(4):
            config.chunkpipe_forward = True
            config.chunkpipe_forward_microbatch = microbatch
            hidden_states = torch.linspace(
                -1.0, 1.0, 16 * 2 * 64, device="cuda"
            ).reshape(16, 2, 64)
            decoder_input = torch.linspace(
                1.0, -1.0, 16 * 2 * 64, device="cuda"
            ).reshape(16, 2, 64)
            attention_mask = torch.ones(
                1, 1, 16, 16, dtype=torch.bool, device="cuda"
            )
            output = block(
                input_ids=None,
                position_ids=None,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
            )[0]
            assert torch.isfinite(output).all()
            output_checksums.append(output.float().sum())

        expected_microbatches = [0, 1, 2]
        assert checkpointed_microbatches == expected_microbatches

        rank = torch.distributed.get_rank()
        calls_tensor = torch.tensor(
            checkpointed_microbatches, dtype=torch.int64, device="cuda"
        )
        gathered_calls = [torch.empty_like(calls_tensor) for _ in range(2)]
        torch.distributed.all_gather(gathered_calls, calls_tensor)
        for peer_calls in gathered_calls[1:]:
            torch.testing.assert_close(peer_calls, gathered_calls[0])

        checksum_tensor = torch.stack(output_checksums).float()
        gathered_checksums = [torch.empty_like(checksum_tensor) for _ in range(2)]
        torch.distributed.all_gather(gathered_checksums, checksum_tensor)
        for peer_checksums in gathered_checksums[1:]:
            torch.testing.assert_close(peer_checksums, gathered_checksums[0], rtol=0.0, atol=1e-5)

        if rank == 0:
            print(
                "mtp_retention_multirank=pass "
                f"checkpointed_microbatches={checkpointed_microbatches}"
            )
    finally:
        Utils.destroy_model_parallel()
