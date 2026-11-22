# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext

import pytest
import torch

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.test_schedule_layer_1f1b import (
    run_transformer_layer_a2a_overlap_with_capture,
    run_transformer_layer_ref_with_capture,
)
from tests.unit_tests.a2a_overlap.utils import (
    build_data,
    compare_captures,
    deterministic_mode,
    get_test_config,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
@pytest.mark.parametrize(
    ("selector", "shared_expert"),
    [
        ("a2a_overlap_attn", False),
        ("a2a_overlap_post_attn", False),
        ("a2a_overlap_mlp", False),
        ("a2a_overlap_mlp", True),
    ],
)
def test_a2a_recompute_selector_matches_reference(selector, shared_expert):
    """Each fine-grained A2A recompute selector must preserve output and gradients."""

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=4,
    )
    try:
        base_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
            "overlap_moe_expert_parallel_comm": True,
        }
        if shared_expert:
            base_kwargs.update(
                moe_shared_expert_intermediate_size=512,
                moe_shared_expert_overlap=False,
            )

        overlap_kwargs = dict(base_kwargs)
        overlap_kwargs.update(
            recompute_granularity="selective",
            recompute_modules=[selector],
        )
        overlap_config = get_test_config(extra_kwargs=overlap_kwargs)

        reference_kwargs = dict(base_kwargs)
        reference_kwargs["overlap_moe_expert_parallel_comm"] = False
        reference_config = get_test_config(extra_kwargs=reference_kwargs)

        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=reference_config, use_transformer_engine=True
            )
            reference_model = GPTModel(
                config=reference_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            reference_model.cuda()
            reference_params = reset_model(reference_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = (
                get_fp8_context(reference_config, 0)
                if reference_config.fp8
                else nullcontext()
            )
            with fp8_context:
                reference_capture = run_transformer_layer_ref_with_capture(
                    reference_model, input_tensors, microbatches
                )
            del reference_model

            overlap_model = GPTModel(
                config=overlap_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            overlap_model.cuda()
            reset_model(overlap_model, reference_params)
            overlap_capture = run_transformer_layer_a2a_overlap_with_capture(
                overlap_model, input_tensors, microbatches
            )

        matched, message = compare_captures(reference_capture, overlap_capture, True)
        assert matched, (
            f"[rank {torch.distributed.get_rank()}] "
            f"selector={selector} shared_expert={shared_expert}: {message}"
        )
    finally:
        Utils.destroy_model_parallel()
