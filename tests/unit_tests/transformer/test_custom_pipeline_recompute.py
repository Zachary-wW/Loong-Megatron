from types import SimpleNamespace

import torch

from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    values = dict(
        num_layers=4,
        num_attention_heads=1,
        hidden_size=8,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        recompute_granularity="full",
        recompute_method="block",
    )
    values.update(overrides)
    return TransformerConfig(**values)


def test_custom_pipeline_recompute_uses_physical_pp_rank():
    config = _config(custom_pipeline_recompute_layers=[1, 3])

    first = SimpleNamespace(config=config, pp_rank=0)
    second = SimpleNamespace(config=config, pp_rank=1)

    assert TransformerBlock._recompute_num_layers.fget(first) == 1
    assert TransformerBlock._recompute_num_layers.fget(second) == 3


def test_custom_pipeline_recompute_falls_back_to_recompute_num_layers():
    config = _config(recompute_num_layers=2)

    block = SimpleNamespace(config=config, pp_rank=1)

    assert TransformerBlock._recompute_num_layers.fget(block) == 2
