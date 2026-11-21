from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer import param_group_identifier_keys


class _RecordingOptimizer:
    def __init__(self):
        self.state = {}
        self.param_groups = [{"params": [0]}]
        self.loaded_state_dict = None

    def state_dict(self):
        group = {key: None for key in param_group_identifier_keys}
        return {"state": {}, "param_groups": [{**group, "params": [0]}]}

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict


def _build_optimizer(precision_aware):
    model_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.config = SimpleNamespace(
        optimizer="adam",
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.float32,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware,
        store_param_remainders=True,
        bf16=True,
        main_params_dtype=torch.float32,
        fp16=False,
        _uses_fp8_cpu_offload_main_params=lambda: False,
    )
    optimizer.optimizer = _RecordingOptimizer()
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.gbuf_ranges = [
        {
            (torch.float32, torch.float32): [
                {"param_map": {model_param: {"gbuf_world": Range(0, 7)}}}
            ]
        }
    ]
    optimizer.model_param_group_index_map = {model_param: (0, 0)}
    optimizer.grad_scaler = None
    optimizer._maybe_materialize_hdo_state_arenas = lambda: None
    return optimizer


@pytest.mark.parametrize(
    ("precision_aware", "expected_dtypes"),
    [
        (
            False,
            {"exp_avg": torch.bfloat16, "exp_avg_sq": torch.float32},
        ),
        (
            True,
            {
                "exp_avg": torch.bfloat16,
                "exp_avg_sq": torch.float32,
                "master_param": torch.int16,
            },
        ),
    ],
)
def test_empty_shard_state_is_allocated_on_cpu(precision_aware, expected_dtypes):
    optimizer = _build_optimizer(precision_aware)
    saved_group = {key: None for key in param_group_identifier_keys}

    optimizer.load_state_dict({"optimizer": {"param_groups": [saved_group]}})

    loaded = optimizer.optimizer.loaded_state_dict
    placeholder_state = loaded["state"][0]
    assert set(placeholder_state) == set(expected_dtypes)
    for key, dtype in expected_dtypes.items():
        tensor = placeholder_state[key]
        assert tensor.device.type == "cpu"
        assert tensor.dtype == dtype
        assert tensor.shape == (7,)
    assert loaded["param_groups"][0]["params"] == [0]
