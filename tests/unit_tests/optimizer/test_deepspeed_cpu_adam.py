# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tensor checks for optional DeepSpeed CPU Adam wiring."""

import builtins
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import _get_cpu_adam_class
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import _extract_hdo_step
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def _sub_optimizer(*steps, group_step=None, fused=False):
    states = {index: {"step": step} for index, step in enumerate(steps)}
    groups = [{"params": [object()], "step": group_step}] if group_step is not None else []
    if fused:
        return SimpleNamespace(state=states, param_groups=groups)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    optimizer.state = states
    optimizer.param_groups = groups
    return optimizer


def _hdo(*sub_optimizers):
    return SimpleNamespace(sub_optimizers=list(sub_optimizers))


def test_extract_hdo_step_accepts_torch_and_deepspeed_scalar_steps(monkeypatch):
    cpu_adam_type = type("DeepSpeedCPUAdam", (SimpleNamespace,), {})
    monkeypatch.setitem(
        sys.modules, "deepspeed.ops.adam", SimpleNamespace(DeepSpeedCPUAdam=cpu_adam_type)
    )
    cpu_adam = cpu_adam_type(state={0: {"step": 3}}, param_groups=[])
    optimizer = _hdo(_sub_optimizer(torch.tensor(3.0)), cpu_adam)

    assert _extract_hdo_step(optimizer) == 3


def test_extract_hdo_step_ignores_empty_sub_optimizer():
    optimizer = _hdo(_sub_optimizer(), _sub_optimizer(7))

    assert _extract_hdo_step(optimizer) == 7


def test_extract_hdo_step_rejects_inconsistent_steps():
    with pytest.raises(AssertionError, match="Inconsistent optimizer steps"):
        _extract_hdo_step(_hdo(_sub_optimizer(3), _sub_optimizer(4)))


def test_extract_hdo_step_accepts_empty_rank():
    assert _extract_hdo_step(_hdo(_sub_optimizer())) is None


def test_extract_hdo_step_uses_group_counter_for_fused_optimizer():
    assert _extract_hdo_step(_hdo(_sub_optimizer(group_step=5, fused=True))) == 5


def test_extract_hdo_step_ignores_stale_restored_group_counter():
    assert _extract_hdo_step(_hdo(_sub_optimizer(6, group_step=5))) == 6


def test_extract_hdo_step_ignores_stale_fused_parameter_counter():
    optimizer = _hdo(_sub_optimizer(11), _sub_optimizer(10, group_step=11, fused=True))
    assert _extract_hdo_step(optimizer) == 11


@pytest.mark.parametrize("torch_only", [False, True])
def test_cpu_adam_opt_out_does_not_import_deepspeed(monkeypatch, torch_only):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("deepspeed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    config = OptimizerConfig(
        use_deepspeed_cpu_adam=torch_only, use_torch_optimizer_for_cpu_offload=torch_only
    )
    assert _get_cpu_adam_class(config) is torch.optim.AdamW


def test_missing_deepspeed_warns_and_selects_torch(monkeypatch):
    real_import = builtins.__import__

    def missing_deepspeed(name, *args, **kwargs):
        if name == "deepspeed.ops.adam":
            raise ImportError("DeepSpeed unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_deepspeed)
    config = OptimizerConfig()
    with pytest.warns(UserWarning, match="falling back"):
        assert _get_cpu_adam_class(config) is torch.optim.AdamW
    assert not config.use_deepspeed_cpu_adam


def test_available_deepspeed_is_selected_lazily(monkeypatch):
    real_import = builtins.__import__
    cpu_adam = type("DeepSpeedCPUAdam", (), {})

    def available_deepspeed(name, *args, **kwargs):
        if name == "deepspeed.ops.adam":
            return SimpleNamespace(DeepSpeedCPUAdam=cpu_adam)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", available_deepspeed)
    assert _get_cpu_adam_class(OptimizerConfig()) is cpu_adam


def test_deepspeed_cpu_adam_forces_fp32_moments_for_offload():
    with pytest.warns(UserWarning, match="FP32 moment states"):
        config = OptimizerConfig(
            optimizer_cpu_offload=True,
            optimizer="adam",
            use_deepspeed_cpu_adam=True,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.float16,
        )

    assert config.exp_avg_dtype == torch.float32
    assert config.exp_avg_sq_dtype == torch.float32


def test_torch_cpu_adam_opt_out_preserves_requested_state_dtypes():
    config = OptimizerConfig(
        optimizer_cpu_offload=True,
        optimizer="adam",
        use_deepspeed_cpu_adam=False,
        use_precision_aware_optimizer=True,
        use_distributed_optimizer=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.float16,
    )

    assert config.exp_avg_dtype == torch.bfloat16
    assert config.exp_avg_sq_dtype == torch.float16


@pytest.mark.parametrize("deepspeed_enabled", [False, True])
def test_hdo_restores_cpu_step_and_moment_types(monkeypatch, deepspeed_enabled):
    param = torch.zeros(2)
    state = {
        "step": torch.tensor(9.0),
        "exp_avg": torch.ones(2, dtype=torch.bfloat16),
        "exp_avg_sq": torch.ones(2, dtype=torch.bfloat16),
    }
    cpu_adam_type = type("DeepSpeedCPUAdam", (SimpleNamespace,), {})
    monkeypatch.setitem(
        sys.modules, "deepspeed.ops.adam", SimpleNamespace(DeepSpeedCPUAdam=cpu_adam_type)
    )
    optimizer_type = cpu_adam_type if deepspeed_enabled else SimpleNamespace
    cpu_optimizer = optimizer_type(state={param: state})
    hdo = HybridDeviceOptimizer.__new__(HybridDeviceOptimizer)
    hdo.cpu_optimizers = [cpu_optimizer]
    hdo.gpu_optimizer = None
    hdo.defaults = {"cpu_optimizer_cls": SimpleNamespace}
    hdo.inner_param_to_orig_param = {param: param}
    hdo.state = {param: state}

    hdo._move_new_state_to_right_device()

    assert state["step"] == 9
    assert isinstance(state["step"], int if deepspeed_enabled else torch.Tensor)
    expected_dtype = torch.float32 if deepspeed_enabled else torch.bfloat16
    for key in ("exp_avg", "exp_avg_sq"):
        assert state[key].device.type == "cpu"
        assert state[key].dtype == expected_dtype
        torch.testing.assert_close(state[key], torch.ones(2, dtype=expected_dtype))


@pytest.mark.parametrize("preserve_step,expected_step", [(True, 11), (False, 10)])
def test_hdo_sync_preserves_live_counter_except_during_load(preserve_step, expected_step):
    param = torch.nn.Parameter(torch.ones(1))
    sub_optimizer = SimpleNamespace(param_groups=[{"params": [param], "step": 11, "lr": 0.1}])
    hdo = HybridDeviceOptimizer.__new__(HybridDeviceOptimizer)
    hdo.cpu_optimizers = []
    hdo.gpu_optimizer = sub_optimizer
    hdo.param_groups = [{"params": [param], "step": 10, "lr": 0.01}]
    hdo.param_to_inner_param = {param: param}

    hdo._sync_hdo_param_groups_to_sub_optimizers(preserve_step=preserve_step)

    assert sub_optimizer.param_groups[0]["step"] == expected_step
    assert sub_optimizer.param_groups[0]["lr"] == 0.01
