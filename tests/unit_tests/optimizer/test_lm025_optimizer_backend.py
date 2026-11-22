import argparse
import builtins
from types import SimpleNamespace

import pytest
import torch
from torch.optim import AdamW

from megatron.core.optimizer import OptimizerConfig, _resolve_cpu_adam_class
from megatron.training.arguments import add_megatron_arguments


def test_deepspeed_cpu_adam_import_failure_falls_back_to_adamw(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("deepspeed"):
            raise ImportError("simulated missing DeepSpeed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    config = SimpleNamespace(use_deepspeed_cpu_adam=True)

    with pytest.warns(UserWarning, match="Falling back to PyTorch AdamW"):
        optimizer_cls = _resolve_cpu_adam_class(config)

    assert optimizer_cls is AdamW
    assert config.use_deepspeed_cpu_adam is False


def test_deepspeed_cpu_adam_can_be_disabled_without_import(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("deepspeed"):
            raise AssertionError("disabled backend must not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    config = SimpleNamespace(use_deepspeed_cpu_adam=False)

    assert _resolve_cpu_adam_class(config) is AdamW


def test_no_use_deepspeed_cpu_adam_cli_sets_false():
    parser = argparse.ArgumentParser()
    add_megatron_arguments(parser)

    args = parser.parse_args(["--no-use-deepspeed-cpu-adam"])
    assert args.use_deepspeed_cpu_adam is False


def test_deepspeed_cpu_adam_forces_fp32_moments():
    with pytest.warns(UserWarning, match="DeepSpeed CPUAdam requires FP32"):
        config = OptimizerConfig(
            optimizer="adam",
            use_distributed_optimizer=False,
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=1.0,
            use_deepspeed_cpu_adam=True,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.float16,
            bf16=True,
        )

    assert config.exp_avg_dtype is torch.float32
    assert config.exp_avg_sq_dtype is torch.float32
