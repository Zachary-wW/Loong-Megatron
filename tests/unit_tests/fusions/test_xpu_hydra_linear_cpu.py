# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU parity checks for the XPU Hydra linear autograd function."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytest


MODULE_PATH = Path(__file__).resolve().parents[3] / "xpu_plugin" / "_hydra" / "_linear.py"


def _install_stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_linear_module(monkeypatch):
    class StatefulConfig:
        def __init__(self, *args, **kwargs):
            pass

        def make_autograd_function_stateful(self, function):
            return function

    hydrax = _install_stub("hydrax", __path__=[])
    xaccelerator = _install_stub("hydrax.xaccelerator", __path__=[])
    hydrax.xaccelerator = xaccelerator
    fc_fusion = _install_stub(
        "hydrax.xaccelerator.fc_fusion", fc_fusion=lambda *args, **kwargs: None
    )
    linear = _install_stub(
        "hydrax.xaccelerator.linear",
        linear_bwd_dgrad=lambda *args, **kwargs: None,
        linear_bwd_wgrad=lambda *args, **kwargs: None,
        linear_fwd=lambda *args, **kwargs: None,
        linear_with_all_gather_x=lambda *args, **kwargs: None,
        linear_with_all_reduce=lambda *args, **kwargs: None,
    )
    linear_impl = _install_stub(
        "hydrax.xaccelerator.linear._linear", view_matrix=lambda tensor: tensor
    )
    stateful = _install_stub(
        "hydrax.xaccelerator.stateful",
        StatefulConfig=StatefulConfig,
        state_of=lambda *args, **kwargs: None,
    )
    xaccelerator.fc_fusion = fc_fusion
    xaccelerator.linear = linear
    xaccelerator.stateful = stateful
    linear._linear = linear_impl

    torch_xmlir = _install_stub("torch_xmlir", __path__=[])
    torch_xmlir.config = _install_stub(
        "torch_xmlir.config",
        config=SimpleNamespace(disable_cast_cache=False, batch_parallel_enabled=False),
    )

    spec = importlib.util.spec_from_file_location(
        f"test_xpu_hydra_linear_{id(monkeypatch)}", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "linear_fwd", lambda x, w, b: F.linear(x, w, b))
    monkeypatch.setattr(
        module,
        "linear_bwd_dgrad",
        lambda grad_output, weight, output: output.copy_(grad_output @ weight),
    )
    monkeypatch.setattr(
        module,
        "linear_bwd_wgrad",
        lambda grad_output, input_tensor, output: output.copy_(
            grad_output.t() @ input_tensor
        ),
    )
    return module


def _reference_grads(x, w, b, grad_output):
    reference = F.linear(x, w, b)
    return torch.autograd.grad(reference, (x, w, b), grad_output)


def test_forward_backward_matches_torch_linear(monkeypatch):
    module = _load_linear_module(monkeypatch)
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(6, 8, requires_grad=True)
    b = torch.randn(6, requires_grad=True)
    grad_output = torch.randn(4, 6)

    actual = module.LinearWithGradAccumulationAndAsyncCommunication.apply(
        x, w, b, False, False, False, None, 0, object(), 1, None
    )
    actual_grads = torch.autograd.grad(actual, (x, w, b), grad_output)
    expected_grads = _reference_grads(x, w, b, grad_output)

    torch.testing.assert_close(actual, F.linear(x, w, b))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_sequence_parallel_collectives_with_one_rank(monkeypatch, tmp_path):
    module = _load_linear_module(monkeypatch)
    torch.manual_seed(1)
    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(6, 8, requires_grad=True)
    b = torch.randn(6, requires_grad=True)
    grad_output = torch.randn(4, 6)

    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'store'}",
            rank=0,
            world_size=1,
        )
    try:
        actual = module.LinearWithGradAccumulationAndAsyncCommunication.apply(
            x, w, b, False, False, True, None, 0, dist.group.WORLD, 1, None
        )
        actual_grads = torch.autograd.grad(actual, (x, w, b), grad_output)
        expected_grads = _reference_grads(x, w, b, grad_output)
    finally:
        if initialized_here:
            dist.destroy_process_group()

    torch.testing.assert_close(actual, F.linear(x, w, b))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_unsupported_gtp_and_output_dtype_fail_fast(monkeypatch):
    module = _load_linear_module(monkeypatch)
    x = torch.randn(2, 8)
    w = torch.randn(6, 8)
    b = torch.randn(6)
    function = module.LinearWithGradAccumulationAndAsyncCommunication

    with pytest.raises(NotImplementedError):
        function.apply(x, w, b, False, False, False, None, 0, object(), 2, None)

    with pytest.raises(NotImplementedError):
        function.apply(
            x, w, b, False, False, False, None, 0, object(), 1, torch.float16
        )
