import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions import fused_linear_cross_entropy
from megatron.core.fusions.linear_cross_entropy.generic import entry as generic_entry


def _configure_generic(monkeypatch):
    monkeypatch.setenv("LCE_FORCE_GENERIC", "1")
    monkeypatch.setenv("LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE", "4")
    monkeypatch.setenv("LCE_GENERIC_BWD_VOCAB_SPLIT_SIZE", "4")
    generic_entry._get_config.cache_clear()
    fused_linear_cross_entropy._get_platform.cache_clear()


def _select_platform(monkeypatch, *, force_generic: bool, capability: tuple[int, int]):
    monkeypatch.setenv("LCE_FORCE_GENERIC", "1" if force_generic else "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    fused_linear_cross_entropy.Platform._instance = None
    fused_linear_cross_entropy._get_platform.cache_clear()
    return fused_linear_cross_entropy._get_platform()


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_generic_lce_matches_torch_cross_entropy(monkeypatch, reduction):
    _configure_generic(monkeypatch)
    torch.manual_seed(17)
    num_tokens, hidden_size, vocab_size = 4, 8, 16
    hidden = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(
        vocab_size, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
    )
    labels = torch.tensor([0, 3, 5, -100], device="cuda")

    actual = fused_linear_cross_entropy.linear_cross_entropy(
        hidden,
        weight,
        labels,
        reduction=reduction,
        ignore_index=-100,
    )
    reference = F.cross_entropy(
        hidden @ weight.t(),
        labels,
        ignore_index=-100,
        reduction=reduction,
    )

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
    actual_grads = torch.autograd.grad(actual.sum(), (hidden, weight))
    reference_grads = torch.autograd.grad(reference.sum(), (hidden, weight))
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-4, atol=1e-4)


def test_generic_lce_bf16_matches_float_reference(monkeypatch):
    _configure_generic(monkeypatch)
    torch.manual_seed(23)
    hidden = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([1, 2, 7, 15], device="cuda")

    actual = fused_linear_cross_entropy.linear_cross_entropy(
        hidden, weight, labels, reduction="mean"
    )
    reference = F.cross_entropy(
        hidden.float() @ weight.float().t(), labels, reduction="mean"
    )

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    for actual_grad, reference_grad in zip(
        torch.autograd.grad(actual, (hidden, weight)),
        torch.autograd.grad(reference, (hidden, weight)),
    ):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=2e-2, atol=2e-2)


def test_generic_lce_platform_selection(monkeypatch):
    _configure_generic(monkeypatch)
    platform = fused_linear_cross_entropy._get_platform()
    assert platform.forward_func is generic_entry.forward
    assert platform.backward_func is generic_entry.backward


def test_platform_selects_blackwell_for_sm100(monkeypatch):
    from megatron.core.fusions.linear_cross_entropy.blackwell import entry as blackwell_entry

    platform = _select_platform(monkeypatch, force_generic=False, capability=(10, 0))
    assert platform.forward_func is blackwell_entry.forward
    assert platform.backward_func is blackwell_entry.backward


def test_platform_selects_generic_on_non_sm100(monkeypatch):
    platform = _select_platform(monkeypatch, force_generic=False, capability=(12, 0))
    assert platform.forward_func is generic_entry.forward
    assert platform.backward_func is generic_entry.backward


def test_platform_force_generic_overrides_sm100(monkeypatch):
    platform = _select_platform(monkeypatch, force_generic=True, capability=(10, 0))
    assert platform.forward_func is generic_entry.forward
    assert platform.backward_func is generic_entry.backward
