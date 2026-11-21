from types import SimpleNamespace

import torch

from megatron.core.pipeline_parallel.schedules import backward_step


def _config():
    return SimpleNamespace(
        timers=None,
        grad_scale_func=None,
        deallocate_pipeline_outputs=False,
    )


def test_backward_step_skips_output_without_grad_fn():
    input_tensor = torch.ones(2, requires_grad=True)
    output_tensor = input_tensor.detach().requires_grad_(True)

    grad = backward_step(
        input_tensor,
        output_tensor,
        torch.ones_like(output_tensor),
        _config(),
    )

    assert grad is None
    assert input_tensor.grad is None


def test_backward_step_runs_for_differentiable_output():
    input_tensor = torch.ones(2, requires_grad=True)
    output_tensor = input_tensor * 2

    grad = backward_step(
        input_tensor,
        output_tensor,
        torch.ones_like(output_tensor),
        _config(),
    )

    assert grad is not None
    torch.testing.assert_close(grad, torch.full_like(input_tensor, 2.0))
