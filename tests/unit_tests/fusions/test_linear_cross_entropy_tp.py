import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.fusions import fused_linear_cross_entropy
from megatron.core.fusions.linear_cross_entropy.generic import entry as generic_entry


pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="requires two distributed ranks",
)


def setup_module():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    os.environ["LCE_FORCE_GENERIC"] = "1"
    os.environ["LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE"] = "4"
    os.environ["LCE_GENERIC_BWD_VOCAB_SPLIT_SIZE"] = "4"
    generic_entry._get_config.cache_clear()
    fused_linear_cross_entropy._get_platform.cache_clear()


def teardown_module():
    dist.destroy_process_group()


def _inputs():
    torch.manual_seed(31)
    hidden = torch.randn(8, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(16, 8, device="cuda", dtype=torch.float32)
    labels = torch.tensor([0, 3, 5, 7, 9, 11, 13, 15], device="cuda")
    return hidden, weight, labels


def test_tp_vocab_shard_matches_full_reference():
    hidden, weight, labels = _inputs()
    rank = dist.get_rank()
    local_weight = weight[rank * 8 : (rank + 1) * 8].detach().clone().requires_grad_(True)
    hidden = hidden.detach().clone().requires_grad_(True)

    actual = fused_linear_cross_entropy.linear_cross_entropy(
        hidden, local_weight, labels, tp_group=dist.group.WORLD, reduction="mean"
    )
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = F.cross_entropy(
        reference_hidden @ reference_weight.t(), labels, reduction="mean"
    )
    assert actual.item() == pytest.approx(reference.item(), abs=1e-4)

    actual_hidden_grad, actual_weight_grad = torch.autograd.grad(
        actual, (hidden, local_weight)
    )
    reference_hidden_grad, reference_weight_grad = torch.autograd.grad(
        reference, (reference_hidden, reference_weight)
    )
    torch.testing.assert_close(actual_hidden_grad, reference_hidden_grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        actual_weight_grad,
        reference_weight_grad[rank * 8 : (rank + 1) * 8],
        atol=1e-4,
        rtol=1e-4,
    )


def test_sequence_parallel_hidden_shard_matches_full_reference():
    hidden, weight, labels = _inputs()
    rank = dist.get_rank()
    local_hidden = hidden[rank * 4 : (rank + 1) * 4].detach().clone().requires_grad_(True)
    local_weight = weight[rank * 8 : (rank + 1) * 8].detach().clone().requires_grad_(True)

    actual = fused_linear_cross_entropy.linear_cross_entropy(
        local_hidden,
        local_weight,
        labels,
        tp_group=dist.group.WORLD,
        reduction="mean",
        sequence_parallel=True,
    )
    reference_hidden = hidden.detach().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = F.cross_entropy(
        reference_hidden @ reference_weight.t(), labels, reduction="mean"
    )
    assert actual.item() == pytest.approx(reference.item(), abs=1e-4)

    actual_hidden_grad, actual_weight_grad = torch.autograd.grad(
        actual, (local_hidden, local_weight)
    )
    reference_hidden_grad, reference_weight_grad = torch.autograd.grad(
        reference, (reference_hidden, reference_weight)
    )
    torch.testing.assert_close(
        actual_hidden_grad,
        reference_hidden_grad[rank * 4 : (rank + 1) * 4],
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        actual_weight_grad,
        reference_weight_grad[rank * 8 : (rank + 1) * 8],
        atol=1e-4,
        rtol=1e-4,
    )
