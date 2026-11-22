import torch

from megatron.core.transformer.moe.offloading_planner import (
    auction_allocation,
    breadth_first_allocation,
    depth_first_allocation,
    gen_assignment,
    gen_intermediate,
    one_shot_greedy_assignment,
)


def test_one_shot_greedy_assignment_preserves_chunk_tokens():
    chunks = torch.tensor([3, 2], dtype=torch.int32)
    buckets = torch.tensor([4, 4], dtype=torch.int32)

    assignment = one_shot_greedy_assignment(chunks, buckets)

    assert assignment.dtype == torch.int64
    assert assignment.shape == (2, 2)
    assert assignment.sum() == chunks.sum()
    torch.testing.assert_close(
        assignment,
        torch.tensor([[3, 0], [1, 1]], dtype=torch.int64),
    )


def test_gen_intermediate_identifies_spillover_and_spare_capacity():
    counts = torch.tensor([[10, 2], [8, 0]], dtype=torch.int32)

    spillover, capacity, avg = gen_intermediate(counts, ep_rank=0, num_ep_ranks=2)

    torch.testing.assert_close(spillover, torch.tensor([8, 0], dtype=torch.int64))
    torch.testing.assert_close(capacity, torch.tensor([0, 8], dtype=torch.int64))
    assert avg.item() == 10


def test_gen_assignment_has_conservative_bucket_capacity():
    counts = torch.tensor([[10, 2], [8, 0]], dtype=torch.int32)

    assignment, spillover, capacity = gen_assignment(
        counts,
        ep_rank=1,
        num_ep_ranks=2,
        num_spare_experts_per_ep_rank=1,
    )

    assert assignment.shape == (2, 2)
    assert spillover.shape == (2,)
    assert capacity.shape == (2,)
    assert assignment.sum() <= spillover.sum()
    assert (assignment.sum(dim=0) <= capacity).all()


def test_breadth_and_depth_first_allocations_conserve_tokens():
    counts = torch.tensor([[3, 5], [1, 4]], dtype=torch.int32)
    assignment = torch.tensor([[5], [3]], dtype=torch.int32)

    first, after_first, remaining = breadth_first_allocation(
        counts, assignment, device="cpu"
    )
    second, after_second = depth_first_allocation(after_first, remaining)

    assert first.shape == assignment.shape
    assert after_first.shape == counts.shape
    assert remaining.shape == assignment.shape
    assert second.shape == (counts.shape[0], assignment.shape[1])
    assert after_second.shape == counts.shape
    torch.testing.assert_close(
        first.sum() + after_first.sum(), counts.sum().float()
    )
    torch.testing.assert_close(
        second.sum() + after_second.sum(), after_first.sum()
    )


def test_auction_allocation_respects_supply_and_demand():
    counts = torch.tensor([[3, 5], [1, 4]], dtype=torch.int32)
    assignment = torch.tensor([[5], [3]], dtype=torch.int32)

    allocation, supply_remaining = auction_allocation(
        counts, assignment, ranks_per_node=1
    )

    assert allocation.shape == (2, 1)
    assert supply_remaining.shape == counts.shape
    assert allocation.sum().item() == 4
    assert (allocation.sum(dim=0) <= assignment.sum(dim=0)).all()
    assert (supply_remaining.sum(dim=1) <= counts.sum(dim=1)).all()
    assert supply_remaining.sum() + allocation.sum() == counts.sum()
