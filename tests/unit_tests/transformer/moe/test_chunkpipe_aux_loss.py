from types import SimpleNamespace

import torch

from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.router import TopKRouter


def _router(config):
    return SimpleNamespace(config=SimpleNamespace(**config), topk=1)


def test_chunkpipe_group_state_and_token_accumulation():
    router = _router(
        {
            "sft_chunkpipe_mode": False,
            "chunkpipe_forward": True,
            "chunk_num_per_seq": 3,
            "chunkpipe_forward_microbatch": 0,
            "chunkpipe_backward_microbatch": 0,
        }
    )
    assert TopKRouter._get_chunkpipe_group_state(router, forward=True) == (0, 3, 0)
    router._get_chunkpipe_group_state = lambda forward: TopKRouter._get_chunkpipe_group_state(
        router, forward
    )

    first = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32
    )
    TopKRouter._accumulate_chunkpipe_tokens_per_expert(router, first, seq_length=4)

    router.config.chunkpipe_forward_microbatch = 1
    second = torch.tensor([[1, 0, 0]] * 4, dtype=torch.float32)
    TopKRouter._accumulate_chunkpipe_tokens_per_expert(router, second, seq_length=4)

    full = TopKRouter._get_chunkpipe_full_tokens_per_expert(router, sequence_layout=False)
    torch.testing.assert_close(full, torch.tensor([6, 1, 1], dtype=torch.float32))
    seq_full = TopKRouter._get_chunkpipe_full_tokens_per_expert(router, sequence_layout=True)
    torch.testing.assert_close(seq_full, torch.tensor([6, 1, 1], dtype=torch.float32))

    TopKRouter._discard_chunkpipe_tokens_per_expert(router, microbatch_key=0)
    assert TopKRouter._get_chunkpipe_full_tokens_per_expert(router, sequence_layout=False) is None
    assert TopKRouter._get_chunkpipe_full_tokens_per_expert(router, sequence_layout=True) is None


def test_chunkpipe_sft_group_state_uses_scheduler_offsets():
    router = _router(
        {
            "sft_chunkpipe_mode": True,
            "chunkpipe_forward": True,
            "chunkpipe_chunk_idx_in_group": 1,
            "chunkpipe_current_group_size": 3,
            "chunkpipe_forward_microbatch": 4,
            "chunkpipe_backward_microbatch": 4,
        }
    )
    assert TopKRouter._get_chunkpipe_group_state(router, forward=True) == (1, 3, 3)


def test_chunkpipe_reduce_produces_global_and_token_counts(monkeypatch):
    router = SimpleNamespace(config=SimpleNamespace(), topk=2)
    calls = []

    def fake_reduce(tensor, group):
        calls.append((tensor.clone(), group))
        return tensor * 2

    monkeypatch.setattr(
        router_module, "reduce_from_tensor_model_parallel_region", fake_reduce
    )
    local_tokens = torch.tensor([3, 0, 5], dtype=torch.float32)

    global_tokens, local_num_tokens, total_num_tokens = (
        TopKRouter._reduce_chunkpipe_tokens_per_expert(
            router, local_tokens, ("tp-group",)
        )
    )

    torch.testing.assert_close(global_tokens, local_tokens * 2)
    assert local_num_tokens == 4
    assert total_num_tokens == 8
    assert len(calls) == 1
    assert calls[0][1] == "tp-group"
