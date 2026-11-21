from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer import multi_latent_attention
from megatron.core.transformer.multi_latent_attention import MultiLatentAttention


def _cache_owner(forward_microbatch=0):
    return SimpleNamespace(
        config=SimpleNamespace(
            enable_chunkpipe=True,
            chunkpipe_forward=True,
            sft_chunkpipe_mode=False,
            chunkpipe_forward_microbatch=forward_microbatch,
            chunksize=4,
            max_chunk_size=4,
            micro_batch_size=1,
            kv_lora_rank=2,
            qk_pos_emb_head_dim=2,
        ),
        pg_collection=SimpleNamespace(tp=None),
        num_chunks_per_seq=4,
        empty_chunk_indices=[0, 1],
        micro_batch_to_cache_chunk_map={},
        kv_compressed_cache=torch.zeros(8, 1, 2),
        key_pos_emb_cache=torch.zeros(8, 1, 2, 2),
        is_enable_grad_chunkpipe=lambda: False,
        layer_number=1,
    )


def test_mla_cache_writes_slot_offset_and_single_rope_head(monkeypatch):
    monkeypatch.setattr(multi_latent_attention, "get_pg_size", lambda _: 1)
    owner = _cache_owner()
    compressed = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)
    pos_emb = torch.arange(4, dtype=torch.float32).reshape(2, 1, 1, 2)

    MultiLatentAttention.append_chunk_key_value_cache_mla(owner, compressed, pos_emb)

    torch.testing.assert_close(owner.kv_compressed_cache[:2], compressed)
    torch.testing.assert_close(owner.key_pos_emb_cache[:2, :, :1, :], pos_emb)
    assert torch.count_nonzero(owner.key_pos_emb_cache[:2, :, 1:, :]) == 0
    assert owner.micro_batch_to_cache_chunk_map == {0: 0}
    assert owner.empty_chunk_indices == [1]


def test_mla_cache_skips_last_chunk(monkeypatch):
    monkeypatch.setattr(multi_latent_attention, "get_pg_size", lambda _: 1)
    owner = _cache_owner(forward_microbatch=3)
    compressed = torch.ones(2, 1, 2)
    pos_emb = torch.ones(2, 1, 1, 2)

    MultiLatentAttention.append_chunk_key_value_cache_mla(owner, compressed, pos_emb)

    assert owner.micro_batch_to_cache_chunk_map == {}
    assert owner.empty_chunk_indices == [0, 1]
    assert torch.count_nonzero(owner.kv_compressed_cache) == 0


def test_mla_cache_rejects_use_without_chunkpipe():
    owner = _cache_owner()
    owner.config.enable_chunkpipe = False

    with pytest.raises(RuntimeError, match="chunkpipe"):
        MultiLatentAttention.append_chunk_key_value_cache_mla(
            owner, torch.ones(2, 1, 2), torch.ones(2, 1, 1, 2)
        )


def test_mla_cache_grad_consumption_fails_on_orphan():
    owner = _cache_owner()
    owner.kv_compressed_cache_grad = {0: torch.zeros(2, 1, 2)}
    owner.key_pos_emb_cache_grad = {}

    with pytest.raises(AssertionError, match="Orphan kv_compressed_cache_grad"):
        MultiLatentAttention.check_kv_cache_grad_consumed(owner)
