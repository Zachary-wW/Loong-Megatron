from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer import multi_latent_attention
from megatron.core.transformer.enums import AttnMaskType
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


def _concat_cache_owner(*, sft_chunkpipe_mode):
    batch = 1
    kv_lora_rank = 2
    pos_dim = 2
    slot_size = 3
    total_slots = 3
    total_tokens = slot_size * total_slots
    if sft_chunkpipe_mode:
        kv_shape = (total_tokens, kv_lora_rank)
        pos_shape = (total_tokens, 1, pos_dim)
    else:
        kv_shape = (total_tokens, batch, kv_lora_rank)
        pos_shape = (total_tokens, batch, 1, pos_dim)

    owner = SimpleNamespace(
        config=SimpleNamespace(
            enable_chunkpipe=True,
            chunkpipe_forward=True,
            sft_chunkpipe_mode=sft_chunkpipe_mode,
            chunkpipe_forward_microbatch=2,
            chunkpipe_backward_microbatch=0,
            chunkpipe_chunk_idx_in_group=2,
            chunk_sizes=[2, 3, 2],
            cu_chunks=[0, 2, 5, 7],
            max_chunk_size=slot_size,
            chunksize=slot_size,
            micro_batch_size=batch,
            v_head_dim=kv_lora_rank,
        ),
        num_chunks_per_seq=3,
        num_attention_heads_per_partition=1,
        key_hidden_size=kv_lora_rank + pos_dim,
        padding_v_head_dim=False,
        pg_collection=SimpleNamespace(tp=None),
        kv_cache_chunk_size=total_slots,
        empty_chunk_indices=[],
        micro_batch_to_cache_chunk_map={0: 0, 1: 1},
        kv_compressed_cache=torch.zeros(kv_shape),
        key_pos_emb_cache=torch.zeros(pos_shape),
        kv_compressed_cache_grad={},
        key_pos_emb_cache_grad={},
        is_enable_grad_chunkpipe=lambda: False,
        layer_number=1,
    )

    first_kv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second_kv = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    first_pos = torch.arange(4, dtype=torch.float32).reshape(2, 1, pos_dim)
    second_pos = (torch.arange(6, dtype=torch.float32) + 10).reshape(3, 1, pos_dim)

    if sft_chunkpipe_mode:
        owner.kv_compressed_cache[0:2] = first_kv
        owner.kv_compressed_cache[3:6] = second_kv
        owner.key_pos_emb_cache[0:2] = first_pos
        owner.key_pos_emb_cache[3:6] = second_pos
    else:
        owner.kv_compressed_cache[0:2, 0, :] = first_kv
        owner.kv_compressed_cache[3:6, 0, :] = second_kv
        owner.key_pos_emb_cache[0:2, 0] = first_pos
        owner.key_pos_emb_cache[3:6, 0] = second_pos

    recovered = []

    def recover_key_value_up_proj_tensors(kv_compressed, key_pos_emb):
        recovered.append((kv_compressed.clone(), key_pos_emb.clone()))
        if sft_chunkpipe_mode:
            key = torch.cat((kv_compressed.unsqueeze(1), key_pos_emb), dim=-1)
            value = kv_compressed.unsqueeze(1) * 100
        else:
            key = torch.cat((kv_compressed.unsqueeze(2), key_pos_emb), dim=-1)
            value = kv_compressed.unsqueeze(2) * 100
        return key, value

    owner.recover_key_value_up_proj_tensors = recover_key_value_up_proj_tensors
    owner.recovered = recovered
    owner.expected_first_kv = first_kv
    owner.expected_second_kv = second_kv
    owner.expected_first_pos = first_pos if sft_chunkpipe_mode else first_pos.unsqueeze(2)
    owner.expected_second_pos = second_pos if sft_chunkpipe_mode else second_pos.unsqueeze(2)
    return owner


def test_mla_cache_concatenates_variable_chunks_and_builds_bottom_right_mask():
    owner = _concat_cache_owner(sft_chunkpipe_mode=True)
    current_kv = torch.tensor([[11.0, 12.0], [13.0, 14.0]])
    current_pos = (torch.arange(4, dtype=torch.float32) + 20).reshape(2, 1, 2)
    current_key = torch.cat((current_kv.unsqueeze(1), current_pos), dim=-1)
    current_value = current_kv.unsqueeze(1)

    key, value, mask = MultiLatentAttention.concat_cached_chunk_key_value_mla(
        owner,
        attention_mask=torch.ones(1, 1, 2, 7, dtype=torch.bool),
        curr_key=current_key,
        curr_value=current_value,
    )

    assert len(owner.recovered) == 2
    torch.testing.assert_close(owner.recovered[0][0], owner.expected_first_kv)
    torch.testing.assert_close(owner.recovered[1][0], owner.expected_second_kv)
    torch.testing.assert_close(owner.recovered[0][1], owner.expected_first_pos)
    torch.testing.assert_close(owner.recovered[1][1], owner.expected_second_pos)

    expected_first_key = torch.cat(
        (owner.expected_first_kv.unsqueeze(1), owner.expected_first_pos), dim=-1
    )
    expected_second_key = torch.cat(
        (owner.expected_second_kv.unsqueeze(1), owner.expected_second_pos), dim=-1
    )
    torch.testing.assert_close(key[:2], expected_first_key)
    torch.testing.assert_close(key[2:5], expected_second_key)
    torch.testing.assert_close(key[5:], current_key)
    torch.testing.assert_close(value[:2], owner.expected_first_kv.unsqueeze(1) * 100)
    torch.testing.assert_close(value[2:5], owner.expected_second_kv.unsqueeze(1) * 100)
    torch.testing.assert_close(value[5:], current_value)

    # True means masked: the first current token may attend the five cached
    # tokens but not the second token in its own chunk.
    assert mask.shape == (1, 1, 2, 7)
    assert not mask[0, 0, 0, :6].any()
    assert mask[0, 0, 0, 6]
    assert not mask[0, 0, 1].any()


def test_mla_cache_concatenation_preserves_single_rope_key_head():
    owner = _concat_cache_owner(sft_chunkpipe_mode=False)
    current_kv = torch.tensor([[[11.0, 12.0]], [[13.0, 14.0]]])
    current_pos = (torch.arange(4, dtype=torch.float32) + 20).reshape(2, 1, 1, 2)
    current_key = torch.cat((current_kv.unsqueeze(2), current_pos), dim=-1)
    current_value = current_kv.unsqueeze(2)

    key, value, mask = MultiLatentAttention.concat_cached_chunk_key_value_mla(
        owner,
        attention_mask=None,
        curr_key=current_key,
        curr_value=current_value,
    )

    assert mask is None
    assert len(owner.recovered) == 2
    for _, recovered_pos in owner.recovered:
        assert recovered_pos.shape[2] == 1
    torch.testing.assert_close(owner.recovered[0][1], owner.expected_first_pos)
    torch.testing.assert_close(owner.recovered[1][1], owner.expected_second_pos)
    expected_first_kv = owner.expected_first_kv.unsqueeze(1).unsqueeze(2)
    expected_second_kv = owner.expected_second_kv.unsqueeze(1).unsqueeze(2)
    torch.testing.assert_close(key[:2], torch.cat(
        (expected_first_kv, owner.expected_first_pos), dim=-1
    ))
    torch.testing.assert_close(key[2:5], torch.cat(
        (expected_second_kv, owner.expected_second_pos), dim=-1
    ))
    torch.testing.assert_close(key[5:], current_key)
    torch.testing.assert_close(value[:2], expected_first_kv * 100)
    torch.testing.assert_close(value[2:5], expected_second_kv * 100)
    torch.testing.assert_close(value[5:], current_value)


def _forward_cache_owner(*, experimental_attention_variant):
    captured = {}
    query = torch.ones(1, 1, 1, 4)
    key = torch.ones(1, 1, 1, 4)
    value = torch.ones(1, 1, 1, 4)
    packed_seq_params = SimpleNamespace(qkv_format="thd")

    def get_qkv(*_args, **_kwargs):
        return query, key, value, None, None

    def adjust_key_value(*_args, **_kwargs):
        return query, key, value, None, AttnMaskType.causal, None

    def concat_cached_chunk_key_value_mla(attention_mask, current_key, current_value):
        captured["concat_called"] = True
        return current_key, current_value, attention_mask

    def run_core_attention(
        q,
        k,
        v,
        attention_mask,
        packed_seq_params=None,
        attn_mask_type=None,
        **_kwargs,
    ):
        captured["attn_mask_type"] = attn_mask_type
        captured["packed_seq_params"] = packed_seq_params
        return torch.zeros(1, 1, 1, 4)

    class _Projection(torch.nn.Module):
        def forward(self, x):
            return x, None

    owner = SimpleNamespace(
        config=SimpleNamespace(
            enable_chunkpipe=True,
            sft_chunkpipe_mode=False,
            chunkpipe_current_group_size=4,
            chunkpipe_chunk_idx_in_group=1,
            experimental_attention_variant=experimental_attention_variant,
            cache_mla_latents=False,
        ),
        training=False,
        cache_mla_latents=False,
        checkpoint_core_attention=False,
        padding_v_head_dim=False,
        recompute_up_proj=False,
        offload_qkv_linear=False,
        offload_core_attention=False,
        offload_attn_proj=False,
        _use_fused_q_uproj=False,
        core_attention=SimpleNamespace(requires_dsa_inputs=False),
        linear_proj=_Projection(),
        get_query_key_value_tensors=get_qkv,
        _adjust_key_value_for_inference=adjust_key_value,
        concat_cached_chunk_key_value_mla=concat_cached_chunk_key_value_mla,
        _run_core_attention=run_core_attention,
    )
    return owner, packed_seq_params, captured


def test_mla_dense_chunkpipe_selects_bottom_right_mask():
    owner, packed_seq_params, captured = _forward_cache_owner(
        experimental_attention_variant=None
    )

    MultiLatentAttention.forward(
        owner,
        hidden_states=torch.ones(1, 1, 4),
        attention_mask=torch.ones(1, 1, 1, 4, dtype=torch.bool),
        packed_seq_params=packed_seq_params,
    )

    assert captured["concat_called"]
    assert captured["attn_mask_type"] is AttnMaskType.causal_bottom_right
    assert captured["packed_seq_params"] is None


def test_mla_dsa_chunkpipe_keeps_causal_mask_and_packed_params():
    owner, packed_seq_params, captured = _forward_cache_owner(
        experimental_attention_variant="dsa"
    )

    MultiLatentAttention.forward(
        owner,
        hidden_states=torch.ones(1, 1, 4),
        attention_mask=torch.ones(1, 1, 1, 4, dtype=torch.bool),
        packed_seq_params=packed_seq_params,
    )

    assert captured["concat_called"]
    assert captured["attn_mask_type"] is AttnMaskType.causal
    assert captured["packed_seq_params"] is packed_seq_params
