from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.attention as attention_module
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer


def _general_attention_owner():
    owner = SimpleNamespace()
    owner.config = SimpleNamespace(
        enable_chunkpipe=True,
        chunkpipe_forward=True,
        sft_chunkpipe_mode=True,
        chunkpipe_forward_microbatch=0,
        chunkpipe_backward_microbatch=2,
        chunkpipe_chunk_idx_in_group=0,
        chunkpipe_current_group_size=3,
        chunksize=2,
        keep_activations_chunks=0,
        micro_batch_size=1,
    )
    owner.training = True
    owner.num_chunks_per_seq = 3
    owner.kv_cache_chunk_size = 3
    owner.num_query_groups_per_partition = 2
    owner.hidden_size_per_attention_head = 3
    owner.key_cache = torch.zeros(6, 1, 2, 3)
    owner.value_cache = torch.zeros(6, 1, 2, 3)
    owner.empty_chunk_indices = [0, 1, 2]
    owner.micro_batch_to_cache_chunk_map = {}
    owner.key_cache_grad = {}
    owner.value_cache_grad = {}
    owner.layer_number = 1
    owner.is_enable_grad_chunkpipe = lambda: False
    return owner


def test_general_attention_cache_append_concat_and_mask():
    owner = _general_attention_owner()
    key0 = torch.arange(12, dtype=torch.float32).reshape(2, 1, 2, 3)
    value0 = key0 + 100
    key1 = key0 + 20
    value1 = value0 + 20
    key2 = key0 + 40
    value2 = value0 + 40

    Attention.append_chunk_key_value_cache(owner, key0, value0)
    owner.config.chunkpipe_forward_microbatch = 1
    owner.config.chunkpipe_chunk_idx_in_group = 1
    Attention.append_chunk_key_value_cache(owner, key1, value1)

    assert owner.micro_batch_to_cache_chunk_map == {0: 0, 1: 1}
    assert owner.empty_chunk_indices == [2]
    torch.testing.assert_close(owner.key_cache[:2], key0)
    torch.testing.assert_close(owner.value_cache[2:4], value1)

    owner.config.chunkpipe_forward = False
    owner.config.chunkpipe_backward_microbatch = 2
    owner.config.chunkpipe_chunk_idx_in_group = 2
    key, value, mask = Attention.concat_cached_chunk_key_value(
        owner, key2, value2, torch.ones(1, 1, 2, 6, dtype=torch.bool)
    )

    torch.testing.assert_close(key[:2], key0)
    torch.testing.assert_close(key[2:4], key1)
    torch.testing.assert_close(key[4:], key2)
    torch.testing.assert_close(value[:2], value0)
    torch.testing.assert_close(value[2:4], value1)
    torch.testing.assert_close(value[4:], value2)
    assert mask.shape == (1, 1, 2, 6)
    assert mask[0, 0, 0, 5]
    assert not mask[0, 0, 1].any()


def test_general_attention_cache_skips_last_chunk_and_rejects_orphan_grad():
    owner = _general_attention_owner()
    owner.config.chunkpipe_chunk_idx_in_group = 2
    Attention.append_chunk_key_value_cache(
        owner,
        torch.ones(2, 1, 2, 3),
        torch.ones(2, 1, 2, 3),
    )
    assert owner.micro_batch_to_cache_chunk_map == {}
    assert owner.empty_chunk_indices == [0, 1, 2]

    owner.key_cache_grad[0] = torch.zeros(2, 1, 2, 3)
    with pytest.raises(AssertionError, match="Orphan key_cache_grad"):
        Attention.check_kv_cache_grad_consumed(owner)


def _dsa_indexer_owner():
    owner = DSAIndexer.__new__(DSAIndexer)
    owner.config = SimpleNamespace(
        enable_chunkpipe=True,
        chunkpipe_forward=True,
        sft_chunkpipe_mode=True,
        chunkpipe_forward_microbatch=0,
        chunkpipe_backward_microbatch=0,
        chunkpipe_chunk_idx_in_group=0,
        chunkpipe_current_group_size=3,
        chunk_num_per_seq=3,
        keep_activations_chunks=0,
    )
    owner.training = True
    owner._chunk_indexer_key_cache = {}
    owner._chunk_indexer_key_cache_grad = {}
    return owner


@pytest.mark.parametrize(
    ("config", "pp_size", "pp_rank", "expected_slots"),
    [
        (
            {
                "chunk_num_per_seq": 4,
                "sft_chunkpipe_mode": False,
                "virtual_pipeline_model_parallel_size": None,
            },
            2,
            1,
            4,
        ),
        (
            {
                "chunk_num_per_seq": 4,
                "sft_chunkpipe_mode": False,
                "virtual_pipeline_model_parallel_size": 3,
            },
            4,
            1,
            16,
        ),
        (
            {
                "chunk_num_per_seq": 4,
                "sft_chunkpipe_mode": True,
                "virtual_pipeline_model_parallel_size": 2,
                "chunkpipe_num_microbatches": 8,
            },
            2,
            0,
            6,
        ),
    ],
)
def test_general_attention_cache_slot_configuration(
    monkeypatch, config, pp_size, pp_rank, expected_slots
):
    owner = SimpleNamespace(
        config=SimpleNamespace(**config),
        pg_collection=SimpleNamespace(pp=object()),
    )
    monkeypatch.setattr(attention_module, "get_pg_size", lambda _: pp_size)
    monkeypatch.setattr(attention_module, "get_pg_rank", lambda _: pp_rank)

    Attention.setup_chunkpipe_kv_cache_config(owner)

    assert owner.num_chunks_per_seq == config["chunk_num_per_seq"]
    assert owner.kv_cache_chunk_size == expected_slots
    assert owner.micro_batch_to_cache_chunk_map == {}
    assert owner.empty_chunk_indices == list(range(expected_slots))


def test_general_attention_cache_delete_and_clear_restore_slots():
    owner = _general_attention_owner()
    owner.kv_cache_chunk_size = 3
    owner.micro_batch_to_cache_chunk_map = {2: 1}
    owner.empty_chunk_indices = [0, 2]

    Attention.delete_chunk_key_value_cache(owner, 2)
    assert owner.micro_batch_to_cache_chunk_map == {}
    assert owner.empty_chunk_indices == [0, 2, 1]

    Attention.delete_chunk_key_value_cache(owner, 99)
    assert owner.empty_chunk_indices == [0, 2, 1]

    owner.micro_batch_to_cache_chunk_map[4] = 2
    Attention.clear_chunk_key_value_cache(owner)
    assert owner.micro_batch_to_cache_chunk_map == {}
    assert owner.empty_chunk_indices == [0, 1, 2]


def test_general_attention_cache_initializes_declared_shape():
    owner = SimpleNamespace(
        config=SimpleNamespace(
            chunksize=3,
            micro_batch_size=2,
        ),
        kv_cache_chunk_size=2,
        num_query_groups_per_partition=4,
        hidden_size_per_attention_head=5,
        parameters=lambda: iter([torch.zeros(1)]),
    )

    Attention.init_chunk_key_value_cache(owner)

    assert owner.key_cache.shape == (6, 2, 4, 5)
    assert owner.value_cache.shape == (6, 2, 4, 5)
    assert owner.key_cache_grad == {}
    assert owner.value_cache_grad == {}


def test_general_attention_cached_chunk_grads_are_collected_and_consumed():
    owner = SimpleNamespace(
        config=SimpleNamespace(
            enable_chunkpipe=True,
            chunkpipe_forward=False,
            chunkpipe_backward_microbatch=1,
            sft_chunkpipe_mode=False,
            chunksize=2,
            max_chunk_size=2,
            cu_chunks=[0, 2, 4],
            chunk_sizes=[2, 2],
            micro_batch_size=1,
        ),
        num_chunks_per_seq=2,
        num_query_groups_per_partition=2,
        hidden_size_per_attention_head=3,
        key_cache=torch.zeros(4, 1, 2, 3),
        value_cache=torch.zeros(4, 1, 2, 3),
        key_cache_grad={},
        value_cache_grad={},
        micro_batch_to_cache_chunk_map={0: 0},
        empty_chunk_indices=[1],
        is_enable_grad_chunkpipe=lambda: True,
        layer_number=1,
    )
    owner.key_cache[:2] = torch.arange(12, dtype=torch.float32).reshape(2, 1, 2, 3)
    owner.value_cache[:2] = torch.arange(12, dtype=torch.float32).reshape(2, 1, 2, 3) + 20
    current_key = torch.ones(2, 1, 2, 3, requires_grad=True)
    current_value = torch.ones(2, 1, 2, 3, requires_grad=True)

    key, value, mask = Attention.concat_cached_chunk_key_value(
        owner,
        current_key,
        current_value,
        attention_mask=None,
    )

    assert mask is None
    (key.sum() + value.sum()).backward()
    torch.testing.assert_close(owner.key_cache_grad[0], torch.ones(2, 1, 2, 3))
    torch.testing.assert_close(owner.value_cache_grad[0], torch.ones(2, 1, 2, 3))

    owner.key_cache_grad.pop(0)
    owner.value_cache_grad.pop(0)
    Attention.check_kv_cache_grad_consumed(owner)


def test_dsa_indexer_cache_prefix_and_last_chunk():
    owner = _dsa_indexer_owner()
    key0 = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    key1 = key0 + 10
    key2 = key0 + 20

    out0 = DSAIndexer.cache_and_concat_chunk_indexer_keys(owner, key0)
    torch.testing.assert_close(out0, key0)
    assert 0 in owner._chunk_indexer_key_cache

    owner.config.chunkpipe_forward_microbatch = 1
    owner.config.chunkpipe_chunk_idx_in_group = 1
    out1 = DSAIndexer.cache_and_concat_chunk_indexer_keys(owner, key1)
    assert out1.shape == (4, 1, 4)
    torch.testing.assert_close(out1[:2], key0)
    torch.testing.assert_close(out1[2:], key1)

    owner.config.chunkpipe_forward_microbatch = 2
    owner.config.chunkpipe_chunk_idx_in_group = 2
    out2 = DSAIndexer.cache_and_concat_chunk_indexer_keys(owner, key2)
    torch.testing.assert_close(out2, key2)
    assert 2 not in owner._chunk_indexer_key_cache


def test_dsa_indexer_cache_rejects_missing_prefix_and_orphan_grad():
    owner = _dsa_indexer_owner()
    owner.config.chunkpipe_chunk_idx_in_group = 1
    owner.config.chunkpipe_forward_microbatch = 1

    with pytest.raises(RuntimeError, match="missing a previous chunk"):
        DSAIndexer.cache_and_concat_chunk_indexer_keys(
            owner, torch.ones(2, 1, 4)
        )

    owner._chunk_indexer_key_cache_grad[0] = torch.zeros(2, 1, 4)
    with pytest.raises(RuntimeError, match="gradient was not consumed"):
        DSAIndexer.delete_chunk_indexer_key_cache(owner, 0)

    DSAIndexer.clear_chunk_indexer_key_cache(owner)
    assert owner._chunk_indexer_key_cache == {}
    assert owner._chunk_indexer_key_cache_grad == {}
