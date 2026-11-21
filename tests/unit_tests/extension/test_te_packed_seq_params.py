from dataclasses import fields
from types import SimpleNamespace

from packaging.version import Version
import torch

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    _get_te_kept_packed_seq_params,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType


INTERNAL_FIELDS = {
    "total_tokens",
    "seq_idx",
    "tokens_per_sample",
    "cp_scatter_cache",
    "cu_seqlens_cpu",
}


def _all_fields():
    return {field.name for field in fields(PackedSeqParams)}


def test_te_packed_seq_param_filter_legacy_1_2():
    kept = _get_te_kept_packed_seq_params(Version("1.2.0"))

    assert kept == _all_fields() - INTERNAL_FIELDS - {
        "max_seqlen_q",
        "max_seqlen_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "pad_between_seqs",
    }


def test_te_packed_seq_param_filter_1_10():
    kept = _get_te_kept_packed_seq_params(Version("1.10.0"))

    assert kept == _all_fields() - INTERNAL_FIELDS - {"pad_between_seqs"}
    assert "max_seqlen_q" in kept
    assert "cu_seqlens_q_padded" in kept


def test_te_packed_seq_param_filter_2_2():
    kept = _get_te_kept_packed_seq_params(Version("2.2.0"))

    assert kept == _all_fields() - INTERNAL_FIELDS
    assert "pad_between_seqs" in kept
    assert "cu_seqlens_cpu" not in kept


def test_te_attention_consumer_drops_cpu_and_internal_metadata(monkeypatch):
    captured = {}
    base_class = TEDotProductAttention.__mro__[1]

    def fake_te_forward(self, query, key, value, attention_mask, **kwargs):
        captured.update(kwargs)
        return query

    monkeypatch.setattr(base_class, "forward", fake_te_forward)
    attention = object.__new__(TEDotProductAttention)
    attention.config = SimpleNamespace(
        window_size=None,
        qk_clip=False,
        log_max_attention_logit=False,
    )
    attention.qkv_format = "sbhd"
    attention.num_splits = None
    attention.te_forward_mask_type = True
    attention.kept_packed_seq_params = _get_te_kept_packed_seq_params(Version("2.2.0"))
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_cpu=torch.tensor([0, 2], dtype=torch.int32),
        max_seqlen_q=2,
        max_seqlen_kv=2,
        total_tokens=2,
        tokens_per_sample=torch.tensor([2]),
    )

    query = torch.zeros(2, 1, 4)
    result = attention._forward(
        query,
        torch.zeros_like(query),
        torch.zeros_like(query),
        None,
        AttnMaskType.causal,
        packed_seq_params=packed,
    )

    assert result is query
    assert captured["cu_seqlens_q"] is packed.cu_seqlens_q
    assert captured["qkv_format"] == "thd"
    assert "cu_seqlens_cpu" not in captured
    assert "total_tokens" not in captured
    assert "tokens_per_sample" not in captured
