from dataclasses import fields

from packaging.version import Version

from megatron.core.extensions.transformer_engine import _get_te_kept_packed_seq_params
from megatron.core.packed_seq_params import PackedSeqParams


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
