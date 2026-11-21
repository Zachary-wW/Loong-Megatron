import torch

from megatron.core.utils import get_batch_on_this_tp_rank


def test_chunkpipe_variable_sequence_and_runtime_metadata_tp_broadcast(monkeypatch):
    sent = []
    receiver_mode = False

    def fake_broadcast(tensor, src, group=None):
        if receiver_mode:
            tensor.copy_(sent.pop(0))
        else:
            sent.append(tensor.clone())

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    source_batch = {
        "tokens": torch.arange(10, dtype=torch.int64).reshape(2, 5),
        "labels": torch.arange(10, 20, dtype=torch.int64).reshape(2, 5),
        "loss_mask": torch.ones(2, 5),
        "position_ids": torch.arange(5).expand(2, 5),
        "cu_seqlens": torch.tensor([[0, 2, 5], [0, 3, 5]], dtype=torch.int32),
        "max_seqlen": torch.tensor([5, 5], dtype=torch.int32),
        "chunkpipe_current_group_size": 2,
        "chunkpipe_composite_group_size": 3,
        "chunkpipe_component_sizes": [2, 1],
    }
    common = dict(
        has_cu_seqlens=True,
        is_hybrid_cp=False,
        create_attention_mask_in_dataloader=False,
        broadcast_src_rank=0,
        broadcast_group=object(),
        cp_size=1,
        micro_batch_size=2,
        seq_length=5,
        mtp_on_this_rank=True,
        pipeline_model_parallel_size=1,
        chunkpipe_variable_seq_lengths=True,
        chunkpipe_runtime_metadata=True,
    )

    get_batch_on_this_tp_rank(source_batch, tp_rank=0, **common)
    receiver_mode = True
    received = get_batch_on_this_tp_rank({}, tp_rank=1, **common)

    assert sent == []
    torch.testing.assert_close(received["tokens"], source_batch["tokens"])
    torch.testing.assert_close(received["labels"], source_batch["labels"])
    torch.testing.assert_close(received["loss_mask"], source_batch["loss_mask"])
    torch.testing.assert_close(received["position_ids"], source_batch["position_ids"])
    torch.testing.assert_close(received["cu_seqlens"], source_batch["cu_seqlens"])
    torch.testing.assert_close(received["max_seqlen"], source_batch["max_seqlen"])
    assert received["chunkpipe_current_group_size"] == 2
    assert received["chunkpipe_composite_group_size"] == 3
    assert received["chunkpipe_component_sizes"] == [2, 1]
