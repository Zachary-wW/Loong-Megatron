from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossLoggingHelper,
)


def test_save_indexer_loss_slots_can_grow_past_decoder_layers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    tracker = {}
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)

    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
        torch.tensor(2.0), layer_number=1, num_layers=3
    )
    assert tracker["values"].shape == (3,)
    torch.testing.assert_close(tracker["values"], torch.tensor([2.0, 0.0, 0.0]))

    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
        torch.tensor(5.0), layer_number=4, num_layers=3
    )
    assert tracker["values"].shape == (4,)
    torch.testing.assert_close(tracker["values"], torch.tensor([2.0, 0.0, 0.0, 5.0]))


def test_save_indexer_loss_without_layer_number_is_a_noop(monkeypatch):
    tracker = {}
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)

    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
        torch.tensor(2.0), layer_number=None, num_layers=3
    )

    assert tracker == {}


@pytest.mark.parametrize(("num_indexer_layers", "expected"), [(2, 4.0), (1, 8.0)])
def test_track_indexer_metrics_logs_average_and_preserves_groups(
    monkeypatch, num_indexer_layers, expected
):
    reduce_group = object()
    avg_group = object()
    tracker = {
        "values": torch.tensor([1.0, 3.0]),
        "reduce_group": reduce_group,
        "avg_group": avg_group,
    }
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
    monkeypatch.setattr(
        DSAIndexerLossLoggingHelper,
        "reduce_loss_in_tracker",
        lambda pg_collection, num_layers=None: None,
    )
    writer = MagicMock()
    wandb_writer = MagicMock()
    total_loss_dict = {}

    DSAIndexerLossLoggingHelper.track_indexer_metrics(
        loss_scale=2.0,
        iteration=11,
        writer=writer,
        pg_collection=SimpleNamespace(),
        wandb_writer=wandb_writer,
        total_loss_dict=total_loss_dict,
        num_indexer_layers=num_indexer_layers,
        preserve_groups=True,
    )

    torch.testing.assert_close(total_loss_dict["indexer loss"], torch.tensor(expected))
    writer.add_scalar.assert_called_once()
    assert writer.add_scalar.call_args.args[0] == "indexer loss"
    assert writer.add_scalar.call_args.args[2] == 11
    wandb_writer.log.assert_called_once()
    torch.testing.assert_close(tracker["values"], torch.zeros(2))
    assert tracker["reduce_group"] is reduce_group
    assert tracker["avg_group"] is avg_group


def test_reduce_indexer_loss_uses_pp_reduce_and_dp_groups(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    pp_group = object()
    dp_group = object()
    reduce_group = object()
    avg_group = object()
    tracker = {
        "values": torch.tensor([1.0, 3.0]),
        "reduce_group": reduce_group,
        "avg_group": avg_group,
        "agreed_size": 2,
    }
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, group=None, op=None: calls.append((group, op)),
    )

    DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
        SimpleNamespace(pp=pp_group, dp=dp_group), num_layers=2
    )

    assert [group for group, _ in calls] == [pp_group, reduce_group, avg_group, dp_group]
    assert calls[0][1] is None
    assert calls[1][1] is None
    assert calls[2][1] is torch.distributed.ReduceOp.AVG
    assert calls[3][1] is torch.distributed.ReduceOp.AVG
