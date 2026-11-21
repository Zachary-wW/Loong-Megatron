from types import SimpleNamespace

import torch

from megatron.training import checkpointing
from megatron.training.checkpointing import maybe_save_dataloader_state


class _DummyIterable:
    is_save_state_rank_independent = False

    def __init__(self):
        self.calls = 0

    def save_state(self):
        self.calls += 1
        return {"cursor": 7}


class _DummyIterator:
    def __init__(self):
        self.iterable = _DummyIterable()


def test_vpp_dataloader_state_unwraps_first_iterator(monkeypatch, tmp_path):
    first = _DummyIterator()
    second = _DummyIterator()
    saved = {}

    monkeypatch.setattr(
        "megatron.core.mpu.get_virtual_pipeline_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "megatron.core.mpu.get_context_parallel_group",
        lambda check_initialized=False: None,
    )
    monkeypatch.setattr(
        "megatron.core.mpu.is_pipeline_first_stage",
        lambda ignore_virtual=True: True,
    )
    monkeypatch.setattr(
        "megatron.core.mpu.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr("megatron.core.mpu.get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "megatron.core.mpu.get_data_parallel_group", lambda: None
    )
    monkeypatch.setattr(
        "megatron.core.mpu.get_pipeline_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(
        checkpointing,
        "get_checkpoint_name",
        lambda *args, **kwargs: str(tmp_path / kwargs["basename"]),
    )
    monkeypatch.setattr(checkpointing, "ensure_directory_exists", lambda _: None)
    monkeypatch.setattr(
        checkpointing.torch,
        "save",
        lambda value, path: saved.update({"value": value, "path": path}),
    )

    maybe_save_dataloader_state(
        [first, second],
        iteration=3,
        dataloader_save_path=str(tmp_path),
    )

    assert first.iterable.calls == 1
    assert second.iterable.calls == 0
    assert saved["path"].endswith("train_dataloader_dprank000.pt")
    assert saved["value"] == {"dataloader_state_dict": {"cursor": 7}}
