from types import SimpleNamespace

import pytest

from megatron.training import checkpointing


class _StopAfterDirectorySelection(Exception):
    pass


@pytest.mark.parametrize(
    ("save_arg", "expected"),
    [("save", "/tmp/default-save"), ("save2", "/tmp/alternate-save")],
)
def test_save_checkpoint_uses_requested_save_argument(monkeypatch, save_arg, expected):
    args = SimpleNamespace(
        async_save=False,
        use_dist_ckpt=False,
        save="/tmp/default-save",
        save2="/tmp/alternate-save",
        ckpt_format="torch",
        freeze_all_layers=False,
        use_distributed_optimizer=False,
        no_save_optim=True,
    )
    captured = {}

    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "on_save_checkpoint_start", lambda _: None)
    monkeypatch.setattr(checkpointing.ft_integration, "on_checkpointing_start", lambda: None)
    monkeypatch.setattr(checkpointing, "unwrap_model", lambda model: model)
    monkeypatch.setattr(checkpointing, "print_rank_0", lambda *_: None)
    monkeypatch.setattr(checkpointing, "get_rng_state", lambda *_, **__: None)
    monkeypatch.setattr(
        checkpointing,
        "get_rerun_state_machine",
        lambda: SimpleNamespace(state_dict=lambda **__: None),
    )

    def stop_at_checkpoint_name(save_dir, *unused, **unused_kwargs):
        captured["save_dir"] = save_dir
        raise _StopAfterDirectorySelection

    monkeypatch.setattr(checkpointing, "get_checkpoint_name", stop_at_checkpoint_name)

    with pytest.raises(_StopAfterDirectorySelection):
        checkpointing.save_checkpoint(
            model=[],
            optimizer=None,
            opt_param_scheduler=None,
            iteration=7,
            num_floating_point_operations_so_far=0,
            tp_group=object(),
            pp_group=object(),
            save_arg=save_arg,
        )

    assert captured["save_dir"] == expected
