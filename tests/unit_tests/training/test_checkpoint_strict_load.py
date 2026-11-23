import pytest

from megatron.training.checkpointing import load_model_state_dict


class _StrictFallbackModule:
    def __init__(self):
        self.calls = []

    def load_state_dict(self, state_dict, strict=True):
        self.calls.append((state_dict, strict))
        if strict:
            raise RuntimeError("missing/unexpected keys")
        return "fallback-loaded"


class _FailingModule:
    def __init__(self):
        self.calls = []

    def load_state_dict(self, state_dict, strict=True):
        self.calls.append((state_dict, strict))
        raise RuntimeError(f"load failed strict={strict}")


def test_strict_load_failure_retries_non_strict(monkeypatch):
    monkeypatch.setattr(
        "megatron.core.tensor_parallel.gtp_api.HAVE_GTP", False
    )
    module = _StrictFallbackModule()
    state_dict = {"weight": 1}

    load_model_state_dict(module, state_dict, strict=True)

    assert module.calls == [(state_dict, True), (state_dict, False)]


def test_non_strict_load_failure_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        "megatron.core.tensor_parallel.gtp_api.HAVE_GTP", False
    )
    module = _FailingModule()

    with pytest.raises(RuntimeError, match="strict=False"):
        load_model_state_dict(module, {"weight": 1}, strict=False)

    assert module.calls == [({"weight": 1}, False)]
