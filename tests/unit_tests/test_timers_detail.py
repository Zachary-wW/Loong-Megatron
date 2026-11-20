import megatron.core.timers as timers_module
from megatron.core.timers import DummyTimer, Timer, Timers


def test_detail_timer_requires_explicit_enable():
    timers = Timers(log_level=0, log_option="minmax")

    assert isinstance(timers("detail", log_level=2), DummyTimer)
    timers.set_show_detail_log(True)
    assert isinstance(timers("detail", log_level=2), Timer)


def test_detail_timer_name_is_cached():
    timers = Timers(log_level=0, log_option="minmax")
    timers.set_show_detail_log(True)

    first = timers("detail", log_level=2)
    second = timers("detail", log_level=2)
    assert first is second


def test_timer_writer_emits_min_and_max_keys(monkeypatch):
    class FakeWriter:
        def __init__(self):
            self.calls = []

        def add_scalar(self, name, value, iteration):
            self.calls.append((name, value, iteration))

    timers = Timers(log_level=0, log_option="minmax")
    monkeypatch.setattr(timers_module, "SummaryWriter", FakeWriter)
    monkeypatch.setattr(
        timers, "_get_global_min_max_time", lambda *args: {"step": (0.001, 0.002)}
    )
    writer = FakeWriter()
    timers.write(["step"], writer, iteration=7)

    assert writer.calls == [
        ("step-time/min", 0.001, 7),
        ("step-time/max", 0.002, 7),
    ]
