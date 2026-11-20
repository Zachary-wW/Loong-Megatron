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
