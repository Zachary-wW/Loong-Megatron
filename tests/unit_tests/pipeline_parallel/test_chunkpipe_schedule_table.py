import megatron.core.pipeline_parallel.schedules as schedule


def test_extended_schedule_table_override_groups_chunks_in_order():
    table = schedule.get_extended_schedule_table(
        num_microbatches=8,
        num_model_chunks=2,
        microbatch_group_size_per_vp_stage=4,
        num_chunks=2,
        override_mgspvs=4,
    )

    assert len(table) == 16
    assert table[:4] == [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]
    assert table[4:8] == [(2, 0, 0), (2, 0, 1), (3, 0, 0), (3, 0, 1)]
    assert table[8:12] == [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]
    assert table[-4:] == [(2, 1, 0), (2, 1, 1), (3, 1, 0), (3, 1, 1)]


def test_extended_schedule_table_derives_group_size_from_pp_world(monkeypatch):
    monkeypatch.setattr(
        schedule.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: 4,
    )

    table = schedule.get_extended_schedule_table(
        num_microbatches=8,
        num_model_chunks=2,
        microbatch_group_size_per_vp_stage=4,
        num_chunks=2,
    )

    assert len(table) == 16
    assert table[:8] == [
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 0),
        (1, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    assert table[8:] == [
        (2, 0, 0),
        (2, 0, 1),
        (3, 0, 0),
        (3, 0, 1),
        (2, 1, 0),
        (2, 1, 1),
        (3, 1, 0),
        (3, 1, 1),
    ]
