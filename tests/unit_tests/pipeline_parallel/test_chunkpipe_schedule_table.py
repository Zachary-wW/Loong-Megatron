from types import SimpleNamespace

import torch

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


def test_noninterleaved_chunkpipe_forward_p2p_follows_chunk_order(monkeypatch):
    events = []
    config = SimpleNamespace(
        timers=None,
        no_sync_func=None,
        grad_sync_func=None,
        chunk_num_per_seq=2,
        sft_chunkpipe_mode=False,
        mtp_num_layers=0,
        overlap_p2p_comm=False,
        num_microbatches_with_partial_activation_checkpoints=None,
        deallocate_pipeline_outputs=False,
        finalize_model_grads_func=None,
        fine_grained_activation_offloading=False,
        calculate_per_token_loss=False,
        barrier_with_L1_time=False,
        cuda_graph_impl="none",
    )
    model = SimpleNamespace(config=config)
    model.model_type = "unit-test"
    fake_group = SimpleNamespace(size=lambda: 1)
    pg_collection = SimpleNamespace(
        tp=fake_group,
        cp=fake_group,
        embd=fake_group,
        pos_embd=fake_group,
        pp=fake_group,
        dp_cp=fake_group,
    )

    monkeypatch.setattr(schedule, "get_model_config", lambda _: config)
    monkeypatch.setattr(schedule, "get_tensor_shapes", lambda **_: [[1]])
    monkeypatch.setattr(schedule, "deallocate_output_tensor", lambda *_: None)
    monkeypatch.setattr(
        schedule,
        "clear_key_value_cache",
        lambda *_: events.append(("clear", config.chunkpipe_forward_microbatch)),
    )
    real_zeros = torch.zeros
    monkeypatch.setattr(
        schedule.torch,
        "zeros",
        lambda *args, **kwargs: real_zeros(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        ),
    )

    def fake_forward_step(*_args, **kwargs):
        microbatch = kwargs["current_microbatch"]
        events.append(
            (
                "forward",
                microbatch,
                config.chunkpipe_chunk_idx_in_group,
            )
        )
        return torch.tensor([microbatch]), torch.tensor(0)

    monkeypatch.setattr(schedule, "forward_step", fake_forward_step)

    class FakeGroup:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 1

    class FakeCommunicator:
        pp_group = FakeGroup()
        virtual_pipeline_model_parallel_size = 2

        def __init__(self):
            self.config = config
            self.recv_index = 0

        def recv_forward(self, *_args, **_kwargs):
            index = self.recv_index
            self.recv_index += 1
            events.append(("recv_forward", index))
            return torch.tensor([index])

        def send_forward(self, tensor, *_args, **_kwargs):
            events.append(("send_forward", int(tensor.flatten()[0])))

    schedule.forward_backward_pipelining_without_interleaving_chunkpipe(
        forward_step_func=None,
        data_iterator=None,
        model=model,
        num_microbatches=4,
        seq_length=1,
        micro_batch_size=1,
        forward_only=True,
        p2p_communicator=FakeCommunicator(),
        pg_collection=pg_collection,
    )

    assert events == [
        ("recv_forward", 0),
        ("forward", 0, 0),
        ("send_forward", 0),
        ("recv_forward", 1),
        ("forward", 1, 1),
        ("send_forward", 1),
        ("recv_forward", 2),
        ("clear", 1),
        ("forward", 2, 0),
        ("send_forward", 2),
        ("recv_forward", 3),
        ("forward", 3, 1),
        ("send_forward", 3),
        ("clear", 3),
    ]
