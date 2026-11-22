from types import SimpleNamespace

import torch

import megatron.core.transformer.multi_token_prediction as mtp_module
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer


def test_chunkpipe_mtp_consumer_preserves_position_ids():
    owner = SimpleNamespace(
        config=SimpleNamespace(
            enable_chunkpipe=True,
            sequence_parallel=False,
            attention_cp_layout="contiguous",
            mtp_detach_heads=False,
        ),
        cp_group=None,
        tp_group=None,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.tensor([[11, 12, 13]])
    hidden_states = torch.zeros(3, 1, 4)
    captured = {}

    def embedding(input_ids, position_ids):
        captured["input_ids"] = input_ids
        captured["position_ids"] = position_ids
        return torch.zeros(3, 1, 4)

    outputs = MultiTokenPredictionLayer._get_embeddings(
        owner,
        input_ids=input_ids,
        position_ids=position_ids,
        embedding=embedding,
        hidden_states=hidden_states,
    )

    assert outputs[0] is input_ids
    assert outputs[1] is position_ids
    assert captured["input_ids"] is input_ids
    assert captured["position_ids"] is position_ids


def test_non_chunkpipe_mtp_rolls_inputs_before_embedding(monkeypatch):
    owner = SimpleNamespace(
        config=SimpleNamespace(
            enable_chunkpipe=False,
            sequence_parallel=False,
            attention_cp_layout="zigzag",
            mtp_detach_heads=False,
        ),
        cp_group=None,
        tp_group=None,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.tensor([[11, 12, 13]])
    hidden_states = torch.zeros(3, 1, 4)
    captured = {}

    def fake_roll(tensor, **kwargs):
        return tensor + 100, None

    def embedding(input_ids, position_ids):
        captured["input_ids"] = input_ids
        captured["position_ids"] = position_ids
        return torch.zeros(3, 1, 4)

    monkeypatch.setattr(mtp_module, "roll_tensor", fake_roll)
    outputs = MultiTokenPredictionLayer._get_embeddings(
        owner,
        input_ids=input_ids,
        position_ids=position_ids,
        embedding=embedding,
        hidden_states=hidden_states,
    )

    torch.testing.assert_close(outputs[0], input_ids + 100)
    torch.testing.assert_close(outputs[1], position_ids + 100)
    torch.testing.assert_close(captured["input_ids"], input_ids + 100)
    torch.testing.assert_close(captured["position_ids"], position_ids + 100)


def test_chunkpipe_mtp_recompute_retains_trailing_chunk():
    owner = SimpleNamespace(
        config=SimpleNamespace(
            enable_chunkpipe=True,
            sft_chunkpipe_mode=False,
            chunk_num_per_seq=4,
            keep_activations_chunks=1,
            chunkpipe_forward_microbatch=0,
            recompute_granularity=None,
        ),
        training=True,
        mtp_layer_pattern=None,
    )
    checkpointed_microbatches = []

    def get_precomputed_embeddings(decoder_input, hidden_states):
        return decoder_input, hidden_states

    def checkpointed_forward(**kwargs):
        checkpointed_microbatches.append(owner.config.chunkpipe_forward_microbatch)
        return kwargs["hidden_states"]

    def project_and_transformer_layer(**kwargs):
        return kwargs["hidden_states"]

    owner._get_precomputed_embeddings = get_precomputed_embeddings
    owner._checkpointed_forward = checkpointed_forward
    owner._proj_and_transformer_layer = project_and_transformer_layer

    for microbatch in range(4):
        owner.config.chunkpipe_forward_microbatch = microbatch
        hidden_states = torch.ones(2, 1, 4)
        outputs = MultiTokenPredictionLayer.forward(
            owner,
            input_ids=None,
            position_ids=None,
            hidden_states=hidden_states,
            attention_mask=None,
            decoder_input=torch.zeros(2, 1, 4),
        )
        torch.testing.assert_close(outputs[0], hidden_states)
        assert outputs[1:] == (None, None, None, None)

    assert checkpointed_microbatches == [0, 1, 2]
