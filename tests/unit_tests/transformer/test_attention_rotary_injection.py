import copy

import torch

import megatron.core.transformer.attention as attention_module
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from tests.unit_tests.test_utilities import Utils


class TestAttentionRotaryInjection:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def _submodules(self):
        return copy.deepcopy(
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
        )

    def test_custom_rotary_callable_is_used_for_query_and_key(self, monkeypatch):
        captured = []

        def custom_rotary(tensor, freqs, **kwargs):
            captured.append((tensor.shape, freqs.shape, kwargs))
            return tensor

        submodules = self._submodules()
        submodules.apply_rotary_fn = custom_rotary
        attention = SelfAttention(
            self.config,
            submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        assert attention.apply_rotary_fn is custom_rotary

        default_attention = SelfAttention(
            self.config,
            self._submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        assert default_attention.apply_rotary_fn is attention_module.apply_rotary_pos_emb

        def fake_core_attention_forward(module_self, query, *args, **kwargs):
            return torch.zeros(
                (query.shape[0], query.shape[1], self.config.hidden_size),
                dtype=query.dtype,
                device=query.device,
            )

        monkeypatch.setattr(
            type(attention.core_attention), "forward", fake_core_attention_forward
        )
        attention.cuda().eval()
        hidden_states = torch.ones(
            (8, 1, self.config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        rotary_pos_emb = RotaryEmbedding(kv_channels=16, rotary_percent=1.0)(8).cuda()

        attention(hidden_states, None, rotary_pos_emb=rotary_pos_emb)

        assert len(captured) == 2
        for tensor_shape, freqs_shape, kwargs in captured:
            assert tensor_shape[-1] == 16
            assert freqs_shape == (8, 1, 1, 16)
            assert kwargs["config"] is self.config
