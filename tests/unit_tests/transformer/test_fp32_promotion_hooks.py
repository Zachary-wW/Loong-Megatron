from types import SimpleNamespace

import torch

from megatron.core.extensions import transformer_engine as transformer_engine_module
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.module import restore_fp16module_inputs_to_fp32


class _ProbeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.seen_input_dtype = None

    def forward(self, value):
        self.seen_input_dtype = value.dtype
        return value


def test_restore_fp16module_inputs_promotes_and_restores_output():
    module = _ProbeModule()
    restore_fp16module_inputs_to_fp32(
        module,
        SimpleNamespace(fp16=False, bf16=True),
        weight_is_fp32=lambda _module: True,
    )

    output = module(torch.ones(2, 3, dtype=torch.bfloat16))

    assert module.seen_input_dtype is torch.float32
    assert output.dtype is torch.bfloat16


def test_restore_fp16module_inputs_can_keep_fp32_output():
    module = _ProbeModule()
    restore_fp16module_inputs_to_fp32(
        module,
        SimpleNamespace(fp16=False, bf16=True),
        weight_is_fp32=lambda _module: True,
        keep_fp32_outputs=True,
    )

    output = module(torch.ones(2, 3, dtype=torch.bfloat16))

    assert module.seen_input_dtype is torch.float32
    assert output.dtype is torch.float32


def test_restore_fp16module_inputs_noop_without_predicate():
    module = _ProbeModule()
    restore_fp16module_inputs_to_fp32(
        module,
        SimpleNamespace(fp16=False, bf16=True),
        weight_is_fp32=None,
    )

    output = module(torch.ones(2, 3, dtype=torch.bfloat16))

    assert module.seen_input_dtype is torch.bfloat16
    assert output.dtype is torch.bfloat16


def test_column_parallel_linear_promotes_input_for_fp32_weight():
    captured = {}
    owner = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32)),
        bias=None,
        skip_bias_add=True,
        allreduce_dgrad=True,
        sequence_parallel=False,
        explicit_expert_comm=False,
        disable_grad_reduce=False,
        gather_output=False,
        gradient_accumulation_fusion=False,
        grad_output_buffer=None,
        gtp_remat_size=1,
        output_dtype=torch.bfloat16,
        output_size_per_partition=2,
        input_size=3,
        tp_group=None,
        config=SimpleNamespace(
            defer_embedding_wgrad_compute=False,
            _cpu_offloading_context=None,
            cpu_offloading=False,
            cpu_offloading_activations=False,
        ),
    )

    def forward_impl(**kwargs):
        captured["input"] = kwargs["input"]
        return kwargs["input"]

    owner._forward_impl = forward_impl
    output, bias = ColumnParallelLinear.forward(
        owner,
        torch.ones(2, 3, dtype=torch.bfloat16),
    )

    assert bias is None
    assert captured["input"].dtype is torch.float32
    assert output.dtype is torch.float32


def test_te_norm_consumer_promotes_fp32_weight_inputs(monkeypatch):
    class DummyLayerNorm(torch.nn.Module):
        def __init__(self, normalized_shape, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.ones(normalized_shape, dtype=torch.float32)
            )
            self.seen_input_dtype = None

        def forward(self, value):
            self.seen_input_dtype = value.dtype
            return value

    monkeypatch.setattr(
        transformer_engine_module.te.pytorch,
        "LayerNorm",
        DummyLayerNorm,
    )
    config = SimpleNamespace(
        normalization="LayerNorm",
        fused_residual_rmsnorm=False,
        sequence_parallel=False,
        layernorm_zero_centered_gamma=False,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        init_model_with_meta_device=False,
        fp16=False,
        bf16=True,
    )

    norm = TENorm(config, hidden_size=8)
    output = norm(torch.ones(2, 8, dtype=torch.bfloat16))

    assert norm.seen_input_dtype is torch.float32
    assert output.dtype is torch.bfloat16
