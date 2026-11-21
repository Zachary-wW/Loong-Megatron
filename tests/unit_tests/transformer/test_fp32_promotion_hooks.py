from types import SimpleNamespace

import torch

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
