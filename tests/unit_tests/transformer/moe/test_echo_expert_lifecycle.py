# Copyright (c) 2026, Baidu, Inc. All rights reserved.

import torch

from megatron.core.transformer.moe.experts import TEGroupedMLP


class _WeightHolder(torch.nn.Module):
    def __init__(self, shapes):
        super().__init__()
        for index, shape in enumerate(shapes):
            setattr(self, f"weight{index}", torch.nn.Parameter(torch.zeros(shape)))
        self.use_bias = False


def test_echo_expert_weight_get_set_free_lifecycle():
    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.linear_fc1 = _WeightHolder([(4, 3), (4, 3)])
    module.linear_fc2 = _WeightHolder([(5, 4), (5, 4)])
    module.fc1_weight_shape = module.linear_fc1.weight0.shape
    module.fc2_weight_shape = module.linear_fc2.weight0.shape

    fc1_original = module.get_expert_weights("fc1", [0])
    assert fc1_original[0] is module.linear_fc1.weight0

    replacement = torch.nn.Parameter(torch.ones(4, 3))
    module.set_expert_weights("fc1", [replacement], [1])
    assert module.linear_fc1.weight1 is replacement

    empty = torch.empty(0, requires_grad=True)
    module.set_expert_weights("fc2", [empty], [0])
    placeholder = module.linear_fc2.weight0
    assert placeholder.shape == module.fc2_weight_shape
    assert placeholder.grad_fn is not None

    module.free_expert_parameters([1])
    assert not hasattr(module.linear_fc1, "weight1")
    assert not hasattr(module.linear_fc2, "weight1")
    assert hasattr(module.linear_fc1, "weight0")
    assert hasattr(module.linear_fc2, "weight0")
