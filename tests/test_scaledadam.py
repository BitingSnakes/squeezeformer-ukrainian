from __future__ import annotations

import torch

from zipformer_pytorch.optim import ScaledAdam


def test_scaledadam_updates_zero_initialized_bias_vector() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.tensor([1.0, -1.0, 0.5, -0.5])

    optimizer = ScaledAdam([parameter], lr=0.045)
    optimizer.step()

    assert torch.all(parameter.detach().abs() > 1.0e-3)
    assert "scale_exp_avg" not in optimizer.state[parameter]


def test_scaledadam_keeps_scaled_updates_for_matrix_parameters() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    parameter.grad = torch.ones_like(parameter)

    optimizer = ScaledAdam([parameter], lr=0.045)
    optimizer.step()

    assert torch.all(parameter.detach() < 1.0)
    assert "scale_exp_avg" in optimizer.state[parameter]
