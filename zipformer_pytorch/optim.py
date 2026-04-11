from __future__ import annotations

import math

import torch
from torch import Tensor


class ScaledAdam(torch.optim.Optimizer):
    """Paper-style ScaledAdam without the extra batching/diagnostics utilities."""

    def __init__(
        self,
        params,
        lr: float = 4.5e-2,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1.0e-8,
        scale_lr: float = 0.1,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "scale_lr": scale_lr,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            scale_lr = float(group["scale_lr"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("ScaledAdam does not support sparse gradients.")
                param_before = parameter.detach().clone()

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if parameter.ndim > 1 and parameter.numel() > 1:
                        state["scale_exp_avg"] = torch.zeros(
                            (), device=parameter.device, dtype=torch.float32
                        )
                        state["scale_exp_avg_sq"] = torch.zeros(
                            (), device=parameter.device, dtype=torch.float32
                        )

                state["step"] += 1
                step = int(state["step"])
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction = math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
                denom = exp_avg_sq.sqrt().add_(eps)

                if parameter.ndim < 2:
                    parameter.add_(exp_avg / denom, alpha=-lr * bias_correction)
                    continue

                param_rms = param_before.float().pow(2).mean().sqrt()
                parameter.add_(
                    exp_avg / denom,
                    alpha=-lr * bias_correction * float(param_rms.clamp_min(1.0e-8)),
                )

                scale_exp_avg: Tensor = state["scale_exp_avg"]
                scale_exp_avg_sq: Tensor = state["scale_exp_avg_sq"]
                scale_grad = torch.sum(grad.float() * param_before.float())
                scale_exp_avg.mul_(beta1).add_(scale_grad, alpha=1.0 - beta1)
                scale_exp_avg_sq.mul_(beta2).add_(scale_grad.square(), alpha=1.0 - beta2)
                scale_denom = scale_exp_avg_sq.sqrt().add_(eps)
                scale_step = -scale_lr * lr * bias_correction * scale_exp_avg / scale_denom
                parameter.add_(param_before, alpha=float(scale_step))

        return loss
