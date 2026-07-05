# Dynamic learning rate scheduler for Transformer training,
# implementing the learning rate schedule described in the original Transformer paper.

import torch


class NoamOpt:
    """Optimizer wrapper implementing the Noam learning-rate schedule.

    lrate = d_model^(-0.5) * min(step_num^(-0.5),
                                  step_num * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    # -- delegate essential Optimizer methods --

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "step": self._step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._step = state_dict["step"]

    # -- step with schedule --

    def step(self) -> None:
        self._step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()

    def _compute_lr(self) -> float:
        step = self._step
        factor = self.d_model ** (-0.5)
        warmup = step * (self.warmup_steps ** (-1.5))
        decay = step ** (-0.5)
        return factor * min(warmup, decay)

    @property
    def current_lr(self) -> float:
        return self._compute_lr()

    @property
    def current_step(self) -> int:
        return self._step
