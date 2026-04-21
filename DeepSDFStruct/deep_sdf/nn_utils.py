import torch
import torch.nn as nn

activations = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
}


class ClampedL1Loss(torch.nn.Module):
    def __init__(self, clamp_val=0.1):
        super().__init__()
        self.clamp_val = clamp_val
        self.loss = torch.nn.L1Loss()

    def forward(self, input, target):
        input_clamped = input.clamp(-self.clamp_val, self.clamp_val)
        target_clamped = target.clamp(-self.clamp_val, self.clamp_val)
        return self.loss(input_clamped, target_clamped)


class LeakyClampedL1Loss(torch.nn.Module):
    """L1 loss that clamps values within [-clamp_val, clamp_val], but applies a
    leaky linear continuation outside that range instead of hard clamping.

    This behaves like a `clamp` in the center and a `LeakyReLU`-style slope
    outside. Both input and target are transformed the same way before L1.
    """

    def __init__(self, clamp_val=0.1, leak=0.01):
        super().__init__()
        self.clamp_val = clamp_val
        self.leak = leak
        self.loss = torch.nn.L1Loss()

    def _leaky_clamp(self, x: torch.Tensor) -> torch.Tensor:
        cv = self.clamp_val
        lk = self.leak
        # If x > c -> c + l*(x-c); if x < -c -> -c + l*(x + c); else x
        pos = torch.where(x > cv, cv + lk * (x - cv), x)
        out = torch.where(pos < -cv, -cv + lk * (pos + cv), pos)
        return out

    def forward(self, input, target):
        input_trans = self._leaky_clamp(input)
        target_trans = self._leaky_clamp(target)
        return self.loss(input_trans, target_trans)


def get_loss_function(loss_fun_spec: str):
    """Return a loss function instance for the given spec string."""
    if loss_fun_spec == "clampedL1":
        return ClampedL1Loss()
    elif loss_fun_spec == "leakyClampedL1":
        return LeakyClampedL1Loss()
    elif loss_fun_spec == "L1":
        return torch.nn.L1Loss()
    elif loss_fun_spec == "MSE":
        return torch.nn.MSELoss()
    elif loss_fun_spec == "huber":
        return torch.nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fun_spec}")
