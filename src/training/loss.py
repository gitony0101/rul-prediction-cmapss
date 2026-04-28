import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)


class LinExLoss(nn.Module):
    def __init__(self, a: float, clip_value: float = 50.0):
        super().__init__()
        self.a = a
        self.clip_value = clip_value

    def forward(self, pred, target):
        z = torch.clamp(
            self.a * (pred - target), min=-self.clip_value, max=self.clip_value
        )
        return torch.mean(torch.exp(z) - z - 1.0)
