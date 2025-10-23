import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()

        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.transpose(-1, -2)
