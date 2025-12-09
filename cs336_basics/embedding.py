import math

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
