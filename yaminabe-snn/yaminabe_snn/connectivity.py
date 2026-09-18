"""Connection presence is independent of trainable connection strength."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def make_mask(n: int, probability: float = 1.0) -> torch.Tensor:
    if n < 1 or not 0 <= probability <= 1:
        raise ValueError("n must be positive and probability must be in [0, 1]")
    mask = (torch.rand(n, n) < probability).float()
    mask.fill_diagonal_(0)
    return mask


def load_mask(path: str | Path, n: int) -> torch.Tensor:
    """Read binary adjacency[post, pre]; not a full connectome converter."""
    array = np.load(path, allow_pickle=False)
    if array.shape != (n, n):
        raise ValueError(f"Expected adjacency shape {(n, n)}, got {array.shape}")
    return validate_mask(torch.as_tensor(array, dtype=torch.float32), n)


def validate_mask(mask: torch.Tensor, n: int) -> torch.Tensor:
    if mask.shape != (n, n) or not torch.isfinite(mask).all():
        raise ValueError("Adjacency must be a finite square matrix of the configured size")
    if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask.diag() != 0):
        raise ValueError("Adjacency must contain only 0/1, with a zero diagonal")
    return mask.detach().clone().float()


class RecurrentConnections(nn.Module):
    def __init__(self, mask: torch.Tensor, row_bound: float | None = 2.0):
        super().__init__()
        n = mask.shape[0]
        self.register_buffer("mask", validate_mask(mask, n))
        if row_bound is not None and (not np.isfinite(row_bound) or row_bound <= 0):
            raise ValueError("row_bound must be positive or None")
        self.row_bound = row_bound
        self.weight = nn.Parameter(torch.empty(n, n))
        nn.init.normal_(self.weight, std=0.2 / max(n, 1) ** 0.5)

    def effective_weight(self) -> torch.Tensor:
        weight = self.weight * self.mask
        if self.row_bound is not None:
            # A transparent engineering constraint, not a physiological law.
            scale = (weight.abs().sum(dim=1, keepdim=True) / self.row_bound).clamp_min(1)
            weight = weight / scale
        return weight

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # F.linear(x, W) computes x @ W.T: W[post, pre].
        return torch.nn.functional.linear(spikes, self.effective_weight())
