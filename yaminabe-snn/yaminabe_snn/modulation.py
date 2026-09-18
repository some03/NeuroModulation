"""Replaceable parameter modulation; no LC/NE physiology is assumed here."""

import torch
from torch import nn


class NoModulation(nn.Module):
    def forward(self, time_ms, previous_spikes, threshold, tau_mem_ms):
        return threshold, tau_mem_ms


class StepThreshold(nn.Module):
    """External threshold intervention for debugging, not an adaptive controller."""

    def __init__(self, at_ms: float, factor: float = 1.5):
        super().__init__()
        if at_ms < 0 or factor <= 0:
            raise ValueError("at_ms must be nonnegative and factor must be positive")
        self.at_ms = at_ms
        self.factor = factor

    def forward(self, time_ms, previous_spikes, threshold, tau_mem_ms):
        factor = self.factor if time_ms >= self.at_ms else 1.0
        return threshold * factor, tau_mem_ms


def positive_parameters(threshold: torch.Tensor, tau_mem_ms: torch.Tensor):
    # Numeric guard for future custom controllers. Values use normalized voltage / ms.
    return threshold.clamp_min(1e-4), tau_mem_ms.clamp_min(1e-4)
