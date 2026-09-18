"""Batched recurrent LIF dynamics. Neuron dimension is vectorized; time is sequential."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .connectivity import RecurrentConnections, make_mask, validate_mask
from .modulation import NoModulation, positive_parameters
from .neurons import lif_step


@dataclass
class NetworkConfig:
    n_inputs: int = 700
    n_hidden: int = 256
    n_outputs: int = 20
    dt_ms: float = 1.0
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    tau_readout_ms: float = 30.0
    threshold: float = 1.0
    connection_probability: float = 1.0
    recurrent_row_bound: float | None = 2.0

    def __post_init__(self):
        for name in ("n_inputs", "n_hidden", "n_outputs"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dt_ms", "tau_mem_ms", "tau_syn_ms", "tau_readout_ms", "threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.connection_probability <= 1:
            raise ValueError("connection_probability must be in [0, 1]")
        if self.recurrent_row_bound is not None and (not math.isfinite(self.recurrent_row_bound) or self.recurrent_row_bound <= 0):
            raise ValueError("recurrent_row_bound must be positive or None")


@dataclass
class NetworkOutput:
    logits: torch.Tensor
    mean_spikes_per_step: torch.Tensor
    trace: dict[str, torch.Tensor] | None


class RecurrentLIF(nn.Module):
    def __init__(self, config: NetworkConfig, mask=None, modulator=None):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.n_inputs, config.n_hidden, bias=False)
        # Initially positive input currents keep the small demo active.
        # Training may change signs; Dale's law is not enforced in this first version.
        nn.init.uniform_(self.input_projection.weight, 0, 2 / math.sqrt(config.n_inputs))
        if mask is None:
            mask = make_mask(config.n_hidden, config.connection_probability)
        self.recurrent = RecurrentConnections(validate_mask(mask, config.n_hidden), config.recurrent_row_bound)
        self.readout = nn.Linear(config.n_hidden, config.n_outputs)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer("threshold", torch.full((config.n_hidden,), config.threshold))
        self.register_buffer("tau_mem_ms", torch.full((config.n_hidden,), config.tau_mem_ms))
        self.modulator = modulator if modulator is not None else NoModulation()

    def forward(self, inputs, record=False):
        """inputs: nonnegative spike counts [batch, time, input_channel].

        Each call starts a new independent trial. Trace records batch item 0 only.
        logits are the final leaky-readout state, not a probability.
        """
        cfg = self.config
        if not inputs.is_floating_point():
            raise ValueError("Input spike counts must use a floating-point tensor")
        if inputs.ndim != 3 or inputs.shape[-1] != cfg.n_inputs or inputs.shape[1] < 1:
            raise ValueError("Expected [batch, nonempty time, n_inputs]")
        batch, steps, _ = inputs.shape
        if batch < 1:
            raise ValueError("Batch must not be empty")
        current = inputs.new_zeros(batch, cfg.n_hidden)
        voltage = torch.zeros_like(current)
        spikes = torch.zeros_like(current)
        output = inputs.new_zeros(batch, cfg.n_outputs)
        beta = math.exp(-cfg.dt_ms / cfg.tau_syn_ms)
        gamma = math.exp(-cfg.dt_ms / cfg.tau_readout_ms)
        spike_total = inputs.new_zeros(())
        # Reuse this graph within the trial rather than normalizing N x N weights T times.
        recurrent_weight = self.recurrent.effective_weight()
        traces = {k: [] for k in ("voltage", "spikes", "threshold", "readout")} if record else None
        for t in range(steps):
            theta, tau = self.modulator(t * cfg.dt_ms, spikes, self.threshold, self.tau_mem_ms)
            theta, tau = positive_parameters(theta, tau)
            # Recurrent drive uses the PREVIOUS step's spikes (one dt delay).
            current = (beta * current + self.input_projection(inputs[:, t])
                       + torch.nn.functional.linear(spikes, recurrent_weight))
            voltage, spikes, before_reset = lif_step(current, voltage, theta, tau, cfg.dt_ms)
            output = gamma * output + (1 - gamma) * self.readout(spikes)
            spike_total = spike_total + spikes.sum()
            if traces is not None:
                traces["voltage"].append(before_reset[0].detach())
                traces["spikes"].append(spikes[0].detach())
                traces["threshold"].append(torch.broadcast_to(theta, voltage.shape)[0].detach())
                traces["readout"].append(output[0].detach())
        trace = {k: torch.stack(v).cpu() for k, v in traces.items()} if traces is not None else None
        return NetworkOutput(output, spike_total / (batch * steps * cfg.n_hidden), trace)
