"""LIF update and surrogate gradient, separated for later neuron-model changes."""

import torch


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, overdrive):
        ctx.save_for_backward(overdrive)
        return (overdrive >= 0).to(overdrive.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (overdrive,) = ctx.saved_tensors
        return grad_output / (1 + 10 * overdrive.abs()).square()


def lif_step(current, voltage, threshold, tau_mem_ms, dt_ms):
    """Exponential leak with current held constant during one integration step."""
    alpha = torch.exp(-dt_ms / tau_mem_ms)
    before_reset = alpha * voltage + (1 - alpha) * current
    spikes = SurrogateSpike.apply(before_reset - threshold)
    # Subtractive reset; detach the entire reset term in backpropagation.
    voltage = before_reset - (spikes * threshold).detach()
    return voltage, spikes, before_reset
