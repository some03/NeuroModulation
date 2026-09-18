from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_dynamics(inputs, baseline, modulated, dt_ms, change_ms, path, trained=False):
    inputs = inputs.detach().cpu().numpy()
    base = {k: v.numpy() for k, v in baseline.items()}
    mod = {k: v.numpy() for k, v in modulated.items()}
    time = np.arange(len(inputs)) * dt_ms
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=[0.8, 1.3, 1.2, 1])
    ax = fig.add_subplot(grid[0, :])
    t, channel = np.nonzero(inputs)
    ax.scatter(time[t], channel, s=5, c="#506479", rasterized=True)
    ax.set(title="Identical input in both conditions", ylabel="Input channel", xlim=(0, len(time) * dt_ms))
    neurons = np.argsort(base["spikes"].sum(axis=0))[-3:][::-1]
    colors = ["#117a8b", "#dd7b35", "#6657a0"]
    for column, (title, data) in enumerate((("Baseline", base), ("Threshold intervention", mod))):
        ax = fig.add_subplot(grid[1, column])
        t, neuron = np.nonzero(data["spikes"])
        ax.scatter(time[t], neuron, s=2, c="#117a8b", rasterized=True)
        ax.axvline(change_ms, color="#dd7b35", ls="--", lw=1)
        ax.set(title=title, ylabel="LIF neuron", xlim=(0, time[-1] + dt_ms))
        ax = fig.add_subplot(grid[2, column])
        for neuron, color in zip(neurons, colors):
            ax.plot(time, data["voltage"][:, neuron], lw=0.9, color=color, label=f"Neuron {neuron}")
        ax.plot(time, data["threshold"][:, neurons[0]], color="#303943", ls="--", lw=1.3, label="Threshold")
        ax.set(ylabel="Voltage before reset\n(normalized)", xlabel="Time (ms)", xlim=(0, len(time) * dt_ms))
        ax.legend(fontsize=8, ncol=2)
    ax = fig.add_subplot(grid[3, :])
    width = max(1, round(10 / dt_ms))
    for label, data, color in (("Baseline", base, "#117a8b"), ("Threshold intervention", mod, "#dd7b35")):
        rates, centers = [], []
        for start in range(0, len(time), width):
            block = data["spikes"][start:start + width]
            rates.append(block.mean() * 1000 / dt_ms)
            centers.append((start + len(block) / 2) * dt_ms)
        ax.plot(centers, rates, marker="o", ms=3, lw=1.5, label=label, color=color)
    ax.axvline(change_ms, color="#303943", ls="--", lw=1)
    ax.set(xlabel="Time (ms)", ylabel="Population mean rate (Hz)", xlim=(0, len(time) * dt_ms))
    ax.legend()
    state = "Loaded checkpoint" if trained else "Untrained network / diagnostic only"
    fig.suptitle(f"Recurrent LIF network — {state}", fontsize=15, fontweight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    epochs = [r["epoch"] for r in history]
    for split, color in (("train", "#117a8b"), ("validation", "#dd7b35")):
        axes[0].plot(epochs, [r[f"{split}_loss"] for r in history], label=split, color=color)
        axes[1].plot(epochs, [r[f"{split}_accuracy"] for r in history], label=split, color=color)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.03))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.15)
    fig.savefig(path, dpi=150)
    plt.close(fig)
