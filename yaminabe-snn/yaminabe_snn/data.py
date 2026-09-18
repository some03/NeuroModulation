"""Synthetic smoke data and a local-file SHD adapter. No automatic downloads."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


def time_steps(duration_ms: float, dt_ms: float) -> int:
    if not np.isfinite(duration_ms) or not np.isfinite(dt_ms) or duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be finite and positive")
    ratio = duration_ms / dt_ms
    if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-7):
        raise ValueError("duration_ms must be an integer multiple of dt_ms")
    return int(round(ratio))


def toy_dataset(n_samples=256, n_inputs=16, duration_ms=160.0, dt_ms=1.0, seed=42):
    """Two channel groups activated in opposite orders. Debug data, not SHD."""
    if n_inputs < 2 or n_samples < 2:
        raise ValueError("Toy data need at least 2 inputs and 2 samples")
    steps = time_steps(duration_ms, dt_ms)
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n_samples) % 2
    rates_hz = torch.full((n_samples, steps, n_inputs), 5.0)
    middle = n_inputs // 2
    for sample, label in enumerate(labels):
        first = slice(0, middle) if label == 0 else slice(middle, n_inputs)
        second = slice(middle, n_inputs) if label == 0 else slice(0, middle)
        rates_hz[sample, steps // 8:steps // 2, first] = 140.0
        rates_hz[sample, steps // 2:7 * steps // 8, second] = 140.0
    # Poisson counts preserve multiple events per bin.
    counts = torch.poisson(rates_hz * dt_ms / 1000, generator=g)
    return TensorDataset(counts, labels.long())


def bin_events(times_seconds, units, n_inputs, duration_ms, dt_ms):
    steps = time_steps(duration_ms, dt_ms)
    times = np.asarray(times_seconds, dtype=np.float64)
    units = np.asarray(units)
    if times.ndim != 1 or units.shape != times.shape:
        raise ValueError("Event times and units must be matching 1-D arrays")
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Event times must be finite, nonnegative seconds")
    if not np.isfinite(units).all() or np.any(units != np.floor(units)):
        raise ValueError("Unit IDs must be finite integers")
    if np.any((units < 0) | (units >= n_inputs)):
        raise ValueError("Unit ID outside configured input range")
    # Half-open trial [0, duration). Keep event counts; do not clip duplicates to 1.
    inside = times < duration_ms / 1000
    bins = np.floor(times[inside] * (1000 / dt_ms)).astype(np.int64)
    valid_bins = bins < steps
    counts = np.zeros((steps, n_inputs), dtype=np.float32)
    np.add.at(counts, (bins[valid_bins], units[inside][valid_bins].astype(np.int64)), 1)
    return torch.from_numpy(counts)


class SHDDataset(Dataset):
    def __init__(self, path: str | Path, duration_ms=1400.0, dt_ms=1.0, n_inputs=700):
        import h5py

        self.duration_ms, self.dt_ms, self.n_inputs = duration_ms, dt_ms, n_inputs
        time_steps(duration_ms, dt_ms)
        # Store raw variable-length events, not a huge dense [samples,time,700] tensor.
        # Closing the file here also makes DataLoader workers safe on Windows.
        with h5py.File(path, "r") as f:
            self.labels = np.asarray(f["labels"], dtype=np.int64)
            self.times = [np.asarray(x, dtype=np.float64) for x in f["spikes/times"]]
            self.units = [np.asarray(x) for x in f["spikes/units"]]
        if not len(self.labels) or len(self.times) != len(self.labels) or len(self.units) != len(self.labels):
            raise ValueError("Inconsistent or empty SHD file")
        if np.any((self.labels < 0) | (self.labels >= 20)):
            raise ValueError("Expected SHD labels in [0, 19]")
        self.truncated_events = sum(int(np.count_nonzero(t >= duration_ms / 1000)) for t in self.times)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        counts = bin_events(self.times[index], self.units[index], self.n_inputs, self.duration_ms, self.dt_ms)
        return counts, torch.tensor(self.labels[index], dtype=torch.long)
