"""python -m yaminabe_snn.demo: plot matched-input threshold intervention."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .connectivity import load_mask
from .data import toy_dataset
from .model import NetworkConfig, RecurrentLIF
from .modulation import NoModulation, StepThreshold
from .plots import plot_dynamics
from .utils import read_config, run_metadata, seed_all, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.json")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint; overrides network config")
    parser.add_argument("--mask", type=Path, help="Binary adjacency .npy, [post, pre]")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/demo"))
    parser.add_argument("--factor", type=float, default=1.5)
    args = parser.parse_args()
    config, duration, seed = read_config(args.config)
    checkpoint = None
    if args.checkpoint:
        if args.mask:
            parser.error("A checkpoint already includes its mask; omit --mask")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = NetworkConfig(**checkpoint["network_config"])
        duration = checkpoint["duration_ms"]
        seed = checkpoint["seed"]
    seed_all(seed)
    torch.set_num_threads(2)
    device = select_device(args.device)
    mask = load_mask(args.mask, config.n_hidden) if args.mask else None
    model = RecurrentLIF(config, mask=mask).to(device).eval()
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    # This demo always uses synthetic input, even if a SHD checkpoint is loaded.
    dataset = toy_dataset(2, config.n_inputs, duration, config.dt_ms, seed + 999)
    inputs = dataset.tensors[0][:1].to(device)
    change_ms = duration / 2
    with torch.inference_mode():
        model.modulator = NoModulation()
        baseline = model(inputs, record=True)
        model.modulator = StepThreshold(change_ms, args.factor)
        modulated = model(inputs, record=True)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_dynamics(inputs[0], baseline.trace, modulated.trace, config.dt_ms,
                  change_ms, args.out / "dynamics.png", bool(checkpoint))
    arrays = {f"baseline_{k}": v.numpy() for k, v in baseline.trace.items()}
    arrays.update({f"modulated_{k}": v.numpy() for k, v in modulated.trace.items()})
    np.savez_compressed(args.out / "traces.npz", inputs=inputs[0].cpu().numpy(), **arrays)
    summary = run_metadata(config, duration, seed, device)
    summary.update({"input": "synthetic only", "trained_checkpoint": str(args.checkpoint) if checkpoint else None,
                    "threshold_factor": args.factor, "change_ms": change_ms,
                    "baseline_rate_hz": float(baseline.mean_spikes_per_step) * 1000 / config.dt_ms,
                    "intervention_rate_hz": float(modulated.mean_spikes_per_step) * 1000 / config.dt_ms})
    write_json(args.out / "summary.json", summary)
    print(f"Saved {args.out / 'dynamics.png'}")
    print(f"Mean rates: baseline={summary['baseline_rate_hz']:.2f} Hz, intervention={summary['intervention_rate_hz']:.2f} Hz")
    print("Diagnostic on synthetic input; this is not evidence of improved classification.")


if __name__ == "__main__":
    main()
