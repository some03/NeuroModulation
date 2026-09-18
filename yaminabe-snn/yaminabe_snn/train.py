"""Train an unmodulated baseline; select checkpoints on validation, not test data."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from .connectivity import load_mask
from .data import SHDDataset, toy_dataset
from .model import RecurrentLIF
from .plots import plot_training
from .utils import read_config, run_metadata, seed_all, select_device, write_json


def run_epoch(model, loader, device, optimizer=None, spike_penalty=0.0):
    model.train(optimizer is not None)
    total_loss = total_correct = total_rate = total_samples = 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            result = model(inputs)
            ce = F.cross_entropy(result.logits, labels)
            objective = ce + spike_penalty * result.mean_spikes_per_step
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite loss; inspect rates, weights, and learning rate")
            if optimizer is not None:
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = len(labels)
            total_loss += float(ce.detach()) * count
            total_correct += int((result.logits.argmax(-1) == labels).sum())
            total_rate += float(result.mean_spikes_per_step.detach()) * 1000 / model.config.dt_ms * count
            total_samples += count
    if total_samples == 0:
        raise ValueError("Empty data split")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples,
            "mean_rate_hz": total_rate / total_samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["toy", "shd"], default="toy")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--shd-dir", type=Path, default=Path("data/shd"))
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--hidden", type=int, help="Optional override of n_hidden")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--spike-penalty", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/train"))
    parser.add_argument("--test", action="store_true", help="Evaluate best checkpoint on held-out test once")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.spike_penalty < 0:
        parser.error("epochs/batch-size/lr must be positive; spike-penalty must be nonnegative")
    path = args.config or Path("configs/shd.json" if args.dataset == "shd" else "configs/demo.json")
    config, duration, seed = read_config(path)
    if args.hidden is not None:
        config.n_hidden = args.hidden
        config.__post_init__()
    seed_all(seed)
    torch.set_num_threads(2)
    device = select_device(args.device)
    if args.dataset == "toy":
        if config.n_outputs != 2:
            parser.error("Toy dataset requires n_outputs=2")
        dataset = toy_dataset(256, config.n_inputs, duration, config.dt_ms, seed)
    else:
        if config.n_inputs != 700 or config.n_outputs != 20:
            parser.error("SHD requires n_inputs=700 and n_outputs=20")
        dataset = SHDDataset(args.shd_dir / "shd_train.h5", duration, config.dt_ms)
        print(f"SHD training examples={len(dataset)}, events truncated by duration={dataset.truncated_events}", flush=True)
    if len(dataset) < 4:
        parser.error("At least 4 samples are needed for a train/validation split")
    validation_size = max(1, round(len(dataset) * 0.2))
    train_data, validation_data = random_split(dataset, [len(dataset) - validation_size, validation_size],
                                              generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    mask = load_mask(args.mask, config.n_hidden) if args.mask else None
    model = RecurrentLIF(config, mask=mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(config, duration, seed, device)
    metadata.update({"dataset": args.dataset, "epochs": args.epochs, "batch_size": args.batch_size,
                     "lr": args.lr, "spike_penalty": args.spike_penalty,
                     "train_indices": train_data.indices, "validation_indices": validation_data.indices,
                     "mask_source": str(args.mask) if args.mask else "generated"})
    write_json(args.out / "config.json", metadata)
    history, best_loss, best_epoch = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer, args.spike_penalty)
        validation = run_epoch(model, validation_loader, device)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                  **{f"validation_{k}": v for k, v in validation.items()}}
        history.append(record)
        write_json(args.out / "history.json", history)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save({"model_state": model.state_dict(), "network_config": asdict(config),
                        "duration_ms": duration, "seed": seed, "epoch": epoch, "dataset": args.dataset},
                       args.out / "best.pt")
        print(f"Epoch {epoch:02d} | train loss={train['loss']:.4f} acc={train['accuracy']:.3f} | "
              f"val loss={validation['loss']:.4f} acc={validation['accuracy']:.3f} | "
              f"val mean rate={validation['mean_rate_hz']:.2f} Hz", flush=True)
    plot_training(history, args.out / "learning.png")
    summary = {"dataset": args.dataset, "best_epoch": best_epoch, "best_validation_loss": best_loss}
    if args.test:
        checkpoint = torch.load(args.out / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        test_data = (toy_dataset(128, config.n_inputs, duration, config.dt_ms, seed + 1000)
                     if args.dataset == "toy" else SHDDataset(args.shd_dir / "shd_test.h5", duration, config.dt_ms))
        test_result = run_epoch(model, DataLoader(test_data, batch_size=args.batch_size), device)
        summary["test"] = test_result
        if args.dataset == "shd":
            summary["test_truncated_events"] = test_data.truncated_events
        print(f"Held-out test: {test_result}", flush=True)
    write_json(args.out / "summary.json", summary)
    print(f"Saved checkpoint and plots in {args.out}", flush=True)


if __name__ == "__main__":
    main()
