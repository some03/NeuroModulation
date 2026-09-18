import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .model import NetworkConfig


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    options = {k: v for k, v in raw.items() if k not in {"duration_ms", "seed"}}
    config = NetworkConfig(**options)
    return config, float(raw.get("duration_ms", 160)), int(raw.get("seed", 42))


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or install a compatible CUDA PyTorch build")
    return device


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_metadata(config, duration_ms, seed, device):
    return {
        "network": asdict(config), "duration_ms": duration_ms, "seed": seed,
        "device": str(device), "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "voltage_units": "normalized; rest=0, default threshold=1",
        "weight_orientation": "[post, pre]", "readout": "final leaky state",
        "reset": "subtractive; reset gradient detached",
    }
