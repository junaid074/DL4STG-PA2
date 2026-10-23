"""Train ConvLSTM models A–D."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WindowDataset, load_moving_mnist
from .metrics import eval_e, mse_over_mask
from .models import model_a, model_d


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(name: str):
    if name in ("A", "B", "C"):
        return model_a()
    if name == "D":
        return model_d()
    raise ValueError(name)


def train_one(name: str, seed: int, root: Path, epochs: int = 30, batch_size: int = 32):
    device = get_device()
    set_seed(seed)
    data_root = root / "data"
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading MovingMNIST → {data_root}", flush=True)
    videos = load_moving_mnist(data_root)
    train_idx = range(0, 8000)
    speed_aug = name in ("C", "D")
    sym_aug = name == "B"
    ds = WindowDataset(
        videos, train_idx, speed_aug=speed_aug, sym_aug=sym_aug, train=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = build_model(name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    P = model.count_params()
    print(f"model {name} seed={seed} P={P} device={device} speed_aug={speed_aug} sym_aug={sym_aug}",
          flush=True)

    epoch_times = []
    for ep in range(1, epochs + 1):
        set_seed(seed * 1000 + ep)  # shuffle / window sampling under seed control
        # re-create loader iterator with seeded random for sampling inside __getitem__
        random.seed(seed * 1000 + ep)
        model.train()
        t0 = time.perf_counter()
        losses = []
        for X, Y, mask in loader:
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            pred = model(X)
            err = (pred - Y).pow(2).mean(dim=(2, 3, 4))  # (B,5)
            denom = mask.sum(dim=1).clamp(min=1e-8)
            loss = ((err * mask).sum(dim=1) / denom).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        print(f"  ep{ep:02d} loss={np.mean(losses):.5f} time={dt:.1f}s", flush=True)

    # final-epoch weights
    path = ckpt_dir / f"model{name}_seed{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "name": name, "seed": seed, "params": P}, path)

    # quick val check
    val = videos[8000:9000]
    model.eval()
    with torch.no_grad():
        metrics = eval_e(model, val, device, k=1, horizons=5, t=1)
    meta = {
        "name": name, "seed": seed, "params": P,
        "epoch_time_mean": float(np.mean(epoch_times)),
        "epoch_times": epoch_times,
        "val_E5": metrics["E"],
        "device": str(device),
        "path": str(path),
    }
    (ckpt_dir / f"model{name}_seed{seed}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {path} val_E5={metrics['E']:.5f}", flush=True)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list("ABCD"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--root", type=str, default=".")
    args = p.parse_args()
    train_one(args.model, args.seed, Path(args.root), args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
