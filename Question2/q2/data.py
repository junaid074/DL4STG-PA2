"""Moving MNIST loading and window sampling."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MovingMNIST

from .transforms import (
    G_ELEMENTS,
    allowed_starts,
    apply_g,
    extract_window,
)


def load_moving_mnist(root: Path) -> torch.Tensor:
    """
    Returns float32 tensor (10000, 20, 1, 64, 64) in [0, 1].
    Videos 0–7999 train, 8000–8999 val, 9000–9999 test.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "moving_mnist_10000.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)

    ds = MovingMNIST(root=str(root), split=None, download=True)
    # Fast path: underlying tensor if exposed
    if hasattr(ds, "data"):
        v = ds.data
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        v = v.float()
        if v.max() > 1.5:
            v = v / 255.0
        if v.ndim == 4:  # (N, 20, H, W)
            v = v.unsqueeze(2)
        torch.save(v, cache)
        return v

    videos = []
    for i in range(len(ds)):
        item = ds[i]
        if isinstance(item, (tuple, list)):
            item = item[0]
        item = item.float()
        if item.max() > 1.5:
            item = item / 255.0
        if item.ndim == 3:
            item = item.unsqueeze(1)
        videos.append(item)
    v = torch.stack(videos, dim=0)
    torch.save(v, cache)
    return v


class WindowDataset(Dataset):
    def __init__(
        self,
        videos: torch.Tensor,
        indices: range | list[int],
        *,
        speed_aug: bool = False,
        sym_aug: bool = False,
        train: bool = True,
        fixed_t: int | None = 1,
        fixed_k: int = 1,
    ):
        self.videos = videos
        self.indices = list(indices)
        self.speed_aug = speed_aug
        self.sym_aug = sym_aug
        self.train = train
        self.fixed_t = fixed_t
        self.fixed_k = fixed_k

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        vid = self.videos[self.indices[i]]
        if self.train:
            if self.speed_aug:
                k = random.choice([1, 2, 3])
            else:
                k = 1
            starts = allowed_starts(k, speed_aug=self.speed_aug)
            t = random.choice(starts)
        else:
            k = self.fixed_k
            t = self.fixed_t if self.fixed_t is not None else 1
        X, Y = extract_window(vid, t, k)
        if self.train and self.sym_aug:
            a, m = random.choice(G_ELEMENTS)
            X = apply_g(X, a, m)
            Y = apply_g(Y, a, m)
        # pad Y to 5 frames with zeros + mask
        n = Y.shape[0]
        if n < 5:
            pad = torch.zeros(5 - n, *Y.shape[1:], dtype=Y.dtype)
            Y_full = torch.cat([Y, pad], dim=0)
        else:
            Y_full = Y
        mask = torch.zeros(5, dtype=torch.float32)
        mask[:n] = 1.0
        return X, Y_full, mask
