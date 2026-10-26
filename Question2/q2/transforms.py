# Mirror, rotation, speed windows for Moving MNIST.
from __future__ import annotations

import torch


def mirror(x: torch.Tensor) -> torch.Tensor:
    # (Mx)[i, j] = x[i, 63 - j]
    return torch.flip(x, dims=(-1,))


def rotate(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    # 90 deg CCW, k times (k can be negative)
    return torch.rot90(x, k=k, dims=(-2, -1))


def apply_g(x: torch.Tensor, a: int, mirrored: bool) -> torch.Tensor:
    # g = R^a or M R^a (rotate first)
    y = rotate(x, k=a)
    if mirrored:
        y = mirror(y)
    return y


def inverse_g(x: torch.Tensor, a: int, mirrored: bool) -> torch.Tensor:
    # R^{-a} undoes R^a; MRa undoes itself
    if mirrored:
        return apply_g(x, a, True)
    return rotate(x, k=-a)


G_ELEMENTS = [(a, m) for m in (False, True) for a in range(4)]


def window_frames(t: int, k: int) -> tuple[list[int], list[int]]:
    """
    1-based frame numbers for speed-k window starting at t.
    Returns (input_frames, target_frames) keeping only targets <= 20.
    """
    inp = [t + i * k for i in range(5)]
    tgt = [t + (5 + i) * k for i in range(5) if t + (5 + i) * k <= 20]
    return inp, tgt


def allowed_starts(k: int, speed_aug: bool) -> list[int]:
    """Allowed 1-based start frames t."""
    if not speed_aug:
        # models A/B: all five targets exist → t + 9k <= 20 → t <= 20 - 9k
        return list(range(1, 20 - 9 * k + 1))
    # C/D: five inputs + at least two targets
    # t + 4k <= 20 and t + 6k <= 20 → t <= 20 - 6k
    return list(range(1, 20 - 6 * k + 1))


def extract_window(video: torch.Tensor, t: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    video: (20, C, H, W) or (20, H, W), frames indexed 0..19 corresponding to frames 1..20.
    Returns X (5, ...), Y (n_tgt, ...) with n_tgt in {2..5}.
    """
    inp_idx, tgt_idx = window_frames(t, k)
    # convert 1-based → 0-based
    X = video[[i - 1 for i in inp_idx]]
    Y = video[[i - 1 for i in tgt_idx]]
    return X, Y
