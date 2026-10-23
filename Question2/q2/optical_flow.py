"""Figure 1 — Farneback optical flow on first 50 training videos."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_flow_stats(videos: torch.Tensor, n_videos: int = 50):
    """
    videos: (N, 20, 1, 64, 64) float in [0,1].
    Returns directions (deg), speeds (px), n_kept.
    """
    dirs = []
    speeds = []
    for vidx in range(n_videos):
        v = videos[vidx, :, 0].numpy()  # (20, 64, 64)
        for s in range(19):
            prev = (v[s] * 255).astype(np.uint8)
            nxt = (v[s + 1] * 255).astype(np.uint8)
            flow = cv2.calcOpticalFlowFarneback(
                prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            dx = flow[..., 0]
            dy = flow[..., 1]
            mag = np.sqrt(dx * dx + dy * dy)
            mask = (v[s] > 0.2) & (v[s + 1] > 0.2) & (mag > 0.5)
            if not mask.any():
                continue
            dx_k, dy_k, mag_k = dx[mask], dy[mask], mag[mask]
            # θ = atan2(-dy, dx) in [0, 360)
            theta = (np.degrees(np.arctan2(-dy_k, dx_k)) + 360.0) % 360.0
            dirs.append(theta)
            speeds.append(mag_k)
    dirs = np.concatenate(dirs) if dirs else np.array([])
    speeds = np.concatenate(speeds) if speeds else np.array([])
    return dirs, speeds


def save_figure1(videos: torch.Tensor, out_path: Path):
    dirs, speeds = compute_flow_stats(videos, 50)
    median = float(np.median(speeds)) if len(speeds) else float("nan")
    n_kept = int(len(speeds))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # (a) direction
    axes[0].hist(dirs, bins=np.arange(0, 361, 20), density=False, weights=np.ones_like(dirs) / max(len(dirs), 1))
    axes[0].axhline(1 / 18, color="gray", ls="--", lw=1, label="1/18")
    axes[0].set_xlabel("flow direction θ (deg)")
    axes[0].set_ylabel("fraction of kept pixels")
    axes[0].set_title("(a) direction")
    axes[0].legend(frameon=False)
    axes[0].set_xlim(0, 360)

    # (b) speed
    axes[1].hist(speeds, bins=40, density=False, weights=np.ones_like(speeds) / max(len(speeds), 1))
    axes[1].axvline(median, color="crimson", lw=1.5, label=f"median={median:.2f}")
    axes[1].set_xlabel("flow speed (px / frame)")
    axes[1].set_ylabel("fraction of kept pixels")
    axes[1].set_title("(b) speed")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Figure 1 — optical flow on first 50 train videos (kept pixels n={n_kept})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    meta = {"median_speed": median, "n_kept": n_kept}
    (out_path.parent / "fig1_meta.json").write_text(
        __import__("json").dumps(meta, indent=2)
    )
    print(meta)
    return meta
