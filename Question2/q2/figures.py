"""Build Figures 2–6 and animated GIFs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .transforms import extract_window, mirror, rotate


def _to_img(frame: torch.Tensor) -> np.ndarray:
    x = frame.detach().cpu().float().numpy()
    if x.ndim == 3:
        x = x[0]
    return np.clip(x, 0, 1)


def save_gif(frames_rows: list[list[np.ndarray]], path: Path, fps: int = 2):
    """
    frames_rows: list over horizons; each item is list of 4 row-images (already 64x64).
    We compose a 4-row panel per horizon and animate.
    """
    panels = []
    for cols in zip(*frames_rows):  # wrong structure — handle below
        pass
    # frames_rows[r][h] = image at row r, horizon h
    n_rows = len(frames_rows)
    n_h = len(frames_rows[0])
    images = []
    for h in range(n_h):
        rows = [frames_rows[r][h] for r in range(n_rows)]
        panel = np.concatenate(rows, axis=0)  # stack vertically
        # pad side-by-side with inputs? GIF only steps forecast horizons with 4 rows
        img = (panel * 255).astype(np.uint8)
        images.append(Image.fromarray(img, mode="L"))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)


@torch.no_grad()
def figure2(model, videos, out: Path, device, indices=(0, 1, 2)):
    model.eval()
    fig, axes = plt.subplots(len(indices), 15, figsize=(18, 3 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, 0)
    for r, idx in enumerate(indices):
        v = videos[idx].to(device)
        X, Y = extract_window(v, 1, 1)
        pred = model(X.unsqueeze(0))[0]
        # show 5 input + 5 pred + 5 gt
        for c in range(5):
            axes[r, c].imshow(_to_img(X[c]), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"x{c+1}")
        for c in range(5):
            axes[r, 5 + c].imshow(_to_img(pred[c]), cmap="gray", vmin=0, vmax=1)
            axes[r, 5 + c].axis("off")
            if r == 0:
                axes[r, 5 + c].set_title(f"pred{c+1}")
        for c in range(5):
            axes[r, 10 + c].imshow(_to_img(Y[c]), cmap="gray", vmin=0, vmax=1)
            axes[r, 10 + c].axis("off")
            if r == 0:
                axes[r, 10 + c].set_title(f"gt{c+1}")
        axes[r, 0].set_ylabel(f"vid {idx}")
    fig.suptitle("Figure 2 — model A (seed 0): input | prediction | ground truth")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


@torch.no_grad()
def figure3_consistency(model, videos, C_per, T: str, out_png: Path, out_gif_dir: Path, device, top_k=2):
    """Largest per-video CT panels."""
    model.eval()
    vals, idxs = torch.topk(C_per, top_k)
    Tf = mirror if T == "M" else (lambda z: rotate(z, 1))
    fig, axes = plt.subplots(top_k * 4, 10, figsize=(14, 1.6 * top_k * 4))
    gif_paths = []
    for i, vid_i in enumerate(idxs.tolist()):
        v = videos[vid_i].to(device)
        X, Y = extract_window(v, 1, 1)
        TX, TY = Tf(X), Tf(Y)
        fX = model(X.unsqueeze(0))[0]
        fTX = model(TX.unsqueeze(0))[0]
        T_fX = Tf(fX)
        rows = [
            list(X) + list(fX),
            list(TX) + list(TY),
            list(TX) + list(fTX),
            list(TX) + list(T_fX),
        ]
        labels = ["(1) X | f(X)", "(2) T(X) | T(Y)", "(3) T(X) | f(T(X))", "(4) T(X) | T(f(X))"]
        for rr, (row, lab) in enumerate(zip(rows, labels)):
            axrow = i * 4 + rr
            for c in range(10):
                axes[axrow, c].imshow(_to_img(row[c]), cmap="gray", vmin=0, vmax=1)
                axes[axrow, c].axis("off")
            axes[axrow, 0].set_ylabel(f"v{vid_i}\n{lab}", fontsize=6)
        # GIF: 4 rows × 5 horizons (cols 6-10), keep inputs in cols 1-5 static? Spec:
        # "steps through the five forecast horizons while keeping the same four rows"
        # So each frame of GIF shows 4 rows, and animates the future columns.
        # Simpler: each GIF frame is the 4×10 panel but only future part changes — 
        # we'll animate 5 panels where each shows input(5)+that horizon focus.
        # Practical approach: 5 frames, each is 4 rows × (5 input + 1 future) from each row.
        frames_for_gif = []
        for h in range(5):
            panel_rows = []
            for row in rows:
                # 5 inputs + current future frame
                strip = np.concatenate([_to_img(row[c]) for c in list(range(5)) + [5 + h]], axis=1)
                panel_rows.append(strip)
            frames_for_gif.append(np.concatenate(panel_rows, axis=0))
        gif_path = out_gif_dir / f"fig3_{T}_vid{vid_i}.gif"
        imgs = [Image.fromarray((p * 255).astype(np.uint8), mode="L") for p in frames_for_gif]
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=400, loop=0)
        gif_paths.append(str(gif_path))
    fig.suptitle(f"Figure 3 — largest C_{T} videos (model A seed 0)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return idxs.tolist(), vals.tolist(), gif_paths


@torch.no_grad()
def figure4_hist(CM, CR, mark_m, mark_r, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(CM.numpy(), bins=40, color="steelblue", alpha=0.85)
    for v in mark_m:
        axes[0].axvline(CM[v].item(), color="crimson", ls="--", lw=1)
    axes[0].set_title("(a) per-video $C_M$")
    axes[0].set_xlabel("$C_M$")
    axes[1].hist(CR.numpy(), bins=40, color="darkorange", alpha=0.85)
    for v in mark_r:
        axes[1].axvline(CR[v].item(), color="crimson", ls="--", lw=1)
    axes[1].set_title("(b) per-video $C_R$")
    axes[1].set_xlabel("$C_R$")
    fig.suptitle("Figure 4 — consistency gap histograms (model A seed 0)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


@torch.no_grad()
def figure5_speed(model, video, out: Path, device):
    model.eval()
    fig, axes = plt.subplots(3, 9, figsize=(14, 5))
    for r, k in enumerate([1, 2, 3]):
        X, Y = extract_window(video.to(device), 1, k)
        pred = model(X.unsqueeze(0))[0]
        # 5 inputs + first 2 pred + first 2 true
        for c in range(5):
            axes[r, c].imshow(_to_img(X[c]), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        for c in range(2):
            axes[r, 5 + c].imshow(_to_img(pred[c]), cmap="gray", vmin=0, vmax=1)
            axes[r, 5 + c].axis("off")
            axes[r, 7 + c].imshow(_to_img(Y[c]), cmap="gray", vmin=0, vmax=1)
            axes[r, 7 + c].axis("off")
        axes[r, 0].set_ylabel(f"{k}×")
    axes[0, 0].set_title("inputs")
    axes[0, 5].set_title("pred")
    axes[0, 7].set_title("true")
    fig.suptitle("Figure 5 — speed stress (model A seed 0)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def pick_nonoverlap_video(videos, n_check=200):
    """Pick a test video whose digits don't overlap in displayed true frames at 1/2/3×."""
    for i in range(min(n_check, len(videos))):
        v = videos[i]
        ok = True
        for k in (1, 2, 3):
            _, Y = extract_window(v, 1, k)
            for h in range(min(2, Y.shape[0])):
                # crude: if too many bright pixels, likely overlap
                bright = (Y[h, 0] > 0.2).sum().item()
                if bright > 120:  # single digit ~40-80
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return i
    return 0
