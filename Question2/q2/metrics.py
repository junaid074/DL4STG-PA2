"""Evaluation metrics for Q2."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .transforms import apply_g, extract_window, mirror, rotate


@torch.no_grad()
def mse_over_mask(pred, target, mask):
    """pred/target (B,5,1,H,W), mask (B,5)."""
    err = (pred - target).pow(2).mean(dim=(2, 3, 4))  # (B,5)
    # average over available horizons per video, then over batch
    denom = mask.sum(dim=1).clamp(min=1e-8)
    per_vid = (err * mask).sum(dim=1) / denom
    return per_vid.mean()


@torch.no_grad()
def eval_e(model, videos, device, k: int = 1, horizons: int = 5, t: int = 1, batch_size: int = 32):
    """
    E(horizons) on test videos at speed k, start t.
    Only first `horizons` available targets are used (and only those that exist).
    """
    model.eval()
    n = videos.shape[0]
    mse_h = [0.0] * 5
    counts = [0] * 5
    for start in range(0, n, batch_size):
        batch = videos[start:start + batch_size].to(device)
        Xs, Ys, masks = [], [], []
        for v in batch:
            X, Y = extract_window(v, t, k)
            n_tgt = Y.shape[0]
            pad = torch.zeros(5 - n_tgt, *Y.shape[1:], device=device, dtype=Y.dtype)
            Y_full = torch.cat([Y, pad], 0) if n_tgt < 5 else Y
            mask = torch.zeros(5, device=device)
            mask[:n_tgt] = 1.0
            Xs.append(X)
            Ys.append(Y_full)
            masks.append(mask)
        X = torch.stack(Xs)
        Y = torch.stack(Ys)
        mask = torch.stack(masks)
        pred = model(X)
        err = (pred - Y).pow(2).mean(dim=(2, 3, 4))  # (B,5)
        for h in range(5):
            m = mask[:, h]
            if m.sum() > 0:
                mse_h[h] += (err[:, h] * m).sum().item()
                counts[h] += m.sum().item()
    mse_h = [mse_h[h] / max(counts[h], 1) for h in range(5)]
    use = min(horizons, sum(1 for c in counts if c > 0))
    # for E(2)/E(5) average first `horizons` that have data
    avail = [mse_h[h] for h in range(5) if counts[h] > 0][:horizons]
    e = sum(avail) / len(avail) if avail else float("nan")
    return {"MSE": mse_h, "E": e, "counts": counts}


@torch.no_grad()
def eval_consistency(model, videos, device, T: str = "M", batch_size: int = 32):
    """
    CT and ET at 1×, t=1 over all videos.
    Also returns per-video CT for histograms.
    """
    model.eval()
    n = videos.shape[0]
    ct_list, et_list = [], []
    for start in range(0, n, batch_size):
        batch = videos[start:start + batch_size].to(device)
        Xs, Ys = [], []
        for v in batch:
            X, Y = extract_window(v, 1, 1)
            Xs.append(X)
            Ys.append(Y)
        X = torch.stack(Xs)
        Y = torch.stack(Ys)
        if T == "M":
            TX, TY = mirror(X), mirror(Y)
            Tf = mirror
        else:
            TX, TY = rotate(X, 1), rotate(Y, 1)
            Tf = lambda z: rotate(z, 1)

        fX = model(X)
        fTX = model(TX)
        T_fX = Tf(fX)
        # MSE over pixels and 5 horizons, per video
        ct = (fTX - T_fX).pow(2).mean(dim=(1, 2, 3, 4))
        et = (fTX - TY).pow(2).mean(dim=(1, 2, 3, 4))
        ct_list.append(ct.cpu())
        et_list.append(et.cpu())
    ct_all = torch.cat(ct_list)
    et_all = torch.cat(et_list)
    return {
        "C": ct_all.mean().item(),
        "E_T": et_all.mean().item(),
        "C_per_video": ct_all,
        "E_per_video": et_all,
    }


@torch.no_grad()
def timed_inference(model, videos, device, batch_size: int = 32, symmetrised: bool = False):
    import time
    model.eval()
    # warmup
    x = videos[:batch_size].to(device)
    Xs = torch.stack([extract_window(v, 1, 1)[0] for v in x])
    _ = model(Xs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    n = videos.shape[0]
    for start in range(0, n, batch_size):
        batch = videos[start:start + batch_size].to(device)
        Xs = torch.stack([extract_window(v, 1, 1)[0] for v in batch])
        _ = model(Xs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - t0
