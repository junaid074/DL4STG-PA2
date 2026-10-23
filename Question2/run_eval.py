#!/usr/bin/env python3
"""Evaluate trained models; write tables JSON + figures."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from q2.data import load_moving_mnist
from q2.figures import (
    figure2,
    figure3_consistency,
    figure4_hist,
    figure5_speed,
    pick_nonoverlap_video,
)
from q2.metrics import eval_consistency, eval_e, timed_inference
from q2.models import (
    BlankPredictor,
    CopyLastPredictor,
    model_a,
    model_d,
    symmetrised,
)
from q2.optical_flow import save_figure1
from q2.transforms import extract_window

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CKPT = ROOT / "checkpoints"
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


def load_model(name: str, seed: int):
    if name == "D":
        m = model_d()
    else:
        m = model_a()
    path = CKPT / f"model{name}_seed{seed}.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(state["state_dict"])
    return m.to(DEVICE).eval()


def fmt(mean, a, b):
    return f"{mean:.3g} ({a:.3g}, {b:.3g})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("device", DEVICE, flush=True)
    videos = load_moving_mnist(ROOT / "data")
    train_v, test_v = videos[:8000], videos[9000:]

    # Figure 1
    print("Figure 1...", flush=True)
    fig1_meta = save_figure1(train_v, OUT / "figure1.pdf")

    # Table 2 — A seeds + blank + copy
    print("Table 2...", flush=True)
    table2 = {}
    for seed in (0, 1):
        m = load_model("A", seed)
        r = eval_e(m, test_v, DEVICE, k=1, horizons=5)
        table2[f"A_seed{seed}"] = r
    blank = BlankPredictor().to(DEVICE)
    copy = CopyLastPredictor().to(DEVICE)
    table2["Blank"] = eval_e(blank, test_v, DEVICE, k=1, horizons=5)
    table2["CopyLast"] = eval_e(copy, test_v, DEVICE, k=1, horizons=5)
    table2["A_params"] = model_a().count_params()

    # Figure 2
    print("Figure 2...", flush=True)
    # pick videos: one with overlap
    overlap_idx = 0
    for i in range(len(test_v)):
        X, Y = extract_window(test_v[i], 1, 1)
        if (Y[0, 0] > 0.2).sum() > 100:
            overlap_idx = i
            break
    idxs = [0, 1, overlap_idx]
    figure2(load_model("A", 0), test_v, OUT / "figure2.pdf", DEVICE, indices=idxs)

    # Consistency Table 3 + Fig 3,4
    print("Table 3 / Fig 3–4...", flush=True)
    table3 = {"Blank": {}, "CopyLast": {}}
    for label, m in [("Blank", blank), ("CopyLast", copy)]:
        for T in ("M", "R"):
            c = eval_consistency(m, test_v, DEVICE, T=T)
            table3[label][T] = {"C": c["C"], "E_T": c["E_T"]}
    A_cons = {}
    for seed in (0, 1):
        m = load_model("A", seed)
        A_cons[seed] = {}
        e5 = eval_e(m, test_v, DEVICE, k=1, horizons=5)["E"]
        A_cons[seed]["E5"] = e5
        for T in ("M", "R"):
            c = eval_consistency(m, test_v, DEVICE, T=T)
            A_cons[seed][T] = c
    table3["A"] = {
        "E5": (A_cons[0]["E5"], A_cons[1]["E5"]),
        "CM": (A_cons[0]["M"]["C"], A_cons[1]["M"]["C"]),
        "EM": (A_cons[0]["M"]["E_T"], A_cons[1]["M"]["E_T"]),
        "CR": (A_cons[0]["R"]["C"], A_cons[1]["R"]["C"]),
        "ER": (A_cons[0]["R"]["E_T"], A_cons[1]["R"]["E_T"]),
    }
    CM = A_cons[0]["M"]["C_per_video"]
    CR = A_cons[0]["R"]["C_per_video"]
    m_idxs, m_vals, gifs_m = figure3_consistency(
        load_model("A", 0), test_v, CM, "M", OUT / "figure3_M.pdf", OUT / "gifs", DEVICE
    )
    r_idxs, r_vals, gifs_r = figure3_consistency(
        load_model("A", 0), test_v, CR, "R", OUT / "figure3_R.pdf", OUT / "gifs", DEVICE
    )
    figure4_hist(CM, CR, m_idxs, r_idxs, OUT / "figure4.pdf")

    # Table 4 speed
    print("Table 4 / Fig 5...", flush=True)
    table4 = {"A": {}, "CopyLast": {}}
    for seed in (0, 1):
        m = load_model("A", seed)
        table4["A"][seed] = {
            "E2_1x": eval_e(m, test_v, DEVICE, k=1, horizons=2)["E"],
            "E2_2x": eval_e(m, test_v, DEVICE, k=2, horizons=2)["E"],
            "E2_3x": eval_e(m, test_v, DEVICE, k=3, horizons=2)["E"],
            "E5_1x": eval_e(m, test_v, DEVICE, k=1, horizons=5)["E"],
            "E5_2x": eval_e(m, test_v, DEVICE, k=2, horizons=5)["E"],
        }
    table4["CopyLast"] = {
        "E2_1x": eval_e(copy, test_v, DEVICE, k=1, horizons=2)["E"],
        "E2_2x": eval_e(copy, test_v, DEVICE, k=2, horizons=2)["E"],
        "E2_3x": eval_e(copy, test_v, DEVICE, k=3, horizons=2)["E"],
        "E5_1x": eval_e(copy, test_v, DEVICE, k=1, horizons=5)["E"],
        "E5_2x": eval_e(copy, test_v, DEVICE, k=2, horizons=5)["E"],
    }
    vid5 = pick_nonoverlap_video(test_v)
    figure5_speed(load_model("A", 0), test_v[vid5], OUT / "figure5.pdf", DEVICE)

    # Table 5 — A–D
    print("Table 5...", flush=True)
    table5 = {}
    for name in list("ABCD"):
        path0 = CKPT / f"model{name}_seed0.json"
        path1 = CKPT / f"model{name}_seed1.json"
        if not path0.exists():
            print(f"skip {name}, no ckpt", flush=True)
            continue
        meta0 = json.loads(path0.read_text())
        meta1 = json.loads(path1.read_text())
        row = {"params": meta0["params"], "train_s_per_ep": (meta0["epoch_time_mean"], meta1["epoch_time_mean"])}
        for seed in (0, 1):
            m = load_model(name, seed)
            row[seed] = {
                "E5_1x": eval_e(m, test_v, DEVICE, k=1, horizons=5)["E"],
                "E2_1x": eval_e(m, test_v, DEVICE, k=1, horizons=2)["E"],
                "E2_2x": eval_e(m, test_v, DEVICE, k=2, horizons=2)["E"],
                "E2_3x": eval_e(m, test_v, DEVICE, k=3, horizons=2)["E"],
            }
            for T, keyC, keyE in [("M", "CM", "EM"), ("R", "CR", "ER")]:
                c = eval_consistency(m, test_v, DEVICE, T=T)
                row[seed][keyC] = c["C"]
                row[seed][keyE] = c["E_T"]
            row[seed]["infer_s"] = timed_inference(m, test_v, DEVICE)
        table5[name] = row

    # Table 6 — symmetrised
    print("Table 6 / Fig 6...", flush=True)
    table6 = {}
    for name in ("B", "D"):
        if name not in table5:
            continue
        for seed in (0, 1):
            base = load_model(name, seed)
            sym = symmetrised(base).to(DEVICE).eval()
            key = f"sym{name}_seed{seed}"
            table6[key] = {
                "E5_1x": eval_e(sym, test_v, DEVICE, k=1, horizons=5)["E"],
                "CM": eval_consistency(sym, test_v, DEVICE, "M")["C"],
                "EM": eval_consistency(sym, test_v, DEVICE, "M")["E_T"],
                "CR": eval_consistency(sym, test_v, DEVICE, "R")["C"],
                "ER": eval_consistency(sym, test_v, DEVICE, "R")["E_T"],
                "E2_1x": eval_e(sym, test_v, DEVICE, k=1, horizons=2)["E"],
                "E2_2x": eval_e(sym, test_v, DEVICE, k=2, horizons=2)["E"],
                "E2_3x": eval_e(sym, test_v, DEVICE, k=3, horizons=2)["E"],
                "infer_s": timed_inference(sym, test_v, DEVICE),
                "params": base.count_params(),
            }
    # Final = sym D; redo fig 3/5 style
    if "D" in table5:
        final = symmetrised(load_model("D", 0)).to(DEVICE).eval()
        # Fig 6a: same videos as fig3
        for T, idxs in [("M", m_idxs), ("R", r_idxs)]:
            # reuse figure3 on those specific videos by faking topk
            C_fake = torch.zeros(len(test_v))
            for rank, vi in enumerate(idxs):
                C_fake[vi] = 1e6 - rank
            figure3_consistency(
                final, test_v, C_fake, T,
                OUT / f"figure6a_{T}.pdf", OUT / "gifs_final", DEVICE, top_k=len(idxs),
            )
        figure5_speed(final, test_v[vid5], OUT / "figure6b.pdf", DEVICE)

    results = {
        "fig1": fig1_meta,
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "table5": table5,
        "table6": table6,
        "fig3_M_videos": m_idxs,
        "fig3_R_videos": r_idxs,
        "fig5_video": vid5,
        "fig2_videos": idxs,
        "gifs_M": gifs_m,
        "gifs_R": gifs_r,
        "device": str(DEVICE),
    }

    # make JSON serializable
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return o

    (OUT / "all_results.json").write_text(json.dumps(conv(results), indent=2))
    print("wrote", OUT / "all_results.json", flush=True)


if __name__ == "__main__":
    main()
