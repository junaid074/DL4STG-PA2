"""Table 1 transformation / window checks."""
from __future__ import annotations

import torch

from .transforms import extract_window, mirror, rotate, window_frames


def run_table1_checks() -> list[dict]:
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1.0

    rows = []

    # (a)
    a = (mirror(mirror(x)) - x).abs().max().item()
    rows.append({"check": "(a)", "expected": "0", "yours": a, "ok": a == 0.0})

    # (b) R^4 = id
    b = (rotate(x, 4) - x).abs().max().item()
    rows.append({"check": "(b)", "expected": "0", "yours": b, "ok": b == 0.0})

    # (c) M R M = R^{-1}
    left = mirror(rotate(mirror(x), 1))
    right = rotate(x, -1)
    c = (left - right).abs().max().item()
    rows.append({"check": "(c)", "expected": "0", "yours": c, "ok": c == 0.0})

    # (d) bright pixel of M(p)
    mp = mirror(p)
    idx = (mp > 0.5).nonzero(as_tuple=False)[0].tolist()
    # frame, row, col — tensor is (T,C,H,W); nonzero gives (t,c,h,w)
    d = (idx[0], idx[2], idx[3])
    rows.append({"check": "(d)", "expected": "(7, 10, 43)", "yours": d, "ok": d == [7, 10, 43] or d == (7, 10, 43)})

    # (e) bright pixel of R(p)
    rp = rotate(p, 1)
    idx = (rp > 0.5).nonzero(as_tuple=False)[0].tolist()
    e = (idx[0], idx[2], idx[3])
    rows.append({"check": "(e)", "expected": "(7, 43, 10)", "yours": e, "ok": e == [7, 43, 10] or e == (7, 43, 10)})

    # (f)–(h) value-filled video
    filled = torch.zeros(20, 1, 64, 64)
    for f in range(20):
        filled[f] = float(f + 1)  # frame number f+1

    def frames_of(k):
        X, Y = extract_window(filled, 1, k)
        # read unique value per frame
        xin = [int(X[i, 0, 0, 0].item()) for i in range(5)]
        yout = [int(Y[i, 0, 0, 0].item()) for i in range(Y.shape[0])]
        return xin, yout

    f_in, f_out = frames_of(1)
    rows.append({
        "check": "(f)", "expected": "1–5 / 6–10",
        "yours": f"{f_in} / {f_out}",
        "ok": f_in == [1, 2, 3, 4, 5] and f_out == [6, 7, 8, 9, 10],
    })
    g_in, g_out = frames_of(2)
    rows.append({
        "check": "(g)", "expected": "1,3,5,7,9 / 11,13,15,17,19",
        "yours": f"{g_in} / {g_out}",
        "ok": g_in == [1, 3, 5, 7, 9] and g_out == [11, 13, 15, 17, 19],
    })
    h_in, h_out = frames_of(3)
    rows.append({
        "check": "(h)", "expected": "1,4,7,10,13 / 16,19",
        "yours": f"{h_in} / {h_out}",
        "ok": h_in == [1, 4, 7, 10, 13] and h_out == [16, 19],
    })

    # also print window_frames helper
    for k, lab in [(1, "f"), (2, "g"), (3, "h")]:
        print(lab, window_frames(1, k))

    return rows


if __name__ == "__main__":
    for r in run_table1_checks():
        print(r)
