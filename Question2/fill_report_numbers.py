#!/usr/bin/env python3
"""Read all_results.json and print LaTeX table snippets for the report."""
from __future__ import annotations

import json
from pathlib import Path

R = json.loads((Path(__file__).parent / "results" / "all_results.json").read_text())


def mean_pair(a, b, sig=3):
    m = 0.5 * (a + b)
    return f"{m:.{sig}g} ({a:.{sig}g}, {b:.{sig}g})"


def sci(x):
    return f"{x:.3e}"


def main():
    t2 = R["table2"]
    a0, a1 = t2["A_seed0"], t2["A_seed1"]
    print("% --- Table 2 ---")
    print("A E5", mean_pair(a0["E"], a1["E"]))
    for h in range(5):
        print(f"A MSE{h+1}", mean_pair(a0["MSE"][h], a1["MSE"][h]))
    print("Blank E5", f"{t2['Blank']['E']:.3g}")
    print("Copy E5", f"{t2['CopyLast']['E']:.3g}")

    t3 = R["table3"]["A"]
    print("% --- Table 3 ---")
    print("E5", mean_pair(*t3["E5"]))
    print("CM", mean_pair(*t3["CM"]))
    print("EM", mean_pair(*t3["EM"]))
    print("CR", mean_pair(*t3["CR"]))
    print("ER", mean_pair(*t3["ER"]))

    t4 = R["table4"]["A"]
    print("% --- Table 4 ---")
    for k in ["E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x"]:
        print(k, mean_pair(t4["0"][k], t4["1"][k]))

    print("% --- Table 5 ---")
    for name, row in R["table5"].items():
        print(name, "params", row["params"])
        print("  E5", mean_pair(row["0"]["E5_1x"], row["1"]["E5_1x"]))
        print("  CM", mean_pair(row["0"]["CM"], row["1"]["CM"]))
        print("  EM", mean_pair(row["0"]["EM"], row["1"]["EM"]))
        print("  CR", mean_pair(row["0"]["CR"], row["1"]["CR"]))
        print("  ER", mean_pair(row["0"]["ER"], row["1"]["ER"]))
        print("  E2@1/2/3", mean_pair(row["0"]["E2_1x"], row["1"]["E2_1x"]),
              mean_pair(row["0"]["E2_2x"], row["1"]["E2_2x"]),
              mean_pair(row["0"]["E2_3x"], row["1"]["E2_3x"]))
        print("  train_s", mean_pair(*row["train_s_per_ep"]))
        print("  infer_s", mean_pair(row["0"]["infer_s"], row["1"]["infer_s"]))

    print("% --- Table 6 ---")
    for k, row in R["table6"].items():
        print(k, {kk: (sci(vv) if "C" in kk and kk.startswith("C") else vv) for kk, vv in row.items() if kk in ("E5_1x", "CM", "EM", "CR", "ER", "E2_3x", "infer_s")})

    print("fig1", R["fig1"])
    print("fig3M", R["fig3_M_videos"], "fig3R", R["fig3_R_videos"], "fig5", R["fig5_video"])


if __name__ == "__main__":
    main()
