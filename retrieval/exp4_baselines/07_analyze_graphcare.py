"""
07_analyze_graphcare.py   --   env: torch_gpu (numpy)   [CPU]
================================================================================
Stay-paired comparison of the GraphCare re-implementation with the MM-KG GNN on the common
test stays: AUROC / AP with 2,000-resample paired bootstrap (seed 0) for every contrast.
Comparators (phase2 results_ft): Full (all channels), CodesKG (codes + PrimeKG, no labs/imaging/report),
Codes-KG (codes only), -KG (all channels, no PrimeKG).  Writes outputs/exp4/graphcare_comparison.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, stats

RES = paths.P2 / "ihm" / "results_ft"; D = paths.RESTRICTED / "exp4" / "graphcare_v1"; PUB = paths.OUT / "exp4"


def load(prefix, name, root):
    try:
        return np.load(root / f"preds_{name}.npy"), np.load(root / f"hadms_{name}.npy"), np.load(root / f"ytest_{name}.npy")
    except FileNotFoundError:
        return None


def main():
    arms = {}
    for v in ("codes", "codes_labs", "codes_nokg"):
        r = load("gc", v, D)
        if r: arms[f"GraphCare_{v}"] = r
    for c in ("Full", "CodesKG", "Codes-KG", "-KG", "-Labs"):
        r = load("mm", c, RES)
        if r: arms[f"MMKG_{c}"] = r
    common = None
    for p, h, y in arms.values():
        common = set(h.tolist()) if common is None else common & set(h.tolist())
    common = np.array(sorted(common)); out = {"n_common_test_stays": int(len(common)), "arms": {}, "contrasts": {}}
    S = {}
    for k, (p, h, y) in arms.items():
        pos = {int(hh): i for i, hh in enumerate(h)}; ii = [pos[int(c)] for c in common]; S[k] = (p[ii], y[ii])
        out["arms"][k] = {"auroc": round(stats.auroc(S[k][1], S[k][0]), 4), "auroc_ci": stats.boot_metric(S[k][1], S[k][0], stats.auroc, n=2000)["ci"],
                          "ap": round(stats.average_precision(S[k][1], S[k][0]), 4)}
    yref = next(iter(S.values()))[1]
    for k, (pk, yk) in S.items():
        assert np.array_equal(yk, yref)
    pairs = [(a, b) for a in S if a.startswith("GraphCare") for b in S if b.startswith("MMKG")] + [("GraphCare_codes_labs", "GraphCare_codes"), ("GraphCare_codes", "GraphCare_codes_nokg")]
    for a, b in pairs:
        if a in S and b in S:
            out["contrasts"][f"{a} - {b}"] = stats.boot_diff(yref, S[a][0], S[b][0], stats.auroc, n=2000, seed=0)
    json.dump(out, open(PUB / "graphcare_comparison.json", "w"), indent=2); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
