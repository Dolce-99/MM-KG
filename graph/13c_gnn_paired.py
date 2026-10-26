"""
13c_gnn_paired.py   --   env: umls_eval   [CPU]
================================================================================
Paired significance tests for the GNN channel ablations, mirroring the LLM
ablation design exactly: contribution = AUROC(Full) - AUROC(-channel),
2,000-resample paired bootstrap over test admissions, 95% CI, two-sided p.
Reads the per-patient ensemble predictions saved by 13_icu_ihm_whole.py
(preds_<cfg>.npy in outputs/ihm/results_ft/). Alignment across configs is
verified on the saved hadm order before any statistic is computed.

Usage: python 13c_gnn_paired.py
Output: outputs/ihm/results_ft/gnn_paired_ablation.json (+ printed table)
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent / "outputs" / "ihm" / "results_ft"
CFGS = ["Full", "-History", "-Imaging", "-Report", "-KG", "-Labs", "-Meds"]


def _rank_avg(s):
    n = len(s); o = np.argsort(s, kind="mergesort"); sv = s[o]
    st = np.empty(n, bool); st[0] = True
    np.not_equal(sv[1:], sv[:-1], out=st[1:])
    g = np.cumsum(st) - 1; gs = np.flatnonzero(st); gc = np.diff(np.append(gs, n))
    r = np.empty(n, float); r[o] = (gs + (gc - 1) / 2.0 + 1.0)[g]
    return r


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    return float((_rank_avg(s)[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def ap(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    o = np.argsort(-s, kind="mergesort"); yy = y[o]
    tp = np.cumsum(yy)
    return float(((tp / np.arange(1, len(yy) + 1)) * yy).sum() / yy.sum())


def paired(y, sa, sb, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a, b = auroc(y[i], sa[i]), auroc(y[i], sb[i])
        if not (math.isnan(a) or math.isnan(b)):
            d.append(b - a)
    d = np.asarray(d); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"mean": round(float(d.mean()), 4),
            "ci": [round(float(np.percentile(d, 2.5)), 4),
                   round(float(np.percentile(d, 97.5)), 4)],
            "p": round(float(min(p, 1.0)), 4)}


def main():
    have = [c for c in CFGS if (RES / f"preds_{c}.npy").exists()]
    missing = [c for c in CFGS if c not in have]
    if missing:
        print(f"WARNING: missing preds for {missing} -- run 13b first")
    assert "Full" in have, "Full config predictions are required"

    ref_h = np.load(RES / "hadms_Full.npy")
    y = np.load(RES / "ytest_Full.npy")
    P = {}
    for c in have:
        h = np.load(RES / f"hadms_{c}.npy")
        assert np.array_equal(h, ref_h), f"test-set order mismatch in {c}"
        P[c] = np.load(RES / f"preds_{c}.npy")
    print(f"aligned test set: n={len(y)}, deaths={int(y.sum())}, "
          f"configs={have}", flush=True)

    out = {"n_test": int(len(y)), "test_pos": int(y.sum()),
           "full": {"auroc": round(auroc(y, P["Full"]), 4),
                    "ap": round(ap(y, P["Full"]), 4)},
           "contributions": {}}
    print(f"\n{'dropped':12s}{'AUROC':>8}{'AP':>8}{'contribution (95% CI)':>26}{'p':>8}")
    print(f"{'Full':12s}{out['full']['auroc']:8.4f}{out['full']['ap']:8.4f}")
    for c in have:
        if c == "Full":
            continue
        d = paired(y, P[c], P["Full"])
        out["contributions"][c] = {
            "auroc": round(auroc(y, P[c]), 4), "ap": round(ap(y, P[c]), 4),
            "delta_auroc": d,
            "delta_ap": round(out["full"]["ap"] - ap(y, P[c]), 4)}
        print(f"{c:12s}{out['contributions'][c]['auroc']:8.4f}"
              f"{out['contributions'][c]['ap']:8.4f}"
              f"   {d['mean']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]"
              f"{d['p']:>8}")
    (RES / "gnn_paired_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> {RES/'gnn_paired_ablation.json'}")


if __name__ == "__main__":
    main()
