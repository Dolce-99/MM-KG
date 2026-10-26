"""Metrics and paired bootstrap used by every phase-4 experiment.

Conventions follow phase3/19_composite_wide_llm.py exactly (rank-based AUROC with
average ranks for ties; 2,000-resample paired bootstrap with seed 0; two-sided
p = 2*min(P(d<=0), P(d>=0)); resamples with a single class are skipped) and add:
  * average precision (sklearn definition, tie-safe),
  * sensitivity / specificity / balanced accuracy at a threshold,
  * dev-selected threshold,
  * cluster (patient) bootstrap,
  * Holm adjustment.
"""
from __future__ import annotations
import math
import numpy as np


# ------------------------------------------------------------------ metrics
def rank_avg(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float); n = len(s)
    o = np.argsort(s, kind="mergesort"); sv = s[o]
    st = np.empty(n, bool); st[0] = True
    np.not_equal(sv[1:], sv[:-1], out=st[1:])
    g = np.cumsum(st) - 1; gs = np.flatnonzero(st); gc = np.diff(np.append(gs, n))
    r = np.empty(n, float); r[o] = (gs + (gc - 1) / 2.0 + 1.0)[g]
    return r


def auroc(y, s) -> float:
    y = np.asarray(y); s = np.asarray(s, float)
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    return float((rank_avg(s)[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def average_precision(y, s) -> float:
    """AP = sum_k (R_k - R_{k-1}) P_k over distinct score thresholds (descending)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return float("nan")
    o = np.argsort(-s, kind="mergesort"); ys = y[o]; ss = s[o]
    tp = np.cumsum(ys); k = np.arange(1, len(y) + 1)
    last = np.r_[ss[1:] != ss[:-1], True]        # last index of each tie block
    prec = tp[last] / k[last]; rec = tp[last] / npos
    drec = np.diff(np.r_[0.0, rec])
    return float((drec * prec).sum())


def sens_spec(y, pred) -> tuple[float, float]:
    y = np.asarray(y); pred = np.asarray(pred).astype(int)
    se = pred[y == 1].mean() if (y == 1).any() else float("nan")
    sp = 1 - pred[y == 0].mean() if (y == 0).any() else float("nan")
    return float(se), float(sp)


def balanced_acc(y, pred) -> float:
    se, sp = sens_spec(y, pred)
    return float((se + sp) / 2)


def at_threshold(y, s, thr: float) -> dict:
    pred = (np.asarray(s, float) >= thr).astype(int)
    se, sp = sens_spec(y, pred)
    return {"threshold": float(thr), "sensitivity": se, "specificity": sp,
            "balanced_accuracy": float((se + sp) / 2)}


def select_threshold(y, s) -> float:
    """Threshold maximising balanced accuracy on (dev) data; midpoint between
    neighbouring scores; ties resolved toward 0.5."""
    y = np.asarray(y); s = np.asarray(s, float)
    u = np.unique(s)
    cands = np.r_[u[0] - 1e-9, (u[1:] + u[:-1]) / 2, u[-1] + 1e-9]
    best = max(cands, key=lambda t: (balanced_acc(y, s >= t), -abs(t - 0.5)))
    return float(best)


# ------------------------------------------------------------------ bootstrap
def _resample_index(rng, n: int, clusters=None) -> np.ndarray:
    if clusters is None:
        return rng.integers(0, n, n)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[c] for c in pick])


def boot(y, cells: dict, fn, n: int = 2000, seed: int = 0, clusters=None, require_both_classes: bool = True) -> dict | None:
    """Paired bootstrap of the scalar fn(y, cells) -> float.

    cells: {name: array aligned with y}; fn receives the resampled arrays.
    clusters: optional array of cluster ids (e.g. patient) for a cluster bootstrap.
    Returns {"mean", "ci", "p"} with the 19_composite_wide_llm.py conventions, or
    None if no valid resample was produced.  The point estimate on the full data
    is left to the caller (fn(y, cells)).
    """
    rng = np.random.default_rng(seed); y = np.asarray(y); v = []
    arrs = {k: np.asarray(c) for k, c in cells.items()}
    for _ in range(n):
        i = _resample_index(rng, len(y), clusters)
        if require_both_classes and len(np.unique(y[i])) < 2:
            continue
        x = fn(y[i], {k: a[i] for k, a in arrs.items()})
        if x is not None and not (isinstance(x, float) and math.isnan(x)):
            v.append(x)
    if not v:
        return None
    v = np.asarray(v, float); p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return {"mean": round(float(v.mean()), 4),
            "ci": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
            "p": round(float(min(p, 1.0)), 4), "n_resamples": int(len(v))}


def boot_metric(y, s, metric=auroc, **kw) -> dict:
    """Point estimate + bootstrap CI of a single-arm metric."""
    est = metric(y, s)
    b = boot(y, {"s": s}, lambda yy, c: metric(yy, c["s"]), **kw)
    return {"estimate": round(float(est), 4), "ci": b["ci"] if b else None}


def boot_diff(y, s_a, s_b, metric=auroc, **kw) -> dict:
    """Paired difference metric(a) - metric(b) with CI and two-sided p."""
    est = metric(y, s_a) - metric(y, s_b)
    b = boot(y, {"a": s_a, "b": s_b}, lambda yy, c: metric(yy, c["a"]) - metric(yy, c["b"]), **kw)
    out = {"estimate": round(float(est), 4)}
    out.update(b or {})
    return out


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a family of tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, (m - i) * p); adj[k] = round(min(run, 1.0), 4)
    return adj
