"""Unit tests for common.stats / common.yamlw (CPU, numpy only)."""
import itertools, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import stats, yamlw


def brute_auroc(y, s):
    pos = [a for a, b in zip(s, y) if b == 1]; neg = [a for a, b in zip(s, y) if b == 0]
    return np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg])


def brute_ap(y, s):
    o = np.argsort(-np.asarray(s), kind="mergesort"); y = np.asarray(y)[o]; s = np.asarray(s)[o]
    npos = y.sum(); ap = 0.0; prev_r = 0.0
    for t in sorted(set(s), reverse=True):
        m = s >= t; tp = y[m].sum(); prec = tp / m.sum(); rec = tp / npos
        ap += (rec - prev_r) * prec; prev_r = rec
    return ap


def test_auroc_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = rng.integers(4, 30); y = rng.integers(0, 2, n)
        if y.min() == y.max(): continue
        s = rng.integers(0, 5, n).astype(float)          # many ties
        assert math.isclose(stats.auroc(y, s), brute_auroc(y, s), abs_tol=1e-12)


def test_ap_matches_brute_force():
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = rng.integers(4, 30); y = rng.integers(0, 2, n)
        if y.min() == y.max(): continue
        s = rng.integers(0, 4, n).astype(float)
        assert math.isclose(stats.average_precision(y, s), brute_ap(y, s), abs_tol=1e-12)


def test_threshold_and_balanced_accuracy():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.4, 0.6, 0.9])
    assert stats.balanced_acc(y, s >= 0.5) == 1.0
    thr = stats.select_threshold(y, s); assert 0.4 < thr < 0.6
    r = stats.at_threshold(y, s, thr); assert r["sensitivity"] == 1.0 and r["specificity"] == 1.0


def test_bootstrap_sign_and_clusters():
    rng = np.random.default_rng(3); n = 400
    y = rng.integers(0, 2, n); a = y + rng.normal(0, 0.8, n); b = rng.normal(0, 1, n)
    d = stats.boot_diff(y, a, b, n=300, seed=0)
    assert d["estimate"] > 0 and d["ci"][0] > 0 and d["p"] < 0.05
    cl = np.repeat(np.arange(n // 4), 4)
    dc = stats.boot_diff(y, a, b, n=300, seed=0, clusters=cl)
    assert dc["n_resamples"] > 0 and abs(dc["mean"] - d["mean"]) < 0.1
    # same seed -> identical result (determinism)
    assert stats.boot_diff(y, a, b, n=100, seed=7) == stats.boot_diff(y, a, b, n=100, seed=7)


def test_holm():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06


def test_yaml_roundtrip_shape(tmp_path=Path("/tmp")):
    obj = {"a": 1, "b": [1, 2, {"c": "x y", "d": None}], "e": {"f": True, "g": 1.5, "h": "yes", "i": []}, "j": {}}
    txt = yamlw.dumps(obj)
    assert 'h: "yes"' in txt and "d: null" in txt and "f: true" in txt and "i: []" in txt and "j: {}" in txt
    try:
        import yaml
        assert yaml.safe_load(txt) == obj
    except ImportError:
        pass


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
