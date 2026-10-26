"""
26_subset_interactions.py   --   env: umls_eval   [CPU, seconds]
================================================================================
Subset rows of the interaction table (Table 18 / Figure 3) recomputed from the
frozen score arrays written by 19_composite_wide_llm.py. Nothing is re-scored.

For every subset the four arms A_none / B_mm / C_kg / D_both are read from
cw_<tag>_<arm>.npy (aligned with the benchmark file order) and

    I_M = [M(D) - M(C)] - [M(B) - M(A)]

is bootstrapped with the same 2,000-resample paired item bootstrap (seed 0) as
the main analysis, with M = AUROC on the continuous P(yes) and M = balanced
accuracy at 0.5.

SUBSETS
  all                    the 3,200 items (reproduces the pooled row)
  <family>               each of the four question families
  drug_controlled        items whose candidate drug occurs as both a positive and
                         a negative within its family (index list codified by
                         phase4 stage0_audit/freeze_subsets_and_splits.py)
  leave_one_family_out   every family except drug_effect
  corroborated /         contra_labs + contra_report positives whose answer-bearing
  contradicted           (drug, condition) contraindication edge lies in the
                         split-half partition of 08a_refine_contraindications.py,
                         each kept together with its same-admission matched
                         negative. Drug strings map to PrimeKG drug nodes through
                         02a's cohort_drug2pkg.tsv; conditions match on the PrimeKG
                         node name recorded in the item's `why` field. A positive
                         with one corroborated and one contradicted condition
                         appears in both subsets.

Usage (defaults assume the original tree):
  python 26_subset_interactions.py --tag widev2_q35
Output: outputs/mm/subset_interactions_<tag>.json (+ printed table)
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ["A_none", "B_mm", "C_kg", "D_both"]
CONTRA = ("contra_labs", "contra_report")


def balanced_acc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    se = p[y == 1].mean() if (y == 1).any() else np.nan
    sp = 1 - p[y == 0].mean() if (y == 0).any() else np.nan
    return float((se + sp) / 2)


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


def boot(y, cells, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        x = fn(y[i], {k: np.asarray(c)[i] for k, c in cells.items()})
        if not math.isnan(x):
            v.append(x)
    if not v:
        return None
    v = np.asarray(v); p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return {"mean": round(float(v.mean()), 4),
            "ci": [round(float(np.percentile(v, 2.5)), 4),
                   round(float(np.percentile(v, 97.5)), 4)],
            "p": round(float(min(p, 1.0)), 4)}


def interaction(metric):
    return lambda yb, c: ((metric(yb, c["D_both"]) - metric(yb, c["C_kg"]))
                          - (metric(yb, c["B_mm"]) - metric(yb, c["A_none"])))


def partition_mask(items, tsv, drug2pkg):
    edges = set()
    for r in csv.reader(open(tsv), delimiter="\t"):
        edges.add((r[0], r[3].lower()))
    negatives = {(it["hadm_id"], it["family"]): k for k, it in enumerate(items) if it["answer"] == "no"}
    mask = np.zeros(len(items), bool)
    for k, it in enumerate(items):
        if it["family"] not in CONTRA or it["answer"] != "yes":
            continue
        d = drug2pkg.get(it["drug"])
        if d and any((d, w.lower()) in edges for w in it["why"]):
            mask[k] = True
            mask[negatives[(it["hadm_id"], it["family"])]] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="widev2_q35")
    ap.add_argument("--mm", default=str(HERE / "outputs" / "mm"))
    ap.add_argument("--benchmark", default="composite_benchmark_wide_v2.jsonl")
    ap.add_argument("--subset", default=str(HERE.parent / "retrieval" / "outputs" / "stage0"
                                            / "drug_balanced_subset_v2.json"))
    ap.add_argument("--audit", default=str(HERE / "outputs" / "audit"))
    ap.add_argument("--kg_cache", default=str(HERE / "outputs" / "medsafety" / "kg_cache"))
    args = ap.parse_args()
    mm, audit, kgc = Path(args.mm), Path(args.audit), Path(args.kg_cache)

    items = [json.loads(l) for l in open(mm / args.benchmark)]
    y = np.array([1 if it["answer"] == "yes" else 0 for it in items])
    S = {a: np.load(mm / f"cw_{args.tag}_{a}.npy") for a in ARMS}
    assert all(len(s) == len(items) for s in S.values()), "score arrays do not match the benchmark"
    fam = np.array([it["family"] for it in items])

    drug2pkg = {r[0]: r[1] for r in csv.reader(open(kgc / "cohort_drug2pkg.tsv"), delimiter="\t")}
    sub = json.load(open(args.subset))
    dc = np.zeros(len(items), bool); dc[sub["item_index"]] = True

    subsets = [("all", np.ones(len(items), bool))]
    subsets += [(f, fam == f) for f in dict.fromkeys(fam)]
    subsets += [("drug_controlled", dc),
                ("leave_one_family_out", fam != "drug_effect"),
                ("corroborated", partition_mask(items, audit / "contraindications_corroborated.tsv", drug2pkg)),
                ("contradicted", partition_mask(items, audit / "contraindications_contradicted.tsv", drug2pkg))]

    res = {"tag": args.tag, "benchmark": args.benchmark, "subsets": {}}
    for name, mask in subsets:
        yy = y[mask]
        cont = {a: S[a][mask] for a in ARMS}
        binr = {a: (S[a][mask] > 0.5).astype(int) for a in ARMS}
        res["subsets"][name] = {
            "n": int(mask.sum()), "positives": int(yy.sum()),
            "auroc": {a: round(auroc(yy, cont[a]), 4) for a in ARMS},
            "bal_acc": {a: round(balanced_acc(yy, binr[a]), 4) for a in ARMS},
            "interaction_auroc": boot(yy, cont, interaction(auroc)),
            "interaction_bal_acc": boot(yy, binr, interaction(balanced_acc)),
        }

    dest = mm / f"subset_interactions_{args.tag}.json"
    json.dump(res, open(dest, "w"), indent=2)

    print(f"{'subset':22s}{'n':>6}{'A':>7}{'B':>7}{'C':>7}{'D':>7}"
          f"{'I_AUROC [95% CI]':>28}{'p':>8}{'I_balacc':>10}")
    for name, b in res["subsets"].items():
        a, i, j = b["auroc"], b["interaction_auroc"], b["interaction_bal_acc"]
        print(f"{name:22s}{b['n']:6d}{a['A_none']:7.3f}{a['B_mm']:7.3f}{a['C_kg']:7.3f}{a['D_both']:7.3f}"
              f"{i['mean']:+.3f} [{i['ci'][0]:+.3f}, {i['ci'][1]:+.3f}]".rjust(28)
              + f"{i['p']:8.4f}{j['mean']:+10.3f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
