"""
freeze_subsets_and_splits.py   --   env: umls_eval   [CPU]
================================================================================
Stage 0: codify two things the paper relies on but never persisted.

1. The drug-controlled subset (1,895 items). In the paper it was computed
   inline: items whose candidate drug appears as BOTH a positive and a negative
   within its family. Here it is written as an index list with a checksum.

2. A frozen patient-level development/test split of the 3,200-item bank for
   the new experiments. The published results used every item with no tuning
   split; the new retrieval and ranking experiments must tune k, token budgets
   and thresholds on development items only. Split unit = subject_id (patient),
   stratified by family; all items of a patient fall on one side; matched
   pairs (same admission) therefore never straddle the split.

Nothing frozen is modified. Outputs (untracked):
  outputs/stage0/drug_balanced_subset_v2.json   {item_index: [...]}
  outputs/stage0/split_v1.json                  {dev: [...], test: [...]} (item indices)
  outputs/stage0/split_v1_stats.yaml            counts only (safe to quote)
"""
from __future__ import annotations
import json, hashlib, collections
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P3   = HERE.parents[1] / "tasks" / "outputs" / "mm"
P2   = HERE.parents[1] / "graph" / "outputs"
OUT  = HERE.parent / "outputs" / "stage0"
OUT.mkdir(parents=True, exist_ok=True)
BENCH = P3 / "composite_benchmark_wide_v2.jsonl"
DEV_FRAC, SEED = 0.25, 20260911


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def main():
    items = [json.loads(l) for l in open(BENCH)]
    fam = [it["family"] for it in items]

    # ---- 1. drug-controlled subset (exactly the paper's rule) ----------------
    keep = []
    for f in sorted(set(fam)):
        idx = [k for k, it in enumerate(items) if it["family"] == f]
        P = {items[k]["drug"] for k in idx if items[k]["answer"] == "yes"}
        N = {items[k]["drug"] for k in idx if items[k]["answer"] == "no"}
        keep += [k for k in idx if items[k]["drug"] in (P & N)]
    keep = sorted(keep)
    (OUT / "drug_balanced_subset_v2.json").write_text(json.dumps(
        {"benchmark": BENCH.name, "benchmark_sha256": sha(BENCH),
         "rule": "candidate drug occurs as both a positive and a negative within its family",
         "n": len(keep), "item_index": keep}, indent=1))
    print(f"drug-balanced subset: {len(keep)} items (paper: 1,895)")

    # ---- 2. patient-level dev/test split -----------------------------------
    icu = pd.read_parquet(P2 / "ihm" / "icu_index.parquet", columns=["hadm_id", "subject_id"])
    h2s = dict(zip(icu.hadm_id, icu.subject_id))
    subj = np.array([h2s[it["hadm_id"]] for it in items])
    rng = np.random.default_rng(SEED)
    dev_subjects = set()
    # stratify by family: within each family, sample ~DEV_FRAC of that family's
    # patients; a patient present in several families is assigned once.
    assigned = set()
    for f in sorted(set(fam)):
        subs = sorted({s for s, ff in zip(subj, fam) if ff == f and s not in assigned})
        rng.shuffle(subs)
        n_dev = int(round(DEV_FRAC * len(subs)))
        dev_subjects |= set(subs[:n_dev]); assigned |= set(subs)
    dev = [k for k in range(len(items)) if subj[k] in dev_subjects]
    test = [k for k in range(len(items)) if subj[k] not in dev_subjects]
    assert not (set(dev) & set(test)) and len(dev) + len(test) == len(items)
    assert not ({subj[k] for k in dev} & {subj[k] for k in test}), "patient leak across split"
    (OUT / "split_v1.json").write_text(json.dumps(
        {"benchmark": BENCH.name, "benchmark_sha256": sha(BENCH), "unit": "subject_id",
         "seed": SEED, "dev_frac_patients": DEV_FRAC, "dev": dev, "test": test}, indent=1))
    stats = {"n_items": len(items), "n_patients": int(len(set(subj))),
             "dev": {"items": len(dev), "patients": int(len({subj[k] for k in dev})),
                     "pos": int(sum(items[k]["answer"] == "yes" for k in dev)),
                     "by_family": dict(collections.Counter(fam[k] for k in dev))},
             "test": {"items": len(test), "patients": int(len({subj[k] for k in test})),
                      "pos": int(sum(items[k]["answer"] == "yes" for k in test)),
                      "by_family": dict(collections.Counter(fam[k] for k in test))},
             "drug_balanced_in_test": int(len(set(keep) & set(test)))}
    def yml(d, ind=0):                       # tiny writer; avoids a pyyaml dependency
        out = []
        for k, v in d.items():
            if isinstance(v, dict):
                out.append(" " * ind + f"{k}:"); out.append(yml(v, ind + 2))
            else:
                out.append(" " * ind + f"{k}: {v}")
        return "\n".join(out)
    (OUT / "split_v1_stats.yaml").write_text(yml(stats) + "\n")
    print(yml(stats))


if __name__ == "__main__":
    main()
