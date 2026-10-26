"""
07_external_validation.py   --   env: umls_eval   [CPU, ~5 min]
================================================================================
EXTERNAL VALIDATION OF PrimeKG's DRUG-DISEASE EDGES — no LLM, no KG-derived label.

The open boundary on every fact-checking result (ours and the ADNI group's) is that
ground truth comes from PrimeKG itself, so the benchmark shows an LLM can USE the
graph, not that the graph is CLINICALLY CORRECT. This closes that gap using a
reference standard that is independent of both the KG and the model: what clinicians
actually did.

  If PrimeKG's `contraindication` edges are real, clinicians should prescribe
  that drug LESS OFTEN to patients who have that condition.
  If PrimeKG's `indication` edges are real, they should prescribe it MORE OFTEN.

For every drug-condition pair observable in the cohort we compute

    log OR = log [ P(drug | has condition) / P(drug | no condition) ]

and compare three groups: PrimeKG contraindications, PrimeKG indications, and
unlabelled pairs (the null). Indications act as a POSITIVE CONTROL — if the method
cannot recover them, it cannot say anything about contraindications either.

This is a real test that can fail: if contraindicated pairs are prescribed at the
same rate as unlabelled ones, PrimeKG's contraindication layer is not reflected in
practice, and the fact-checking results must be reported as grounding-only.

Output: outputs/audit/external_validation.json  (+ printed report)
"""
from __future__ import annotations
import os
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
HERE    = Path(__file__).resolve().parent
OUT     = HERE / "outputs" / "audit"
MS      = HERE / "outputs" / "medsafety"
V2      = HERE / "outputs" / "bridge_v2"
MIN_N   = 150          # min patients with the condition, for a stable rate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx2name, idx2type = {}, {}
    with open(PRIMEKG / "nodes.tab") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["node_index"]); idx2name[i] = r["node_name"]; idx2type[i] = r["node_type"]

    contra, indic = set(), set()
    dz = pd.read_parquet(MS / "kg_cache" / "drug_disease.parquet")
    for rel, x, xt, y in zip(dz.relation, dz.x_index, dz.x_type, dz.y_index):
        pair = (int(x), int(y)) if xt == "drug" else (int(y), int(x))
        if rel == "contraindication": contra.add(pair)
        elif rel == "indication":     indic.add(pair)

    drug2pkg = {}
    for line in open(V2 / "drug2primekg_v2.tsv"):
        d, i, r, n = line.rstrip("\n").split("\t"); drug2pkg[d] = int(i)
    cond_name2idx = {nm: i for i, nm in idx2name.items() if idx2type.get(i) == "disease"}

    # whole-cohort audit base (09a), not the prediction-filtered subset
    recs = [json.loads(l) for l in open(OUT / "audit_cohort.jsonl")]
    pats = [(set(r["drug_nodes"]), set(r["cond_nodes"])) for r in recs]
    N = len(pats)
    print(f"patients with mapped conditions: {N:,}")

    drug_n = defaultdict(int); cond_n = defaultdict(int); joint = defaultdict(int)
    for dn, cn in pats:
        for d in dn: drug_n[d] += 1
        for c in cn:
            cond_n[c] += 1
            for d in dn: joint[(d, c)] += 1
    conds = [c for c, n in cond_n.items() if n >= MIN_N]
    drugs = [d for d, n in drug_n.items() if n >= MIN_N]
    print(f"drugs >= {MIN_N} patients: {len(drugs):,} | conditions >= {MIN_N}: {len(conds):,}")

    rows = []
    for d in drugs:
        for c in conds:
            a = joint.get((d, c), 0)            # has condition & got drug
            b = cond_n[c] - a                   # has condition, no drug
            cc = drug_n[d] - a                  # no condition, got drug
            dd = N - cond_n[c] - cc             # neither
            if min(a, b, cc, dd) < 0: continue
            lor = math.log(((a + .5) / (b + .5)) / ((cc + .5) / (dd + .5)))
            se = math.sqrt(1/(a+.5) + 1/(b+.5) + 1/(cc+.5) + 1/(dd+.5))
            lab = ("contraindication" if (d, c) in contra else
                   "indication" if (d, c) in indic else "unlabelled")
            rows.append({"drug": idx2name.get(d, d), "cond": idx2name.get(c, c),
                         "label": lab, "log_or": lor, "z": lor / se, "n_cond": cond_n[c]})
    df = pd.DataFrame(rows)
    print(f"drug-condition pairs evaluated: {len(df):,} "
          f"({(df.label=='contraindication').sum():,} contraindicated, "
          f"{(df.label=='indication').sum():,} indicated)")

    def summ(lab):
        s = df[df.label == lab]["log_or"]
        return {"n": int(len(s)), "mean_log_or": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "pct_negative": round(float((s < 0).mean()), 4)}
    res = {k: summ(k) for k in ("contraindication", "indication", "unlabelled")}

    # permutation test: is the contraindication mean below the unlabelled mean?
    rng = np.random.default_rng(0)
    ct = df[df.label == "contraindication"]["log_or"].to_numpy()
    ul = df[df.label == "unlabelled"]["log_or"].to_numpy()
    ind = df[df.label == "indication"]["log_or"].to_numpy()
    obs_c = ct.mean() - ul.mean(); obs_i = ind.mean() - ul.mean()
    pool_c = np.concatenate([ct, ul]); pool_i = np.concatenate([ind, ul])
    pc = pi = 0
    for _ in range(2000):
        rng.shuffle(pool_c); rng.shuffle(pool_i)
        if pool_c[:len(ct)].mean() - pool_c[len(ct):].mean() <= obs_c: pc += 1
        if pool_i[:len(ind)].mean() - pool_i[len(ind):].mean() >= obs_i: pi += 1
    res["contraindication_vs_unlabelled"] = {"delta_mean_log_or": round(float(obs_c), 4),
                                             "p_one_sided_lower": round(pc / 2000, 4)}
    res["indication_vs_unlabelled"] = {"delta_mean_log_or": round(float(obs_i), 4),
                                       "p_one_sided_higher": round(pi / 2000, 4)}
    json.dump(res, open(OUT / "external_validation.json", "w"), indent=2)

    print("\n=== DO CLINICIANS BEHAVE AS PrimeKG SAYS THEY SHOULD? ===")
    print("log OR > 0 = drug prescribed MORE to patients with the condition\n")
    print(f"  {'edge type':20s} {'n pairs':>9} {'mean logOR':>11} {'median':>9} {'% negative':>11}")
    for k in ("indication", "unlabelled", "contraindication"):
        r = res[k]
        print(f"  {k:20s} {r['n']:>9,} {r['mean_log_or']:>11.3f} {r['median']:>9.3f} {r['pct_negative']:>11.1%}")
    print(f"\n  POSITIVE CONTROL  indication vs unlabelled : "
          f"{res['indication_vs_unlabelled']['delta_mean_log_or']:+.3f} "
          f"(p={res['indication_vs_unlabelled']['p_one_sided_higher']}) -- should be POSITIVE")
    print(f"  TEST              contraindication vs unlabelled : "
          f"{res['contraindication_vs_unlabelled']['delta_mean_log_or']:+.3f} "
          f"(p={res['contraindication_vs_unlabelled']['p_one_sided_lower']}) -- should be NEGATIVE")
    print("\n  READ: if the positive control fails, the method is uninformative and neither")
    print("  direction should be reported. If indications are prescribed more AND")
    print("  contraindications less, PrimeKG's drug-disease layer is corroborated by")
    print("  real prescribing behaviour, independently of the KG and of the LLM.")

    strongest = df[df.label == "contraindication"].nsmallest(8, "log_or")
    print("\n  contraindications most avoided in practice:")
    for r in strongest.itertuples(index=False):
        print(f"    {r.drug[:26]:26s} in {r.cond[:30]:30s} logOR {r.log_or:+.2f}")
    df.to_csv(OUT / "external_validation_pairs.csv", index=False)
    print(f"\nwrote -> {OUT}/external_validation.json, external_validation_pairs.csv")


if __name__ == "__main__":
    main()
