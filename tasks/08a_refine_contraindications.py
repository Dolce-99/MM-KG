"""
08a_refine_contraindications.py   --   env: umls_eval   [CPU, ~5 min]
================================================================================
EMPIRICALLY-GROUNDED LINK REFINEMENT of PrimeKG's contraindication layer.

07 showed the layer as a whole is not reflected in prescribing behaviour (mean log-OR
-0.024 vs -0.092 for unlabelled pairs, p=1.0) even though the indication layer clearly
is (+0.827, p<0.001). Individual entries are nonetheless correct and strongly avoided.
So the layer is not wrong, it is DILUTED -- and the useful move is to separate the
entries clinicians act on from the ones they ignore.

SPLIT-HALF DESIGN (this is the part that matters)
Selecting pairs by their effect in the same data we later evaluate on would select
partly on noise -- with 5,316 pairs, hundreds will look "avoided" by chance. So:

  patients are split in half BY PATIENT (no pair appears in both halves)
    discovery half   -> require a significant avoidance signal (z < -Z_DISC)
    validation half  -> require the SAME DIRECTION independently (log-OR < 0)

Only pairs that replicate are called corroborated. The replication rate is compared
with the ~50% expected if the discovery signal were pure noise, which is what tells us
whether the refinement found anything real.

The same procedure run in reverse flags CONTRADICTED edges: pairs PrimeKG marks
contraindicated that are reliably prescribed MORE to affected patients. Spot-checking
these finds genuine PrimeKG errors, e.g. fondaparinux in heparin-induced
thrombocytopenia (fondaparinux is a treatment for HIT), clozapine in tardive
dyskinesia, cytarabine in neutropenia (the drug causes it; direction inverted).

Outputs (outputs/audit/):
  contraindications_corroborated.tsv   replicated, clinician-avoided
  contraindications_contradicted.tsv   replicated, clinician-preferred -> likely wrong
  refinement_stats.json
  refined_benchmark.jsonl              contraindication questions on the corroborated
                                       vs contradicted subsets, for the LLM run
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
MIN_N, Z_DISC, NEIGH = 60, 1.96, 12


def lor_table(pats, pairs_of_interest):
    """log-OR of P(drug | has condition) vs P(drug | not), for the given pairs."""
    N = len(pats)
    drug_n, cond_n, joint = defaultdict(int), defaultdict(int), defaultdict(int)
    for dn, cn in pats:
        for d in dn: drug_n[d] += 1
        for c in cn:
            cond_n[c] += 1
            for d in dn: joint[(d, c)] += 1
    out = {}
    for (d, c) in pairs_of_interest:
        if cond_n[c] < MIN_N or drug_n[d] < MIN_N: continue
        a = joint.get((d, c), 0); b = cond_n[c] - a
        cc = drug_n[d] - a; dd = N - cond_n[c] - cc
        if min(a, b, cc, dd) < 0: continue
        lor = math.log(((a + .5) / (b + .5)) / ((cc + .5) / (dd + .5)))
        se = math.sqrt(1/(a+.5) + 1/(b+.5) + 1/(cc+.5) + 1/(dd+.5))
        out[(d, c)] = (lor, lor / se, cond_n[c])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx2name, idx2type = {}, {}
    with open(PRIMEKG / "nodes.tab") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["node_index"]); idx2name[i] = r["node_name"]; idx2type[i] = r["node_type"]

    dz = pd.read_parquet(MS / "kg_cache" / "drug_disease.parquet")
    contra = set()
    contra_by_drug = defaultdict(set); indic_by_drug = defaultdict(set)
    for rel, x, xt, y in zip(dz.relation, dz.x_index, dz.x_type, dz.y_index):
        pair = (int(x), int(y)) if xt == "drug" else (int(y), int(x))
        if rel == "contraindication": contra.add(pair); contra_by_drug[pair[0]].add(pair[1])
        elif rel == "indication": indic_by_drug[pair[0]].add(pair[1])

    drug2pkg = {}
    for line in open(V2 / "drug2primekg_v2.tsv"):
        d, i, r, n = line.rstrip("\n").split("\t"); drug2pkg[d] = int(i)
    cond_name2idx = {nm: i for i, nm in idx2name.items() if idx2type.get(i) == "disease"}
    # whole-cohort audit base (09a), not the prediction-filtered subset
    recs = [json.loads(l) for l in open(OUT / "audit_cohort.jsonl")]
    pats = [(set(r["drug_nodes"]), set(r["cond_nodes"])) for r in recs]

    # ---- split BY PATIENT so no pair's evidence is shared across halves -----
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pats))
    half = len(order) // 2
    disc = [pats[i] for i in order[:half]]
    vald = [pats[i] for i in order[half:]]
    print(f"patients: {len(pats):,} -> discovery {len(disc):,} / validation {len(vald):,}")

    D = lor_table(disc, contra); V = lor_table(vald, contra)
    both = sorted(set(D) & set(V))
    print(f"contraindication pairs measurable in BOTH halves: {len(both):,}")

    corr, contra_wrong, rows = [], [], []
    for p in both:
        ld, zd, nd = D[p]; lv, zv, nv = V[p]
        rows.append({"drug": idx2name.get(p[0], p[0]), "cond": idx2name.get(p[1], p[1]),
                     "disc_log_or": round(ld, 3), "disc_z": round(zd, 2),
                     "val_log_or": round(lv, 3), "val_z": round(zv, 2)})
        if zd < -Z_DISC and lv < 0: corr.append(p)
        elif zd > Z_DISC and lv > 0: contra_wrong.append(p)

    # replication rate vs the ~50% expected if the discovery signal were noise
    disc_avoid = [p for p in both if D[p][1] < -Z_DISC]
    disc_prefer = [p for p in both if D[p][1] > Z_DISC]
    rep_avoid = len(corr) / max(len(disc_avoid), 1)
    rep_prefer = len(contra_wrong) / max(len(disc_prefer), 1)

    print(f"\n=== SPLIT-HALF REPLICATION (chance = 0.50) ===")
    print(f"  discovery says AVOIDED   : {len(disc_avoid):>5,}  replicated in held-out half: "
          f"{len(corr):>5,}  ({rep_avoid:.1%})")
    print(f"  discovery says PREFERRED : {len(disc_prefer):>5,}  replicated in held-out half: "
          f"{len(contra_wrong):>5,}  ({rep_prefer:.1%})")

    def dump(pairs, path):
        with open(path, "w") as w:
            for p in sorted(pairs, key=lambda q: D[q][0]):
                w.write(f"{p[0]}\t{p[1]}\t{idx2name.get(p[0],p[0])}\t{idx2name.get(p[1],p[1])}\t"
                        f"{D[p][0]:.3f}\t{V[p][0]:.3f}\n")
    dump(corr, OUT / "contraindications_corroborated.tsv")
    dump(contra_wrong, OUT / "contraindications_contradicted.tsv")
    pd.DataFrame(rows).to_csv(OUT / "contraindication_split_half.csv", index=False)

    print(f"\n  corroborated (clinicians avoid, replicated) : {len(corr):,}")
    for p in sorted(corr, key=lambda q: D[q][0])[:6]:
        print(f"    {idx2name.get(p[0],'')[:24]:24s} in {idx2name.get(p[1],'')[:30]:30s} "
              f"disc {D[p][0]:+.2f} / val {V[p][0]:+.2f}")
    print(f"\n  CONTRADICTED (clinicians prefer, replicated -> likely PrimeKG errors): {len(contra_wrong):,}")
    for p in sorted(contra_wrong, key=lambda q: -D[q][0])[:6]:
        print(f"    {idx2name.get(p[0],'')[:24]:24s} in {idx2name.get(p[1],'')[:30]:30s} "
              f"disc {D[p][0]:+.2f} / val {V[p][0]:+.2f}")

    # ---- benchmark on the two subsets --------------------------------------
    # Same question wording as the main audit, so results are directly comparable.
    # Negatives are drawn per subset from pairs that co-occur but carry no edge.
    cooc = defaultdict(set)
    for dn, cn in pats:
        for d in dn:
            for c in cn: cooc[d].add(c)
    items = []
    for subset, tag in ((corr, "corroborated"), (contra_wrong, "contradicted")):
        pos = list(subset)
        neg = [(d, c) for d in {p[0] for p in subset} for c in list(cooc[d])[:40]
               if c not in contra_by_drug.get(d, set())]
        k = min(len(pos), len(neg), 120)
        if k < 20:
            print(f"  (skipping {tag}: only {k} usable items)"); continue
        pi = rng.choice(len(pos), k, replace=False); ni = rng.choice(len(neg), k, replace=False)
        for src, idxs, ans in ((pos, pi, "yes"), (neg, ni, "no")):
            for i in idxs:
                d, c = src[i]
                edges = []
                if contra_by_drug.get(d):
                    edges.append(f"{idx2name[d]} --contraindicated_in--> " +
                                 "; ".join(idx2name.get(x, str(x)) for x in sorted(contra_by_drug[d])[:NEIGH]))
                if indic_by_drug.get(d):
                    edges.append(f"{idx2name[d]} --indicated_for--> " +
                                 "; ".join(idx2name.get(x, str(x)) for x in sorted(indic_by_drug[d])[:NEIGH]))
                items.append({"family": f"contra_{tag}", "hops": 1,
                              "question": f"Is {idx2name[d]} contraindicated in a patient with {idx2name[c]}?",
                              "answer": ans, "drugs": [d], "conds": [c],
                              "relations": ["contraindication"], "hadm_id": None,
                              "edges_true": edges,
                              "entity_names": [idx2name.get(d, ""), idx2name.get(c, "")],
                              "prevalence_stratum": "n/a"})
        print(f"  benchmark {tag}: {k} yes / {k} no")

    # length-matched wrong-fact and name-scrambled controls, as in 06a
    pool = [it["edges_true"] for it in items if it["edges_true"]]
    names = [idx2name[i] for i in list(idx2name)[:60000] if idx2type.get(i) in ("disease", "drug")]
    for it in items:
        src = pool[int(rng.integers(0, len(pool)))] if pool else []
        it["edges_shuffled"] = src[:max(len(it["edges_true"]), 1)]
        it["edges_corrupted"] = [e.partition("--> ")[0] + "--> " +
                                 "; ".join(names[int(rng.integers(0, len(names)))] for _ in range(3))
                                 for e in it["edges_true"]]
    rng.shuffle(items)
    with open(OUT / "refined_benchmark.jsonl", "w") as w:
        for it in items: w.write(json.dumps(it) + "\n")

    json.dump({"pairs_both_halves": len(both), "disc_avoided": len(disc_avoid),
               "corroborated": len(corr), "replication_rate_avoided": round(rep_avoid, 4),
               "disc_preferred": len(disc_prefer), "contradicted": len(contra_wrong),
               "replication_rate_preferred": round(rep_prefer, 4),
               "benchmark_items": len(items)},
              open(OUT / "refinement_stats.json", "w"), indent=2)
    print(f"\nwrote -> {OUT}/contraindications_{{corroborated,contradicted}}.tsv, refined_benchmark.jsonl")


if __name__ == "__main__":
    main()
