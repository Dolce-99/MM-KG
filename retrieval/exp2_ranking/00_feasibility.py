"""
00_feasibility.py   --   env: umls_eval   [CPU, ~1 min]
================================================================================
Feasibility counts for Experiment 2 (candidate-drug ranking with path-supported
explanations) over ALL 28,582 eligible ICU stays of the frozen cohort, using the
same frozen tables and bridge-v1 alignment as the join benchmark.

An admission is item-eligible when it received >= 5 PrimeKG-aligned drugs, at
least one of them has a PrimeKG contraindication (family "contra") or
adverse-effect (family "adverse") edge to a condition supported by eligible
evidence (prior coded diagnosis, early chest-film report finding, first-48h
laboratory state, BiomedCLIP imaging finding), and >= 4 other received drugs
have NO such edge to ANY supported condition (clean hard negatives).

Output: outputs/exp2/feasibility_v1.json (counts only; no identifiers).
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, frozen as fz

OUT = paths.OUT / "exp2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f2p, d2p, rep_by_study, img_by_study = fz.load_alignment_and_findings(verify=False)
    rel = fz.load_rel_cache(verify=False); ihm = fz.load_ihm(verify=False)
    contra, effect = rel["contra"], rel["effect"]
    split = paths.load_split()
    idxp = ihm["idx"][["hadm_id", "subject_id"]]
    h2s = dict(zip(idxp.hadm_id.astype(int), idxp.subject_id.astype(int)))
    bench = paths.load_benchmark(verify=False)
    dev_pat = {h2s[int(bench[i]["hadm_id"])] for i in split["dev"]}
    test_pat = {h2s[int(bench[i]["hadm_id"])] for i in split["test"]}

    C = Counter(); n_pos_dist = Counter(); n_clean_dist = Counter(); channel_support = Counter()
    elig_pat = set(); cand_pos_drugs = Counter(); cand_all_drugs = Counter(); fam_elig = Counter()
    for h, rec in ihm["recs"].items():
        drug_pkg = fz.drug_packages(h, ihm, d2p)
        C["stays"] += 1
        if len(drug_pkg) < 5:
            continue
        C["stays_ge5_aligned_drugs"] += 1
        coded = set()
        for cu in ihm["pdx"].get(h, []):
            coded |= f2p.get(cu, set())
        st = ihm["sof"].get(h)
        rep = set(); img = set()
        if st is not None:
            for cu in rep_by_study.get(st, set()):
                rep |= f2p.get(cu, set())
            for cu in img_by_study.get(st, set()):
                img |= f2p.get(cu, set())
        labs = set()
        for sname, (dis, col, test, tmpl) in fz.LAB_STATES.items():
            v = rec.get(col)
            if not pd.isna(v) and test(float(v)):
                labs |= set(dis)
        cond = {"codes": coded, "report": rep, "labs": labs, "imaging": img}
        all_cond = coded | rep | labs | img
        pos, clean = {}, []
        for p, dr in drug_pkg.items():
            hits_c = contra.get(p, set()) & all_cond; hits_e = effect.get(p, set()) & all_cond
            if hits_c or hits_e:
                fams = ("contra" if hits_c else "") + ("+adverse" if hits_e else "")
                chans = sorted(ch for ch, s in cond.items() if (hits_c | hits_e) & s)
                pos[p] = (fams.strip("+"), chans)
            elif not (contra.get(p, set()) | effect.get(p, set())):
                continue                                     # drug without any relation in the layers: not a valid clean negative
            else:
                clean.append(p)
        n_pos_dist[min(len(pos), 6)] += 1; n_clean_dist[min(len(clean), 10)] += 1
        if pos and len(clean) >= 4:
            C["item_eligible_stays"] += 1; elig_pat.add(h2s[h])
            if len(pos) == 1:
                C["item_eligible_exactly_one_positive"] += 1
            for p, (fam, chans) in pos.items():
                fam_elig[fam] += 1; channel_support["+".join(chans)] += 1
                cand_pos_drugs[p] += 1
            for p in drug_pkg:
                cand_all_drugs[p] += 1
            if len(clean) >= 9:
                C["item_eligible_candidate_set_10"] += 1
            if len(clean) >= 2:
                C["item_eligible_candidate_set_3"] += 1
    C["eligible_patients"] = len(elig_pat)
    C["eligible_patients_in_exp1_dev"] = len(elig_pat & dev_pat); C["eligible_patients_in_exp1_test"] = len(elig_pat & test_pat)
    out = {"counts": dict(C), "positives_per_eligible_stay(0-6+)": dict(sorted(n_pos_dist.items())),
           "clean_negatives_per_stay(0-10+)": dict(sorted(n_clean_dist.items())),
           "positive_family": dict(fam_elig), "support_channels": dict(channel_support.most_common(12)),
           "distinct_positive_drugs": len(cand_pos_drugs), "distinct_candidate_drugs": len(cand_all_drugs),
           "top_positive_drug_share": round(max(cand_pos_drugs.values()) / max(1, sum(cand_pos_drugs.values())), 3)}
    json.dump(out, open(OUT / "feasibility_v1.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
