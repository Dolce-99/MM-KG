"""
01_build_ranking_benchmark.py   --   env: umls_eval   [CPU, ~2 min]
================================================================================
Experiment 2 benchmark: CANDIDATE-DRUG RANKING WITH PATH-SUPPORTED EXPLANATIONS
(a graph-reasoning benchmark; not treatment recommendation).

One item = one ICU stay (one per patient) + K=5 candidate drugs the patient
actually received in the first 48 h, exactly one of which ("positive") has a
PrimeKG contraindication or adverse-effect edge to a condition supported by
eligible patient evidence.  The 4 hard negatives received the same admission,
have NO such edge to ANY condition the patient has (coded history, report,
laboratory state, imaging), carry no condition name that is a lexical
near-synonym of anything visible in the record, and are matched to the positive
on prescription frequency, PrimeKG degree, relation-packet length and drug class.

Support policy (primary): the supporting evidence must include a first-48h
laboratory state or a chest-film report finding (the multimodal join); whether the
condition is ALSO in the coded history is stored as the `readable_in_codes` stratum.
Candidate identity is balanced with a per-drug cap; candidate order is shuffled
with a fixed seed; the split is by patient and extends split_v1 (same seed/rule).

Outputs
  outputs/restricted/exp2/ranking_v1/items.jsonl        (RESTRICTED: stay/patient ids, evidence)
  outputs/restricted/exp2/ranking_v1/patient_units.jsonl
  outputs/exp2/ranking_v1_audit.json                     (counts only)
  outputs/exp2/ranking_v1_split.json                     (item indices per split; no identifiers)
"""
from __future__ import annotations
import importlib.util, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, frozen as fz, runlog

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--max_total_dist", type=int, default=12, help="sum over the K-1 decoys of the bin distance to the positive")
_ap.add_argument("--version", default=None)
_ap.add_argument("--trainbank_of", default=None, help="build a patient-disjoint TRAINING bank excluding the patients of this frozen version and of split_v1")
_args = _ap.parse_args()
SEED = 20260911; K = 5; N_MAX = 2000 if not _args.trainbank_of else 100000; CAP_FRAC = 0.03
MAX_TOTAL_DIST = _args.max_total_dist
VERSION = _args.version or f"ranking_v1_d{MAX_TOTAL_DIST}"
DEST = paths.RESTRICTED / "exp2" / VERSION; PUB = paths.OUT / "exp2"
_spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parents[1] / "exp1_retrieval" / "01_build_evidence_collection.py")
ev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ev)

_STOP = {"disease", "diseases", "disorder", "disorders", "syndrome", "finding", "findings", "chronic", "acute", "other",
         "unspecified", "value", "qualifier", "morphologic", "abnormality", "structure", "primary", "secondary", "left",
         "right", "increase", "answer", "question", "combination", "bilateral", "moderate", "severe", "small", "large"}


def content_tokens(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 5 and w not in _STOP}


def near_synonym(a, b):
    return bool(a) and bool(b) and (a <= b or b <= a)


def lbin(x):
    return int(math.log2(x + 1))


def bins_of(p, freq, degree, contra, effect, dclass):
    return (lbin(freq[p]), lbin(degree.get(p, 0)), lbin(len(contra.get(p, ())) + len(effect.get(p, ()))), dclass.get(p, "unknown"))


def heuristic_hits1(items, key, reverse):
    """Hits@1 of ranking candidates by a single label-blind attribute (chance = 1/K)."""
    hit = 0
    for it in items:
        order = sorted(it["candidates"], key=lambda c: (key(c), c["name"]), reverse=reverse)
        hit += order[0]["positive"]
    return round(hit / len(items), 4)


def main():
    run = runlog.Run("exp2", f"build_{VERSION}", {"seed": SEED, "K": K, "N_MAX": N_MAX, "cap_frac": CAP_FRAC, "max_total_dist": MAX_TOTAL_DIST,
                                                  "support_policy": "labs_or_report_required; codes/imaging recorded as strata",
                                                  "negatives": "clean vs all channels + lexical guard; matched on freq/degree/packet/class"})
    f2p, d2p, rep_by_study, img_by_study = fz.load_alignment_and_findings()
    findings = (rep_by_study, img_by_study)
    rel = fz.load_rel_cache(); ihm = fz.load_ihm(); cui_name = fz.load_cui_names(); name = rel["name"]
    contra, effect = rel["contra"], rel["effect"]
    onehop = pd.read_parquet(paths.OUT / "exp1" / "kg_onehop_v1.parquet", columns=["x_index", "hop_role"])
    degree = onehop[onehop.hop_role == "linked"].groupby("x_index").size().to_dict()
    dfeat = pd.read_csv(paths.PRIMEKG / "drug_features.tab", sep="\t", usecols=["node_index", "atc_1"], dtype=str)
    dclass = {}
    for r in dfeat.itertuples():
        m = re.search(r"anatomically related to (.+?)(?: and |\.|$)", str(r.atc_1))
        dclass[int(r.node_index)] = m.group(1).strip() if m else "unknown"
    h2s = dict(zip(ihm["idx"].hadm_id.astype(int), ihm["idx"].subject_id.astype(int)))
    # prescription frequency at PrimeKG-drug level (number of cohort stays)
    freq = Counter()
    for h in ihm["recs"]:
        for p in fz.drug_packages(h, ihm, d2p):
            freq[p] += 1

    # ---- per-stay candidate analysis --------------------------------------------------
    rng = np.random.default_rng(SEED)
    stays = []
    for h, rec in ihm["recs"].items():
        drug_pkg = fz.drug_packages(h, ihm, d2p)
        if len(drug_pkg) < K:
            continue
        coded = set()
        for cu in ihm["pdx"].get(h, []):
            coded |= f2p.get(cu, set())
        st = ihm["sof"].get(h)
        rep_cu = {}; img = set()
        if st is not None:
            for cu in rep_by_study.get(st, set()):
                if f2p.get(cu):
                    rep_cu[cu] = f2p[cu]
            for cu in img_by_study.get(st, set()):
                img |= f2p.get(cu, set())
        rep = set().union(*rep_cu.values()) if rep_cu else set()
        labs = {}
        for sname, (dis, col, test, tmpl) in fz.LAB_STATES.items():
            v = rec.get(col)
            if not pd.isna(v) and test(float(v)):
                labs[col.split("_")[1]] = (set(dis), sname)
        lab_all = set().union(*(s for s, _ in labs.values())) if labs else set()
        all_cond = coded | rep | lab_all | img
        mm_cond = rep | lab_all
        vis_names = [cui_name.get(cu, cu) for cu in (rep_by_study.get(st, set()) | img_by_study.get(st, set()))] if st is not None else []
        vis_names += [cui_name.get(cu, cu) for cu in ihm["pdx"].get(h, [])]
        vis_keys = [k for k in (content_tokens(v) for v in vis_names) if k]
        pos, clean = {}, []
        for p, dr in drug_pkg.items():
            rels = {"contraindication": contra.get(p, set()), "drug_effect": effect.get(p, set())}
            hits = {r: s & all_cond for r, s in rels.items()}
            if any(hits.values()):
                mm_hits = {r: s & mm_cond for r, s in hits.items()}
                if any(mm_hits.values()):
                    pos[p] = (dr, mm_hits, {r: s & coded for r, s in hits.items()})
                continue
            if not (rels["contraindication"] or rels["drug_effect"]):
                continue                                              # no packet at all: not a valid decoy
            kg_keys = (content_tokens(name.get(x, str(x))) for x in (rels["contraindication"] | rels["drug_effect"]))
            if any(near_synonym(k, v) for k in kg_keys for v in vis_keys):
                continue
            clean.append((p, dr))
        if pos and len(clean) >= K - 1:
            stays.append({"hadm": h, "subject": h2s[h], "pos": pos, "clean": clean, "coded": coded, "rep_cu": rep_cu,
                          "labs": labs, "study": st, "rec": rec})
    print(f"eligible stays {len(stays)} / patients {len({s['subject'] for s in stays})}", flush=True)

    # ---- one stay per patient, seeded order, per-drug cap -------------------------------
    by_pat = defaultdict(list)
    for s in stays:
        by_pat[s["subject"]].append(s)
    if _args.trainbank_of:
        excl = {it["subject"] for it in map(json.loads, open(paths.RESTRICTED / "exp2" / _args.trainbank_of / "items.jsonl"))}
        split1 = paths.load_split(); bench = paths.load_benchmark(verify=False)
        excl |= {h2s[int(bench[i]["hadm_id"])] for i in split1["dev"] + split1["test"]}
        by_pat = {k: v for k, v in by_pat.items() if k not in excl}
        print(f"training bank: {len(by_pat)} eligible patients after excluding {len(excl)} dev/test patients", flush=True)
    chosen = [max(v, key=lambda s: (len(s["clean"]), -s["hadm"])) for v in by_pat.values()]
    order = rng.permutation(len(chosen)); chosen = [chosen[i] for i in order]
    def best_dist(s):
        out = []
        for p in s["pos"]:
            pb_ = bins_of(p, freq, degree, contra, effect, dclass)
            ds = sorted(abs(pb_[0] - qb[0]) + abs(pb_[1] - qb[1]) + abs(pb_[2] - qb[2]) + (0 if pb_[3] == qb[3] else 1)
                        for qb in (bins_of(q, freq, degree, contra, effect, dclass) for q, _ in s["clean"]))
            out.append(sum(ds[:K - 1]))
        return min(out)
    feasible = sum(best_dist(s) <= MAX_TOTAL_DIST for s in chosen)
    cap = max(5, int(CAP_FRAC * min(N_MAX, feasible))) if not _args.trainbank_of else 10**9   # training bank: no identity cap (more data matters more than balance)
    print(f"stays passing the distance filter: {feasible}; per-drug cap {cap}", flush=True)
    use_pos, use_neg = Counter(), Counter()
    items = []; skipped_dist = 0
    for s in chosen:
        if len(items) >= N_MAX:
            break
        # joint choice: the positive whose best K-1 clean decoys are closest in (freq, degree, packet, class) bins;
        # ties broken by usage balance, then seeded randomness.  Usage of negatives is only a tie-breaker.
        best = None
        for p in s["pos"]:
            if use_pos[p] >= cap:
                continue
            pb_ = bins_of(p, freq, degree, contra, effect, dclass)
            sc = []
            for q, dq in s["clean"]:
                qb = bins_of(q, freq, degree, contra, effect, dclass)
                dist = abs(pb_[0] - qb[0]) + abs(pb_[1] - qb[1]) + abs(pb_[2] - qb[2]) + (0 if pb_[3] == qb[3] else 1)
                sc.append((dist, 0.01 * use_neg[q], rng.random(), q, dq, qb))
            sc.sort(); top = sc[:K - 1]; D = sum(x[0] for x in top)
            key = (D, use_pos[p], rng.random())
            if best is None or key < best[0]:
                best = (key, p, pb_, top)
        if best is None:
            continue
        (D, _, _), p, pb, negs = best
        if D > MAX_TOTAL_DIST:
            skipped_dist += 1; continue
        dr, mm_hits, coded_hits = s["pos"][p]
        use_pos[p] += 1
        for _, _, _, q, _, _ in negs:
            use_neg[q] += 1
        # gold: supported conditions and their evidence units
        gold_paths, cond_names, ev_uids, chans = [], set(), set(), set()
        for r, conds in mm_hits.items():
            for c in sorted(conds):
                cond_names.add(name.get(c, str(c)))
                for k, (dis, sname) in s["labs"].items():
                    if c in dis:
                        gold_paths.append({"patient_uid": f"lab:{k}", "condition": c, "condition_name": name.get(c, str(c)),
                                           "edges": [[p, r, c]], "channel": "labs", "rule": f"{k} threshold -> {sname}"})
                        ev_uids.add(f"lab:{k}"); chans.add("labs")
                for cu, t in s["rep_cu"].items():
                    if c in t:
                        gold_paths.append({"patient_uid": f"rep:{cu}", "condition": c, "condition_name": name.get(c, str(c)),
                                           "edges": [[p, r, c]], "channel": "report"})
                        ev_uids.add(f"rep:{cu}"); chans.add("report")
        readable = any(coded_hits.values())
        cand_list = [{"pkg": p, "name": name.get(p, dr), "mimic_name": dr, "positive": True, "bins": pb}] + \
                    [{"pkg": q, "name": name.get(q, dq), "mimic_name": dq, "positive": False, "bins": qb, "match_dist": int(d)}
                     for d, _, _, q, dq, qb in negs]
        perm = rng.permutation(K); cand_list = [cand_list[i] for i in perm]
        items.append({"idx": len(items), "hadm": s["hadm"], "subject": s["subject"], "study": s["study"], "candidates": cand_list,
                      "positive_position": int(np.flatnonzero(perm == 0)[0]),
                      "gold": {"selected_pkg": p, "selected_name": name.get(p, dr), "condition_pkg": sorted({c for cs in mm_hits.values() for c in cs}),
                               "condition_names": sorted(cond_names), "patient_uids": sorted(ev_uids), "paths": gold_paths,
                               "kg_edges": sorted({tuple(e) for gp in gold_paths for e in gp["edges"]}),
                               "support_channels": sorted(chans), "readable_in_codes": readable,
                               "relation_families": sorted(r for r, cs in mm_hits.items() if cs)},
                      "n_positive_eligible": len(s["pos"]), "n_clean": len(s["clean"])})
    # ---- split by patient, extending split_v1 ------------------------------------------
    split1 = paths.load_split(); bench = paths.load_benchmark(verify=False)
    dev_pat = {h2s[int(bench[i]["hadm_id"])] for i in split1["dev"]}; test_pat = {h2s[int(bench[i]["hadm_id"])] for i in split1["test"]}
    new_pat = sorted({it["subject"] for it in items} - dev_pat - test_pat)
    r2 = np.random.default_rng(SEED); pick = set(np.array(new_pat)[r2.permutation(len(new_pat))[: int(0.25 * len(new_pat))]].tolist())
    for it in items:
        it["split"] = "train" if _args.trainbank_of else ("dev" if (it["subject"] in dev_pat or it["subject"] in pick) else "test")
    assert not ({it["subject"] for it in items if it["split"] == "dev"} & {it["subject"] for it in items if it["split"] == "test"})

    DEST.mkdir(parents=True, exist_ok=True); PUB.mkdir(parents=True, exist_ok=True)
    with open(DEST / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    with open(DEST / "patient_units.jsonl", "w") as f:
        for s in chosen:
            if any(it["hadm"] == s["hadm"] for it in items[:0]):
                pass
        done = set()
        for it in items:
            if it["hadm"] in done:
                continue
            units, st = ev.patient_units(it["hadm"], ihm["recs"][it["hadm"]], ihm, f2p, d2p, findings, cui_name)
            f.write(json.dumps({"hadm": it["hadm"], "study": st, "units": units}) + "\n"); done.add(it["hadm"])
    json.dump({"train": [it["idx"] for it in items if it["split"] == "train"], "dev": [it["idx"] for it in items if it["split"] == "dev"], "test": [it["idx"] for it in items if it["split"] == "test"],
               "seed": SEED, "extends": "stage0/split_v1.json"}, open(PUB / f"{VERSION}_split.json", "w"))
    audit = {"version": VERSION, "n_items": len(items), "n_patients": len({it["subject"] for it in items}),
             "eligible_stays": len(stays), "eligible_patients": len(by_pat), "cap_per_drug": cap,
             "split": {s: sum(it["split"] == s for it in items) for s in ("train", "dev", "test")},
             "positive_position": dict(Counter(it["positive_position"] for it in items)),
             "distinct_positive_drugs": len(use_pos), "max_positive_share": round(max(use_pos.values()) / len(items), 4),
             "distinct_negative_drugs": len(use_neg), "max_negative_share": round(max(use_neg.values()) / (len(items) * (K - 1)), 4),
             "support_channels": dict(Counter("+".join(it["gold"]["support_channels"]) for it in items)),
             "relation_families": dict(Counter("+".join(it["gold"]["relation_families"]) for it in items)),
             "readable_in_codes": dict(Counter(it["gold"]["readable_in_codes"] for it in items)),
             "gold_paths_per_item": {"mean": round(float(np.mean([len(it["gold"]["paths"]) for it in items])), 2), "max": max(len(it["gold"]["paths"]) for it in items)},
             "negative_match_distance": {"mean": round(float(np.mean([c["match_dist"] for it in items for c in it["candidates"] if not c["positive"]])), 3),
                                         "exact_bins": round(float(np.mean([c["match_dist"] == 0 for it in items for c in it["candidates"] if not c["positive"]])), 3)},
             "stays_skipped_for_distance": skipped_dist,
             "heuristic_hits_at_1(chance 0.2)": {"packet_length_desc": heuristic_hits1(items, lambda c: c["bins"][2], True),
                                                 "packet_length_asc": heuristic_hits1(items, lambda c: c["bins"][2], False),
                                                 "degree_desc": heuristic_hits1(items, lambda c: c["bins"][1], True),
                                                 "frequency_desc": heuristic_hits1(items, lambda c: c["bins"][0], True),
                                                 "frequency_asc": heuristic_hits1(items, lambda c: c["bins"][0], False)},
             "positive_vs_negative_bins": {"freq_bin_mean": [round(float(np.mean([c["bins"][0] for it in items for c in it["candidates"] if c["positive"] == pos_])), 2) for pos_ in (True, False)],
                                           "degree_bin_mean": [round(float(np.mean([c["bins"][1] for it in items for c in it["candidates"] if c["positive"] == pos_])), 2) for pos_ in (True, False)],
                                           "packet_bin_mean": [round(float(np.mean([c["bins"][2] for it in items for c in it["candidates"] if c["positive"] == pos_])), 2) for pos_ in (True, False)]}}
    json.dump(audit, open(PUB / f"{VERSION}_audit.json", "w"), indent=2)
    for p_ in (DEST / "items.jsonl", DEST / "patient_units.jsonl", PUB / f"{VERSION}_split.json", PUB / f"{VERSION}_audit.json"):
        run.add_output(p_)
    run.finish(counts=audit); print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
