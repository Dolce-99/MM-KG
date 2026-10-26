"""
01_build_evidence_collection.py   --   env: umls_eval   [CPU, ~5 min]
================================================================================
Builds the COMMON ADMISSIBLE EVIDENCE COLLECTION for Experiment 1 from the
frozen sources behind the 3,200-item join benchmark (composite_benchmark_wide_v2).

Every piece of patient evidence becomes a unit with a stable id, a channel, a
rendered text, a normalised concept, the PrimeKG nodes it aligns to (bridge v1,
exactly as the frozen graph), and provenance (source table, study).  Units are
admission-scoped and question-independent: prior-visit coded diagnoses,
first-48h medications, first-48h laboratory extremes, the early chest-film
report findings and BiomedCLIP imaging findings -- the same windows the frozen
benchmark used, so all evidence precedes the question time.

Validation: the frozen prompt blocks of all 3,200 items are RE-RENDERED from the
units with the verbatim phase-3 renderer and compared byte-for-byte.  Gold
(answer-bearing) evidence is derived from the graph side only (never from model
output) and checked against the frozen `evidence` / `why` fields.

Outputs
  outputs/restricted/exp1/evidence_v1/patient_units.jsonl   (per admission; RESTRICTED)
  outputs/restricted/exp1/evidence_v1/items.jsonl           (per item, label-blind; RESTRICTED)
  outputs/restricted/exp1/evidence_v1/gold.jsonl            (per item, answer-bearing units; RESTRICTED)
  outputs/exp1/linked_nodes_v1.json                          (PrimeKG node ids only)
  outputs/exp1/evidence_v1_validation.json                   (aggregate counts only)
No frozen file is modified.
"""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, frozen as fz, runlog

import os
VERSION = os.environ.get("MMKG_EVIDENCE_VERSION", "evidence_v1")
DEST = paths.RESTRICTED / "exp1" / VERSION
PUB = paths.OUT / "exp1"
HEADER_RE = re.compile(r"^Facts from a curated biomedical knowledge graph\. (.+?) "
                       r"(is contraindicated in|has the following documented adverse effects|"
                       r"acts on proteins implicated in)")


def patient_units(h, rec, ihm, f2p, d2p, findings, cui_name):
    """Question-independent, label-blind evidence units for one admission."""
    rep_by_study, img_by_study = findings
    units = []
    age = int(rec["age"]) if pd.notna(rec.get("age")) else None
    sex = {"M": "male", "F": "female"}.get(str(rec.get("gender")), "patient")
    units.append({"uid": "demo", "channel": "demo", "text": (f"A {age}-year-old {sex} patient in intensive care."
                  if age else f"A {sex} patient in intensive care."), "concept": None, "pkg": [],
                  "prov": {"table": "icu_index", "fields": ["age", "gender"]}})
    for cu in sorted(set(ihm["pdx"].get(h, []))):
        units.append({"uid": f"dx:{cu}", "channel": "codes", "text": cui_name.get(cu, cu),
                      "concept": {"type": "cui", "id": cu}, "pkg": sorted(f2p.get(cu, ())),
                      "prov": {"table": "prior_dx", "note": "diagnoses of admissions before the current one"}})
    for dr in dict.fromkeys(ihm["meds"].get(h, [])):
        units.append({"uid": f"med:{dr}", "channel": "meds", "text": dr,
                      "concept": {"type": "drug_name", "id": dr.lower()}, "pkg": sorted(d2p.get(dr.lower(), ())),
                      "prov": {"table": "icu_early_meds", "note": "first 48 h of the ICU stay"}})
    for k, txt in fz.lab_entries(rec):
        units.append({"uid": f"lab:{k}", "channel": "labs", "text": txt,
                      "concept": {"type": "lab", "id": k}, "pkg": [],
                      "prov": {"table": "icu_index", "fields": [f"lab_{k}_min", f"lab_{k}_max"], "note": "first 48 h extremes"}})
    st = ihm["sof"].get(h)
    if st is not None:
        for cu in sorted(rep_by_study.get(st, set())):
            units.append({"uid": f"rep:{cu}", "channel": "report", "text": cui_name.get(cu, cu),
                          "concept": {"type": "cui", "id": cu}, "pkg": sorted(f2p.get(cu, ())),
                          "prov": {"table": "REPORTED_FINDING", "study": int(st)}})
        for cu in sorted(img_by_study.get(st, set())):
            units.append({"uid": f"img:{cu}", "channel": "imaging", "text": cui_name.get(cu, cu),
                          "concept": {"type": "cui", "id": cu}, "pkg": sorted(f2p.get(cu, ())),
                          "prov": {"table": "IMAGING_FINDING", "study": int(st), "note": "BiomedCLIP zero-shot"}})
    return units, st


def gold_for_item(it, pidx, rec, h, st, ihm, f2p, findings, rel, cui_name):
    """Answer-bearing evidence, derived from the graph side exactly as 18 did."""
    fam, name = it["family"], rel["name"]
    table = rel["layer"][fam]
    rep_by_study, _ = findings
    coded = set()
    for cu in ihm["pdx"].get(h, []):
        coded |= f2p.get(cu, set())
    g = {"idx": None, "answer": it["answer"], "family": fam, "patient_uids": [], "condition_pkg": [],
         "kg_edges": [], "paths": [], "checks": {}}
    if it["answer"] != "yes":
        g["checks"]["negative_clean"] = not bool(table.get(pidx, set()) & _all_cond(rec, h, st, ihm, f2p, findings, coded))
        g["n_drug_relations"] = len(table.get(pidx, ()))
        return g
    if fam == "contra_labs":
        lab_states = []
        for sname, (dis, col, test, tmpl) in fz.LAB_STATES.items():
            v = rec.get(col)
            if pd.isna(v) or not test(float(v)) or (coded & set(dis)):
                continue
            lab_states.append((sname, set(dis), tmpl.format(v=float(v)), col))
        sname, target, evid, col = lab_states[0]
        g["checks"]["state_match"] = (sname == it["state"]); g["checks"]["evidence_match"] = (evid == it["evidence"])
        analyte = col.split("_")[1]
        g["patient_uids"] = [f"lab:{analyte}"]
        hits = sorted(table.get(pidx, set()) & target)
        g["condition_pkg"] = hits
        g["kg_edges"] = [[pidx, "contraindication", d] for d in hits]
        g["paths"] = [{"patient_uid": f"lab:{analyte}", "rule": f"{analyte} threshold -> {sname}", "condition": d,
                       "edges": [[pidx, "contraindication", d]]} for d in hits]
        g["checks"]["why_match"] = (sorted(name.get(x, str(x)) for x in hits)[:3] == it["why"])
        g["visible_in_frozen_prompt"] = True          # laboratory block is never capped
    else:
        rep_occult, rep_names, bearing = set(), [], []
        for cu in rep_by_study.get(st, set()) if st is not None else ():
            t = f2p.get(cu, set())
            if t and not (t & coded):
                rep_occult |= t; rep_names.append(cui_name.get(cu, cu))
                if t & table.get(pidx, set()):
                    bearing.append(cu)
        hits = sorted(table.get(pidx, set()) & rep_occult)
        g["patient_uids"] = [f"rep:{cu}" for cu in sorted(bearing)]
        g["condition_pkg"] = hits
        g["checks"]["evidence_match"] = ("; ".join(sorted(rep_names)[:4]) == it["evidence"])
        g["checks"]["why_match"] = (sorted(name.get(x, str(x)) for x in hits)[:3] == it["why"])
        relname = {"contra_report": "contraindication", "drug_effect": "drug_effect"}.get(fam)
        for cu in sorted(bearing):
            for d in sorted(f2p[cu] & table[pidx]):
                if fam == "mechanism":
                    for pr in sorted(rel["drug2prot"].get(pidx, ())):
                        if d in rel["prot2dis"].get(pr, ()):
                            e = [[pidx, "drug_protein", pr], [pr, "disease_protein", d]]
                            g["kg_edges"] += [x for x in e if x not in g["kg_edges"]]
                            g["paths"].append({"patient_uid": f"rep:{cu}", "condition": d, "edges": e})
                else:
                    e = [[pidx, relname, d]]
                    if e[0] not in g["kg_edges"]:
                        g["kg_edges"].append(e[0])
                    g["paths"].append({"patient_uid": f"rep:{cu}", "condition": d, "edges": e})
        shown = set(sorted({cui_name.get(x, x) for x in rep_by_study.get(st, set())})[:20]) if st is not None else set()
        g["visible_in_frozen_prompt"] = any(cui_name.get(cu, cu) in shown for cu in bearing)
    return g


def _all_cond(rec, h, st, ihm, f2p, findings, coded):
    rep_by_study, _ = findings
    all_cond = set(coded)
    for cu in rep_by_study.get(st, set()) if st is not None else ():
        t = f2p.get(cu, set())
        if t and not (t & coded):
            all_cond |= t
    for sname, (dis, col, test, tmpl) in fz.LAB_STATES.items():
        v = rec.get(col)
        if pd.isna(v) or not test(float(v)) or (coded & set(dis)):
            continue
        all_cond |= set(dis)
    return all_cond


def main():
    run = runlog.Run("exp1", "build_evidence_v1", {"version": VERSION, "alignment": "bridge_v1 (frozen graph ALIGNS_TO_CUI)",
                                                    "sources": ["icu_index", "icu_early_meds", "prior_dx", "icu_early_cxr",
                                                                "edges.parquet(REPORTED_FINDING, IMAGING_FINDING, ALIGNS_TO_CUI)",
                                                                "primekg_rel_cache", "cui2name_full"]})
    for k in ("bench_v2", "rel_cache", "graph_edges", "icu_index"):
        run.add_input(paths.FROZEN[k][0])
    for p in (paths.ICU_MEDS, paths.PRIOR_DX, paths.ICU_CXR, paths.CUI2NAME):
        run.add_input(p)
    print("loading frozen sources ...", flush=True)
    items = paths.load_benchmark()
    f2p, d2p, rep_by_study, img_by_study = fz.load_alignment_and_findings()
    findings = (rep_by_study, img_by_study)
    rel = fz.load_rel_cache(); ihm = fz.load_ihm(); cui_name = fz.load_cui_names()
    name = rel["name"]
    DEST.mkdir(parents=True, exist_ok=True); PUB.mkdir(parents=True, exist_ok=True)

    if VERSION != "evidence_v1":
        print(f"NOTE: building {VERSION} under alignment override {os.environ.get('MMKG_ALIGNMENT_OVERRIDE')}; frozen-block validation is expected to differ only in kg blocks", flush=True)
    val = {"n_items": len(items), "block_mismatch": Counter(), "pidx_ambiguous": 0, "checks_failed": Counter(),
           "units_per_channel": defaultdict(list), "visible_in_frozen_prompt": Counter(), "gold_paths_per_pos": [],
           "families": Counter(it["family"] for it in items)}
    linked = {"queried_drugs": set(), "med_drugs": set(), "concepts": set(), "proteins": set()}
    adm_cache = {}
    with open(DEST / "items.jsonl", "w") as fi, open(DEST / "gold.jsonl", "w") as fg:
        for i, it in enumerate(items):
            h = int(it["hadm_id"]); rec = ihm["recs"][h]; fam = it["family"]
            if h not in adm_cache:
                units, st = patient_units(h, rec, ihm, f2p, d2p, findings, cui_name)
                adm_cache[h] = (units, st)
                for ch in fz.CHANNELS:
                    val["units_per_channel"][ch].append(sum(u["channel"] == ch for u in units))
            units, st = adm_cache[h]
            # recover the PrimeKG drug index exactly as 18 chose it
            drug_pkg = fz.drug_packages(h, ihm, d2p)
            m = HEADER_RE.match(it["blocks"]["kg"]); hdr_name = m.group(1) if m else None
            cands = [p for p, dr in drug_pkg.items() if dr == it["drug"] and name.get(p, dr) == hdr_name]
            if len(cands) != 1:
                val["pidx_ambiguous"] += 1
                if not cands and os.environ.get("MMKG_ALIGNMENT_OVERRIDE"):
                    cands = [p for p, n_ in name.items() if n_ == hdr_name]   # keep the frozen item's drug node under an override
                cands = cands or [p for p, dr in drug_pkg.items() if dr == it["drug"]]
            pidx = cands[0]
            # byte-for-byte re-rendering of the frozen v1 blocks
            blocks = fz.make_blocks(rec, h, fam, pidx, it["drug"], ihm, findings, cui_name, rel)
            for k, v in blocks.items():
                if it["blocks"].get(k, "") != v:
                    val["block_mismatch"][k] += 1
            g = gold_for_item(it, pidx, rec, h, st, ihm, f2p, findings, rel, cui_name); g["idx"] = i
            for k, ok in g["checks"].items():
                if not ok:
                    val["checks_failed"][k] += 1
            if g["answer"] == "yes":
                val["visible_in_frozen_prompt"][str(g.get("visible_in_frozen_prompt"))] += 1
                val["gold_paths_per_pos"].append(len(g["paths"]))
            linked["queried_drugs"].add(pidx); linked["med_drugs"].update(drug_pkg.keys())
            for u in units:
                linked["concepts"].update(u["pkg"] if u["channel"] != "meds" else [])
            linked["proteins"].update(rel["drug2prot"].get(pidx, ()))
            fi.write(json.dumps({"idx": i, "family": fam, "hadm": h, "study": st, "drug": it["drug"], "drug_pkg": pidx,
                                 "drug_pkg_name": name.get(pidx, it["drug"]), "question": it["question"],
                                 "n_units": {ch: sum(u["channel"] == ch for u in units) for ch in fz.CHANNELS},
                                 "linked_pkg": sorted({p for u in units for p in u["pkg"]}),
                                 "med_pkg": sorted(drug_pkg.keys())}) + "\n")
            fg.write(json.dumps(g) + "\n")
    with open(DEST / "patient_units.jsonl", "w") as fu:
        for h, (units, st) in adm_cache.items():
            fu.write(json.dumps({"hadm": h, "study": st, "units": units}) + "\n")
    json.dump({k: sorted(v) for k, v in linked.items()}, open(PUB / ("linked_nodes_v1.json" if VERSION == "evidence_v1" else f"linked_nodes_{VERSION}.json"), "w"))

    upc = {ch: {"mean": round(float(np.mean(v)), 2), "median": float(np.median(v)), "max": int(np.max(v)), "zero": int(sum(x == 0 for x in v))}
           for ch, v in val["units_per_channel"].items()}
    summary = {"version": VERSION, "n_items": val["n_items"], "n_admissions": len(adm_cache),
               "families": dict(val["families"]), "block_mismatch": dict(val["block_mismatch"]),
               "pidx_ambiguous": val["pidx_ambiguous"], "gold_checks_failed": dict(val["checks_failed"]),
               "positives_visible_in_frozen_prompt": dict(val["visible_in_frozen_prompt"]),
               "gold_paths_per_positive": {"mean": round(float(np.mean(val["gold_paths_per_pos"])), 2), "max": int(max(val["gold_paths_per_pos"]))},
               "units_per_admission": upc,
               "linked_nodes": {k: len(v) for k, v in linked.items()}}
    json.dump(summary, open(PUB / f"{VERSION}_validation.json", "w"), indent=2)
    for p in (DEST / "items.jsonl", DEST / "gold.jsonl", DEST / "patient_units.jsonl", PUB / "linked_nodes_v1.json", PUB / f"{VERSION}_validation.json"):
        run.add_output(p)
    run.finish(counts=summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
