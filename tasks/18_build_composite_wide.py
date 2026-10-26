"""
18_build_composite_wide.py   --   env: umls_eval   [CPU]
================================================================================
The composite benchmark, widened to FOUR question families across THREE PrimeKG
relation layers, and carrying EVERY patient modality so two experiments can be
run from one file:

  (1) the 2x2          multimodal on/off  x  PrimeKG on/off
  (2) leave-one-out    with PrimeKG ALWAYS ON, drop one patient modality at a
                       time -- which channel actually carries the signal once
                       the knowledge graph is available

FAMILIES  (each: the patient state is occult -- present in a modality, absent
from the coded diagnoses -- and the rule that interprets it lives in PrimeKG)

  contra_labs    organ state from lab values      | contraindication layer
  contra_report  finding from the radiology report| contraindication layer
  drug_effect    occult finding is a documented adverse effect of the drug
                                                  | drug_effect layer
  mechanism      drug targets a protein implicated in the occult condition
                                                  | drug_protein + disease_protein
                                                  | (a genuine two-hop join)

Widening across relation layers matters: if the interaction only appeared for
contraindications it could be an artifact of that one construction.

BLOCKS emitted per item (arms are built by including/excluding these)
  demo    always on
  codes   coded diagnoses -- the channel a conventional system already has
  meds    medication list
  labs / report / imaging   the patient modalities
  kg      the queried drug's COMPLETE relevant list, never truncated

MATCHED NEGATIVES: same patient, a drug that is also actually prescribed but
whose PrimeKG list touches none of the patient's conditions, chosen to have a
list of comparable length so block size cannot stand in for the answer.

Output: outputs/mm/composite_benchmark_wide.jsonl
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# --- near-synonym guard ------------------------------------------------------
# The label is computed by PrimeKG index, but the model reads STRINGS, and the
# graph carries distinct indices whose names overlap ("Edema" vs "Pulmonary
# edema"). A negative whose list shares a content word with something the
# patient visibly has looks like a match to any reader, so the item is
# mislabelled from the reader's point of view rather than merely hard. Negatives
# must therefore be clean lexically as well as by index.
_STOP = {"disease", "diseases", "disorder", "disorders", "syndrome", "finding",
         "findings", "chronic", "acute", "other", "unspecified", "value",
         "qualifier", "morphologic", "abnormality", "structure", "primary",
         "secondary", "left", "right", "increase", "answer", "question",
         "combination", "bilateral", "moderate", "severe", "small", "large"}


def content_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) >= 5 and w not in _STOP}


def near_synonym(a: set[str], b: set[str]) -> bool:
    """True when two concept names would read as the same thing.

    Subset, not mere overlap. "Edema" {edema} sits inside "Pulmonary edema"
    {pulmonary, edema} and is a genuine near-synonym; "Pleural effusion" and
    "Pleural mesothelioma" merely share a region word and are plainly different
    findings. Testing overlap rather than containment rejects the second case
    too and starves the report-based families of usable decoys.
    """
    return bool(a) and bool(b) and (a <= b or b <= a)

HERE = Path(__file__).resolve().parent
P2   = HERE.parent / "graph" / "outputs"
OUT  = HERE / "outputs" / "mm"
KGC  = HERE / "outputs" / "medsafety" / "kg_cache"

PER_FAMILY = 800      # items per family, balanced 50/50
SEED       = 0

LAB_STATES = {
    "renal impairment":   ([35764, 36253, 36040, 36694], "lab_creatinine_max", lambda v: v > 2.0, "creatinine {v:g}"),
    "hepatic impairment": ([35565, 39531, 37765], "lab_bilirubin_max", lambda v: v > 2.0, "bilirubin {v:g}"),
    "thrombocytopenia":   ([27626], "lab_platelet_min", lambda v: v < 100, "platelets {v:g}"),
    "anemia":             ([33632, 33623], "lab_hemoglobin_min", lambda v: v < 8.0, "haemoglobin {v:g}"),
}
LAB_LABELS = {"creatinine": "creatinine", "bun": "BUN", "sodium": "sodium", "potassium": "potassium",
              "bicarbonate": "bicarbonate", "glucose": "glucose", "wbc": "WBC",
              "hemoglobin": "haemoglobin", "platelet": "platelet", "lactate": "lactate",
              "bilirubin": "bilirubin", "inr": "INR", "aniongap": "anion gap",
              "albumin": "albumin", "ph": "pH", "po2": "pO2", "pco2": "pCO2"}

QUESTION = {
    "contra_labs":   "Is {d} contraindicated for this patient, given their documented condition? Answer yes or no.",
    "contra_report": "Is {d} contraindicated for this patient, given their documented condition? Answer yes or no.",
    "drug_effect":   "Could any of this patient's documented findings be a known adverse effect of {d}? Answer yes or no.",
    "mechanism":     "Does {d} act on a protein implicated in a condition this patient has? Answer yes or no.",
}


def render_labs(rec):
    p = []
    for k, lab in LAB_LABELS.items():
        lo, hi = rec.get(f"lab_{k}_min"), rec.get(f"lab_{k}_max")
        if pd.isna(lo) and pd.isna(hi):
            continue
        if pd.isna(lo) or pd.isna(hi) or abs(float(hi) - float(lo)) < 1e-9:
            p.append(f"{lab} {float(hi if pd.isna(lo) else lo):g}")
        else:
            p.append(f"{lab} {float(lo):g}-{float(hi):g}")
    return "Laboratory values, first 24 hours: " + ", ".join(p) + "." if p else ""


def main():
    rng = np.random.default_rng(SEED)

    print("loading graph ...", flush=True)
    ed = pd.read_parquet(P2 / "patient_graph" / "edges.parquet", columns=["src", "dst", "etype"])
    al = ed[ed.etype == "ALIGNS_TO_CUI"]
    f2p, d2p = defaultdict(set), defaultdict(set)
    for s, d in zip(al.src.values, al.dst.values):
        k = s.split(":", 1)[1]; v = int(d.split(":", 1)[1])
        (f2p if s.startswith("finding:") else d2p)[k].add(v)
    rep_by_study = defaultdict(set)
    for s, d in zip(*[ed.loc[ed.etype == "REPORTED_FINDING", c].values for c in ("src", "dst")]):
        rep_by_study[int(s.split(":", 1)[1])].add(d.split(":", 1)[1])
    img_by_study = defaultdict(set)
    for s, d in zip(*[ed.loc[ed.etype == "IMAGING_FINDING", c].values for c in ("src", "dst")]):
        img_by_study[int(s.split(":", 1)[1])].add(d.split(":", 1)[1])
    del ed, al

    kg = pd.read_parquet(OUT / "primekg_rel_cache.parquet")
    name = {}
    contra, effect, drug2prot, prot2dis = (defaultdict(set), defaultdict(set),
                                           defaultdict(set), defaultdict(set))
    for r, xi, xn, yi, yn in zip(kg.relation, kg.x_index, kg.x_name, kg.y_index, kg.y_name):
        xi, yi = int(xi), int(yi)
        name.setdefault(xi, str(xn)); name.setdefault(yi, str(yn))
        if r == "contraindication":  contra[xi].add(yi)
        elif r == "drug_effect":     effect[xi].add(yi)
        elif r == "drug_protein":    drug2prot[xi].add(yi)
        elif r == "disease_protein": prot2dis[xi].add(yi)     # x = protein, y = disease

    reach = {}                       # drug -> diseases reachable via its protein targets
    for d, ps in drug2prot.items():
        s = set()
        for p in ps:
            s |= prot2dis.get(p, set())
        if s:
            reach[d] = s

    idx  = pd.read_parquet(P2 / "ihm" / "icu_index.parquet")
    meds = pd.read_parquet(P2 / "ihm" / "icu_early_meds.parquet").groupby("hadm_id").drug.apply(list).to_dict()
    pdx  = pd.read_parquet(P2 / "ihm" / "prior_dx.parquet").groupby("hadm_id").cui.apply(list).to_dict()
    cxr  = pd.read_parquet(P2 / "ihm" / "icu_early_cxr.parquet")
    sof  = dict(zip(cxr.hadm_id.astype(int), cxr.study_id.astype(int)))
    c2n  = pd.read_csv(OUT / "cui2name_full.tsv", sep="\t", header=None, names=["cui", "name"], dtype=str)
    cui_name = dict(zip(c2n.cui, c2n.name))

    # relation layer used by each family: (lookup dict, KG-block phrasing)
    LAYER = {
        "contra_labs":   (contra, "is contraindicated in the following conditions"),
        "contra_report": (contra, "is contraindicated in the following conditions"),
        "drug_effect":   (effect, "has the following documented adverse effects"),
        "mechanism":     (reach,  "acts on proteins implicated in the following conditions"),
    }

    cands = defaultdict(list)
    for rec in idx.to_dict("records"):
        h = int(rec["hadm_id"])
        drug_pkg = {}
        for dr in meds.get(h, []):
            for p in d2p.get(dr.lower(), ()):
                drug_pkg.setdefault(p, dr)
        if len(drug_pkg) < 2:
            continue

        coded = set()
        for cu in pdx.get(h, []):
            coded |= f2p.get(cu, set())

        lab_states = []
        for sname, (dis, col, test, tmpl) in LAB_STATES.items():
            v = rec.get(col)
            if pd.isna(v) or not test(float(v)) or (coded & set(dis)):
                continue
            lab_states.append((sname, set(dis), tmpl.format(v=float(v))))

        st = sof.get(h)
        rep_occult, rep_names = set(), []
        for cu in rep_by_study.get(st, set()) if st is not None else ():
            t = f2p.get(cu, set())
            if t and not (t & coded):
                rep_occult |= t; rep_names.append(cui_name.get(cu, cu))

        # every finding name the model will actually SEE in a modality block
        vis_names = list(rep_names)
        vis_names += [cui_name.get(c, c) for c in
                      (rep_by_study.get(st, set()) if st is not None else ())]
        vis_names += ([cui_name.get(x, x) for x in img_by_study.get(st, set())]
                      if st is not None else [])
        vis_keys = [k for k in (content_tokens(v) for v in vis_names) if k]

        all_cond = set(coded) | rep_occult
        for _n, s, _e in lab_states:
            all_cond |= s

        for fam in ("contra_labs", "contra_report", "drug_effect", "mechanism"):
            table, _ = LAYER[fam]
            if fam == "contra_labs":
                if not lab_states:
                    continue
                target, evid = lab_states[0][1], lab_states[0][2]
                state = lab_states[0][0]
            else:
                if not rep_occult:
                    continue
                target, evid, state = rep_occult, "; ".join(sorted(rep_names)[:4]), "report finding"

            hits = [(p, dr) for p, dr in drug_pkg.items() if table.get(p, set()) & target]
            # negatives: clean by index AND lexically clean against everything
            # the patient visibly has (see the near-synonym guard above)
            safe = []
            for p2, dr2 in drug_pkg.items():
                lst = table.get(p2)
                if not lst or (lst & all_cond):
                    continue
                kg_keys = (content_tokens(name.get(x, str(x))) for x in lst)
                if any(near_synonym(k, v) for k in kg_keys for v in vis_keys):
                    continue
                safe.append((p2, dr2))
            if not hits or not safe:
                continue
            p, dr = hits[rng.integers(len(hits))]
            # Match on RENDERED CHARACTERS, not set size: entry names differ in
            # length (a "mechanism" disease name is far longer than a drug's),
            # so equal-sized lists can still differ markedly in block length,
            # and block length must not stand in for the answer.
            def rlen(k):
                return sum(len(name.get(x, str(x))) + 3 for x in table.get(k, ()))
            want = rlen(p)
            np_, ndr = min(safe, key=lambda s: abs(rlen(s[0]) - want))
            if not (0.85 <= (rlen(np_) + 1) / (want + 1) <= 1.18):
                continue
            cands[fam].append({"hadm_id": h, "rec": rec, "state": state, "evidence": evid,
                               "pos": (p, dr), "neg": (np_, ndr),
                               "why": sorted(name.get(x, str(x)) for x in (table[p] & target))[:3]})

    # ---------------------------------------------------------------- render
    def make_blocks(c, fam, pidx, dstr):
        rec, h = c["rec"], c["hadm_id"]
        age = int(rec["age"]) if pd.notna(rec.get("age")) else None
        sex = {"M": "male", "F": "female"}.get(str(rec.get("gender")), "patient")
        demo = (f"A {age}-year-old {sex} patient in intensive care."
                if age else f"A {sex} patient in intensive care.")
        dxn = sorted({cui_name.get(x, x) for x in pdx.get(h, [])})[:25]
        codes = ("Coded diagnoses on record: " + ", ".join(dxn) + "."
                 if dxn else "No coded diagnoses on record.")
        ml = sorted(dict.fromkeys(meds.get(h, [])))[:40]
        meds_b = f"Current medications ({len(ml)}): " + ", ".join(ml) + "."
        labs_b = render_labs(rec)
        st = sof.get(h)
        rn = sorted({cui_name.get(x, x) for x in rep_by_study.get(st, set())})[:20] if st else []
        rep_b = "Radiology report for the chest film describes: " + ", ".join(rn) + "." if rn else ""
        # IMAGING_FINDING edges carry CUIs, not names -- map them, or the block
        # renders as "C0034063, C0264545" and is unreadable to the model.
        inm = sorted({cui_name.get(x, x) for x in img_by_study.get(st, set())}) if st else []
        img_b = ("Chest radiograph, automated image analysis reports: "
                 + ", ".join(inm) + ".") if inm else ""
        table, phrase = LAYER[fam]
        lst = sorted(name.get(x, str(x)) for x in table.get(pidx, ()))
        kg_b = ""
        if lst:
            kg_b = (f"Facts from a curated biomedical knowledge graph. "
                    f"{name.get(pidx, dstr)} {phrase} (complete list, {len(lst)} entries):\n"
                    + "\n".join("  " + x for x in lst))
        return {"demo": demo, "codes": codes, "meds": meds_b, "labs": labs_b,
                "report": rep_b, "imaging": img_b, "kg": kg_b}

    items = []
    print()
    for fam in ("contra_labs", "contra_report", "drug_effect", "mechanism"):
        pool = cands[fam]; rng.shuffle(pool)
        take = pool[:PER_FAMILY // 2]
        print(f"  {fam:14s} {len(pool):5d} eligible patients -> {len(take)} matched pairs")
        for c in take:
            for ans, (pidx, dstr) in (("yes", c["pos"]), ("no", c["neg"])):
                items.append({"family": fam, "hadm_id": c["hadm_id"], "answer": ans,
                              "drug": dstr, "state": c["state"], "evidence": c["evidence"],
                              "why": c["why"], "blocks": make_blocks(c, fam, pidx, dstr),
                              "question": "Question: " + QUESTION[fam].format(d=dstr)})
    rng.shuffle(items)

    dest = OUT / "composite_benchmark_wide.jsonl"
    with open(dest, "w") as w:
        for it in items:
            w.write(json.dumps(it) + "\n")
    print(f"\nwrote {len(items)} items -> {dest}\n")

    print(f"{'family':15s}{'n':>6}{'yes%':>7}   kg chars (yes/no)   block presence")
    for fam in dict.fromkeys(i["family"] for i in items):
        f = [i for i in items if i["family"] == fam]
        ky = int(np.median([len(i["blocks"]["kg"]) for i in f if i["answer"] == "yes"]))
        kn = int(np.median([len(i["blocks"]["kg"]) for i in f if i["answer"] == "no"]))
        pres = " ".join(f"{b}:{np.mean([bool(i['blocks'][b]) for i in f]):.0%}"
                        for b in ("labs", "report", "imaging", "kg"))
        print(f"{fam:15s}{len(f):6d}{np.mean([i['answer']=='yes' for i in f]):7.0%}"
              f"   {ky:5d} / {kn:<5d}      {pres}")


if __name__ == "__main__":
    main()
