"""
12_build_mm_bundles.py   --   env: umls_eval   [CPU]
================================================================================
Build per-admission MULTIMODAL bundles for the LLM modality-ablation experiment.

QUESTION THIS SERVES
  Does multimodal patient data help an LLM?  We answer it on ICU in-hospital
  mortality -- an outcome label that is completely independent of any KG edge
  (anti-circularity), and the same task/cohort/labels the phase-2 GNN uses, so
  the LLM ablation is directly comparable to the GNN ablation table.

ENTIRE COHORT.  Every ICU stay in the whole graph is emitted (n ~ 28.6k). No
subsampling, no eligibility filter beyond having an ICU stay with a label.
Availability flags are written per modality so the analysis can report BOTH
  (a) the entire cohort, and
  (b) the stratum where every modality is actually present
without re-running anything.

MODALITY BLOCKS (each rendered independently so arms can drop one cleanly)
  demo      age, sex, admission type, prior-admission count
  history   prior-visit diagnoses (CUI names) + prior-visit drugs   [longitudinal]
  meds      drugs given early this admission
  labs      first-24h lab ranges
  imaging   BiomedCLIP findings on the early chest film             [image modality]
  report    findings extracted from the radiology report text       [text modality]
  kg        PrimeKG facts about THIS patient's drugs and conditions [linked graph]

The imaging and report blocks come from the two distinct edge types the graph
keeps separate (IMAGING_FINDING vs REPORTED_FINDING) precisely so that "what the
model saw" and "what the radiologist wrote" can be ablated apart.

Output: outputs/mm/mm_bundles_ihm.jsonl
"""
from __future__ import annotations
import os
import json, pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE    = Path(__file__).resolve().parent
P2      = HERE.parent / "graph" / "outputs"
OUT     = HERE / "outputs" / "mm"
KGC     = HERE / "outputs" / "medsafety" / "kg_cache"
BR      = HERE / "outputs" / "bridge_v2"
META    = Path(os.environ.get("UMLS_META_DIR", os.path.expandvars("$MMKG_HOME/data/umls/2026AA/META")))

OUT.mkdir(parents=True, exist_ok=True)

MAX_HISTORY_DX = 25      # prior conditions listed
MAX_MEDS       = 40      # drugs listed
MAX_KG_FACTS   = 40      # PrimeKG facts listed (see note in the write-up)

LAB_LABELS = {
    "creatinine": "creatinine", "bun": "BUN", "sodium": "sodium",
    "potassium": "potassium", "bicarbonate": "bicarbonate", "glucose": "glucose",
    "wbc": "WBC", "hemoglobin": "hemoglobin", "platelet": "platelet",
    "lactate": "lactate", "bilirubin": "bilirubin", "inr": "INR",
    "aniongap": "anion gap", "albumin": "albumin", "ph": "pH", "po2": "pO2",
    "pco2": "pCO2",
}


# ------------------------------------------------------------------ CUI names
def cui_names(needed: set[str]) -> dict[str, str]:
    """CUI -> preferred English name, from MRCONSO. Cached."""
    cache = OUT / "cui2name_full.tsv"
    if cache.exists():
        d = pd.read_csv(cache, sep="\t", header=None, names=["cui", "name"], dtype=str)
        return dict(zip(d.cui, d.name))

    print("streaming MRCONSO for concept names ...", flush=True)
    best: dict[str, str] = {}
    with open(META / "MRCONSO.RRF", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split("|")
            if len(p) < 15 or p[1] != "ENG":
                continue
            cui = p[0]
            if cui not in needed or cui in best:
                continue
            # p[2]=TS, p[4]=STT, p[6]=ISPREF -- prefer the preferred atom
            if p[2] == "P" and p[4] == "PF" and p[6] == "Y":
                best[cui] = p[14]
    # second pass for anything with no preferred atom
    missing = needed - set(best)
    if missing:
        with open(META / "MRCONSO.RRF", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.split("|")
                if len(p) < 15 or p[1] != "ENG":
                    continue
                if p[0] in missing and p[0] not in best:
                    best[p[0]] = p[14]
    pd.DataFrame(sorted(best.items()), columns=["cui", "name"]).to_csv(
        cache, sep="\t", header=False, index=False)
    print(f"  named {len(best)}/{len(needed)} CUIs -> {cache}", flush=True)
    return best


# ------------------------------------------------------------------ renderers
def render_labs(row) -> str:
    parts = []
    for key, label in LAB_LABELS.items():
        lo, hi = row.get(f"lab_{key}_min"), row.get(f"lab_{key}_max")
        if pd.isna(lo) and pd.isna(hi):
            continue
        if pd.isna(lo) or pd.isna(hi) or abs(float(hi) - float(lo)) < 1e-9:
            parts.append(f"{label} {float(hi if pd.isna(lo) else lo):g}")
        else:
            parts.append(f"{label} {float(lo):g}-{float(hi):g}")
    if not parts:
        return ""
    n_abn = row.get("n_abnormal")
    tail = f" ({int(n_abn)} abnormal results)" if pd.notna(n_abn) else ""
    return "Laboratory values, first 24 hours: " + ", ".join(parts) + tail + "."


def render_demo(row) -> str:
    a = row.get("age")
    age = f"{int(a)}-year-old" if pd.notna(a) else "adult"
    sex = {"M": "male", "F": "female"}.get(str(row.get("gender")), "patient")
    at = str(row.get("admission_type", "")).lower()
    s = f"A {age} {sex} patient, {at} admission, now in intensive care."
    npa = row.get("n_prior_adm")
    if pd.notna(npa):
        s += (" No previous hospital admissions on record." if int(npa) == 0
              else f" {int(npa)} previous hospital admission(s) on record.")
    return s


def join_trunc(items: list[str], cap: int, head: str, tail: str = ".") -> str:
    if not items:
        return ""
    shown = items[:cap]
    extra = f" (and {len(items) - cap} more)" if len(items) > cap else ""
    return f"{head}{', '.join(shown)}{extra}{tail}"


# ------------------------------------------------------------------ main
def main():
    print("loading cohort ...", flush=True)
    idx = pd.read_parquet(P2 / "ihm" / "icu_index.parquet")
    print(f"  ICU stays: {len(idx)}  patients {idx.subject_id.nunique()}  "
          f"pos-rate {idx.label.mean():.4f}", flush=True)

    # ---- patient-side modality tables ------------------------------------
    pdx  = pd.read_parquet(P2 / "ihm" / "prior_dx.parquet")
    pdrg = pd.read_parquet(P2 / "ihm" / "prior_drug.parquet")
    meds = pd.read_parquet(P2 / "ihm" / "icu_early_meds.parquet")
    cxr  = pd.read_parquet(P2 / "ihm" / "icu_early_cxr.parquet")

    # imaging: BiomedCLIP positive findings, keyed by study (string ids!)
    img = pd.read_parquet(P2 / "imaging" / "biomedclip_findings.parquet",
                          columns=["study_id", "present"])
    img["study_id"] = img.study_id.astype("int64")
    img_by_study = {int(s): list(p) for s, p in zip(img.study_id, img.present) if len(p)}

    # report: REPORTED_FINDING edges from the materialised patient graph
    print("loading report-finding edges ...", flush=True)
    ed = pd.read_parquet(P2 / "patient_graph" / "edges.parquet",
                         columns=["src", "dst", "etype"])
    rep = ed[ed.etype == "REPORTED_FINDING"]
    rep_by_study: dict[int, list[str]] = defaultdict(list)
    for s, d in zip(rep.src.values, rep.dst.values):
        rep_by_study[int(s.split(":", 1)[1])].append(d.split(":", 1)[1])
    del ed, rep

    # ---- concept names ----------------------------------------------------
    needed = set(pdx.cui.unique())
    for v in rep_by_study.values():
        needed.update(v)
    c2n = cui_names(needed)

    # ---- KG side ----------------------------------------------------------
    # Schemas (verified): drug_disease / drug_effect carry
    #   relation, x_index, x_type, x_name, y_index, y_type, y_name   (x = drug)
    # so names come straight out of the table; the join key is x_index.
    print("loading PrimeKG cache ...", flush=True)
    dd = pd.read_parquet(KGC / "drug_disease.parquet")
    de = pd.read_parquet(KGC / "drug_effect.parquet")
    d2p = pd.read_csv(KGC / "cohort_drug2pkg.tsv", sep="\t", header=None,
                      names=["drug", "pkg", "route"], dtype=str)
    drug2pkg = dict(zip(d2p.drug.str.lower(), d2p.pkg))
    print(f"  drug_disease relations: {dd.relation.value_counts().to_dict()}", flush=True)

    contra, indic, effect, pkg_name = (defaultdict(list), defaultdict(list),
                                       defaultdict(list), {})
    for r, xi, xn, yn in zip(dd.relation.astype(str), dd.x_index.astype(str),
                             dd.x_name.astype(str), dd.y_name.astype(str)):
        pkg_name.setdefault(xi, xn)
        if "contraindication" in r:
            contra[xi].append(yn)
        elif "indication" in r:
            indic[xi].append(yn)
    for xi, xn, yn in zip(de.x_index.astype(str), de.x_name.astype(str),
                          de.y_name.astype(str)):
        pkg_name.setdefault(xi, xn)
        effect[xi].append(yn)
    print(f"  drugs with contraindications {len(contra)}, indications {len(indic)}, "
          f"effects {len(effect)}", flush=True)

    # ---- group patient tables --------------------------------------------
    print("grouping patient tables ...", flush=True)
    g_pdx  = pdx.groupby("hadm_id").cui.apply(list).to_dict()
    g_pdrg = pdrg.groupby("hadm_id").drug.apply(list).to_dict()
    g_meds = meds.groupby("hadm_id").drug.apply(list).to_dict()
    g_cxr  = dict(zip(cxr.hadm_id, cxr.study_id))

    # ---- emit -------------------------------------------------------------
    out_f = OUT / "mm_bundles_ihm.jsonl"
    n_av = defaultdict(int)
    with open(out_f, "w") as w:
        for rec in idx.to_dict("records"):
            h = int(rec["hadm_id"])
            demo = render_demo(rec)
            labs = render_labs(rec)

            dx_names = [c2n.get(c, c) for c in g_pdx.get(h, [])]
            dx_names = sorted(dict.fromkeys(n for n in dx_names if n))
            prior_rx = sorted(dict.fromkeys(g_pdrg.get(h, [])))
            hist = " ".join(x for x in [
                join_trunc(dx_names, MAX_HISTORY_DX, "Conditions documented at previous visits: "),
                join_trunc(prior_rx, MAX_MEDS, "Medications at previous visits: "),
            ] if x)

            cur_meds = sorted(dict.fromkeys(g_meds.get(h, [])))
            med_block = join_trunc(cur_meds, MAX_MEDS,
                                   f"Medications given early in this ICU stay ({len(cur_meds)}): ")

            st = g_cxr.get(h)
            imaging = report = ""
            if st is not None:
                f_img = img_by_study.get(int(st), [])
                if f_img:
                    imaging = ("Chest radiograph, automated image analysis reports: "
                               + ", ".join(f_img) + ".")
                f_rep = [c2n.get(c, c) for c in rep_by_study.get(int(st), [])]
                f_rep = sorted(dict.fromkeys(n for n in f_rep if n))
                if f_rep:
                    report = join_trunc(f_rep, 20,
                                        "Radiology report for the chest film describes: ")

            # ---- KG block: facts about THIS patient's drugs -----------------
            facts, seen = [], set()
            for d in cur_meds:
                p = drug2pkg.get(d.lower())
                if not p:
                    continue
                dn = pkg_name.get(p, d)
                for c in contra.get(p, [])[:3]:
                    f = f"{dn} --contraindicated_in--> {c}"
                    if f not in seen:
                        seen.add(f); facts.append(f)
                for c in indic.get(p, [])[:2]:
                    f = f"{dn} --indicated_for--> {c}"
                    if f not in seen:
                        seen.add(f); facts.append(f)
                for c in effect.get(p, [])[:3]:
                    f = f"{dn} --can_cause--> {c}"
                    if f not in seen:
                        seen.add(f); facts.append(f)
            kg = ""
            if facts:
                kg = ("Facts from a curated biomedical knowledge graph about this "
                      "patient's medications:\n"
                      + "\n".join("  " + f for f in facts[:MAX_KG_FACTS]))

            blocks = {"demo": demo, "history": hist, "meds": med_block,
                      "labs": labs, "imaging": imaging, "report": report, "kg": kg}
            avail = {k: bool(v) for k, v in blocks.items()}
            for k, v in avail.items():
                n_av[k] += v

            w.write(json.dumps({
                "hadm_id": h, "subject_id": int(rec["subject_id"]),
                "label": int(rec["label"]), "blocks": blocks, "avail": avail,
            }) + "\n")

    n = len(idx)
    print(f"\nwrote {n} bundles -> {out_f}")
    print("modality availability over the ENTIRE cohort:")
    for k in ["demo", "history", "meds", "labs", "imaging", "report", "kg"]:
        print(f"  {k:9s} {n_av[k]:6d}  ({n_av[k]/n:.1%})")


if __name__ == "__main__":
    main()
