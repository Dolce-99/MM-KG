"""Loaders for the frozen phase-2/phase-3 sources and the VERBATIM block
renderers of phase3/18_build_composite_wide.py.

The renderers are copied unchanged so that the evidence collection can be
validated by re-rendering every frozen prompt block byte-for-byte.  Do not
"improve" them here; new renderings belong in the experiment scripts.
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd
from . import paths

# ---- constants copied from 18_build_composite_wide.py ----------------------
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
PHRASE = {
    "contra_labs":   "is contraindicated in the following conditions",
    "contra_report": "is contraindicated in the following conditions",
    "drug_effect":   "has the following documented adverse effects",
    "mechanism":     "acts on proteins implicated in the following conditions",
}
FAMILIES = ("contra_labs", "contra_report", "drug_effect", "mechanism")
CHANNELS = ("demo", "codes", "meds", "labs", "report", "imaging")


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


def lab_entries(rec) -> list[tuple[str, str]]:
    """(analyte key, rendered entry) pairs in the exact order of render_labs."""
    out = []
    for k, lab in LAB_LABELS.items():
        lo, hi = rec.get(f"lab_{k}_min"), rec.get(f"lab_{k}_max")
        if pd.isna(lo) and pd.isna(hi):
            continue
        if pd.isna(lo) or pd.isna(hi) or abs(float(hi) - float(lo)) < 1e-9:
            out.append((k, f"{lab} {float(hi if pd.isna(lo) else lo):g}"))
        else:
            out.append((k, f"{lab} {float(lo):g}-{float(hi):g}"))
    return out


# ---- loaders ----------------------------------------------------------------
def load_alignment_and_findings(verify: bool = True):
    """bridge-v1 alignment (f2p: CUI->pkg set, d2p: drug name->pkg set) and the
    per-study finding sets (report CUIs, imaging CUIs) from the frozen graph."""
    ed = pd.read_parquet(paths.frozen("graph_edges", verify), columns=["src", "dst", "etype"])
    import os
    alt = os.environ.get("MMKG_ALIGNMENT_OVERRIDE")          # downstream validation of a NEW alignment version (never overwrites the frozen graph)
    al = pd.read_parquet(alt, columns=["src", "dst", "etype"]) if alt else ed[ed.etype == "ALIGNS_TO_CUI"]
    f2p, d2p = defaultdict(set), defaultdict(set)
    for s, d in zip(al.src.values, al.dst.values):
        k = s.split(":", 1)[1]; v = int(d.split(":", 1)[1])
        (f2p if s.startswith("finding:") else d2p)[k].add(v)
    rep_by_study, img_by_study = defaultdict(set), defaultdict(set)
    for et, tgt in (("REPORTED_FINDING", rep_by_study), ("IMAGING_FINDING", img_by_study)):
        sub = ed[ed.etype == et]
        for s, d in zip(sub.src.values, sub.dst.values):
            tgt[int(s.split(":", 1)[1])].add(d.split(":", 1)[1])
    return f2p, d2p, rep_by_study, img_by_study


def load_rel_cache(verify: bool = True):
    kg = pd.read_parquet(paths.frozen("rel_cache", verify))
    name = {}
    contra, effect, drug2prot, prot2dis = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    for r, xi, xn, yi, yn in zip(kg.relation, kg.x_index, kg.x_name, kg.y_index, kg.y_name):
        xi, yi = int(xi), int(yi)
        name.setdefault(xi, str(xn)); name.setdefault(yi, str(yn))
        if r == "contraindication":  contra[xi].add(yi)
        elif r == "drug_effect":     effect[xi].add(yi)
        elif r == "drug_protein":    drug2prot[xi].add(yi)
        elif r == "disease_protein": prot2dis[xi].add(yi)     # x = protein, y = disease
    reach = {}
    for d, ps in drug2prot.items():
        s = set()
        for p in ps:
            s |= prot2dis.get(p, set())
        if s:
            reach[d] = s
    layer = {"contra_labs": contra, "contra_report": contra, "drug_effect": effect, "mechanism": reach}
    return {"df": kg, "name": name, "contra": contra, "effect": effect,
            "drug2prot": drug2prot, "prot2dis": prot2dis, "reach": reach, "layer": layer}


def load_ihm(verify: bool = True):
    idx = pd.read_parquet(paths.frozen("icu_index", verify))
    meds = pd.read_parquet(paths.ICU_MEDS).groupby("hadm_id").drug.apply(list).to_dict()
    pdx = pd.read_parquet(paths.PRIOR_DX).groupby("hadm_id").cui.apply(list).to_dict()
    cxr = pd.read_parquet(paths.ICU_CXR)
    sof = dict(zip(cxr.hadm_id.astype(int), cxr.study_id.astype(int)))
    recs = {int(r["hadm_id"]): r for r in idx.to_dict("records")}
    return {"idx": idx, "recs": recs, "meds": meds, "pdx": pdx, "sof": sof}


def load_cui_names() -> dict:
    c2n = pd.read_csv(paths.CUI2NAME, sep="\t", header=None, names=["cui", "name"], dtype=str)
    return dict(zip(c2n.cui, c2n.name))


# ---- verbatim renderer ------------------------------------------------------
def make_blocks(rec, h, fam, pidx, dstr, ihm, findings, cui_name, rel):
    """Exact copy of 18_build_composite_wide.make_blocks (v1 blocks; kg_alt and
    the drug_effect rephrase were added afterwards by 23_prepare_v2_benchmark)."""
    pdx, meds, sof = ihm["pdx"], ihm["meds"], ihm["sof"]
    rep_by_study, img_by_study = findings
    name, table, phrase = rel["name"], rel["layer"][fam], PHRASE[fam]
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
    inm = sorted({cui_name.get(x, x) for x in img_by_study.get(st, set())}) if st else []
    img_b = ("Chest radiograph, automated image analysis reports: "
             + ", ".join(inm) + ".") if inm else ""
    lst = sorted(name.get(x, str(x)) for x in table.get(pidx, ()))
    kg_b = ""
    if lst:
        kg_b = (f"Facts from a curated biomedical knowledge graph. "
                f"{name.get(pidx, dstr)} {phrase} (complete list, {len(lst)} entries):\n"
                + "\n".join("  " + x for x in lst))
    return {"demo": demo, "codes": codes, "meds": meds_b, "labs": labs_b,
            "report": rep_b, "imaging": img_b, "kg": kg_b}


def drug_packages(h, ihm, d2p) -> dict:
    """pkg drug index -> MIMIC drug string, in the exact order 18 built it."""
    drug_pkg = {}
    for dr in ihm["meds"].get(h, []):
        for p in d2p.get(dr.lower(), ()):
            drug_pkg.setdefault(p, dr)
    return drug_pkg
