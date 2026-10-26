"""
04_patient_tables.py   --   env: umls_eval (pandas, pyarrow)
============================================================
WHOLE-COHORT patient node/edge tables (analog of the subgraph `03`, but VECTORIZED
for ~52k patients / millions of rows -- no per-row python loops). Consumes the
merged imaging (02b) and report (03c) outputs.

Outputs (outputs/patient_graph/):
  nodes.parquet   key, ntype, name, attrs(json)
  edges.parquet   src, dst, etype, attrs(json)

Node keys: patient:<sid> adm:<hadm> cxr:<study> finding:<CUI> proc:<ver>_<icd>
           drug:<name> dx:<ver>_<icd>(unmapped diagnosis fallback)
Edges: HAS_ADMISSION, HAS_CXR, DIAGNOSED_WITH, HAS_PROCEDURE, PRESCRIBED,
       IMAGING_FINDING, REPORTED_FINDING, ALIGNS_TO_CUI (Finding/Drug -> pkg:<idx>)
"""
from __future__ import annotations
import os
import json, re
from pathlib import Path
import pandas as pd
import numpy as np

import imaging_findings as IMG

# --- drug-string normalization (MIMIC drug strings -> candidate ingredient names) ---
_DOSE = re.compile(r"\b(\d+(\.\d+)?)\s*(mg|ml|mcg|gm?|g|units?|meq|%|iu|mmol)\b")
_FORM = re.compile(r"\b(tab|tablet|cap|capsule|soln|solution|inj|injection|oral|iv|po|sc|im|"
                   r"syringe|vial|bag|drops?|ointment|cream|patch|er|xr|sr|cr|ec|dr|hcl|hbr|premix|"
                   r"flush|neb|nebu|supp|lozenge|elixir|susp|suspension|conc|kit|pca|human|immediate|"
                   r"release|iso-?osmotic|preservative|free)\b")


def drug_candidates(s: str) -> set[str]:
    """Ingredient-name candidates from a MIMIC drug string: base (dose/form stripped),
    parenthetical (often the brand/generic), and the leading ingredient token."""
    out = set()
    paren = re.findall(r"\(([^)]+)\)", s); base = re.sub(r"\([^)]*\)", " ", s)
    for x in [base] + paren:
        x = _DOSE.sub(" ", x); x = re.sub(r"[0-9.%]", " ", x); x = _FORM.sub(" ", x)
        x = re.sub(r"[^a-z\s/-]", " ", x); x = re.sub(r"\s+", " ", x).strip()
        if x:
            out.add(x)
            toks = x.split()
            if toks:
                out.add(toks[0])
    return out

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", os.path.expandvars("$MMKG_HOME/data/mimiciv/3.1"))) / "hosp"
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "patient_graph"
NOTE_WINDOW = pd.Timedelta(hours=24)


def study_dt(date, time):
    d = pd.to_datetime(date.astype(str).str.split(".").str[0], format="%Y%m%d", errors="coerce")
    t = time.astype(str).str.split(".").str[0].str.zfill(6).str[:6]
    hh = pd.to_numeric(t.str[:2], errors="coerce"); mm = pd.to_numeric(t.str[2:4], errors="coerce"); ss = pd.to_numeric(t.str[4:6], errors="coerce")
    ok = (hh < 24) & (mm < 60) & (ss < 60)
    delta = pd.to_timedelta(np.where(ok, hh * 3600 + mm * 60 + ss, 12 * 3600).astype("float"), unit="s")
    return d + delta


def load_bridge():
    cui2pkg = {}
    for line in open(PRIMEKG / "bridge" / "cui2primekg.tsv"):
        c, idxs = line.rstrip("\n").split("\t"); cui2pkg[c] = [int(i) for i in idxs.split("|")]
    drug2pkg = {}
    for line in open(PRIMEKG / "bridge" / "drug2primekg.tsv"):
        name, idx = line.rstrip("\n").split("\t"); drug2pkg[name] = int(idx)
    icd2cui = {}
    for line in open(PRIMEKG / "mrconso_subset.tsv"):
        cui, sab, code = line.rstrip("\n").split("\t")
        if sab in ("ICD10CM", "ICD9CM"):
            icd2cui.setdefault(code.replace(".", ""), set()).add(cui)
    icd2pkg = {}   # direct ICD -> PrimeKG (MONDO), used with 3-char category roll-up
    for line in open(PRIMEKG / "bridge" / "icd2primekg.tsv"):
        code, idxs = line.rstrip("\n").split("\t"); icd2pkg[code.replace(".", "")] = [int(i) for i in idxs.split("|")]
    return cui2pkg, drug2pkg, icd2cui, icd2pkg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subj_df = pd.read_parquet(HERE / "outputs" / "whole_cohort_subjects.parquet")
    subj = set(subj_df.subject_id)
    studies = pd.read_parquet(HERE / "outputs" / "whole_cohort_studies.parquet")
    imaging = pd.read_parquet(HERE / "outputs" / "imaging" / "biomedclip_findings.parquet")
    reports = pd.read_parquet(HERE / "outputs" / "reports" / "note_findings.parquet")
    studies["study_id"] = studies.study_id.astype(str)   # keep study_id string everywhere
    imaging["study_id"] = imaging.study_id.astype(str)
    cui2pkg, drug2pkg, icd2cui, icd2pkg = load_bridge()
    print(f"cohort {len(subj)} | studies {len(studies)} | imaging {len(imaging)} | report-notes {len(reports)}", flush=True)

    node_frames, edge_frames = [], []
    def add_nodes(keys, ntype, names, attrs=None):
        node_frames.append(pd.DataFrame({"key": keys, "ntype": ntype, "name": names,
                                         "attrs": attrs if attrs is not None else "{}"}))
    def add_edges(src, dst, etype, attrs=None):
        edge_frames.append(pd.DataFrame({"src": src, "dst": dst, "etype": etype,
                                         "attrs": attrs if attrs is not None else "{}"}))

    # ---- Patient ----
    add_nodes("patient:" + subj_df.subject_id.astype(str), "Patient", subj_df.subject_id.astype(str),
              subj_df.apply(lambda r: json.dumps({"gender": r.gender, "anchor_age": int(r.anchor_age)}), axis=1))

    # ---- Admission (+ labels) ----
    adm = pd.read_csv(MIMIC / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "hospital_expire_flag"],
                      parse_dates=["admittime", "dischtime"])
    adm = adm[adm.subject_id.isin(subj)].sort_values(["subject_id", "admittime"]).copy()
    adm["next_admit"] = adm.groupby("subject_id")["admittime"].shift(-1)
    adm["readmit_30d"] = (((adm.next_admit - adm.dischtime).dt.days <= 30) & adm.next_admit.notna()).astype(int)
    adm["los_days"] = ((adm.dischtime - adm.admittime).dt.total_seconds() / 86400).round(2)
    add_nodes("adm:" + adm.hadm_id.astype(str), "Admission", adm.hadm_id.astype(str),
              adm.apply(lambda r: json.dumps({"admission_type": r.admission_type, "los_days": float(r.los_days),
                                              "hospital_expire_flag": int(r.hospital_expire_flag),
                                              "readmit_30d": int(r.readmit_30d),
                                              "admittime": str(r.admittime), "dischtime": str(r.dischtime)}), axis=1))
    add_edges("patient:" + adm.subject_id.astype(str), "adm:" + adm.hadm_id.astype(str), "HAS_ADMISSION")
    adm_win = adm[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()

    # ---- CXR studies (+ imaging scores) + link to admission by datetime window ----
    st = studies.merge(imaging, on="study_id", how="left")
    st["sdt"] = study_dt(st.StudyDate, st.StudyTime)
    score_cols = [f"score_{l.replace(' ', '_')}" for l in IMG.CHEXPERT_FINDINGS]
    def cxr_attr(r):
        sc = {l: (float(r[f"score_{l.replace(' ', '_')}"]) if pd.notna(r.get(f"score_{l.replace(' ', '_')}")) else 0.0)
              for l in IMG.CHEXPERT_FINDINGS}
        return json.dumps({"view": r.ViewPosition, "study_date": str(r.StudyDate),
                           "image_path": r.image_path, "biomedclip_scores": sc})
    add_nodes("cxr:" + st.study_id.astype(str), "CXRStudy", st.study_id.astype(str), st.apply(cxr_attr, axis=1))
    # study -> admission window (merge_asof nearest admit with containment check)
    sj = st[["subject_id", "study_id", "sdt"]].dropna(subset=["sdt"]).sort_values("sdt")
    aw = adm_win.dropna(subset=["admittime"]).sort_values("admittime")
    m = pd.merge_asof(sj, aw, left_on="sdt", right_on="admittime", by="subject_id", direction="backward")
    contained = m.dischtime.notna() & (m.sdt <= m.dischtime)
    hascxr = m[contained]
    add_edges("adm:" + hascxr.hadm_id.astype(int).astype(str), "cxr:" + hascxr.study_id.astype(str), "HAS_CXR")
    outp = m[~contained]
    add_edges("patient:" + outp.subject_id.astype(str), "cxr:" + outp.study_id.astype(str), "HAS_CXR")

    # ---- IMAGING_FINDING (present labels -> CUI) ----
    imp = imaging[["study_id", "present"]].explode("present").dropna(subset=["present"])
    imp = imp[imp.study_id.isin(set(studies.study_id))]
    lab_cui = {lab: IMG.LABEL_CUI[lab][0] for lab in IMG.LABEL_CUI}
    imp["cui"] = imp.present.map(lab_cui)
    imp = imp.dropna(subset=["cui"])
    add_edges("cxr:" + imp.study_id.astype(str), "finding:" + imp.cui, "IMAGING_FINDING",
              imp.present.map(lambda l: json.dumps({"source": "biomedclip", "label": l})))

    # ---- REPORTED_FINDING (note CUIs matched to nearest study +/-24h) ----
    rep = reports.copy(); rep["ct"] = pd.to_datetime(rep.charttime, errors="coerce")
    rep = rep.dropna(subset=["ct"]).sort_values("ct")
    stt = st[["subject_id", "study_id", "sdt"]].dropna(subset=["sdt"]).sort_values("sdt")
    match = pd.merge_asof(rep[["subject_id", "note_id", "ct", "cuis"]], stt,
                          left_on="ct", right_on="sdt", by="subject_id", direction="nearest", tolerance=NOTE_WINDOW)
    match = match.dropna(subset=["study_id"])
    rf = match[["study_id", "note_id", "cuis"]].explode("cuis").dropna(subset=["cuis"])
    add_edges("cxr:" + rf.study_id.astype(str), "finding:" + rf.cuis, "REPORTED_FINDING",
              rf.note_id.map(lambda n: json.dumps({"source": "radiology_report", "note_id": str(n)})))

    # ---- Diagnoses -> Finding(CUI) or Diagnosis fallback ----
    dx = pd.read_csv(MIMIC / "diagnoses_icd.csv.gz", usecols=["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"])
    dx = dx[dx.subject_id.isin(subj)].copy(); dx["code"] = dx.icd_code.astype(str)
    dx["cuis"] = dx.code.map(lambda c: sorted(icd2cui.get(c, ())))
    mapped = dx[dx.cuis.map(len) > 0].explode("cuis")
    add_edges("adm:" + mapped.hadm_id.astype(int).astype(str), "finding:" + mapped.cuis, "DIAGNOSED_WITH",
              mapped.apply(lambda r: json.dumps({"icd": r.code, "icd_version": int(r.icd_version), "seq_num": int(r.seq_num)}), axis=1))
    dname = pd.read_csv(MIMIC / "d_icd_diagnoses.csv.gz")
    dtitle = dict(zip(zip(dname.icd_version, dname.icd_code), dname.long_title))
    unm = dx[dx.cuis.map(len) == 0].copy()
    unm["dxkey"] = "dx:" + unm.icd_version.astype(str) + "_" + unm.code
    add_nodes(unm.dxkey, "Diagnosis", unm.apply(lambda r: dtitle.get((r.icd_version, r.code), r.code), axis=1))
    add_edges("adm:" + unm.hadm_id.astype(int).astype(str), unm.dxkey, "DIAGNOSED_WITH",
              unm.code.map(lambda c: json.dumps({"icd": c})))

    # ---- Procedures ----
    px = pd.read_csv(MIMIC / "procedures_icd.csv.gz", usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    px = px[px.subject_id.isin(subj)].copy()
    pname = pd.read_csv(MIMIC / "d_icd_procedures.csv.gz")
    ptitle = dict(zip(zip(pname.icd_version, pname.icd_code.astype(str)), pname.long_title))
    px["pkey"] = "proc:" + px.icd_version.astype(str) + "_" + px.icd_code.astype(str)
    add_nodes(px.pkey, "Procedure", px.apply(lambda r: ptitle.get((r.icd_version, str(r.icd_code)), str(r.icd_code)), axis=1))
    add_edges("adm:" + px.hadm_id.astype(int).astype(str), px.pkey, "HAS_PROCEDURE")

    # ---- Medications -> Drug (chunked; dedup hadm+drug) + DrugBank align ----
    drug_pairs = []
    for ch in pd.read_csv(MIMIC / "prescriptions.csv.gz", usecols=["subject_id", "hadm_id", "drug"], chunksize=2_000_000):
        ch = ch[ch.subject_id.isin(subj)].dropna(subset=["drug", "hadm_id"])
        ch["name"] = ch.drug.astype(str).str.strip().str.lower()
        drug_pairs.append(ch[["hadm_id", "name"]].drop_duplicates())
    rx = pd.concat(drug_pairs, ignore_index=True).drop_duplicates()
    add_edges("adm:" + rx.hadm_id.astype(int).astype(str), "drug:" + rx.name, "PRESCRIBED")
    drugs = rx.name.drop_duplicates()
    add_nodes("drug:" + drugs, "Drug", drugs)
    # heuristic: exact DrugBank name, else normalized ingredient candidate (Solution 2)
    dsrc, ddst, dvia = [], [], []
    for d in drugs:
        if d in drug2pkg:
            dsrc.append("drug:" + d); ddst.append("pkg:" + str(drug2pkg[d])); dvia.append("drugbank_exact")
        else:
            for c in drug_candidates(d):
                if c in drug2pkg:
                    dsrc.append("drug:" + d); ddst.append("pkg:" + str(drug2pkg[c])); dvia.append("drugbank_normalized"); break
    add_edges(pd.Series(dsrc), pd.Series(ddst), "ALIGNS_TO_CUI",
              pd.Series([json.dumps({"via": v}) for v in dvia]))
    print(f"drug ALIGNS: {len(dsrc)} of {len(drugs)} drugs ({len(dsrc)/max(len(drugs),1):.0%})", flush=True)

    # ---- Finding nodes (all CUIs seen) + ALIGNS_TO_CUI ----
    all_cuis = pd.unique(pd.concat([imp.cui, rf.cuis, mapped.cuis], ignore_index=True).dropna())
    fdf = pd.Series(all_cuis, name="cui")
    add_nodes("finding:" + fdf, "Finding", fdf, fdf.map(lambda c: json.dumps({"cui": c})))
    falign = fdf[fdf.isin(cui2pkg)]
    fa = pd.DataFrame({"cui": falign})
    fa["idx"] = fa.cui.map(cui2pkg)
    fa = fa.explode("idx")
    add_edges("finding:" + fa.cui, "pkg:" + fa.idx.astype(str), "ALIGNS_TO_CUI", json.dumps({"via": "cui"}))

    # ---- diagnosis ANCESTOR ROLL-UP: finding(cui) -> PrimeKG via its ICD, exact else 3-char category (Solution 1) ----
    def icd_pkg(code):
        if code in icd2pkg:
            return icd2pkg[code], "exact"
        if code[:3] in icd2pkg:
            return icd2pkg[code[:3]], "category"
        return [], None
    pairs = mapped[["cuis", "code"]].drop_duplicates()
    rsrc, rdst, rlvl = [], [], []
    for cui, code in zip(pairs.cuis, pairs.code):
        idxs, lvl = icd_pkg(str(code))
        for idx in idxs:
            rsrc.append("finding:" + cui); rdst.append("pkg:" + str(idx)); rlvl.append(lvl)
    add_edges(pd.Series(rsrc, dtype=object), pd.Series(rdst, dtype=object), "ALIGNS_TO_CUI",
              pd.Series([json.dumps({"via": "icd", "align_level": l}) for l in rlvl], dtype=object))
    print(f"dx roll-up ALIGNS: {len(rsrc)} finding->pkg edges from {len(pairs)} (cui,icd) pairs", flush=True)

    # ---- write ----
    nodes = pd.concat(node_frames, ignore_index=True).drop_duplicates("key")
    edges = pd.concat(edge_frames, ignore_index=True).drop_duplicates(["src", "dst", "etype"])
    nodes.to_parquet(OUT / "nodes.parquet", index=False)
    edges.to_parquet(OUT / "edges.parquet", index=False)
    print("\nnodes by type:\n", nodes.ntype.value_counts().to_string())
    print("\nedges by type:\n", edges.etype.value_counts().to_string())
    print(f"\nwrote -> {OUT}/nodes.parquet ({len(nodes)}), edges.parquet ({len(edges)})")


if __name__ == "__main__":
    main()
