"""
13_icu_ihm_features.py   --   env: umls_eval (pandas)
================================================================================
WHOLE-COHORT version of the subgraph's 09a: leakage-safe adult-ICU in-hospital-
mortality (IHM) features over the 51,844-patient cohort.
  index      = first ICU stay per admission
  window     = [icu_intime, icu_intime + 48h]
  eligibility= min(deathtime,dischtime) - intime >= 48h  (outcome after window)
  label      = hospital_expire_flag
  current-visit EARLY (in-window): labs, meds, CXR
  PRIOR-visit history (admissions before current admittime): dx->CUI, procedures, meds
  EXCLUDED   : current admission's discharge dx/proc, and LOS
Outputs (outputs/ihm/): icu_index, icu_early_meds, icu_early_cxr, prior_dx,
prior_proc, prior_drug (.parquet).
"""
from __future__ import annotations
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", os.path.expandvars("$MMKG_HOME/data/mimiciv/3.1")))
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
HERE = Path(__file__).resolve().parent
CO = HERE / "outputs"
OUT = CO / "ihm"
WINDOW_H = 48
LAB_PANEL = {50912: "creatinine", 51006: "bun", 50983: "sodium", 50971: "potassium",
             50882: "bicarbonate", 50931: "glucose", 51301: "wbc", 51222: "hemoglobin",
             51265: "platelet", 50813: "lactate", 50885: "bilirubin", 51237: "inr",
             50868: "aniongap", 50862: "albumin", 50820: "ph", 50821: "po2", 50818: "pco2"}


def load_icd2cui():
    icd2cui = defaultdict(set)
    for line in open(PRIMEKG / "mrconso_subset.tsv"):
        cui, sab, code = line.rstrip("\n").split("\t")
        if sab in ("ICD10CM", "ICD9CM"):
            icd2cui[code.replace(".", "")].add(cui)
    return icd2cui


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coh = pd.read_parquet(CO / "whole_cohort_subjects.parquet")
    subj = set(coh.subject_id)
    ptmeta = coh.set_index("subject_id")[["anchor_age", "gender"]].to_dict("index")

    adm = pd.read_csv(MIMIC / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                               "admission_type", "hospital_expire_flag"],
                      parse_dates=["admittime", "dischtime", "deathtime"])
    adm = adm[adm.subject_id.isin(subj)].copy()
    icu = pd.read_csv(MIMIC / "icu" / "icustays.csv.gz",
                      usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
                      parse_dates=["intime", "outtime"])
    icu = icu[icu.subject_id.isin(subj)].sort_values(["hadm_id", "intime"]).drop_duplicates("hadm_id", keep="first")
    m = icu.merge(adm, on=["subject_id", "hadm_id"])
    m["obs_end"] = m.deathtime.fillna(m.dischtime)
    m["obs_h"] = (m.obs_end - m.intime).dt.total_seconds() / 3600
    m = m[m.obs_h >= WINDOW_H].copy()
    m["win_end"] = m.intime + pd.Timedelta(hours=WINDOW_H)
    elig = set(m.hadm_id)
    win = m.set_index("hadm_id")[["intime", "win_end", "admittime"]].to_dict("index")
    print(f"eligible ICU index stays: {len(m)} | deaths {int(m.hospital_expire_flag.sum())} "
          f"(base {m.hospital_expire_flag.mean():.3f})", flush=True)

    # ---- prior-visit history (admissions strictly before current admittime) ----
    print("prior-visit history ...", flush=True)
    icd2cui = load_icd2cui()
    dx = pd.read_csv(MIMIC / "hosp" / "diagnoses_icd.csv.gz", usecols=["subject_id", "hadm_id", "icd_code"])
    dx = dx[dx.subject_id.isin(subj)]
    px = pd.read_csv(MIMIC / "hosp" / "procedures_icd.csv.gz", usecols=["subject_id", "hadm_id", "icd_version", "icd_code"])
    px = px[px.subject_id.isin(subj)]
    hadm_dx = dx.groupby("hadm_id")["icd_code"].apply(list).to_dict()
    hadm_px = {h: [f"{v}_{c}" for v, c in zip(g.icd_version, g.icd_code)] for h, g in px.groupby("hadm_id")}
    adm_sorted = adm.sort_values(["subject_id", "admittime"])
    subj_adms = adm_sorted.groupby("subject_id").apply(lambda g: list(zip(g.hadm_id, g.admittime))).to_dict()

    prior_dx, prior_proc, n_prior = [], [], {}
    for _, r in m.iterrows():
        cur_h, cur_t = r.hadm_id, r.admittime
        priors = [h for h, t in subj_adms.get(r.subject_id, []) if t < cur_t]
        n_prior[cur_h] = len(priors)
        for ph in priors:
            for code in hadm_dx.get(ph, []):
                for cui in icd2cui.get(str(code), ()):
                    prior_dx.append((cur_h, cui))
            for pc in hadm_px.get(ph, []):
                prior_proc.append((cur_h, pc))
    pd.DataFrame(prior_dx, columns=["hadm_id", "cui"]).drop_duplicates().to_parquet(OUT / "prior_dx.parquet", index=False)
    pd.DataFrame(prior_proc, columns=["hadm_id", "proc"]).drop_duplicates().to_parquet(OUT / "prior_proc.parquet", index=False)

    # ---- meds: current-early (window) + prior-visit ----
    print("medications ...", flush=True)
    early_med = []
    for ch in pd.read_csv(MIMIC / "hosp" / "prescriptions.csv.gz",
                          usecols=["subject_id", "hadm_id", "drug", "starttime"],
                          parse_dates=["starttime"], chunksize=2_000_000):
        cur = ch[ch.subject_id.isin(subj) & ch.hadm_id.isin(elig)].dropna(subset=["drug"])
        for r in cur.itertuples(index=False):
            w = win.get(r.hadm_id)
            if w is not None and pd.notna(r.starttime) and r.starttime <= w["win_end"]:
                early_med.append((r.hadm_id, str(r.drug).strip().lower()))
    pd.DataFrame(early_med, columns=["hadm_id", "drug"]).drop_duplicates().to_parquet(OUT / "icu_early_meds.parquet", index=False)
    rx = pd.read_csv(MIMIC / "hosp" / "prescriptions.csv.gz", usecols=["subject_id", "hadm_id", "drug"])
    rx = rx[rx.subject_id.isin(subj)].dropna(subset=["drug"])
    hadm_drug = rx.assign(drug=lambda d: d.drug.str.strip().str.lower()).groupby("hadm_id")["drug"].apply(set).to_dict()
    prior_drug = []
    for _, r in m.iterrows():
        cur_h, cur_t = r.hadm_id, r.admittime
        for ph in [h for h, t in subj_adms.get(r.subject_id, []) if t < cur_t]:
            for d in hadm_drug.get(ph, ()):
                prior_drug.append((cur_h, d))
    pd.DataFrame(prior_drug, columns=["hadm_id", "drug"]).drop_duplicates().to_parquet(OUT / "prior_drug.parquet", index=False)

    # ---- current-visit early labs (ICU window) ----
    print("scanning labevents (chunked) ...", flush=True)
    agg = {}; n_abn = {h: 0 for h in elig}; n_lab = {h: 0 for h in elig}
    for ci, ch in enumerate(pd.read_csv(MIMIC / "hosp" / "labevents.csv.gz",
                            usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "flag"],
                            parse_dates=["charttime"], chunksize=2_000_000)):
        ch = ch[ch.subject_id.isin(subj) & ch.itemid.isin(LAB_PANEL) & ch.hadm_id.isin(elig)]
        for r in ch.itertuples(index=False):
            w = win.get(r.hadm_id)
            if w is None or pd.isna(r.charttime) or not (w["intime"] <= r.charttime <= w["win_end"]):
                continue
            n_lab[r.hadm_id] += 1
            if str(r.flag).lower() == "abnormal":
                n_abn[r.hadm_id] += 1
            if pd.notna(r.valuenum):
                agg.setdefault(r.hadm_id, {}).setdefault(LAB_PANEL[r.itemid], []).append(float(r.valuenum))
        if ci % 20 == 0:
            print(f"  chunk {ci} ...", flush=True)

    rows = []
    for _, r in m.iterrows():
        h = r.hadm_id
        rec = {"subject_id": r.subject_id, "hadm_id": h, "stay_id": r.stay_id,
               "label": int(r.hospital_expire_flag), "admission_type": r.admission_type,
               "age": float(ptmeta.get(r.subject_id, {}).get("anchor_age", 0) or 0),
               "gender": str(ptmeta.get(r.subject_id, {}).get("gender", "")),
               "n_prior_adm": n_prior.get(h, 0), "n_abnormal": n_abn.get(h, 0), "n_labs": n_lab.get(h, 0)}
        labs = agg.get(h, {})
        for name in LAB_PANEL.values():
            v = labs.get(name, [])
            rec[f"lab_{name}_min"] = min(v) if v else float("nan")
            rec[f"lab_{name}_max"] = max(v) if v else float("nan")
        rows.append(rec)
    pd.DataFrame(rows).to_parquet(OUT / "icu_index.parquet", index=False)

    # ---- current-visit early CXR (study datetime within window) ----
    st = pd.read_parquet(CO / "whole_cohort_studies.parquet")
    def sdt(d, t):
        dd = pd.to_datetime(str(d).split(".")[0], format="%Y%m%d", errors="coerce")
        if pd.isna(dd):
            return pd.NaT
        tt = str(t).split(".")[0].zfill(6)[:6]
        try:
            h, mi, s = int(tt[:2]), int(tt[2:4]), int(tt[4:6])
            if h < 24 and mi < 60 and s < 60:
                return dd + pd.Timedelta(hours=h, minutes=mi, seconds=s)
        except ValueError:
            pass
        return dd + pd.Timedelta(hours=12)
    st["sdt"] = [sdt(d, t) for d, t in zip(st.StudyDate, st.StudyTime)]
    j = st.merge(m[["subject_id", "hadm_id", "intime", "win_end"]], on="subject_id")
    j = j[(j.sdt >= j.intime) & (j.sdt <= j.win_end)]
    j[["hadm_id", "study_id"]].drop_duplicates().to_parquet(OUT / "icu_early_cxr.parquet", index=False)

    idx = pd.read_parquet(OUT / "icu_index.parquet")
    print(f"wrote outputs/ihm/: index {len(idx)} stays, deaths {int(idx.label.sum())} | "
          f"early_cxr {j[['hadm_id','study_id']].drop_duplicates().shape[0]} | "
          f"stays w/ prior visit {int((idx.n_prior_adm>0).sum())}", flush=True)


if __name__ == "__main__":
    main()
