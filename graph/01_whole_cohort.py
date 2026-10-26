"""
01_whole_cohort.py   --   env: umls_eval (pandas)
==================================================
WHOLE-GRAPH cohort (no sampling): every patient with a chest x-ray AND a hospital
admission AND >=1 coded diagnosis. All mimic-cxr-jpg images are now on disk, so no
availability filter is needed.

Outputs (outputs/):
  whole_cohort_subjects.parquet  subject_id, gender, anchor_age
  whole_cohort_studies.parquet   subject_id, study_id, dicom_id, ViewPosition,
                                 StudyDate, StudyTime, image_path
                                 (one representative FRONTAL image per study)
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", os.path.expandvars("$MMKG_HOME/data/mimiciv/3.1"))) / "hosp"
CXR   = Path(os.environ.get("MIMIC_CXR_DIR", os.path.expandvars("$MMKG_HOME/data/mimic-cxr-jpg/2.1.0")))
CXR_FILES = CXR / "files"
OUT   = Path(__file__).resolve().parent / "outputs"
FRONTAL = ("PA", "AP")


def image_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    sid = str(subject_id)
    return str(CXR_FILES / f"p{sid[:2]}" / f"p{sid}" / f"s{study_id}" / f"{dicom_id}.jpg")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading CXR metadata, admissions, diagnoses, patients ...", flush=True)
    meta = pd.read_csv(CXR / "mimic-cxr-2.0.0-metadata.csv.gz",
                       usecols=["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate", "StudyTime"])
    adm = pd.read_csv(MIMIC / "admissions.csv.gz", usecols=["subject_id"])
    dx  = pd.read_csv(MIMIC / "diagnoses_icd.csv.gz", usecols=["subject_id"])
    pts = pd.read_csv(MIMIC / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])

    cohort = set(meta.subject_id) & set(adm.subject_id) & set(dx.subject_id)
    print(f"cohort (CXR & admission & diagnosis): {len(cohort)} patients", flush=True)

    subj = pts[pts.subject_id.isin(cohort)][["subject_id", "gender", "anchor_age"]].drop_duplicates("subject_id")
    subj.to_parquet(OUT / "whole_cohort_subjects.parquet", index=False)

    # one representative frontal image per study
    m = meta[meta.subject_id.isin(cohort)].copy()
    m["is_frontal"] = m.ViewPosition.isin(FRONTAL)
    m = m.sort_values(["study_id", "is_frontal"], ascending=[True, False])
    rep = m.drop_duplicates("study_id", keep="first").copy()
    rep["image_path"] = [image_path(s, st, d) for s, st, d in zip(rep.subject_id, rep.study_id, rep.dicom_id)]
    rep = rep[["subject_id", "study_id", "dicom_id", "ViewPosition", "StudyDate", "StudyTime", "image_path"]]
    rep.to_parquet(OUT / "whole_cohort_studies.parquet", index=False)

    n_frontal = int(rep.ViewPosition.isin(FRONTAL).sum())
    print(f"subjects -> {OUT/'whole_cohort_subjects.parquet'} ({len(subj)} rows)", flush=True)
    print(f"studies  -> {OUT/'whole_cohort_studies.parquet'} ({len(rep)} studies; "
          f"{n_frontal} frontal, {len(rep)-n_frontal} other-view fallback)", flush=True)


if __name__ == "__main__":
    main()
