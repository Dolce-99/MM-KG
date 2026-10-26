"""
03a_prep_reports.py   --   env: umls_eval (pandas)
===================================================
Collect the whole cohort's CHEST radiology notes once, so the QuickUMLS array
(03b) can shard over a compact parquet instead of re-scanning the big note file.
Chest filter mirrors the subgraph pipeline (exam_name contains 'chest', excludes
ct/mr/angio).

Output: outputs/reports/cohort_chest_notes.parquet  note_id, subject_id, charttime, text
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

NOTE = Path(os.environ.get("MIMIC_NOTE_DIR", os.path.expandvars("$MMKG_HOME/data/mimic-iv-note/2.2/note")))
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "reports"


def read_filtered(path, usecols, subjects, chunksize=1_000_000):
    parts = []
    for ch in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        parts.append(ch[ch.subject_id.isin(subjects)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subj = set(pd.read_parquet(HERE / "outputs" / "whole_cohort_subjects.parquet").subject_id)
    print(f"cohort subjects: {len(subj)}", flush=True)

    print("scanning radiology_detail for chest exams ...", flush=True)
    det = read_filtered(NOTE / "radiology_detail.csv.gz",
                        ["note_id", "subject_id", "field_name", "field_value"], subj)
    exam = det[det.field_name == "exam_name"]
    chest_ids = set(exam.loc[exam.field_value.str.contains("chest", case=False, na=False) &
                             ~exam.field_value.str.contains(r"\bct\b|mr |angio", case=False, na=False),
                             "note_id"])
    print(f"chest note_ids: {len(chest_ids)}", flush=True)

    print("reading radiology notes ...", flush=True)
    rad = read_filtered(NOTE / "radiology.csv.gz",
                        ["note_id", "subject_id", "charttime", "text"], subj)
    rad = rad[rad.note_id.isin(chest_ids)].dropna(subset=["text"]).copy()
    rad.to_parquet(OUT / "cohort_chest_notes.parquet", index=False)
    print(f"cohort chest notes: {len(rad)} -> {OUT/'cohort_chest_notes.parquet'}", flush=True)


if __name__ == "__main__":
    main()
