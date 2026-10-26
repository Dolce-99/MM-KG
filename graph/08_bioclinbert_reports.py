"""
08_bioclinbert_reports.py   --   env: torch_gpu   (SLURM ARRAY, the long pole)
==============================================================================
BioClinicalBERT report-text embeddings, 768-d, per CXRStudy.

Matches the SAME nearest note per study as 04's REPORTED_FINDING edges
(merge_asof nearest, tolerance +/-24h, by subject_id), then embeds that note's
free text. When several notes map to one study we keep the temporally closest.

Sharded like the BioMedCLIP / QuickUMLS arrays (WG_NSHARDS strided by
SLURM_ARRAY_TASK_ID). Each task writes outputs/embeddings/reports/shard_<id>.npz
(keys = "cxr:<study_id>", vecs float32 M x 768). Resumable.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

from embed_utils import load_encoder, embed_texts

HERE = Path(__file__).resolve().parent
CO = HERE / "outputs"
RDIR = CO / "embeddings" / "reports"
NOTE_WINDOW = pd.Timedelta(hours=24)

NSHARDS = int(os.environ.get("WG_NSHARDS", "16"))
SHARD = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))


def study_dt(date, time):
    d = pd.to_datetime(date.astype(str).str.split(".").str[0], format="%Y%m%d", errors="coerce")
    t = time.astype(str).str.split(".").str[0].str.zfill(6).str[:6]
    hh = pd.to_numeric(t.str[:2], errors="coerce"); mm = pd.to_numeric(t.str[2:4], errors="coerce"); ss = pd.to_numeric(t.str[4:6], errors="coerce")
    ok = (hh < 24) & (mm < 60) & (ss < 60)
    delta = pd.to_timedelta(np.where(ok, hh * 3600 + mm * 60 + ss, 12 * 3600).astype("float"), unit="s")
    return d + delta


def study_report_map() -> pd.DataFrame:
    """One row per study that has a matched report: study_id, note_id (nearest)."""
    studies = pd.read_parquet(CO / "whole_cohort_studies.parquet")
    studies["sdt"] = study_dt(studies.StudyDate, studies.StudyTime)
    stt = studies[["subject_id", "study_id", "sdt"]].dropna(subset=["sdt"]).sort_values("sdt")
    stt["study_id"] = stt.study_id.astype(str)

    rep = pd.read_parquet(CO / "reports" / "note_findings.parquet")[["subject_id", "note_id", "charttime"]]
    rep["ct"] = pd.to_datetime(rep.charttime, errors="coerce")
    rep = rep.dropna(subset=["ct"]).sort_values("ct")

    m = pd.merge_asof(rep, stt, left_on="ct", right_on="sdt", by="subject_id",
                      direction="nearest", tolerance=NOTE_WINDOW).dropna(subset=["study_id"])
    m["gap"] = (m.ct - m.sdt).abs()
    m = m.sort_values("gap").drop_duplicates("study_id", keep="first")
    return m[["study_id", "note_id"]]


def main():
    RDIR.mkdir(parents=True, exist_ok=True)
    outp = RDIR / f"shard_{SHARD}.npz"
    if outp.exists():
        print(f"shard {SHARD} already done -> {outp}"); return

    m = study_report_map()
    m = m.iloc[SHARD::NSHARDS].reset_index(drop=True)          # strided shard
    print(f"shard {SHARD}/{NSHARDS}: {len(m)} studies with reports", flush=True)

    notes = pd.read_parquet(CO / "reports" / "cohort_chest_notes.parquet")[["note_id", "text"]]
    notes["note_id"] = notes.note_id.astype(str)
    m["note_id"] = m.note_id.astype(str)
    m = m.merge(notes, on="note_id", how="left")
    texts = m.text.fillna("").tolist()

    tok, mdl = load_encoder("bioclinical")
    vecs = embed_texts(tok, mdl, texts, max_len=512, batch_size=32, log_every=50)

    keys = ("cxr:" + m.study_id.astype(str)).tolist()
    np.savez(outp, keys=np.array(keys, dtype=object), vecs=vecs)
    print(f"shard {SHARD}: wrote {len(keys)} report vectors -> {outp}")


if __name__ == "__main__":
    main()
