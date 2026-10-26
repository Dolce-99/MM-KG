"""
03b_report_findings.py   --   env: umls_eval   (run as a SLURM array)
=====================================================================
QuickUMLS finding-CUI extraction over the whole cohort's chest notes (from 03a),
sharded for a SLURM array. Each task extracts CUIs for its strided slice and
writes its own jsonl. Report<->study linkage (+/-24h) happens later in the
patient-table build (04-side), so here we just persist per-note CUIs + charttime.

  WG_NSHARDS / SLURM_ARRAY_TASK_ID   sharding (as in 02)
Output: outputs/reports/shard_<id>.jsonl
          {note_id, subject_id, charttime, cuis:[...]}
"""
from __future__ import annotations
import os, json
from pathlib import Path
import pandas as pd
import nlp_cui

HERE = Path(__file__).resolve().parent
NOTES = HERE / "outputs" / "reports" / "cohort_chest_notes.parquet"
OUTDIR = HERE / "outputs" / "reports"
NSHARDS = int(os.environ.get("WG_NSHARDS", "1"))
SHARD = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))


def load_done(path):
    done = set()
    if path.exists():
        for line in open(path):
            try:
                done.add(str(json.loads(line)["note_id"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(NOTES)
    shard = df.iloc[SHARD::NSHARDS]
    out_path = OUTDIR / f"shard_{SHARD:03d}.jsonl"
    done = load_done(out_path)
    todo = [r for r in shard.itertuples(index=False) if str(r.note_id) not in done]
    print(f"shard {SHARD}/{NSHARDS}: {len(shard)} notes, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        print("nothing to do", flush=True); return

    nlp = nlp_cui.build_nlp()
    logf = open(out_path, "a")
    n = 0
    for i, r in enumerate(todo, 1):
        cuis = list(nlp_cui.extract_cuis(nlp, nlp_cui.findings_sections(r.text)).keys())
        logf.write(json.dumps({"note_id": str(r.note_id), "subject_id": int(r.subject_id),
                               "charttime": str(r.charttime), "cuis": cuis}) + "\n")
        n += 1
        if i % 500 == 0:
            logf.flush(); print(f"  [{i}/{len(todo)}]", flush=True)
    logf.close()
    print(f"=== shard {SHARD} done: {n} notes -> {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
