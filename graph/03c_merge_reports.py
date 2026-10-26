"""
03c_merge_reports.py   --   env: umls_eval (pandas)
====================================================
Merge the QuickUMLS report-CUI array shards (03b) into one parquet.
Output: outputs/reports/note_findings.parquet  note_id, subject_id, charttime, cuis(list)
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RDIR = HERE / "outputs" / "reports"


def main():
    rows, seen, n_files = [], set(), 0
    for fp in sorted(glob.glob(str(RDIR / "shard_*.jsonl"))):
        n_files += 1
        for line in open(fp):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            nid = str(d.get("note_id"))
            if nid in seen:
                continue
            seen.add(nid)
            rows.append({"note_id": nid, "subject_id": int(d["subject_id"]),
                         "charttime": d.get("charttime"), "cuis": d.get("cuis", [])})
    df = pd.DataFrame(rows)
    out = RDIR / "note_findings.parquet"
    df.to_parquet(out, index=False)
    nf = int((df.cuis.apply(len) > 0).sum()) if len(df) else 0
    print(f"merged {n_files} shards -> {len(df)} notes ({nf} with >=1 CUI) -> {out}")


if __name__ == "__main__":
    main()
