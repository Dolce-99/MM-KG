"""
02b_merge_imaging.py   --   env: torch_gpu / umls_eval (pandas)
===============================================================
Merge the BioMedCLIP array shards into one parquet. Dedups by study_id (keeps the
'ok' row). Emits per-study 13 probabilities + present-label list.

Output: outputs/imaging/biomedclip_findings.parquet
          study_id, present(list), lab_<13 finding scores>
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import pandas as pd
from imaging_findings import CHEXPERT_FINDINGS

HERE = Path(__file__).resolve().parent
IMGDIR = HERE / "outputs" / "imaging"


def main():
    rows, seen = [], set()
    n_files = 0
    for fp in sorted(glob.glob(str(IMGDIR / "shard_*.jsonl"))):
        n_files += 1
        for line in open(fp):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(d.get("study_id"))
            if sid in seen or d.get("status") != "ok":
                continue
            seen.add(sid)
            rec = {"study_id": sid, "present": d.get("present", [])}
            sc = d.get("scores", {})
            for lab in CHEXPERT_FINDINGS:
                rec[f"score_{lab.replace(' ', '_')}"] = float(sc.get(lab, 0.0))
            rows.append(rec)
    df = pd.DataFrame(rows)
    out = IMGDIR / "biomedclip_findings.parquet"
    df.to_parquet(out, index=False)
    n_present = int((df.present.apply(len) > 0).sum()) if len(df) else 0
    print(f"merged {n_files} shards -> {len(df)} studies ({n_present} with >=1 finding) -> {out}")


if __name__ == "__main__":
    main()
