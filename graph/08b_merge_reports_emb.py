"""
08b_merge_reports_emb.py   --   env: torch_gpu (numpy)
======================================================
Concatenate the BioClinicalBERT report shard_*.npz into one table.
Output: outputs/embeddings/reports_emb.npz  keys(str[]="cxr:<id>"), vecs(N x 768).
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RDIR = HERE / "outputs" / "embeddings" / "reports"
OUT = HERE / "outputs" / "embeddings" / "reports_emb.npz"


def main():
    keys, vecs, nshard = [], [], 0
    for fp in sorted(glob.glob(str(RDIR / "shard_*.npz"))):
        d = np.load(fp, allow_pickle=True)
        keys.append(d["keys"]); vecs.append(d["vecs"]); nshard += 1
    keys = np.concatenate(keys); vecs = np.concatenate(vecs)
    _, uniq = np.unique(keys, return_index=True)              # de-dup, keep first
    keys, vecs = keys[uniq], vecs[uniq]
    np.savez(OUT, keys=keys, vecs=vecs)
    print(f"merged {nshard} shards -> {len(keys)} study report vectors -> {OUT}")


if __name__ == "__main__":
    main()
