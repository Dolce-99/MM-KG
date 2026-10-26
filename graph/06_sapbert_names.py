"""
06_sapbert_names.py   --   env: torch_gpu
=========================================
SapBERT (entity-identity) name embeddings, 768-d, for every currently-featureless
concept + PrimeKG node:
  Finding    -> name is a raw CUI  -> resolve to UMLS preferred term via MRCONSO
  Drug/Procedure/Diagnosis -> node `name` is already a human string
  kg_*       -> PrimeKG node_name (from nodes.tab)

Writes outputs/embeddings/sapbert_names.npz : keys(str[]), vecs(float32 N x 768).
Keys use the SAME node keys as the graph (finding:CUI, drug:.., proc:.., dx:..,
pkg:<node_index>) so 09 can join directly.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

from embed_utils import load_encoder, embed_texts

HERE = Path(__file__).resolve().parent
PG = HERE / "outputs" / "patient_graph"
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
MRCONSO = Path(os.environ.get("UMLS_META_DIR", os.path.expandvars("$MMKG_HOME/data/umls/2026AA/META"))) / "MRCONSO.RRF"
OUT = HERE / "outputs" / "embeddings"
CLINICAL = {"disease", "effect/phenotype", "drug", "exposure"}


def resolve_cui_names(cuis: set[str]) -> dict[str, str]:
    """Stream MRCONSO once, keep the best English term per requested CUI.
    Priority score = ISPREF=='Y' + TS=='P' + STT=='PF' (higher = more preferred)."""
    best_score: dict[str, int] = {}
    best_name: dict[str, str] = {}
    with open(MRCONSO, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            f = line.split("|")
            if len(f) < 17:
                continue
            cui = f[0]
            if cui not in cuis or f[1] != "ENG":
                continue
            if f[16] == "O":                       # suppressed obsolete
                continue
            score = (f[6] == "Y") + (f[2] == "P") + (f[4] == "PF")
            if score > best_score.get(cui, -1):
                best_score[cui] = score
                best_name[cui] = f[14]
    return best_name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(PG / "nodes.parquet")

    keys, texts = [], []

    # ---- Finding: CUI -> preferred term ----
    find = nodes[nodes.ntype == "Finding"]
    cuis = set(find.name.astype(str))
    print(f"resolving {len(cuis)} Finding CUIs via MRCONSO ...", flush=True)
    cui2name = resolve_cui_names(cuis)
    n_res = sum(c in cui2name for c in cuis)
    print(f"  resolved {n_res}/{len(cuis)} ({n_res/len(cuis):.0%}); unresolved -> keep CUI string", flush=True)
    for k, c in zip(find.key, find.name.astype(str)):
        keys.append(k); texts.append(cui2name.get(c, c))

    # ---- Drug / Procedure / Diagnosis: name is already human text ----
    for nt in ("Drug", "Procedure", "Diagnosis"):
        sub = nodes[nodes.ntype == nt]
        for k, nm in zip(sub.key, sub.name.astype(str)):
            keys.append(k); texts.append(nm)

    # ---- PrimeKG clinical nodes: node_name ----
    pn = pd.read_csv(PRIMEKG / "nodes.tab", sep="\t", usecols=["node_index", "node_type", "node_name"])
    pn = pn[pn.node_type.isin(CLINICAL)]
    for idx, nm in zip(pn.node_index, pn.node_name.astype(str)):
        keys.append(f"pkg:{idx}"); texts.append(nm)

    print(f"embedding {len(texts)} names with SapBERT ...", flush=True)
    tok, mdl = load_encoder("sapbert")
    vecs = embed_texts(tok, mdl, texts, max_len=64, batch_size=128, log_every=50)

    np.savez(OUT / "sapbert_names.npz", keys=np.array(keys, dtype=object), vecs=vecs)
    print(f"wrote {len(keys)} vectors -> {OUT/'sapbert_names.npz'}  (dim {vecs.shape[1]})")


if __name__ == "__main__":
    main()
