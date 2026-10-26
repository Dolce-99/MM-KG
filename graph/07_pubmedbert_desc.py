"""
07_pubmedbert_desc.py   --   env: torch_gpu
===========================================
PubMedBERT (semantic-content) description embeddings, 768-d, for kg_disease and
kg_drug nodes (the only PrimeKG clinical types with natural-language descriptions).

Disease text (user spec): coalesce a definition via fallback chain, then append
clinical detail; exclude non-descriptive / numeric fields.
  definition = first_nonempty(mondo_definition, umls_description, orphanet_definition)
  clinical   = mayo_symptoms + mayo_causes + mayo_risk_factors + mayo_complications
               + orphanet_clinical_description
  (excluded: mayo_prevention, mayo_see_doc, orphanet_prevalence, orphanet_epidemiology)
Drug text (analogous): description + indication + mechanism_of_action
  (excluded: numeric/ATC/pharmacokinetic fields).

Truncate at 512 tokens. Writes outputs/embeddings/pubmedbert_desc.npz :
  keys(str[] = "pkg:<idx>"), vecs(float32 N x 768), has_desc(int8 N).
Only nodes present in OUR graph (clinical disease/drug node_index) are embedded.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

from embed_utils import load_encoder, embed_texts

HERE = Path(__file__).resolve().parent
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
OUT = HERE / "outputs" / "embeddings"

DIS_DEF = ["mondo_definition", "umls_description", "orphanet_definition"]
DIS_CLIN = ["mayo_symptoms", "mayo_causes", "mayo_risk_factors",
            "mayo_complications", "orphanet_clinical_description"]
DRUG_DEF = ["description"]
DRUG_CLIN = ["indication", "mechanism_of_action"]


def _clean(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none", "") else s


def build_text(row, defs, clin) -> str:
    definition = next((v for c in defs if (v := _clean(row.get(c)))), "")
    detail = " ".join(v for c in clin if (v := _clean(row.get(c))))
    return f"{definition} {detail}".strip()


def collect(feat_path, defs, clin, keep_idx):
    df = pd.read_csv(feat_path, sep="\t")
    df = df[df.node_index.isin(keep_idx)]
    keys = [f"pkg:{i}" for i in df.node_index]
    texts = [build_text(r, defs, clin) for _, r in df.iterrows()]
    return keys, texts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pn = pd.read_csv(PRIMEKG / "nodes.tab", sep="\t", usecols=["node_index", "node_type"])
    dis_idx = set(pn[pn.node_type == "disease"].node_index)
    drug_idx = set(pn[pn.node_type == "drug"].node_index)

    k1, t1 = collect(PRIMEKG / "disease_features.tab", DIS_DEF, DIS_CLIN, dis_idx)
    k2, t2 = collect(PRIMEKG / "drug_features.tab", DRUG_DEF, DRUG_CLIN, drug_idx)
    keys = k1 + k2
    texts = t1 + t2
    has = np.array([1 if t.strip() else 0 for t in texts], dtype=np.int8)
    print(f"disease rows {len(k1)} ({int(has[:len(k1)].sum())} with text), "
          f"drug rows {len(k2)} ({int(has[len(k1):].sum())} with text)", flush=True)

    print(f"embedding {len(texts)} descriptions with PubMedBERT (trunc 512) ...", flush=True)
    tok, mdl = load_encoder("pubmedbert")
    vecs = embed_texts(tok, mdl, texts, max_len=512, batch_size=32, log_every=50)
    vecs[has == 0] = 0.0                                     # empty text -> zero block

    np.savez(OUT / "pubmedbert_desc.npz",
             keys=np.array(keys, dtype=object), vecs=vecs, has_desc=has)
    print(f"wrote {len(keys)} description vectors -> {OUT/'pubmedbert_desc.npz'}")


if __name__ == "__main__":
    main()
