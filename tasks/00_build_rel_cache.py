"""
00_build_rel_cache.py   --   env: umls_eval   [CPU, ~1 min]
================================================================================
Cache of the PrimeKG relations used by the join benchmark and the retrieval
experiments (contraindication, drug_effect, drug_protein, disease_protein):
a projection of kg.csv onto those relations, in file order, both edge
directions kept, no deduplication.

Output: outputs/mm/primekg_rel_cache.parquet
"""
import os
from pathlib import Path

import pandas as pd

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
OUT = Path(__file__).resolve().parent / "outputs" / "mm"
REL = {"contraindication", "drug_effect", "drug_protein", "disease_protein"}
COLS = ["relation", "x_index", "x_type", "x_name", "y_index", "y_type", "y_name"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kg = pd.read_csv(PRIMEKG / "kg.csv", usecols=COLS)
    kg = kg[kg.relation.isin(REL)].reset_index(drop=True)
    kg.to_parquet(OUT / "primekg_rel_cache.parquet", index=False)
    print(f"{len(kg)} rows -> {OUT / 'primekg_rel_cache.parquet'}")
    print(kg.relation.value_counts().to_string())


if __name__ == "__main__":
    main()
