"""
02a_medsafety_kg.py   --   env: umls_eval (pandas)   [CPU, run once]
================================================================================
PrimeKG MEDICATION-SAFETY resource cache for the whole-cohort phase-3 task.
Reuses the WHOLE-GRAPH drug normalization (04_patient_tables.py `drug_candidates`)
so we inherit its enriched coverage (~72% of prescription rows, 44% of distinct
drug strings) instead of the old subgraph bridge (~16%).

Builds (outputs/medsafety/kg_cache/):
  cohort_drug2pkg.tsv    MIMIC drug string -> PrimeKG drug node_index (+ how matched)
  ddi_pairs.parquet      drug_drug edges BETWEEN drugs our cohort actually prescribes
                         (PrimeKG has 2.67M drug_drug edges over 7,957 drugs; we keep
                         only the cohort-relevant induced subgraph)
  drug_disease.parquet   contraindication / indication / off-label edges (drug<->disease)
  drug_effect.parquet    drug -> effect/phenotype edges (the ADE-knowledge layer)
  pkg_drug_names.tsv     node_index -> drug name (for prompt verbalization)

NOTE on task validity: everything here is an INPUT FEATURE. The label lives in
02b_ade_labels.py (observed lab events). Never train/evaluate on "a KG edge exists".
"""
from __future__ import annotations
import os
import re
from pathlib import Path
import pandas as pd

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
WG      = Path(__file__).resolve().parent.parent / "graph"
HERE    = Path(__file__).resolve().parent
OUT     = HERE / "outputs" / "medsafety"
CACHE   = OUT / "kg_cache"

# --- drug-string normalization: copied verbatim from whole-graph 04_patient_tables.py ---
_DOSE = re.compile(r"\b(\d+(\.\d+)?)\s*(mg|ml|mcg|gm?|g|units?|meq|%|iu|mmol)\b")
_FORM = re.compile(r"\b(tab|tablet|cap|capsule|soln|solution|inj|injection|oral|iv|po|sc|im|"
                   r"syringe|vial|bag|drops?|ointment|cream|patch|er|xr|sr|cr|ec|dr|hcl|hbr|premix|"
                   r"flush|neb|nebu|supp|lozenge|elixir|susp|suspension|conc|kit|pca|human|immediate|"
                   r"release|iso-?osmotic|preservative|free)\b")


def drug_candidates(s: str) -> set[str]:
    out = set()
    paren = re.findall(r"\(([^)]+)\)", s); base = re.sub(r"\([^)]*\)", " ", s)
    for x in [base] + paren:
        x = _DOSE.sub(" ", x); x = re.sub(r"[0-9.%]", " ", x); x = _FORM.sub(" ", x)
        x = re.sub(r"[^a-z\s/-]", " ", x); x = re.sub(r"\s+", " ", x).strip()
        if x:
            out.add(x)
            toks = x.split()
            if toks:
                out.add(toks[0])
    return out


DRUG_RELATIONS = {"drug_drug", "contraindication", "indication", "off-label use", "drug_effect"}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    # ---- 1. cohort drug strings -> PrimeKG drug nodes ---------------------
    nodes = pd.read_parquet(WG / "outputs" / "patient_graph" / "nodes.parquet", columns=["key", "ntype", "name"])
    drugs = nodes.loc[nodes.ntype == "Drug", "name"].astype(str).tolist()
    print(f"cohort distinct drug strings: {len(drugs):,}")

    drug2pkg = {}
    for line in open(PRIMEKG / "bridge" / "drug2primekg.tsv"):
        n, i = line.rstrip("\n").split("\t"); drug2pkg[n] = int(i)
    print(f"DrugBank name->node entries: {len(drug2pkg):,}")

    mapped = {}
    for d in drugs:
        if d in drug2pkg:
            mapped[d] = (drug2pkg[d], "drugbank_exact")
            continue
        for c in sorted(drug_candidates(d), key=len, reverse=True):
            if c in drug2pkg:
                mapped[d] = (drug2pkg[c], "drugbank_normalized")
                break
    with open(CACHE / "cohort_drug2pkg.tsv", "w") as w:
        for d, (i, how) in sorted(mapped.items()):
            w.write(f"{d}\t{i}\t{how}\n")
    n_exact = sum(1 for v in mapped.values() if v[1] == "drugbank_exact")
    print(f"mapped drugs: {len(mapped):,}/{len(drugs):,} ({len(mapped)/len(drugs):.1%}) "
          f"[exact {n_exact:,} | normalized {len(mapped)-n_exact:,}]")
    cohort_nodes = {i for i, _ in mapped.values()}
    print(f"distinct PrimeKG drug nodes reached: {len(cohort_nodes):,}")

    # ---- 2. scan kg.csv once for all drug-relevant relations ---------------
    print("\nscanning kg.csv for drug relations ...", flush=True)
    cols = ["relation", "x_index", "x_type", "x_name", "y_index", "y_type", "y_name"]
    ddi, ddis, deff = [], [], []
    n_seen = 0
    for ci, ch in enumerate(pd.read_csv(PRIMEKG / "kg.csv", usecols=cols,
                                        dtype={"x_index": "int32", "y_index": "int32"},
                                        chunksize=1_000_000)):
        n_seen += len(ch)
        ch = ch[ch.relation.isin(DRUG_RELATIONS)]
        if ch.empty:
            continue
        # DDI: keep only pairs where BOTH drugs occur in our cohort (induced subgraph)
        dd = ch[(ch.relation == "drug_drug") &
                ch.x_index.isin(cohort_nodes) & ch.y_index.isin(cohort_nodes)]
        if len(dd): ddi.append(dd[["x_index", "x_name", "y_index", "y_name"]])
        # drug <-> disease (contraindication / indication / off-label)
        dz = ch[ch.relation.isin({"contraindication", "indication", "off-label use"})]
        if len(dz): ddis.append(dz)
        # drug -> effect/phenotype (ADE knowledge)
        de = ch[(ch.relation == "drug_effect") & ch.x_index.isin(cohort_nodes | set(ch.y_index))]
        if len(de): deff.append(de)
        if ci % 5 == 0:
            print(f"  chunk {ci} ({n_seen:,} rows) ...", flush=True)

    ddi_df = pd.concat(ddi, ignore_index=True).drop_duplicates() if ddi else pd.DataFrame()
    ddis_df = pd.concat(ddis, ignore_index=True).drop_duplicates() if ddis else pd.DataFrame()
    deff_df = pd.concat(deff, ignore_index=True).drop_duplicates() if deff else pd.DataFrame()
    ddi_df.to_parquet(CACHE / "ddi_pairs.parquet", index=False)
    ddis_df.to_parquet(CACHE / "drug_disease.parquet", index=False)
    deff_df.to_parquet(CACHE / "drug_effect.parquet", index=False)

    # ---- 3. PrimeKG drug node names ---------------------------------------
    nt = pd.read_csv(PRIMEKG / "nodes.tab", sep="\t",
                     usecols=["node_index", "node_type", "node_name"], dtype={"node_index": "int32"})
    dn = nt[nt.node_type == "drug"]
    with open(CACHE / "pkg_drug_names.tsv", "w") as w:
        for i, n in zip(dn.node_index, dn.node_name):
            w.write(f"{i}\t{n}\n")

    print("\n=== medication-safety KG cache ===")
    print(f"DDI pairs (cohort-induced)  : {len(ddi_df):,}")
    print(f"drug-disease edges          : {len(ddis_df):,}")
    if len(ddis_df):
        print(ddis_df.relation.value_counts().to_string())
    print(f"drug_effect edges           : {len(deff_df):,}")
    print(f"wrote -> {CACHE}")


if __name__ == "__main__":
    main()
