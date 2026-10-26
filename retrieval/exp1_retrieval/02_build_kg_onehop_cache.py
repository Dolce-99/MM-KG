"""
02_build_kg_onehop_cache.py   --   env: umls_eval   [CPU, ~3 min, one pass over kg.csv]
================================================================================
Typed one-hop neighbourhoods in raw PrimeKG for every node that Experiment 1 may
link to (queried drugs, the patients' aligned concepts, the patients' aligned
medications) plus the protein->disease hop needed for the mechanism family.
Also collects PrimeKG node text (drug and disease descriptions) for those nodes.

kg.csv lists every undirected edge in both directions, so filtering on x_index
returns the complete one-hop set.  Nothing here is patient-level: the input is
a set of PrimeKG node indices.

Outputs (outputs/exp1/):  kg_onehop_v1.parquet, kg_node_text_v1.parquet, kg_onehop_v1_stats.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, runlog

PUB = paths.OUT / "exp1"
COLS = ["relation", "display_relation", "x_index", "x_id", "x_type", "x_name", "x_source",
        "y_index", "y_id", "y_type", "y_name", "y_source"]


def main():
    import os
    LN = os.environ.get("MMKG_LINKED_NODES", "linked_nodes_v1.json"); KGV = os.environ.get("MMKG_KG_ONEHOP_VERSION", "kg_onehop_v1")
    linked = json.load(open(PUB / LN))
    S = set(linked["queried_drugs"]) | set(linked["med_drugs"]) | set(linked["concepts"])
    P = set(linked["proteins"])
    cfg = {"version": "kg_onehop_v1", "n_linked": len(S), "n_proteins": len(P), "source": str(paths.PRIMEKG / "kg.csv")}
    run = runlog.Run("exp1", "build_kg_onehop_v1", cfg)
    run.add_input(paths.PRIMEKG / "kg.csv"); run.add_input(PUB / "linked_nodes_v1.json")
    parts = []
    for ci, ch in enumerate(pd.read_csv(paths.PRIMEKG / "kg.csv", usecols=COLS, chunksize=1_000_000,
                                        dtype={"x_id": str, "y_id": str})):
        keep = ch[ch.x_index.isin(S) | (ch.x_index.isin(P) & (ch.relation == "disease_protein"))]
        parts.append(keep); print(f"  chunk {ci}: kept {len(keep)}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["hop_role"] = ["linked" if x in S else "protein_bridge" for x in df.x_index]
    df.to_parquet(PUB / f"{KGV}.parquet", index=False)

    ids = S | P | set(df.y_index)
    dfeat = pd.read_csv(paths.PRIMEKG / "drug_features.tab", sep="\t",
                        usecols=["node_index", "description", "indication", "mechanism_of_action"], dtype=str)
    dfeat = dfeat[dfeat.node_index.astype(int).isin(ids)]
    disf = pd.read_csv(paths.PRIMEKG / "disease_features.tab", sep="\t",
                       usecols=["node_index", "mondo_id", "mondo_name", "mondo_definition", "umls_description"], dtype=str)
    disf = disf[disf.node_index.astype(int).isin(ids)]
    text = pd.concat([
        pd.DataFrame({"node_index": dfeat.node_index.astype(int), "node_type": "drug",
                      "text": dfeat.description.fillna(""), "indication": dfeat.indication.fillna(""),
                      "mechanism": dfeat.mechanism_of_action.fillna("")}),
        pd.DataFrame({"node_index": disf.node_index.astype(int), "node_type": "disease",
                      "text": disf.mondo_definition.fillna("").where(disf.mondo_definition.notna(), disf.umls_description.fillna("")),
                      "indication": "", "mechanism": ""})], ignore_index=True)
    text.to_parquet(PUB / f"{KGV.replace('kg_onehop', 'kg_node_text')}.parquet", index=False)

    deg = df[df.hop_role == "linked"].groupby("x_index").size()
    stats = {"rows": int(len(df)), "linked_nodes_with_edges": int(deg.shape[0]), "linked_nodes_requested": len(S),
             "degree_quantiles": {q: int(deg.quantile(q)) for q in (0.5, 0.9, 0.99)}, "degree_max": int(deg.max()),
             "rows_by_relation": {k: int(v) for k, v in df.relation.value_counts().items()},
             "node_text_rows": int(len(text)), "node_text_nonempty": int((text.text.str.len() > 0).sum())}
    json.dump(stats, open(PUB / f"{KGV}_stats.json", "w"), indent=2)
    for p in (PUB / f"{KGV}.parquet", PUB / f"{KGV.replace('kg_onehop', 'kg_node_text')}.parquet", PUB / f"{KGV}_stats.json"):
        run.add_output(p)
    run.finish(counts=stats); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
