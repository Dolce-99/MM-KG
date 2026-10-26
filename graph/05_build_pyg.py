"""
05_build_pyg.py   --   env: torch_gpu (pandas, torch_geometric)
===============================================================
Merge the patient tables (04) with the FULL CLINICAL PrimeKG (disease /
effect-phenotype / drug / exposure; keeping drug_drug) and export a PyG
HeteroData built DIRECTLY from parquet tables (no NetworkX -- does not fit at
whole-graph scale).

Outputs (outputs/graph/):
  whole_graph_hetero.pt        PyG HeteroData
  whole_graph_summary.json     node/edge-type counts + label stats

Node features: Patient=[is_M,is_F,age/100]; Admission=[admission_type one-hot +
log1p(LOS)]; CXRStudy=[13 BioMedCLIP probs]; concept/kg types = 1-d placeholder
(learnable embeddings added by the downstream model). Labels y_mortality /
y_readmit30 on Admission.
"""
from __future__ import annotations
import os
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

import imaging_findings as IMG

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
HERE = Path(__file__).resolve().parent
PG = HERE / "outputs" / "patient_graph"
OUT = HERE / "outputs" / "graph"
CLINICAL = {"disease", "effect/phenotype", "drug", "exposure"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(PG / "nodes.parquet")
    edges = pd.read_parquet(PG / "edges.parquet")
    print(f"patient tables: {len(nodes)} nodes, {len(edges)} edges", flush=True)

    # ---- full clinical PrimeKG: add all clinical nodes + clinical-clinical edges ----
    pnodes = pd.read_csv(PRIMEKG / "nodes.tab", sep="\t", usecols=["node_index", "node_type", "node_name"])
    pcl = pnodes[pnodes.node_type.isin(CLINICAL)].copy()
    pcl["key"] = "pkg:" + pcl.node_index.astype(str)
    pcl["ntype"] = "kg_" + pcl.node_type
    add_pkg = pd.DataFrame({"key": pcl.key, "ntype": pcl.ntype, "name": pcl.node_name.astype(str), "attrs": "{}"})
    nodes = pd.concat([nodes, add_pkg], ignore_index=True).drop_duplicates("key")
    idx2type = dict(zip(pcl.node_index, "kg_" + pcl.node_type))
    clinical_idx = set(pcl.node_index)

    kg = pd.read_csv(PRIMEKG / "kg.csv", usecols=["relation", "x_index", "y_index"])
    kg = kg[kg.x_index.isin(clinical_idx) & kg.y_index.isin(clinical_idx)]     # keep drug_drug
    kg_edges = pd.DataFrame({"src": "pkg:" + kg.x_index.astype(str), "dst": "pkg:" + kg.y_index.astype(str),
                             "etype": kg.relation, "attrs": "{}"})
    # keep only patient ALIGNS edges whose pkg endpoint is a clinical node
    ok_pkg = set(add_pkg.key)
    al = edges.etype == "ALIGNS_TO_CUI"
    edges = pd.concat([edges[~al], edges[al & edges.dst.isin(ok_pkg)]], ignore_index=True)
    edges = pd.concat([edges, kg_edges], ignore_index=True)
    print(f"+ full clinical PrimeKG: {len(add_pkg)} kg nodes, {len(kg_edges)} kg edges "
          f"-> total {len(nodes)} nodes, {len(edges)} edges", flush=True)

    # ---- per-type local index ----
    by_type = {nt: df.reset_index(drop=True) for nt, df in nodes.groupby("ntype")}
    local = {}   # key -> local idx (within its type)
    for nt, df in by_type.items():
        local.update(dict(zip(df.key, range(len(df)))))
    ntype_of = dict(zip(nodes.key, nodes.ntype))

    data = HeteroData()
    FEAT = {"Patient", "Admission", "CXRStudy"}
    adm_df = by_type.get("Admission")
    adm_types = sorted(adm_df["attrs"].map(lambda a: json.loads(a).get("admission_type", "UNK")).unique()) if adm_df is not None else []
    at_idx = {t: i for i, t in enumerate(adm_types)}

    for nt, df in by_type.items():
        data[nt].num_nodes = len(df)
        data[nt].node_key = df.key.tolist()
        if nt == "Patient":
            a = df["attrs"].map(json.loads)
            data[nt].x = torch.tensor([[float(x.get("gender", "").upper() == "M"),
                                        float(x.get("gender", "").upper() == "F"),
                                        float(x.get("anchor_age", 0) or 0) / 100.0] for x in a], dtype=torch.float)
        elif nt == "Admission":
            a = df["attrs"].map(json.loads)
            feats, mort, readm = [], [], []
            for x in a:
                oh = [0.0] * len(adm_types); oh[at_idx.get(x.get("admission_type", "UNK"), 0)] = 1.0
                feats.append(oh + [math.log1p(max(float(x.get("los_days", 0) or 0), 0.0)) / 10.0])
                mort.append(int(x.get("hospital_expire_flag", 0))); readm.append(int(x.get("readmit_30d", 0)))
            data[nt].x = torch.tensor(feats, dtype=torch.float)
            data[nt].y_mortality = torch.tensor(mort); data[nt].y_readmit30 = torch.tensor(readm)
        elif nt == "CXRStudy":
            a = df["attrs"].map(json.loads)
            data[nt].x = torch.tensor([[float((x.get("biomedclip_scores") or {}).get(l, 0.0)) for l in IMG.CHEXPERT_FINDINGS] for x in a], dtype=torch.float)
        else:
            data[nt].x = torch.ones((len(df), 1))

    # ---- edges per relation ----
    edges = edges[edges.src.isin(local) & edges.dst.isin(local)].copy()
    edges["st"] = edges.src.map(ntype_of); edges["dt"] = edges.dst.map(ntype_of)
    edges["si"] = edges.src.map(local); edges["di"] = edges.dst.map(local)
    rel_counts = {}
    for (st, et, dt), g in edges.groupby(["st", "etype", "dt"]):
        data[st, et, dt].edge_index = torch.tensor([g.si.values, g.di.values], dtype=torch.long)
        rel_counts[f"{st}--{et}-->{dt}"] = len(g)

    torch.save(data, OUT / "whole_graph_hetero.pt")
    summary = {
        "nodes_total": int(len(nodes)), "edges_total": int(len(edges)),
        "node_types": {nt: len(df) for nt, df in by_type.items()},
        "edge_types": rel_counts,
        "labels": {"n_admissions": int(len(adm_df)) if adm_df is not None else 0,
                   "mortality_pos": int(data["Admission"].y_mortality.sum()) if adm_df is not None else 0,
                   "readmit30_pos": int(data["Admission"].y_readmit30.sum()) if adm_df is not None else 0},
        "primekg": "full clinical (disease/effect-phenotype/drug/exposure), drug_drug kept",
    }
    (OUT / "whole_graph_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nnode types:", summary["node_types"])
    print("labels:", summary["labels"])
    print(f"PyG relations: {len(rel_counts)} | wrote -> {OUT}/whole_graph_hetero.pt, whole_graph_summary.json")


if __name__ == "__main__":
    main()
