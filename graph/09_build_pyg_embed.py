"""
09_build_pyg_embed.py   --   env: torch_gpu (pandas, torch_geometric)
=====================================================================
Same graph as 05, but the previously-featureless concept / PrimeKG nodes now
carry real pretrained text embeddings, and CXRStudy gains its report embedding.
Writes a SEPARATE file so the placeholder graph (05) is preserved.

Per-node-type features (each type may differ in width -- fine for HeteroData):
  Patient            [is_M, is_F, age/100]                          (3)   unchanged
  Admission          [admission_type one-hot + log1p(LOS)/10]       (10)  unchanged
  CXRStudy           [13 BioMedCLIP probs | report_768 | has_report](782)
  Finding/Drug/
    Procedure/Diagnosis   SapBERT(name)_768                         (768)
  kg_disease/kg_drug SapBERT(name)_768 | PubMedBERT(desc)_768 | has_desc (1537)
  kg_effect/phenotype,
    kg_exposure      SapBERT(name)_768                              (768)

Each text block is already L2-normalised (embed_utils). Missing sub-features are
zero-filled and flagged (has_report / has_desc).

Outputs (outputs/graph/):
  whole_graph_hetero_embed.pt
  whole_graph_embed_summary.json
"""
from __future__ import annotations
import os
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

import imaging_findings as IMG

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
HERE = Path(__file__).resolve().parent
PG = HERE / "outputs" / "patient_graph"
EMB = HERE / "outputs" / "embeddings"
OUT = HERE / "outputs" / "graph"
CLINICAL = {"disease", "effect/phenotype", "drug", "exposure"}
D = 768


def load_emb(fp, want_flag=False):
    d = np.load(fp, allow_pickle=True)
    vecs = d["vecs"].astype(np.float32)
    flag = d["has_desc"] if (want_flag and "has_desc" in d) else None
    # dedup duplicate keys, preferring a row that actually carries text (flag==1)
    idx = {}
    for i, k in enumerate(d["keys"]):
        cur = idx.get(k)
        if cur is None or (flag is not None and flag[i] and not flag[cur]):
            idx[k] = i
    return idx, vecs, flag


def gather(keys, idx, vecs, flag=None):
    """Return (N x dim) block aligned to `keys`; missing -> zeros. Plus has-mask."""
    n = len(keys)
    out = np.zeros((n, vecs.shape[1]), dtype=np.float32)
    has = np.zeros((n, 1), dtype=np.float32)
    for i, k in enumerate(keys):
        j = idx.get(k)
        if j is not None:
            out[i] = vecs[j]
            has[i] = 1.0 if (flag is None or flag[j]) else 0.0
    return out, has


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(PG / "nodes.parquet")
    edges = pd.read_parquet(PG / "edges.parquet")

    # ---- full clinical PrimeKG (identical to 05) ----
    pnodes = pd.read_csv(PRIMEKG / "nodes.tab", sep="\t", usecols=["node_index", "node_type", "node_name"])
    pcl = pnodes[pnodes.node_type.isin(CLINICAL)].copy()
    pcl["key"] = "pkg:" + pcl.node_index.astype(str)
    pcl["ntype"] = "kg_" + pcl.node_type
    add_pkg = pd.DataFrame({"key": pcl.key, "ntype": pcl.ntype, "name": pcl.node_name.astype(str), "attrs": "{}"})
    nodes = pd.concat([nodes, add_pkg], ignore_index=True).drop_duplicates("key")
    clinical_idx = set(pcl.node_index)
    kg = pd.read_csv(PRIMEKG / "kg.csv", usecols=["relation", "x_index", "y_index"])
    kg = kg[kg.x_index.isin(clinical_idx) & kg.y_index.isin(clinical_idx)]
    kg_edges = pd.DataFrame({"src": "pkg:" + kg.x_index.astype(str), "dst": "pkg:" + kg.y_index.astype(str),
                             "etype": kg.relation, "attrs": "{}"})
    ok_pkg = set(add_pkg.key)
    al = edges.etype == "ALIGNS_TO_CUI"
    edges = pd.concat([edges[~al], edges[al & edges.dst.isin(ok_pkg)]], ignore_index=True)
    edges = pd.concat([edges, kg_edges], ignore_index=True)

    # ---- load embedding tables ----
    sap_idx, sap_vecs, _ = load_emb(EMB / "sapbert_names.npz")
    desc_idx, desc_vecs, desc_flag = load_emb(EMB / "pubmedbert_desc.npz", want_flag=True)
    rep_idx, rep_vecs, _ = load_emb(EMB / "reports_emb.npz")
    print(f"loaded emb: sapbert {len(sap_idx)}, desc {len(desc_idx)}, reports {len(rep_idx)}", flush=True)

    # ---- per-type local index ----
    by_type = {nt: df.reset_index(drop=True) for nt, df in nodes.groupby("ntype")}
    local = {}
    for nt, df in by_type.items():
        local.update(dict(zip(df.key, range(len(df)))))
    ntype_of = dict(zip(nodes.key, nodes.ntype))

    data = HeteroData()
    adm_df = by_type.get("Admission")
    adm_types = sorted(adm_df["attrs"].map(lambda a: json.loads(a).get("admission_type", "UNK")).unique())
    at_idx = {t: i for i, t in enumerate(adm_types)}

    SAP_TYPES = {"Finding", "Drug", "Procedure", "Diagnosis", "kg_effect/phenotype", "kg_exposure"}
    DESC_TYPES = {"kg_disease", "kg_drug"}
    feat_dims = {}
    for nt, df in by_type.items():
        data[nt].num_nodes = len(df)
        data[nt].node_key = df.key.tolist()
        keys = df.key.tolist()
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
            probs = np.array([[float((x.get("biomedclip_scores") or {}).get(l, 0.0)) for l in IMG.CHEXPERT_FINDINGS] for x in a], dtype=np.float32)
            rep, has_rep = gather(keys, rep_idx, rep_vecs)
            data[nt].x = torch.tensor(np.concatenate([probs, rep, has_rep], axis=1), dtype=torch.float)
        elif nt in SAP_TYPES:
            name, _ = gather(keys, sap_idx, sap_vecs)
            data[nt].x = torch.tensor(name, dtype=torch.float)
        elif nt in DESC_TYPES:
            name, _ = gather(keys, sap_idx, sap_vecs)
            desc, has_d = gather(keys, desc_idx, desc_vecs, desc_flag)
            data[nt].x = torch.tensor(np.concatenate([name, desc, has_d], axis=1), dtype=torch.float)
        else:
            data[nt].x = torch.ones((len(df), 1))
        feat_dims[nt] = int(data[nt].x.shape[1])

    # ---- edges ----
    edges = edges[edges.src.isin(local) & edges.dst.isin(local)].copy()
    edges["st"] = edges.src.map(ntype_of); edges["dt"] = edges.dst.map(ntype_of)
    edges["si"] = edges.src.map(local); edges["di"] = edges.dst.map(local)
    rel_counts = {}
    for (st, et, dt), g in edges.groupby(["st", "etype", "dt"]):
        data[st, et, dt].edge_index = torch.tensor([g.si.values, g.di.values], dtype=torch.long)
        rel_counts[f"{st}--{et}-->{dt}"] = len(g)

    torch.save(data, OUT / "whole_graph_hetero_embed.pt")
    summary = {
        "nodes_total": int(len(nodes)), "edges_total": int(len(edges)),
        "node_types": {nt: len(df) for nt, df in by_type.items()},
        "feature_dims": feat_dims,
        "edge_types": rel_counts,
        "labels": {"n_admissions": int(len(adm_df)),
                   "mortality_pos": int(data["Admission"].y_mortality.sum()),
                   "readmit30_pos": int(data["Admission"].y_readmit30.sum())},
        "encoders": {"names": "SapBERT-from-PubMedBERT-fulltext (768, L2)",
                     "descriptions": "PubMedBERT abstract (768, L2, kg_disease/kg_drug)",
                     "reports": "Bio_ClinicalBERT (768, L2, CXRStudy nearest note +/-24h)"},
    }
    (OUT / "whole_graph_embed_summary.json").write_text(json.dumps(summary, indent=2))
    print("feature dims:", feat_dims)
    print("labels:", summary["labels"])
    print(f"wrote -> {OUT}/whole_graph_hetero_embed.pt")


if __name__ == "__main__":
    main()
