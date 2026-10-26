"""
03_graphcare_data.py   --   env: torch_gpu   [CPU, ~20 min]
================================================================================
Faithful local re-implementation of the GraphCare data pipeline (Jiang et al., ICLR 2024)
on OUR frozen ICU-mortality cohort (28,582 first ICU stays, outcome after a 48 h window),
with the SAME patient-stratified test split (seed 42, 20%) and the same 5 seeds as the
MM-KG GNN (phase2/13_icu_ihm_whole.py), so predictions are paired at the stay level.

GraphCare ingredients and what stands in for them here (every substitution is a design
note for the paper):
  concept-level KGs   : GPT-4-generated + UMLS-sampled subgraphs in the paper;
                        here PrimeKG one-hop (clinical types) of the concept's aligned node,
                        at most K_NB neighbours per concept (seeded sample) -- both methods
                        therefore see the SAME external knowledge source
  node/relation text  : GPT embeddings in the paper; here SapBERT name embeddings (frozen
                        table shared with the MM-KG GNN) and learned relation embeddings
  node clustering     : agglomerative on embeddings in the paper; here MiniBatchKMeans with
                        K_CLUS clusters on the same SapBERT space (their vocabulary compression)
  visits              : the paper uses the visit sequence; our leakage-safe tables pool prior
                        visits, so max_visit = 2: [prior history (dx, procedures, drugs)],
                        [current stay: first-48h medications]
  EHR features        : conditions / procedures / drugs only (GraphCare sees no labs or images);
                        the "+labs" variant adds the 36 standardised admission lab features to
                        the prediction head, to separate architecture from modality
Outputs (RESTRICTED, outputs/restricted/exp4/graphcare_v1/): graphs.pkl, split.json, labs.npy
Aggregate: outputs/exp4/graphcare_v1_stats.json
"""
from __future__ import annotations
import json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, runlog

K_NB = 30; K_CLUS = 4000; SEED = 20260918
TEST_FRAC = 0.20; VAL_FRAC = 0.125; SEEDS = [0, 1, 2, 3, 4]; SPLIT_SEED = 42
OUT = paths.RESTRICTED / "exp4" / "graphcare_v1"; PUB = paths.OUT / "exp4"
LAB_NAMES = ["creatinine", "bun", "sodium", "potassium", "bicarbonate", "glucose", "wbc", "hemoglobin", "platelet", "lactate",
             "bilirubin", "inr", "aniongap", "albumin", "ph", "po2", "pco2"]
LAB_COLS = [f"lab_{n}_{s}" for n in LAB_NAMES for s in ("min", "max")] + ["n_abnormal", "n_labs"]


def main():
    OUT.mkdir(parents=True, exist_ok=True); PUB.mkdir(parents=True, exist_ok=True)
    run = runlog.Run("exp4", "graphcare_data_v1", {"K_NB": K_NB, "K_CLUS": K_CLUS, "seed": SEED, "split_seed": SPLIT_SEED})
    rng = np.random.default_rng(SEED)
    IHM = paths.P2 / "ihm"
    idx = pd.read_parquet(IHM / "icu_index.parquet")
    early_meds = pd.read_parquet(IHM / "icu_early_meds.parquet").groupby("hadm_id")["drug"].apply(list).to_dict()
    pdx = pd.read_parquet(IHM / "prior_dx.parquet").groupby("hadm_id")["cui"].apply(list).to_dict()
    pproc = pd.read_parquet(IHM / "prior_proc.parquet").groupby("hadm_id")["proc"].apply(list).to_dict()
    pdrug = pd.read_parquet(IHM / "prior_drug.parquet").groupby("hadm_id")["drug"].apply(list).to_dict()
    ed = pd.read_parquet(paths.frozen("graph_edges", verify=False), columns=["src", "dst", "etype"])
    al = ed[ed.etype == "ALIGNS_TO_CUI"]; align = defaultdict(list)
    for s, d in zip(al.src, al.dst):
        align[s].append(int(d[4:]))
    ce = pd.read_parquet(PUB / "clinical_edges.parquet", columns=["x_index", "relation", "y_index"])
    nb = defaultdict(list)
    for x, r, y in zip(ce.x_index.values, ce.relation.values, ce.y_index.values):
        nb[int(x)].append((r, int(y)))
    rels = sorted(set(ce.relation)) + ["aligns_to"]; rel_id = {r: i for i, r in enumerate(rels)}
    # ---- per-stay raw graphs ----
    sap = np.load(paths.P2 / "embeddings" / "sapbert_names.npz", allow_pickle=True)
    keys = [str(k) for k in sap["keys"]]; kidx = {k: i for i, k in enumerate(keys)}; V = sap["vecs"].astype(np.float32)
    stays = []; used_keys = set(); n_missing = Counter()
    for r in idx.itertuples():
        h = int(r.hadm_id)
        v1 = [f"finding:{c}" for c in dict.fromkeys(pdx.get(h, []))] + [f"proc:{p}" for p in dict.fromkeys(pproc.get(h, []))] + [f"drug:{d}" for d in dict.fromkeys(pdrug.get(h, []))]
        v2 = [f"drug:{d}" for d in dict.fromkeys(early_meds.get(h, []))]
        visits = [[k for k in v if k in kidx] for v in (v1, v2)]
        for v, raw in ((visits[0], v1), (visits[1], v2)):
            n_missing["no_sapbert"] += len(raw) - len(v)
        edges = []; kg_nodes = set()
        for v in visits:
            for k in v:
                for p in align.get(k, []):
                    pk = f"pkg:{p}"
                    if pk not in kidx: continue
                    edges.append((k, "aligns_to", pk)); kg_nodes.add(pk)
                    nbrs = nb.get(p, [])
                    if len(nbrs) > K_NB:
                        nbrs = [nbrs[i] for i in rng.choice(len(nbrs), K_NB, replace=False)]
                    for rel, y in nbrs:
                        yk = f"pkg:{y}"
                        if yk in kidx:
                            edges.append((pk, rel, yk)); kg_nodes.add(yk)
        allk = set(visits[0]) | set(visits[1]) | kg_nodes; used_keys |= allk
        stays.append({"hadm": h, "subject": int(r.subject_id), "y": int(r.label), "visits": visits, "edges": edges, "kg_nodes": sorted(kg_nodes)})
    print(f"stays {len(stays)}; distinct node keys {len(used_keys)}; missing sapbert {dict(n_missing)}", flush=True)
    # ---- cluster the node vocabulary (GraphCare's compression step) ----
    ukeys = sorted(used_keys); X = V[[kidx[k] for k in ukeys]]
    km = MiniBatchKMeans(n_clusters=K_CLUS, random_state=SEED, batch_size=4096, n_init=3).fit(X)
    clus = dict(zip(ukeys, km.labels_.astype(int))); centroids = km.cluster_centers_.astype(np.float32)
    # ---- clustered per-stay graphs ----
    graphs = []
    for s in stays:
        vis = [sorted({clus[k] for k in v}) for v in s["visits"]]
        ehr = sorted(set(vis[0]) | set(vis[1]))
        e = {(clus[a], rel_id[rel], clus[b]) for a, rel, b in s["edges"] if clus[a] != clus[b]}
        nodes = sorted(set(ehr) | {a for a, _, _ in e} | {b for _, _, b in e})
        # nodes reachable from a visit's EHR nodes belong to that visit (GraphCare: visit-level node sets include their KG expansion)
        adj = defaultdict(set)
        for a_, _, b_ in s["edges"]:
            adj[a_].add(b_)
        vis_nodes = []; node_set = set(nodes)
        for v_keys in s["visits"]:
            reach = {clus[k] for k in v_keys}
            for k in v_keys:
                for pk in adj.get(k, ()):
                    reach.add(clus[pk]); reach |= {clus[y] for y in adj.get(pk, ())}
            vis_nodes.append(sorted(reach & node_set))
        graphs.append({"hadm": s["hadm"], "subject": s["subject"], "y": s["y"], "nodes": nodes, "ehr": ehr, "visit_nodes": vis_nodes,
                       "edges": sorted(e)})
    # ---- split identical to 13_icu_ihm_whole.py ----
    idx["patient"] = "patient:" + idx.subject_id.astype(str)
    pat_pos = idx.groupby("patient")["label"].max(); P = pat_pos.index.values; yP = pat_pos.values
    tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC, random_state=SPLIT_SEED).split(P, yP))
    test_pats = sorted(P[te]); trainval = P[tv]; yTV = yP[tv]; folds = {}
    for sd in SEEDS:
        a, b = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=sd).split(trainval, yTV))
        folds[sd] = {"train": sorted(trainval[a]), "val": sorted(trainval[b])}
    labs = idx.set_index("hadm_id")[LAB_COLS].astype(float)
    mu, sd_ = labs.mean(), labs.std() + 1e-6; labs = ((labs - mu) / sd_).fillna(0.0)
    with open(OUT / "graphs.pkl", "wb") as f:
        pickle.dump({"graphs": graphs, "num_nodes": K_CLUS, "rels": rels, "centroids": centroids, "labs": {int(h): labs.loc[h].values.astype(np.float32) for h in labs.index}}, f)
    json.dump({"test_patients": test_pats, "folds": folds}, open(OUT / "split.json", "w"))
    stats = {"stays": len(graphs), "deaths": int(sum(g["y"] for g in graphs)), "node_vocab_raw": len(ukeys), "clusters": K_CLUS,
             "nodes_per_stay_mean": float(np.mean([len(g["nodes"]) for g in graphs])), "edges_per_stay_mean": float(np.mean([len(g["edges"]) for g in graphs])),
             "ehr_nodes_per_stay_mean": float(np.mean([len(g["ehr"]) for g in graphs])), "relations": len(rels), "test_patients": len(test_pats),
             "missing_sapbert_mentions": dict(n_missing)}
    json.dump(stats, open(PUB / "graphcare_v1_stats.json", "w"), indent=2)
    run.add_output(OUT / "graphs.pkl"); run.finish(counts=stats); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
