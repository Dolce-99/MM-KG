"""
08_graphcare_rank_data.py   --   env: torch_gpu   [CPU, ~15 min]
================================================================================
"GraphCare-BAT adapted to PrimeKG and the composite task": one personalised graph per
(patient stay, candidate drug) pair of the ranking benchmark (d12: train bank / dev / test),
with the SAME raw patient observations every other method sees:
  nodes   coded history findings, prior procedures, prior drugs, first-48h medications,
          chest-film report findings (CUIs), imaging findings (CUIs), laboratory analytes as
          cohort-tertile bins ("creatinine high"), the candidate drug, the aligned PrimeKG nodes of
          all of these, and the candidate's PrimeKG one-hop neighbourhood (clinical relations,
          drug_drug excluded, capped at K_CAND) plus K_NB sampled neighbours per patient concept
  visits  2 (prior history; current stay = medications + labs + report + imaging + candidate)
  label   1 if the candidate is the item's positive
GraphCare's compression: SapBERT name embeddings + MiniBatchKMeans (K_CLUS) fitted on all node
strings that occur in any pair (unsupervised, no labels).  Lab-bin strings are embedded with
SapBERT on the fly (CPU).  No gold condition or answer path is used anywhere.
Outputs (RESTRICTED): outputs/restricted/exp4/graphcare_rank_v1/{pairs.pkl}; aggregate stats to outputs/exp4/.
"""
from __future__ import annotations
import json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, runlog, frozen as fz

K_NB = 30; K_CAND = 200; K_CLUS = 4000; SEED = 20260918
OUT = paths.RESTRICTED / "exp4" / "graphcare_rank_v1"; PUB = paths.OUT / "exp4"
LAB_NAMES = ["creatinine", "bun", "sodium", "potassium", "bicarbonate", "glucose", "wbc", "hemoglobin", "platelet", "lactate", "bilirubin", "inr", "aniongap", "albumin", "ph", "po2", "pco2"]


def sapbert_embed(texts):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", local_files_only=True)
    m = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", local_files_only=True).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=32, return_tensors="pt"); out.append(m(**b).last_hidden_state[:, 0].numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    run = runlog.Run("exp4", "graphcare_rank_data_v1", {"K_NB": K_NB, "K_CAND": K_CAND, "K_CLUS": K_CLUS, "seed": SEED})
    ihm = fz.load_ihm(verify=False); f2p, d2p, rep_by_study, img_by_study = fz.load_alignment_and_findings(verify=False)
    pproc = pd.read_parquet(paths.P2 / "ihm" / "prior_proc.parquet").groupby("hadm_id")["proc"].apply(list).to_dict()
    pdrug = pd.read_parquet(paths.P2 / "ihm" / "prior_drug.parquet").groupby("hadm_id")["drug"].apply(list).to_dict()
    ce = pd.read_parquet(PUB / "clinical_edges.parquet", columns=["x_index", "relation", "y_index"])
    nb = defaultdict(list)
    for x, r, y in zip(ce.x_index.values, ce.relation.values, ce.y_index.values):
        nb[int(x)].append((r, int(y)))
    rels = sorted(set(ce.relation)) + ["aligns_to", "has_lab"]; rel_id = {r: i for i, r in enumerate(rels)}
    sap = np.load(paths.P2 / "embeddings" / "sapbert_names.npz", allow_pickle=True)
    keys = [str(k) for k in sap["keys"]]; kidx = {k: i for i, k in enumerate(keys)}; V = sap["vecs"].astype(np.float32)
    # lab tertile bins over the cohort
    idx = ihm["idx"]; bins = {}
    for n in LAB_NAMES:
        for s in ("min", "max"):
            v = idx[f"lab_{n}_{s}"].dropna(); bins[(n, s)] = (float(v.quantile(1 / 3)), float(v.quantile(2 / 3)))
    lab_keys = [f"lab:{n}:{s}:{b}" for n in LAB_NAMES for s in ("min", "max") for b in ("low", "mid", "high")]
    lab_vec = dict(zip(lab_keys, sapbert_embed([f"{n} {s} {b}" for n in LAB_NAMES for s in ("min", "max") for b in ("low", "mid", "high")])))
    def lab_nodes(rec):
        out = []
        for n in LAB_NAMES:
            for s in ("min", "max"):
                v = rec.get(f"lab_{n}_{s}")
                if pd.isna(v): continue
                lo, hi = bins[(n, s)]; out.append(f"lab:{n}:{s}:" + ("low" if v < lo else "high" if v > hi else "mid"))
        return out
    items = []
    for ver, splits in (("ranking_v1_d12", ("dev", "test")), ("ranking_v1_d12_trainbank", ("train",))):
        for it in map(json.loads, open(paths.RESTRICTED / "exp2" / ver / "items.jsonl")):
            if it["split"] in splits: items.append(it | {"version": ver})
    print(f"{len(items)} items", flush=True)
    pairs = []; used = set(); extra_vec = {}
    for it in items:
        h = it["hadm"]; rec = ihm["recs"][h]; st = ihm["sof"].get(h)
        v1 = [f"finding:{c}" for c in dict.fromkeys(ihm["pdx"].get(h, []))] + [f"proc:{p}" for p in dict.fromkeys(pproc.get(h, []))] + [f"drug:{d}" for d in dict.fromkeys(pdrug.get(h, []))]
        v2 = [f"drug:{d}" for d in dict.fromkeys(ihm["meds"].get(h, []))] + lab_nodes(rec)
        if st is not None:
            v2 += [f"finding:{c}" for c in sorted(rep_by_study.get(st, set()) | img_by_study.get(st, set()))]
        v1 = [k for k in v1 if k in kidx]; v2 = [k for k in v2 if k in kidx or k in lab_vec]
        base_edges = []; kg_nodes = set()
        for k in v1 + v2:
            tgt = f2p.get(k[8:], set()) if k.startswith("finding:") else (d2p.get(k[5:].lower(), set()) if k.startswith("drug:") else set())
            for p in tgt:
                pk = f"pkg:{p}"
                if pk not in kidx: continue
                base_edges.append((k, "aligns_to", pk)); kg_nodes.add(pk)
                nbrs = [x for x in nb.get(p, []) if x[0] != "drug_drug"]
                if len(nbrs) > K_NB: nbrs = [nbrs[i] for i in rng.choice(len(nbrs), K_NB, replace=False)]
                for rel, y in nbrs:
                    yk = f"pkg:{y}"
                    if yk in kidx: base_edges.append((pk, rel, yk)); kg_nodes.add(yk)
        for c in it["candidates"]:
            ck = f"drug:{c['mimic_name']}"; pk = f"pkg:{c['pkg']}"
            edges = list(base_edges); kn = set(kg_nodes)
            if ck in kidx and pk in kidx:
                edges.append((ck, "aligns_to", pk)); kn.add(pk)
                nbrs = [x for x in nb.get(c["pkg"], []) if x[0] != "drug_drug"]
                if len(nbrs) > K_CAND: nbrs = [nbrs[i] for i in rng.choice(len(nbrs), K_CAND, replace=False)]
                for rel, y in nbrs:
                    yk = f"pkg:{y}"
                    if yk in kidx: edges.append((pk, rel, yk)); kn.add(yk)
            vis = [list(v1), list(v2) + ([ck] if ck in kidx else [])]
            allk = set(vis[0]) | set(vis[1]) | kn; used |= allk
            pairs.append({"version": it["version"], "idx": it["idx"], "split": it["split"], "hadm": h, "subject": it["subject"], "cand": c["pkg"], "cand_name": c["name"],
                          "y": int(c["positive"]), "visits": vis, "edges": edges, "cand_key": ck})
    ukeys = sorted(used); X = np.stack([V[kidx[k]] if k in kidx else lab_vec[k] for k in ukeys])
    km = MiniBatchKMeans(n_clusters=K_CLUS, random_state=SEED, batch_size=4096, n_init=3).fit(X)
    clus = dict(zip(ukeys, km.labels_.astype(int))); cent = km.cluster_centers_.astype(np.float32)
    graphs = []
    for pr in pairs:
        vis = [sorted({clus[k] for k in v}) for v in pr["visits"]]; ehr = sorted(set(vis[0]) | set(vis[1]))
        e = {(clus[a], rel_id[r], clus[b]) for a, r, b in pr["edges"] if clus[a] != clus[b]}
        nodes = sorted(set(ehr) | {a for a, _, _ in e} | {b for _, _, b in e}); node_set = set(nodes)
        adj = defaultdict(set)
        for a, _, b in pr["edges"]: adj[a].add(b)
        vn = []
        for v in pr["visits"]:
            reach = {clus[k] for k in v}
            for k in v:
                for pk in adj.get(k, ()): reach.add(clus[pk]); reach |= {clus[y] for y in adj.get(pk, ())}
            vn.append(sorted(reach & node_set))
        graphs.append({k: pr[k] for k in ("version", "idx", "split", "hadm", "subject", "cand", "cand_name", "y")} | {"nodes": nodes, "ehr": ehr, "visit_nodes": vn, "edges": sorted(e), "cand_node": clus.get(pr["cand_key"], -1)})
    with open(OUT / "pairs.pkl", "wb") as f:
        pickle.dump({"graphs": graphs, "num_nodes": K_CLUS, "rels": rels, "centroids": cent}, f)
    stats = {"pairs": len(graphs), "items": len(items), "by_split": dict(Counter(g["split"] for g in graphs)), "node_vocab_raw": len(ukeys),
             "nodes_per_pair_mean": float(np.mean([len(g["nodes"]) for g in graphs])), "edges_per_pair_mean": float(np.mean([len(g["edges"]) for g in graphs]))}
    json.dump(stats, open(PUB / "graphcare_rank_v1_stats.json", "w"), indent=2); run.add_output(OUT / "pairs.pkl"); run.finish(counts=stats); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
