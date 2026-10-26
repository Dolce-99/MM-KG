"""
11_readmit_whole.py   --   env: torch_gpu
================================================================================
WHOLE-COHORT 30-day-readmission prediction + ablations. Same vetted harness as
the subgraph 07 (leakage-safe temporal ego-subgraphs: index admission + PRIOR
admissions only; readmission is predicted AT DISCHARGE, so LOS and discharge-coded
diagnoses/procedures are legitimately available), scaled to all 219,720 admissions
of the 51,844-patient cohort -- with ENRICHED node features:

  * concept / PrimeKG nodes (Finding/Drug/Procedure/Diagnosis + kg_*) carry FROZEN
    pretrained text embeddings (SapBERT names; +PubMedBERT desc for kg_disease/kg_drug).
  * CXRStudy feature = [13 BioMedCLIP probs | BioClinicalBERT report 768 | has_report].

Ablations (as in 07): Full / -KG / -Imaging / -History, + LR baseline
(admission_type + LOS). Label-stratified PATIENT split, 5 seeds, bootstrap CI.
One CONFIG per SLURM array task (0=Full 1=-KG 2=-Imaging 3=-History); task 0 also
writes the LR baseline.
"""
from __future__ import annotations
import os
import json, math, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

import imaging_findings as IMG

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8)))

HERE = Path(__file__).resolve().parent
PG = HERE / "outputs" / "patient_graph"
EMB = HERE / "outputs" / "embeddings"
OUTDIR = HERE / "outputs" / "readmit"
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))

KG_TYPES = {"kg_disease", "kg_effect/phenotype", "kg_drug", "kg_exposure"}
CONCEPT_TYPES = {"Finding", "Drug", "Procedure", "Diagnosis"} | KG_TYPES
CLINICAL = {"disease", "effect/phenotype", "drug", "exposure"}
PREFIX_NT = {"patient": "Patient", "adm": "Admission", "cxr": "CXRStudy",
             "finding": "Finding", "drug": "Drug", "proc": "Procedure", "dx": "Diagnosis"}
PKG_HOP_CAP = 10
HIDDEN = 32; LAYERS = 2; DROPOUT = 0.5
LR_ = 3e-3; WD = 1e-3; MAX_EPOCHS = 100; PATIENCE = 12; BATCH = 512
TEST_FRAC = 0.20; VAL_FRAC = 0.125; SEEDS = [0, 1, 2, 3, 4]; SPLIT_SEED = 42; N_BOOT = 2000
N_CXR_PROBS = len(IMG.CHEXPERT_FINDINGS); CXR_DIM = N_CXR_PROBS + 768 + 1

CONFIGS = [
    {"name": "Full",     "kg": True,  "img": True,  "hist": True},
    {"name": "-KG",      "kg": False, "img": True,  "hist": True},
    {"name": "-Imaging", "kg": True,  "img": False, "hist": True},
    {"name": "-History", "kg": True,  "img": True,  "hist": False},
]


# ---------------------------------------------------------- structure from tables
def load_structure():
    nodes = pd.read_parquet(PG / "nodes.parquet")
    edges = pd.read_parquet(PG / "edges.parquet")
    pn = pd.read_csv(PRIMEKG / "nodes.tab", sep="\t", usecols=["node_index", "node_type"])
    pn = pn[pn.node_type.isin(CLINICAL)]
    kg_type = {f"pkg:{i}": f"kg_{t}" for i, t in zip(pn.node_index, pn.node_type)}
    clinical_idx = set(pn.node_index)

    def nt_of(key):
        return kg_type.get(key) if key.startswith("pkg:") else PREFIX_NT.get(key.split(":", 1)[0])

    patient_of = {}; pat_adms = defaultdict(list)
    adm_adj = defaultdict(list); cxr_find = defaultdict(list)
    find_kg = defaultdict(list); drug_kg = defaultdict(list)
    real_triples = set()
    for et, src, dst in zip(edges.etype.values, edges.src.values, edges.dst.values):
        if et == "HAS_ADMISSION":
            patient_of[dst] = src; pat_adms[src].append(dst)
            real_triples.add(("Patient", "HAS_ADMISSION", "Admission"))
        elif et in ("DIAGNOSED_WITH", "HAS_PROCEDURE", "PRESCRIBED"):
            adm_adj[src].append((dst, et)); real_triples.add(("Admission", et, nt_of(dst)))
        elif et == "HAS_CXR" and src.startswith("adm:"):
            adm_adj[src].append((dst, et)); real_triples.add(("Admission", "HAS_CXR", "CXRStudy"))
        elif et in ("IMAGING_FINDING", "REPORTED_FINDING"):
            cxr_find[src].append((dst, et)); real_triples.add(("CXRStudy", et, "Finding"))
        elif et == "ALIGNS_TO_CUI":
            if src.startswith("drug:"):
                drug_kg[src].append(dst); real_triples.add(("Drug", "ALIGNS_TO_CUI", nt_of(dst)))
            else:
                find_kg[src].append(dst); real_triples.add(("Finding", "ALIGNS_TO_CUI", nt_of(dst)))

    kg = pd.read_csv(PRIMEKG / "kg.csv", usecols=["relation", "x_index", "y_index"])
    kg = kg[(kg.x_index.isin(clinical_idx)) & (kg.y_index.isin(clinical_idx)) & (kg.relation != "drug_drug")]
    kg_adj = defaultdict(list)
    for rel, x, y in zip(kg.relation.values, kg.x_index.values, kg.y_index.values):
        kx, ky = f"pkg:{x}", f"pkg:{y}"
        kg_adj[kx].append((rel, ky)); real_triples.add((kg_type.get(kx), rel, kg_type.get(ky)))

    # admission + patient attrs
    adm_attr = {}; adm_time = {}
    ad = nodes[nodes.ntype == "Admission"]
    for k, a in zip(ad.key, ad["attrs"]):
        d = json.loads(a); adm_attr[k] = d
        adm_time[k] = pd.to_datetime(d.get("admittime"), errors="coerce")
    pat_attr = {}
    pt = nodes[nodes.ntype == "Patient"]
    for k, a in zip(pt.key, pt["attrs"]):
        pat_attr[k] = json.loads(a)

    vocab = {t: {} for t in CONCEPT_TYPES}
    for nt in ("Finding", "Drug", "Procedure", "Diagnosis"):
        for k in nodes[nodes.ntype == nt].key:
            vocab[nt][k] = len(vocab[nt])
    for k, t in kg_type.items():
        if t in vocab:
            vocab[t][k] = len(vocab[t])

    cxr_scores = {}
    cx = nodes[nodes.ntype == "CXRStudy"]
    for k, a in zip(cx.key, cx["attrs"]):
        sc = (json.loads(a).get("biomedclip_scores") or {})
        cxr_scores[k] = [float(sc.get(l, 0.0)) for l in IMG.CHEXPERT_FINDINGS]

    return dict(patient_of=patient_of, pat_adms=pat_adms, adm_adj=adm_adj, cxr_find=cxr_find,
                find_kg=find_kg, drug_kg=drug_kg, kg_adj=kg_adj, adm_attr=adm_attr, adm_time=adm_time,
                pat_attr=pat_attr, vocab=vocab, kg_type=kg_type, cxr_scores=cxr_scores,
                real_triples=real_triples, nt_of=nt_of)


def build_emb_tables(vocab):
    def load(fp):
        d = np.load(fp, allow_pickle=True); idx = {}
        flag = d["has_desc"] if "has_desc" in d else None
        for i, k in enumerate(d["keys"]):
            cur = idx.get(k)
            if cur is None or (flag is not None and flag[i] and not flag[cur]):
                idx[k] = i
        return idx, d["vecs"].astype(np.float32), flag
    sap_i, sap_v, _ = load(EMB / "sapbert_names.npz")
    des_i, des_v, des_f = load(EMB / "pubmedbert_desc.npz")

    def sap(k):
        j = sap_i.get(k); return sap_v[j] if j is not None else np.zeros(768, np.float32)

    def desc_block(k):
        j = des_i.get(k)
        if j is None:
            return np.zeros(769, np.float32)
        return np.concatenate([des_v[j], [1.0 if (des_f is None or des_f[j]) else 0.0]]).astype(np.float32)

    tables = {}
    for nt, vmap in vocab.items():
        n = len(vmap)
        if nt in ("kg_disease", "kg_drug"):
            M = np.zeros((n, 768 + 769), np.float32)
            for k, i in vmap.items():
                M[i] = np.concatenate([sap(k), desc_block(k)])
        else:
            M = np.zeros((n, 768), np.float32)
            for k, i in vmap.items():
                M[i] = sap(k)
        tables[nt] = torch.from_numpy(M)
    return tables


def canon_rels(cfg, real_triples):
    rels = set()
    for (su, et, sv) in real_triples:
        if su is None or sv is None:
            continue
        if et == "drug_drug":
            continue
        if not cfg["kg"] and (su in KG_TYPES or sv in KG_TYPES):
            continue
        if et == "IMAGING_FINDING" and not cfg["img"]:
            continue
        rels.add((su, et, sv))
    return sorted(rels)


def node_types_for(cfg):
    nts = {"Patient", "Admission", "CXRStudy", "Finding", "Drug", "Procedure", "Diagnosis"}
    if cfg["kg"]:
        nts |= KG_TYPES
    return nts


# ------------------------------------------------------- temporal keep-set + build
def temporal_nodes(S, A, cfg):
    adm_time = S["adm_time"]; t = adm_time.get(A); p = S["patient_of"].get(A)
    keep = {A}
    if p is not None:
        keep.add(p)
        if cfg["hist"]:
            for B in S["pat_adms"].get(p, []):
                if B != A and pd.notna(adm_time.get(B)) and pd.notna(t) and adm_time[B] < t:
                    keep.add(B)
    findings_drugs = set()
    for adm in [n for n in list(keep) if n.startswith("adm:")]:
        for tgt, et in S["adm_adj"].get(adm, []):
            if et in ("DIAGNOSED_WITH", "PRESCRIBED", "HAS_PROCEDURE"):
                keep.add(tgt)
                if tgt.startswith("finding:") or tgt.startswith("drug:"):
                    findings_drugs.add(tgt)
            elif et == "HAS_CXR":
                keep.add(tgt)
                for w, r in S["cxr_find"].get(tgt, []):
                    if r == "REPORTED_FINDING" or (r == "IMAGING_FINDING" and cfg["img"]):
                        keep.add(w); findings_drugs.add(w)
    if cfg["kg"]:
        pkg = set()
        for fd in findings_drugs:
            for kk in S["find_kg"].get(fd, []) + S["drug_kg"].get(fd, []):
                pkg.add(kk)
        hop = set()
        for k in pkg:
            c = 0
            for rel, v in S["kg_adj"].get(k, []):
                hop.add(v); c += 1
                if c >= PKG_HOP_CAP:
                    break
        keep |= pkg | hop
    return keep


def build_subgraph(S, A, keep, cfg, canon, all_nts, feat_fns):
    nt_of = S["nt_of"]; vocab = S["vocab"]
    pat_feat, adm_feat, cxr_feat, adm_dim_ = feat_fns
    by_type = defaultdict(list)
    for n in keep:
        nt = nt_of(n)
        if nt in all_nts:
            by_type[nt].append(n)
    local = {}
    for nt, ns in by_type.items():
        for i, n in enumerate(ns):
            local[n] = (nt, i)

    data = HeteroData()
    for nt in all_nts:
        ns = by_type.get(nt, [])
        data[nt].num_nodes = len(ns)
        if nt == "Patient":
            data[nt].x = torch.tensor([pat_feat(n) for n in ns], dtype=torch.float).reshape(len(ns), 3)
        elif nt == "Admission":
            data[nt].x = torch.tensor([adm_feat(n) for n in ns], dtype=torch.float).reshape(len(ns), adm_dim_)
            data[nt].is_target = torch.tensor([n == A for n in ns], dtype=torch.bool)
        elif nt == "CXRStudy":
            data[nt].x = torch.tensor([cxr_feat(n, cfg) for n in ns], dtype=torch.float).reshape(len(ns), CXR_DIM)
        else:
            data[nt].gid = torch.tensor([vocab[nt][n] for n in ns], dtype=torch.long)

    buckets = {r: [[], []] for r in canon}

    def add(u, v, et):
        if u in local and v in local:
            su, iu = local[u]; sv, iv = local[v]; r = (su, et, sv)
            if r in buckets:
                buckets[r][0].append(iu); buckets[r][1].append(iv)

    for n in keep:
        if n.startswith("patient:"):
            for adm in S["pat_adms"].get(n, []):
                add(n, adm, "HAS_ADMISSION")
        elif n.startswith("adm:"):
            for tgt, et in S["adm_adj"].get(n, []):
                add(n, tgt, et)
        elif n.startswith("cxr:"):
            for w, r in S["cxr_find"].get(n, []):
                if r == "IMAGING_FINDING" and not cfg["img"]:
                    continue
                add(n, w, r)
        elif n.startswith("finding:"):
            for kk in S["find_kg"].get(n, []):
                add(n, kk, "ALIGNS_TO_CUI")
        elif n.startswith("drug:"):
            for kk in S["drug_kg"].get(n, []):
                add(n, kk, "ALIGNS_TO_CUI")
        elif n.startswith("pkg:"):
            for rel, kk2 in S["kg_adj"].get(n, []):
                add(n, kk2, rel)
    for r, (s, o) in buckets.items():
        data[r].edge_index = torch.tensor([s, o], dtype=torch.long) if s else torch.zeros((2, 0), dtype=torch.long)
    return T.ToUndirected(merge=False)(data)


# ------------------------------------------------------------------ model (same as 10)
class HGNN(nn.Module):
    def __init__(self, tables, feat_dims, edge_types, adm_dim):
        super().__init__()
        self.emb = nn.ModuleDict({nt.replace("/", "_"): nn.Embedding.from_pretrained(M, freeze=True) for nt, M in tables.items()})
        self.emb_proj = nn.ModuleDict({nt.replace("/", "_"): nn.Linear(M.shape[1], HIDDEN) for nt, M in tables.items()})
        self.proj = nn.ModuleDict({nt: nn.Linear(d, HIDDEN) for nt, d in feat_dims.items()})
        self.convs = nn.ModuleList([HeteroConv({et: SAGEConv((-1, -1), HIDDEN) for et in edge_types}, aggr="sum") for _ in range(LAYERS)])
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(HIDDEN + adm_dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))

    def forward(self, b):
        h = {}
        for nt in b.node_types:
            k = nt.replace("/", "_")
            h[nt] = self.emb_proj[k](self.emb[k](b[nt].gid)) if k in self.emb else self.proj[nt](b[nt].x)
            h[nt] = self.drop(F.relu(h[nt]))
        for conv in self.convs:
            o = conv(h, b.edge_index_dict)
            h = {nt: self.drop(F.relu(o[nt])) if nt in o else h[nt] for nt in h}
        m = b["Admission"].is_target
        return self.head(torch.cat([h["Admission"][m], b["Admission"].x[m]], -1)).squeeze(-1)


def safe_auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")


def bootstrap_ci(y, p, n=N_BOOT):
    rng = np.random.default_rng(0); a = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if 0 < y[s].sum() < len(s):
            a.append(roc_auc_score(y[s], p[s]))
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (float("nan"), float("nan"))


def evaluate(model, loader, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device); ps.append(torch.sigmoid(model(b)).cpu()); ys.append(b["Admission"].y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train_once(tr, va, te, tables, feat_dims, edge_types, adm_dim, seed, device):
    torch.manual_seed(seed)
    model = HGNN(tables, feat_dims, edge_types, adm_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_, weight_decay=WD)
    ytr = torch.tensor([d["Admission"].y.item() for d in tr])
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True); vl = DataLoader(va, batch_size=BATCH); sl = DataLoader(te, batch_size=BATCH)
    best = {"v": -1, "s": None}; bad = 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for b in tl:
            b = b.to(device); opt.zero_grad(); loss_fn(model(b), b["Admission"].y).backward(); opt.step()
        yv, pv = evaluate(model, vl, device); v = safe_auc(yv, pv)
        if v == v and v > best["v"]:
            best = {"v": v, "s": {k: val.cpu().clone() for k, val in model.state_dict().items()}}; bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    if best["s"] is not None:
        model.load_state_dict(best["s"])
    yte, pte = evaluate(model, sl, device)
    return best["v"], yte, pte


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    S = load_structure()
    vocab = S["vocab"]; adm_attr = S["adm_attr"]; pat_attr = S["pat_attr"]; patient_of = S["patient_of"]
    print("vocab:", {k: len(v) for k, v in vocab.items()}, flush=True)
    tables = build_emb_tables(vocab)

    adms = list(adm_attr.keys())
    at = sorted({str(adm_attr[a].get("admission_type", "UNK")) for a in adms}); at_idx = {t: i for i, t in enumerate(at)}
    adm_dim = len(at) + 1
    feat_dims = {"Patient": 3, "Admission": adm_dim, "CXRStudy": CXR_DIM}

    rep = np.load(EMB / "reports_emb.npz", allow_pickle=True)
    rep_idx = {k: i for i, k in enumerate(rep["keys"])}; rep_vecs = rep["vecs"].astype(np.float32)

    def pat_feat(p):
        d = pat_attr.get(p, {}); g = str(d.get("gender", "")).upper()
        return [float(g == "M"), float(g == "F"), float(d.get("anchor_age", 0) or 0) / 100.0]

    def adm_feat(a):
        d = adm_attr[a]; oh = [0.0] * len(at); oh[at_idx.get(str(d.get("admission_type", "UNK")), 0)] = 1.0
        return oh + [math.log1p(max(float(d.get("los_days", 0) or 0), 0.0)) / 10.0]

    def cxr_feat(ck, cfg):
        probs = S["cxr_scores"].get(ck, [0.0] * N_CXR_PROBS)
        v = probs if cfg["img"] else [0.0] * N_CXR_PROBS
        j = rep_idx.get(ck)
        return v + (list(rep_vecs[j]) + [1.0] if j is not None else [0.0] * 769)

    feat_fns = (pat_feat, adm_feat, cxr_feat, adm_dim)
    readmit = {a: int(adm_attr[a].get("readmit_30d", 0)) for a in adms}

    # patient-level stratified split (patient has >=1 readmitted admission)
    patients = sorted({patient_of.get(a) for a in adms if patient_of.get(a)})
    pat_pos = {p: 0 for p in patients}
    for a in adms:
        p = patient_of.get(a)
        if p and readmit[a]:
            pat_pos[p] = 1
    P = np.array(patients); yP = np.array([pat_pos[p] for p in patients])
    tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC, random_state=SPLIT_SEED).split(P, yP))
    test_pats = set(P[te]); trainval = P[tv]; yTV = yP[tv]
    folds = {}
    for s in SEEDS:
        a, b = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=s).split(trainval, yTV))
        folds[s] = (set(trainval[a]), set(trainval[b]))
    print(f"admissions {len(adms)} readmit-pos {sum(readmit.values())} | patients {len(P)} "
          f"test patients {len(te)} pos-rate {yP[te].mean():.3f}", flush=True)

    arr = os.environ.get("SLURM_ARRAY_TASK_ID")
    configs = [CONFIGS[int(arr)]] if arr is not None else CONFIGS
    RES = OUTDIR / "results"; RES.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        all_nts = node_types_for(cfg)
        cfg_tables = {nt: tables[nt] for nt in (CONCEPT_TYPES & all_nts)}
        canon = canon_rels(cfg, S["real_triples"])
        print(f"\n=== build subgraphs: {cfg['name']} (rels {len(canon)}) ===", flush=True)
        sub = {}
        for a in adms:
            keep = temporal_nodes(S, a, cfg)
            d = build_subgraph(S, a, keep, cfg, canon, all_nts, feat_fns)
            d["Admission"].y = torch.tensor([float(readmit[a])])
            sub[a] = d
        edge_types = next(iter(sub.values())).edge_types
        test_g = [sub[a] for a in adms if patient_of.get(a) in test_pats]
        yref = np.array([readmit[a] for a in adms if patient_of.get(a) in test_pats])
        aucs, aps, probs = [], [], []
        for s in SEEDS:
            trp, vap = folds[s]
            tr = [sub[a] for a in adms if patient_of.get(a) in trp]; va = [sub[a] for a in adms if patient_of.get(a) in vap]
            v, yte, pte = train_once(tr, va, test_g, cfg_tables, feat_dims, edge_types, adm_dim, s, device)
            aucs.append(safe_auc(yte, pte)); aps.append(average_precision_score(yte, pte)); probs.append(pte)
            print(f"  {cfg['name']} seed {s}: val {v:.3f} test AUROC {aucs[-1]:.3f}", flush=True)
        ens = np.mean(np.stack(probs), 0); lo, hi = bootstrap_ci(yref, ens)
        res = {"auroc_mean": float(np.nanmean(aucs)), "auroc_std": float(np.nanstd(aucs)),
               "ap_mean": float(np.mean(aps)), "ensemble_auroc": float(safe_auc(yref, ens)),
               "ci95": [lo, hi], "n_test": int(len(yref)), "test_pos": int(yref.sum())}
        (RES / f"cfg_{cfg['name']}.json").write_text(json.dumps({cfg["name"]: res}, indent=2))
        print(f"  => {cfg['name']}: {res['auroc_mean']:.3f}+-{res['auroc_std']:.3f} ens {res['ensemble_auroc']:.3f} CI[{lo:.3f},{hi:.3f}]", flush=True)

    if arr is None or int(arr) == 0:
        teA = [a for a in adms if patient_of.get(a) in test_pats]
        trA = [a for a in adms if patient_of.get(a) not in test_pats]
        Xtr = np.array([adm_feat(a) for a in trA]); ytr = np.array([readmit[a] for a in trA])
        Xte = np.array([adm_feat(a) for a in teA]); yte = np.array([readmit[a] for a in teA])
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
        plr = lr.predict_proba(Xte)[:, 1]; lo, hi = bootstrap_ci(yte, plr)
        lrres = {"auroc": float(safe_auc(yte, plr)), "ap": float(average_precision_score(yte, plr)),
                 "ci95": [lo, hi], "base": float(yte.mean()), "n": int(len(yte)), "pos": int(yte.sum())}
        (RES / "cfg_LR.json").write_text(json.dumps({"LR": lrres}, indent=2))
        print(f"  LR AUROC {lrres['auroc']:.3f} AP {lrres['ap']:.3f} base {lrres['base']:.3f}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
