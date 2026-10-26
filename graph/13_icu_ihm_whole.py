"""
13_icu_ihm_whole.py   --   env: torch_gpu
================================================================================
WHOLE-COHORT adult-ICU in-hospital-mortality (IHM), leakage-safe (features from
13_icu_ihm_features.py). Same vetted harness as subgraph 09, scaled to all
eligible ICU index stays of the 51,844-patient cohort, with ENRICHED node
features (frozen or fine-tunable pretrained text embeddings; CXR 782-d w/ report).

Per index ICU stay the subgraph is:
  Patient --HAS_ADMISSION--> Admission[admission_type + 48h ICU labs]
  Patient --HAD_FINDING/HAD_PROCEDURE/HAD_DRUG--> prior-visit concepts (history)
  Admission --HAS_CXR--> CXRStudy(early) --IMAGING/REPORTED_FINDING--> Finding --ALIGNS--> KG
  Admission --PRESCRIBED--> Drug(early) --ALIGNS--> KG
No current-admission discharge dx/proc, no LOS.

Ablations: Full / -History / -Imaging / -Report / -KG, + LabsLR.
Env: WG_FINETUNE=1 fine-tune embeddings; WG_IMAGING_ONLY=1 restrict to CXR-present.
One config per SLURM array task (0..4); task 0 writes LabsLR.
"""
from __future__ import annotations
import os
import json, os
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
IHM = HERE / "outputs" / "ihm"
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))

KG_TYPES = {"kg_disease", "kg_effect/phenotype", "kg_drug", "kg_exposure"}
CONCEPT_TYPES = {"Finding", "Drug", "Procedure"} | KG_TYPES
CLINICAL = {"disease", "effect/phenotype", "drug", "exposure"}
PREFIX_NT = {"patient": "Patient", "adm": "Admission", "cxr": "CXRStudy",
             "finding": "Finding", "drug": "Drug", "proc": "Procedure", "dx": "Diagnosis"}
PKG_HOP_CAP = 10
HIDDEN = 32; LAYERS = 2; DROPOUT = 0.5
LR_ = 3e-3; WD = 1e-3; MAX_EPOCHS = 100; PATIENCE = 12; BATCH = 512
TEST_FRAC = 0.20; VAL_FRAC = 0.125; SEEDS = [0, 1, 2, 3, 4]; SPLIT_SEED = 42; N_BOOT = 2000
N_CXR_PROBS = len(IMG.CHEXPERT_FINDINGS); CXR_DIM = N_CXR_PROBS + 768 + 1

FINETUNE = os.environ.get("WG_FINETUNE") == "1"
IMAGING_ONLY = os.environ.get("WG_IMAGING_ONLY") == "1"
FT_LR = 5e-4
SUFFIX = ("_ft" if FINETUNE else "") + ("_img" if IMAGING_ONLY else "")

LAB_NAMES = ["creatinine", "bun", "sodium", "potassium", "bicarbonate", "glucose",
             "wbc", "hemoglobin", "platelet", "lactate", "bilirubin", "inr",
             "aniongap", "albumin", "ph", "po2", "pco2"]
LAB_COLS = [f"lab_{n}_{s}" for n in LAB_NAMES for s in ("min", "max")] + ["n_abnormal", "n_labs"]

CONFIGS = [
    {"name": "Full",     "kg": True,  "img": True,  "report": True,  "hist": True},
    {"name": "-History", "kg": True,  "img": True,  "report": True,  "hist": False},
    {"name": "-Imaging", "kg": True,  "img": False, "report": True,  "hist": True},
    {"name": "-Report",  "kg": True,  "img": True,  "report": False, "hist": True},
    {"name": "-KG",      "kg": False, "img": True,  "report": True,  "hist": True},
    # added for the ICLR paper: complete the channel ablation.
    # -Labs zeroes the admission lab features (dimension kept, architecture
    # unchanged). -Meds drops CURRENT-stay PRESCRIBED edges only; HAD_DRUG
    # prior-medication history is untouched. Note -Meds also severs the
    # drug-mediated KG path for the index stay; since -KG measures ~0, the
    # estimate is effectively the medication channel itself.
    {"name": "-Labs", "kg": True, "img": True, "report": True, "hist": True, "labs": False},
    {"name": "-Meds", "kg": True, "img": True, "report": True, "hist": True, "meds": False},
    # added for Experiment 4 (comparison with GraphCare, which sees coded EHR only):
    # index 7 = coded history + medications + PrimeKG, no labs / imaging / report;
    # index 8 = the same without PrimeKG. Same regime, split and seeds as the rest.
    {"name": "CodesKG",  "kg": True,  "img": False, "report": False, "hist": True, "labs": False},
    {"name": "Codes-KG", "kg": False, "img": False, "report": False, "hist": True, "labs": False},
]
HAD = {"HAD_FINDING", "HAD_PROCEDURE", "HAD_DRUG"}
ALLOW = {"HAS_ADMISSION", "HAS_CXR", "PRESCRIBED", "ALIGNS_TO_CUI"} | HAD


def load_structure():
    nodes = pd.read_parquet(PG / "nodes.parquet")
    edges = pd.read_parquet(PG / "edges.parquet")
    pn = pd.read_csv(PRIMEKG / "nodes.tab", sep="\t", usecols=["node_index", "node_type"])
    pn = pn[pn.node_type.isin(CLINICAL)]
    kg_type = {f"pkg:{i}": f"kg_{t}" for i, t in zip(pn.node_index, pn.node_type)}
    clinical_idx = set(pn.node_index)

    cxr_find = defaultdict(list); find_kg = defaultdict(list); drug_kg = defaultdict(list)
    real_triples = {("Patient", "HAS_ADMISSION", "Admission"),
                    ("Patient", "HAD_FINDING", "Finding"), ("Patient", "HAD_PROCEDURE", "Procedure"),
                    ("Patient", "HAD_DRUG", "Drug"), ("Admission", "HAS_CXR", "CXRStudy"),
                    ("Admission", "PRESCRIBED", "Drug"),
                    ("CXRStudy", "IMAGING_FINDING", "Finding"), ("CXRStudy", "REPORTED_FINDING", "Finding")}
    for et, src, dst in zip(edges.etype.values, edges.src.values, edges.dst.values):
        if et in ("IMAGING_FINDING", "REPORTED_FINDING"):
            cxr_find[src].append((dst, et))
        elif et == "ALIGNS_TO_CUI":
            kt = kg_type.get(dst)
            if src.startswith("drug:"):
                drug_kg[src].append(dst); real_triples.add(("Drug", "ALIGNS_TO_CUI", kt))
            else:
                find_kg[src].append(dst); real_triples.add(("Finding", "ALIGNS_TO_CUI", kt))

    kg = pd.read_csv(PRIMEKG / "kg.csv", usecols=["relation", "x_index", "y_index"])
    kg = kg[(kg.x_index.isin(clinical_idx)) & (kg.y_index.isin(clinical_idx)) & (kg.relation != "drug_drug")]
    kg_adj = defaultdict(list)
    for rel, x, y in zip(kg.relation.values, kg.x_index.values, kg.y_index.values):
        kx, ky = f"pkg:{x}", f"pkg:{y}"
        kg_adj[kx].append((rel, ky)); real_triples.add((kg_type.get(kx), rel, kg_type.get(ky)))

    vocab = {t: {} for t in CONCEPT_TYPES}
    for nt in ("Finding", "Drug", "Procedure"):
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
    return cxr_find, find_kg, drug_kg, kg_adj, vocab, kg_type, cxr_scores, real_triples


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
        if not cfg["kg"] and (su in KG_TYPES or sv in KG_TYPES):
            continue
        if et == "IMAGING_FINDING" and not cfg["img"]:
            continue
        if et == "REPORTED_FINDING" and not cfg["report"]:
            continue
        if et in HAD and not cfg["hist"]:
            continue
        if et == "PRESCRIBED" and not cfg.get("meds", True):
            continue
        if et in ALLOW or et in ("IMAGING_FINDING", "REPORTED_FINDING"):
            rels.add((su, et, sv))
        elif cfg["kg"] and su in KG_TYPES and sv in KG_TYPES:
            rels.add((su, et, sv))
    return sorted(rels)


def node_types_for(cfg):
    nts = {"Patient", "Admission", "CXRStudy", "Finding", "Drug", "Procedure"}
    if cfg["kg"]:
        nts |= KG_TYPES
    return nts


def build_sub(struct, hadm, adm_x, pat_x, cxr_feat_fn, cfg, canon, all_nts,
              early_cxr, early_meds, pdx, pproc, pdrug):
    cxr_find, find_kg, drug_kg, kg_adj, vocab, kg_type, _, _ = struct
    edges = defaultdict(lambda: [[], []])
    concept = {t: {} for t in (CONCEPT_TYPES & all_nts)}
    concept_order = {t: [] for t in concept}
    cxr_feats = []; cxr_index = {}; included_kg = set()

    def add_concept(nt, key):
        if nt not in concept or key not in vocab.get(nt, {}):
            return None
        if key not in concept[nt]:
            concept[nt][key] = vocab[nt][key]; concept_order[nt].append(key)
        return concept_order[nt].index(key)

    def align(src_type, src_i, key, kgmap):
        if not cfg["kg"]:
            return
        for kk in kgmap.get(key, []):
            kt = kg_type.get(kk)
            ki = add_concept(kt, kk)
            if ki is not None:
                included_kg.add(kk)
                edges[src_type, "ALIGNS_TO_CUI", kt][0].append(src_i)
                edges[src_type, "ALIGNS_TO_CUI", kt][1].append(ki)

    edges["Patient", "HAS_ADMISSION", "Admission"][0].append(0)
    edges["Patient", "HAS_ADMISSION", "Admission"][1].append(0)

    if cfg["hist"]:
        for cui in pdx.get(hadm, []):
            fi = add_concept("Finding", f"finding:{cui}")
            if fi is not None:
                edges["Patient", "HAD_FINDING", "Finding"][0].append(0)
                edges["Patient", "HAD_FINDING", "Finding"][1].append(fi)
                align("Finding", fi, f"finding:{cui}", find_kg)
        for pc in pproc.get(hadm, []):
            pi = add_concept("Procedure", f"proc:{pc}")
            if pi is not None:
                edges["Patient", "HAD_PROCEDURE", "Procedure"][0].append(0)
                edges["Patient", "HAD_PROCEDURE", "Procedure"][1].append(pi)
        for name in pdrug.get(hadm, []):
            di = add_concept("Drug", f"drug:{name}")
            if di is not None:
                edges["Patient", "HAD_DRUG", "Drug"][0].append(0)
                edges["Patient", "HAD_DRUG", "Drug"][1].append(di)
                align("Drug", di, f"drug:{name}", drug_kg)

    for study in early_cxr.get(hadm, []):
        ck = f"cxr:{study}"
        if ck not in cxr_index:
            cxr_index[ck] = len(cxr_feats); cxr_feats.append(cxr_feat_fn(ck, cfg))
        cur = cxr_index[ck]
        edges["Admission", "HAS_CXR", "CXRStudy"][0].append(0)
        edges["Admission", "HAS_CXR", "CXRStudy"][1].append(cur)
        for fk, et in cxr_find.get(ck, []):
            if et == "IMAGING_FINDING" and not cfg["img"]:
                continue
            if et == "REPORTED_FINDING" and not cfg["report"]:
                continue
            fi = add_concept("Finding", fk)
            if fi is None:
                continue
            edges["CXRStudy", et, "Finding"][0].append(cur)
            edges["CXRStudy", et, "Finding"][1].append(fi)
            align("Finding", fi, fk, find_kg)

    for name in (early_meds.get(hadm, []) if cfg.get("meds", True) else []):
        dk = f"drug:{name}"
        di = add_concept("Drug", dk)
        if di is None:
            continue
        edges["Admission", "PRESCRIBED", "Drug"][0].append(0)
        edges["Admission", "PRESCRIBED", "Drug"][1].append(di)
        align("Drug", di, dk, drug_kg)

    if cfg["kg"]:
        hop = set()
        for kk in list(included_kg):
            c = 0
            for rel, kk2 in kg_adj.get(kk, []):
                hop.add(kk2); c += 1
                if c >= PKG_HOP_CAP:
                    break
        for kk2 in hop:
            kt = kg_type.get(kk2)
            if add_concept(kt, kk2) is not None:
                included_kg.add(kk2)
        for kk in included_kg:
            kt = kg_type.get(kk)
            if kt not in concept:
                continue
            src_i = concept_order[kt].index(kk)
            for rel, kk2 in kg_adj.get(kk, []):
                if kk2 in included_kg and kg_type.get(kk2) in concept:
                    di = concept_order[kg_type[kk2]].index(kk2)
                    edges[kt, rel, kg_type[kk2]][0].append(src_i)
                    edges[kt, rel, kg_type[kk2]][1].append(di)

    data = HeteroData()
    data["Admission"].x = torch.tensor([adm_x], dtype=torch.float)
    data["Admission"].is_target = torch.tensor([True])
    data["Patient"].x = torch.tensor([pat_x], dtype=torch.float)
    data["CXRStudy"].x = (torch.tensor(cxr_feats, dtype=torch.float) if cxr_feats
                          else torch.zeros((0, CXR_DIM)))
    for nt in (CONCEPT_TYPES & all_nts):
        data[nt].gid = torch.tensor([vocab[nt][k] for k in concept_order[nt]], dtype=torch.long)
        data[nt].num_nodes = len(concept_order[nt])
    for r in canon:
        s, o = edges.get(r, ([], []))
        data[r].edge_index = torch.tensor([s, o], dtype=torch.long) if s else torch.zeros((2, 0), dtype=torch.long)
    return T.ToUndirected(merge=False)(data)


class HGNN(nn.Module):
    def __init__(self, tables, feat_dims, edge_types, adm_dim):
        super().__init__()
        self.emb = nn.ModuleDict({nt.replace("/", "_"): nn.Embedding.from_pretrained(M, freeze=not FINETUNE) for nt, M in tables.items()})
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
    if FINETUNE:
        emb_p = [p for n, p in model.named_parameters() if n.startswith("emb.") and p.requires_grad]
        other = [p for n, p in model.named_parameters() if not n.startswith("emb.")]
        opt = torch.optim.Adam([{"params": other, "lr": LR_}, {"params": emb_p, "lr": FT_LR}], weight_decay=WD)
    else:
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
    print(f"device={device} | FINETUNE={FINETUNE} IMAGING_ONLY={IMAGING_ONLY} suffix='{SUFFIX}'", flush=True)
    struct = load_structure()
    vocab = struct[4]; real_triples = struct[7]
    print("vocab:", {k: len(v) for k, v in vocab.items()}, flush=True)
    tables = build_emb_tables(vocab)

    idx = pd.read_parquet(IHM / "icu_index.parquet")
    early_meds = pd.read_parquet(IHM / "icu_early_meds.parquet").groupby("hadm_id")["drug"].apply(list).to_dict()
    early_cxr = pd.read_parquet(IHM / "icu_early_cxr.parquet").groupby("hadm_id")["study_id"].apply(list).to_dict()
    pdx = pd.read_parquet(IHM / "prior_dx.parquet").groupby("hadm_id")["cui"].apply(list).to_dict()
    pproc = pd.read_parquet(IHM / "prior_proc.parquet").groupby("hadm_id")["proc"].apply(list).to_dict()
    pdrug = pd.read_parquet(IHM / "prior_drug.parquet").groupby("hadm_id")["drug"].apply(list).to_dict()

    if IMAGING_ONLY:
        n0 = len(idx)
        idx = idx[idx.hadm_id.isin(set(early_cxr))].reset_index(drop=True)
        print(f"IMAGING_ONLY: {n0} -> {len(idx)} stays (deaths {int(idx.label.sum())})", flush=True)

    idx["patient"] = "patient:" + idx.subject_id.astype(str)
    pat_pos = idx.groupby("patient")["label"].max()
    P = pat_pos.index.values; yP = pat_pos.values
    tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC, random_state=SPLIT_SEED).split(P, yP))
    test_pats = set(P[te]); trainval = P[tv]; yTV = yP[tv]
    folds = {}
    for s in SEEDS:
        a, b = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=s).split(trainval, yTV))
        folds[s] = (set(trainval[a]), set(trainval[b]))
    print(f"index stays {len(idx)} deaths {int(idx.label.sum())} | patients {len(P)} "
          f"test patients {len(te)} pos-rate {yP[te].mean():.3f}", flush=True)

    at = sorted(idx.admission_type.fillna("UNK").unique()); at_idx = {t: i for i, t in enumerate(at)}
    tvmask = idx.patient.isin(set(trainval))
    labX = idx[LAB_COLS].astype(float)
    med = labX[tvmask].median(); labX = labX.fillna(med)
    mu = labX[tvmask].mean(); sd = labX[tvmask].std().replace(0, 1)
    labZ = ((labX - mu) / sd).clip(-8, 8).values

    def adm_feat(i, row, use_labs=True):
        oh = [0.0] * len(at); oh[at_idx.get(row.admission_type if pd.notna(row.admission_type) else "UNK", 0)] = 1.0
        return oh + (list(labZ[i]) if use_labs else [0.0] * labZ.shape[1])

    def pat_feat(row):
        g = str(row.gender).upper(); return [float(g == "M"), float(g == "F"), float(row.age) / 100.0]

    rep = np.load(EMB / "reports_emb.npz", allow_pickle=True)
    rep_idx = {k: i for i, k in enumerate(rep["keys"])}; rep_vecs = rep["vecs"].astype(np.float32)

    def cxr_feat_fn(ck, cfg):
        probs = struct[6].get(ck, [0.0] * N_CXR_PROBS)
        v = probs if cfg["img"] else [0.0] * N_CXR_PROBS
        j = rep_idx.get(ck)
        return v + (list(rep_vecs[j]) + [1.0] if (cfg["report"] and j is not None) else [0.0] * 769)

    adm_dim = len(at) + labZ.shape[1]
    feat_dims = {"Patient": 3, "Admission": adm_dim, "CXRStudy": CXR_DIM}
    rows = list(idx.itertuples(index=False))
    labels = {r.hadm_id: int(r.label) for r in rows}
    pat_of = {r.hadm_id: r.patient for r in rows}

    arr = os.environ.get("SLURM_ARRAY_TASK_ID")
    configs = [CONFIGS[int(arr)]] if arr is not None else CONFIGS
    RES = IHM / f"results{SUFFIX}"; RES.mkdir(exist_ok=True)

    for cfg in configs:
        all_nts = node_types_for(cfg)
        cfg_tables = {nt: tables[nt] for nt in (CONCEPT_TYPES & all_nts)}
        canon = canon_rels(cfg, real_triples)
        print(f"\n=== build subgraphs: {cfg['name']} (rels {len(canon)}) ===", flush=True)
        sub = {}
        for i, r in enumerate(rows):
            ax = adm_feat(i, r, use_labs=cfg.get("labs", True)); px = pat_feat(r)
            d = build_sub(struct, r.hadm_id, ax, px, cxr_feat_fn, cfg, canon, all_nts,
                          early_cxr, early_meds, pdx, pproc, pdrug)
            d["Admission"].y = torch.tensor([float(labels[r.hadm_id])])
            sub[r.hadm_id] = d
        edge_types = next(iter(sub.values())).edge_types
        test_g = [sub[h] for h in sub if pat_of[h] in test_pats]
        yref = np.array([labels[h] for h in sub if pat_of[h] in test_pats])
        aucs, aps, probs = [], [], []
        for s in SEEDS:
            trp, vap = folds[s]
            tr = [sub[h] for h in sub if pat_of[h] in trp]; va = [sub[h] for h in sub if pat_of[h] in vap]
            v, yte, pte = train_once(tr, va, test_g, cfg_tables, feat_dims, edge_types, adm_dim, s, device)
            aucs.append(safe_auc(yte, pte)); aps.append(average_precision_score(yte, pte)); probs.append(pte)
            print(f"  {cfg['name']} seed {s}: val {v:.3f} test AUROC {aucs[-1]:.3f}", flush=True)
        ens = np.mean(np.stack(probs), 0); lo, hi = bootstrap_ci(yref, ens)
        # persist per-patient ensemble predictions so paired significance tests
        # between configs (the same design as the LLM ablation) can be computed
        # post hoc; hadm order is saved alongside for alignment verification.
        test_hadms = np.array([h for h in sub if pat_of[h] in test_pats])
        np.save(RES / f"preds_{cfg['name']}.npy", ens)
        np.save(RES / f"ytest_{cfg['name']}.npy", np.asarray(yref))
        np.save(RES / f"hadms_{cfg['name']}.npy", test_hadms)
        res = {"auroc_mean": float(np.nanmean(aucs)), "auroc_std": float(np.nanstd(aucs)),
               "ap_mean": float(np.mean(aps)), "ensemble_auroc": float(safe_auc(yref, ens)),
               "ci95": [lo, hi], "n_test": int(len(yref)), "test_pos": int(yref.sum())}
        (RES / f"cfg_{cfg['name']}.json").write_text(json.dumps({cfg["name"]: res}, indent=2))
        print(f"  => {cfg['name']}: {res['auroc_mean']:.3f}+-{res['auroc_std']:.3f} ens {res['ensemble_auroc']:.3f} CI[{lo:.3f},{hi:.3f}]", flush=True)

    if arr is None or int(arr) == 0:
        oh = np.zeros((len(rows), len(at)))
        for i, r in enumerate(rows):
            oh[i, at_idx.get(r.admission_type if pd.notna(r.admission_type) else "UNK", 0)] = 1.0
        sex = np.array([[1.0 if str(r.gender).upper() == "M" else 0.0, float(r.age) / 100.0] for r in rows])
        Xall = np.hstack([sex, oh, labZ]); yall = np.array([labels[r.hadm_id] for r in rows])
        pmask = np.array([pat_of[r.hadm_id] in test_pats for r in rows])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xall[~pmask], yall[~pmask])
        plr = lr.predict_proba(Xall[pmask])[:, 1]; ylr = yall[pmask]
        lo, hi = bootstrap_ci(ylr, plr)
        lrres = {"auroc": float(safe_auc(ylr, plr)), "ap": float(average_precision_score(ylr, plr)),
                 "ci95": [lo, hi], "base": float(ylr.mean()), "n": int(pmask.sum()), "pos": int(ylr.sum())}
        (RES / "cfg_LabsLR.json").write_text(json.dumps({"LabsLR": lrres}, indent=2))
        print(f"  LabsLR AUROC {lrres['auroc']:.3f} AP {lrres['ap']:.3f} base {lrres['base']:.3f}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
