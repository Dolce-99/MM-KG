"""
04_graphcare_train.py   --   env: torch_gpu   [GPU; ~1-2 h for 5 seeds]
================================================================================
Trains the reference GraphCare model (BAT: bi-attention augmented GNN; graphcare_model.py is the
authors' model.py with the PyHealth import removed and an optional side-feature head) on the
graphs of 03_graphcare_data.py with the authors' default hyper-parameters (hidden 128, 3 layers,
dropout 0.5, decay 0.01, lr 1e-3, wd 1e-5, batch 64, alpha/beta/edge attention on, joint patient
mode), early stopping on validation AUROC (patience 12, max 100 epochs, like the MM-KG GNN),
5 seeds ensembled by probability averaging.  Test predictions are saved aligned to the MM-KG
GNN's test stays so that the comparison is stay-paired.

  python 04_graphcare_train.py --variant codes      # GraphCare as published (coded EHR + KG)
  python 04_graphcare_train.py --variant codes_labs # + admission lab features in the head
  python 04_graphcare_train.py --variant codes_nokg # ablation: KG expansion edges removed
Outputs: outputs/restricted/exp4/graphcare_v1/preds_<variant>.npy, hadms_<variant>.npy, ytest_<variant>.npy
         outputs/exp4/graphcare_<variant>.json (aggregate)
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths, runlog, stats
from graphcare_model import GraphCare

D = paths.RESTRICTED / "exp4" / "graphcare_v1"; PUB = paths.OUT / "exp4"
HID = 128; LAYERS = 3; DROPOUT = 0.5; DECAY = 0.01; LR = 1e-3; WD = 1e-5; BATCH = 64; MAX_EPOCHS = 100; PATIENCE = 12


def collate(gs, K, dev, nokg=False, labs=None):
    node_ids, batch, ei, rel, off = [], [], [], [], 0
    vis = torch.zeros(len(gs), 2, K); ehr = torch.zeros(len(gs), K)
    for i, g in enumerate(gs):
        nodes = g["ehr"] if nokg else g["nodes"]
        if not nodes:                          # a stay with no coded history and no medications: one placeholder node keeps the batch aligned
            nodes = [0]
        pos = {n: j for j, n in enumerate(nodes)}
        node_ids += nodes; batch += [i] * len(nodes)
        if not nokg:
            for a, r, b in g["edges"]:
                if a in pos and b in pos:
                    ei.append((pos[a] + off, pos[b] + off)); rel.append(r)
        for v, vn in enumerate(g["visit_nodes"]):
            vis[i, v, [n for n in vn if n in pos]] = 1.0
        ehr[i, g["ehr"] if g["ehr"] else [0]] = 1.0; off += len(nodes)   # empty EHR set -> placeholder, avoids 0/0 in the node readout
    if not ei:
        ei = [(0, 0)]; rel = [0]
    out = {"node_ids": torch.tensor(node_ids), "batch": torch.tensor(batch), "edge_index": torch.tensor(ei).T.contiguous(),
           "rel_ids": torch.tensor(rel), "visit_node": vis, "ehr_nodes": ehr, "y": torch.tensor([g["y"] for g in gs], dtype=torch.float32)}
    if labs is not None:
        out["side"] = torch.tensor(np.stack([labs[g["hadm"]] for g in gs]))
    return {k: v.to(dev) for k, v in out.items()}


def run_epoch(model, gs, K, dev, opt=None, nokg=False, labs=None, bs=BATCH):
    ys, ps = [], []; model.train(opt is not None)
    order = np.random.permutation(len(gs)) if opt is not None else np.arange(len(gs))
    for i in range(0, len(gs), bs):
        b = collate([gs[j] for j in order[i:i + bs]], K, dev, nokg, labs)
        with torch.set_grad_enabled(opt is not None):
            logit = model(b["node_ids"], b["rel_ids"], b["edge_index"], b["batch"], b["visit_node"], b["ehr_nodes"], side=b.get("side")).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, b["y"])
            if opt is not None:
                opt.zero_grad(); loss.backward(); opt.step()
        ys.append(b["y"].cpu().numpy()); ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="codes", choices=["codes", "codes_labs", "codes_nokg"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", type=int, default=0, help="use N stays and 2 epochs (code check only)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pickle.load(open(D / "graphs.pkl", "rb")); split = json.load(open(D / "split.json"))
    graphs = data["graphs"]; K = data["num_nodes"]; cent = torch.tensor(data["centroids"])
    global MAX_EPOCHS
    if a.smoke:
        graphs = graphs[: a.smoke]; MAX_EPOCHS = 2
    labs = data["labs"] if a.variant == "codes_labs" else None; nokg = a.variant == "codes_nokg"
    run = runlog.Run("exp4", f"graphcare_{a.variant}", vars(a) | {"hidden": HID, "layers": LAYERS, "dropout": DROPOUT, "decay": DECAY, "lr": LR, "wd": WD, "batch": BATCH})
    test_p = set(split["test_patients"]); te = [g for g in graphs if f"patient:{g['subject']}" in test_p]
    te = sorted(te, key=lambda g: g["hadm"]); yte = np.array([g["y"] for g in te]); hadms = np.array([g["hadm"] for g in te])
    probs, aucs, aps = [], [], []
    for sd in a.seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        trp = set(split["folds"][str(sd)]["train"]); vap = set(split["folds"][str(sd)]["val"])
        tr = [g for g in graphs if f"patient:{g['subject']}" in trp]; va = [g for g in graphs if f"patient:{g['subject']}" in vap]
        model = GraphCare(num_nodes=K, num_rels=len(data["rels"]), max_visit=2, embedding_dim=cent.shape[1], hidden_dim=HID, out_channels=1, layers=LAYERS,
                          dropout=DROPOUT, decay_rate=DECAY, node_emb=cent, freeze=False, patient_mode="joint", side_dim=(36 if labs is not None else 0)).to(dev)
        model.lambda_j = model.lambda_j.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        best = (-1, None, 0); bad = 0; t0 = time.time()
        for ep in range(MAX_EPOCHS):
            run_epoch(model, tr, K, dev, opt, nokg, labs)
            yv, pv = run_epoch(model, va, K, dev, None, nokg, labs); auc = roc_auc_score(yv, pv)
            if auc > best[0]:
                best = (auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep); bad = 0
            else:
                bad += 1
            if bad >= PATIENCE: break
        model.load_state_dict(best[1]); yt, pt = run_epoch(model, te, K, dev, None, nokg, labs)
        aucs.append(roc_auc_score(yt, pt)); aps.append(average_precision_score(yt, pt)); probs.append(pt)
        print(f"seed {sd}: best val AUROC {best[0]:.4f} @ep{best[2]} | test AUROC {aucs[-1]:.4f} AP {aps[-1]:.4f} | {time.time()-t0:.0f}s", flush=True)
    ens = np.mean(probs, axis=0)
    np.save(D / f"preds_{a.variant}.npy", ens); np.save(D / f"hadms_{a.variant}.npy", hadms); np.save(D / f"ytest_{a.variant}.npy", yte)
    res = {"variant": a.variant, "n_test": int(len(yte)), "deaths": int(yte.sum()), "auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)),
           "ap_mean": float(np.mean(aps)), "ensemble_auroc": float(roc_auc_score(yte, ens)), "ensemble_ap": float(average_precision_score(yte, ens)),
           "ensemble_auroc_ci": stats.boot_metric(yte, ens, stats.auroc, n=2000)["ci"]}
    json.dump(res, open(PUB / f"graphcare_{a.variant}.json", "w"), indent=2); run.finish(metrics=res); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
