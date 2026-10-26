"""
09_graphcare_rank_train.py   --   env: torch_gpu   [GPU, ~1 h for 5 seeds]
================================================================================
Trains GraphCare-BAT (authors' model, defaults) with a binary head on the per-pair graphs of
08_graphcare_rank_data.py: train bank -> fit; dev (421 items) -> epoch and learning-rate
selection (Hits@1); test (1,244 items) scored once per seed; five seeds averaged.
Ranking = candidates of an item ordered by predicted probability.  Also stores per-pair
attention summaries for the qualitative comparison (method-native attribution outputs).
  python 09_graphcare_rank_train.py [--lrs 1e-3 5e-4]
Outputs: outputs/restricted/exp4/graphcare_rank_v1/scores_{dev,test}.jsonl (idx, cand_pkg, cand_name, positive, score)
         outputs/exp4/graphcare_rank_v1_result.json
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths, runlog, stats
from graphcare_model import GraphCare
import importlib.util
_s = importlib.util.spec_from_file_location("gct", Path(__file__).resolve().parent / "04_graphcare_train.py"); gct = importlib.util.module_from_spec(_s); _s.loader.exec_module(gct)

D = paths.RESTRICTED / "exp4" / "graphcare_rank_v1"; PUB = paths.OUT / "exp4"
HID = 128; LAYERS = 3; DROPOUT = 0.5; DECAY = 0.01; WD = 1e-5; BATCH = 64; MAX_EPOCHS = 60; PATIENCE = 8


def hits(rows):
    by = defaultdict(list)
    for r in rows: by[(r["version"], r["idx"])].append(r)
    h1 = mrr = 0.0
    for rs in by.values():
        rs = sorted(rs, key=lambda r: (-r["score"], r["cand_name"])); pos = next(i for i, r in enumerate(rs) if r["y"]); h1 += pos == 0; mrr += 1 / (pos + 1)
    return h1 / len(by), mrr / len(by)


def predict(model, gs, K, dev):
    out = []
    for i in range(0, len(gs), BATCH):
        chunk = gs[i:i + BATCH]; b = gct.collate(chunk, K, dev)
        with torch.no_grad():
            p = torch.sigmoid(model(b["node_ids"], b["rel_ids"], b["edge_index"], b["batch"], b["visit_node"], b["ehr_nodes"]).squeeze(-1)).cpu().numpy()
        out += [{"version": g["version"], "idx": g["idx"], "cand_pkg": g["cand"], "cand_name": g["cand_name"], "y": g["y"], "score": float(s)} for g, s in zip(chunk, p)]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 5e-4]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--variant", default="full", choices=["full", "nokg", "candidate_only"],
                    help="nokg: no PrimeKG edges/nodes (EHR + candidate nodes only); candidate_only: patient side removed")
    a = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pickle.load(open(D / "pairs.pkl", "rb")); G = data["graphs"]; K = data["num_nodes"]; cent = torch.tensor(data["centroids"])
    tr = [g for g in G if g["split"] == "train"]; dv = [g for g in G if g["split"] == "dev"]; te = [g for g in G if g["split"] == "test"]
    if a.variant != "full":
        for g in G:
            if a.variant == "nokg":
                g["nodes"] = list(g["ehr"]); g["edges"] = []; g["visit_nodes"] = [sorted(set(v) & set(g["ehr"])) for v in g["visit_nodes"]]
            else:                                   # candidate_only: the candidate node and its own neighbourhood, nothing from the patient
                c = g["cand_node"]; e = [x for x in g["edges"] if x[0] == c]; keep = {c} | {x[2] for x in e}
                e2 = [x for x in g["edges"] if x[0] in keep and x[2] in keep]
                g["nodes"] = sorted(keep); g["edges"] = e2; g["ehr"] = [c] if c >= 0 else []; g["visit_nodes"] = [[], sorted(keep)]
    global MAX_EPOCHS
    if a.smoke:
        tr, dv, te = tr[: a.smoke], dv[: a.smoke], te[: a.smoke]; MAX_EPOCHS = 2
    sfx = "" if a.variant == "full" else f"_{a.variant}"
    run = runlog.Run("exp4", f"graphcare_rank_train{sfx}", vars(a) | {"hidden": HID, "layers": LAYERS, "batch": BATCH, "n_train_pairs": len(tr), "n_dev_pairs": len(dv), "n_test_pairs": len(te)})
    # learning rate chosen on dev with seed 0 only (prespecified), then fixed for all seeds
    best_lr, best_h = None, -1
    def fit(seed, lr):
        torch.manual_seed(seed); np.random.seed(seed)
        m = GraphCare(num_nodes=K, num_rels=len(data["rels"]), max_visit=2, embedding_dim=cent.shape[1], hidden_dim=HID, out_channels=1, layers=LAYERS,
                      dropout=DROPOUT, decay_rate=DECAY, node_emb=cent, freeze=False, patient_mode="joint").to(dev); m.lambda_j = m.lambda_j.to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=WD); best = (-1, None, 0); bad = 0
        for ep in range(MAX_EPOCHS):
            gct.run_epoch(m, tr, K, dev, opt); m.eval(); h1, mr = hits(predict(m, dv, K, dev)); m.train()
            if h1 > best[0]: best = (h1, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep); bad = 0
            else: bad += 1
            if bad >= PATIENCE: break
        m.load_state_dict(best[1]); m.eval(); return m, best
    for lr in a.lrs:
        m, best = fit(a.seeds[0], lr); print(f"lr {lr}: dev Hits@1 {best[0]:.4f} @ep{best[2]}", flush=True)
        if best[0] > best_h: best_h, best_lr = best[0], lr
    dev_rows, test_rows = defaultdict(list), defaultdict(list); per_seed = []
    for sd in a.seeds:
        t0 = time.time(); m, best = fit(sd, best_lr)
        pd_, pt = predict(m, dv, K, dev), predict(m, te, K, dev); h1, mr = hits(pt); per_seed.append({"seed": sd, "dev_hits1": best[0], "epoch": best[2], "test_hits1": h1, "test_mrr": mr})
        print(f"seed {sd}: dev {best[0]:.4f} | test Hits@1 {h1:.4f} MRR {mr:.4f} | {time.time()-t0:.0f}s", flush=True)
        for r in pd_: dev_rows[(r["idx"], r["cand_pkg"])].append(r)
        for r in pt: test_rows[(r["idx"], r["cand_pkg"])].append(r)
    for name, rows in (("dev", dev_rows), ("test", test_rows)):
        with open(D / f"scores_{name}{sfx}.jsonl", "w") as f:
            for rs in rows.values():
                r = dict(rs[0]); r["score"] = float(np.mean([x["score"] for x in rs])); r["positive"] = bool(r.pop("y")); f.write(json.dumps(r) + "\n")
    ens = [json.loads(l) for l in open(D / f"scores_test{sfx}.jsonl")]
    for r in ens: r["y"] = int(r["positive"])
    h1, mr = hits(ens)
    res = {"lr": best_lr, "per_seed": per_seed, "test_hits1_mean": float(np.mean([s["test_hits1"] for s in per_seed])), "test_hits1_std": float(np.std([s["test_hits1"] for s in per_seed])),
           "ensemble_test_hits1": h1, "ensemble_test_mrr": mr, "n_test_items": len({r["idx"] for r in ens})}
    json.dump(res, open(PUB / f"graphcare_rank_v1_result{sfx}.json", "w"), indent=2); run.finish(metrics=res); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
