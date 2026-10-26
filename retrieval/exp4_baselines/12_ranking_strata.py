"""
12_ranking_strata.py   --   env: umls_eval   [CPU]
================================================================================
Prespecified fairness analyses on the primary ranking endpoint (d12 test), all methods scored on the same items:
  S1 identity-controlled subset : items whose positive drug also occurs as a decoy in other TEST items
  S2 novel-drug subset          : items whose positive drug is never a positive in GraphCare's training bank
  S3 hybrid with supervision    : 2-feature logistic (method score, identity prior) fitted on the DEV items only,
                                  applied to test -- gives each LLM method the base-rate information GraphCare learned
  S4 support-channel strata     : lab-supported vs report-supported positives
  S5 cost / supervision table   : prompt tokens, generated tokens, GPU seconds per item, labelled examples
Outputs: outputs/exp4/analysis/strata.csv, hybrid.csv, cost.csv (+ printed)
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, stats

OUT = paths.OUT / "exp4" / "analysis"; SC2 = paths.RESTRICTED / "exp2" / "scores" / "q35"
METHODS = {"MM-KG (D-R4)": "ranking_v1_d12_{split}_DR4_none_bnatural", "MindMap-PrimeKG (score)": "MM_score_{split}",
           "Patient-only LLM (D-R0)": "ranking_v1_d12_{split}_DR0_none_bnatural", "Flat patient+KG prompt (D-R3)": "ranking_v1_d12_{split}_DR3_none_bnatural"}


def load_scores(tag):
    d = defaultdict(dict)
    for r in map(json.loads, open(SC2 / f"{tag}_evalA.jsonl")): d[r["idx"]][r["cand_pkg"]] = r["score"]
    return d


def rank_of_pos(it, score):
    order = sorted(it["candidates"], key=lambda c: (-score(c), c["name"])); return next(i for i, c in enumerate(order) if c["positive"])


def metrics(items, score):
    r = np.array([rank_of_pos(it, lambda c, it=it: score(it, c)) for it in items]); h = (r == 0).astype(float)
    ci = stats.boot(np.ones(len(r)), {"h": h}, lambda yy, c: float(c["h"].mean()), n=2000, require_both_classes=False)["ci"]
    return {"n": len(items), "hits@1": round(float(h.mean()), 4), "ci": ci, "mrr": round(float((1 / (r + 1)).mean()), 4)}, h


def main():
    items = {it["idx"]: it for it in map(json.loads, open(paths.RESTRICTED / "exp2" / "ranking_v1_d12" / "items.jsonl"))}
    test = [it for it in items.values() if it["split"] == "test"]; dev = [it for it in items.values() if it["split"] == "dev"]
    tr = [json.loads(l) for l in open(paths.RESTRICTED / "exp2" / "ranking_v1_d12_trainbank" / "items.jsonl")]
    pos, seen = Counter(), Counter()
    for it in tr:
        for c in it["candidates"]: seen[c["pkg"]] += 1; pos[c["pkg"]] += c["positive"]
    prior = {p: (pos[p] + 0.5) / (seen[p] + 1) for p in seen}
    S = {name: {sp: load_scores(tag.format(split=sp)) for sp in ("dev", "test")} for name, tag in METHODS.items()}
    gc = {sp: {} for sp in ("dev", "test")}
    for sp in ("dev", "test"):
        for r in map(json.loads, open(paths.RESTRICTED / "exp4" / "graphcare_rank_v1" / f"scores_{sp}.jsonl")): gc[sp].setdefault(r["idx"], {})[r["cand_pkg"]] = r["score"]
    S["GraphCare-BAT adapted"] = gc
    S["Candidate-identity prior"] = {sp: {it["idx"]: {c["pkg"]: prior.get(c["pkg"], 0.1) for c in it["candidates"]} for it in (dev if sp == "dev" else test)} for sp in ("dev", "test")}
    # ---- subsets ----
    decoy_test = {c["pkg"] for it in test for c in it["candidates"] if not c["positive"]}
    train_pos = {p for p, n in pos.items() if n > 0}
    subsets = {"all": test, "S1 identity-controlled (positive drug also a decoy in test)": [it for it in test if it["gold"]["selected_pkg"] in decoy_test],
               "S2 novel-drug (positive never positive in train bank)": [it for it in test if it["gold"]["selected_pkg"] not in train_pos],
               "S4 lab-supported": [it for it in test if "labs" in it["gold"]["support_channels"]], "S4 report-supported": [it for it in test if "report" in it["gold"]["support_channels"]],
               "identity-hard (prior picks wrong drug)": [it for it in test if not max(it["candidates"], key=lambda c: prior.get(c["pkg"], 0.1))["positive"]]}
    rows = []; ref = "MM-KG (D-R4)"
    for sname, sub in subsets.items():
        hv = {}
        for name in S:
            m, h = metrics(sub, lambda it, c, name=name: S[name]["test"][it["idx"]][c["pkg"]]); hv[name] = h
            rows.append({"subset": sname, "method": name, **m})
        for name in S:
            if name != ref:
                d = stats.boot(np.ones(len(sub)), {"a": hv[name], "b": hv[ref]}, lambda yy, c: float(c["a"].mean() - c["b"].mean()), n=2000, require_both_classes=False)
                rows[-len(S) + list(S).index(name)]["d_vs_MMKG"] = d["mean"]; rows[-len(S) + list(S).index(name)]["d_ci"] = d["ci"]; rows[-len(S) + list(S).index(name)]["p"] = d["p"]
    pd.DataFrame(rows).to_csv(OUT / "strata.csv", index=False)
    # ---- S3 hybrid: logistic on dev (score, log prior) -> test ----
    from numpy.linalg import lstsq
    hyb = []
    def fit_logistic(X, y, iters=500, lr=0.1):
        w = np.zeros(X.shape[1]); b = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(X @ w + b))); g = p - y; w -= lr * X.T @ g / len(y); b -= lr * g.mean()
        return w, b
    for name in ("MM-KG (D-R4)", "MindMap-PrimeKG (score)", "GraphCare-BAT adapted"):
        Xd = np.array([[S[name]["dev"][it["idx"]][c["pkg"]], np.log(prior.get(c["pkg"], 0.1))] for it in dev for c in it["candidates"]]); yd = np.array([c["positive"] for it in dev for c in it["candidates"]], float)
        mu, sd = Xd.mean(0), Xd.std(0) + 1e-9; w, b = fit_logistic((Xd - mu) / sd, yd)
        sc = lambda it, c: float((np.array([S[name]["test"][it["idx"]][c["pkg"]], np.log(prior.get(c["pkg"], 0.1))]) - mu) / sd @ w + b)
        m, h = metrics(test, sc); mh, _ = metrics(subsets["identity-hard (prior picks wrong drug)"], sc)
        hyb.append({"method": f"{name} + identity prior (logistic fitted on dev)", **m, "hits@1_identity_hard": mh["hits@1"], "weights": [round(float(x), 3) for x in w]})
    pd.DataFrame(hyb).to_csv(OUT / "hybrid.csv", index=False)
    # ---- S5 cost ----
    cost = []
    for name, tag, gen in (("MM-KG (D-R4)", "ranking_v1_d12_test_DR4_none_bnatural", False), ("MindMap-PrimeKG (score)", "MM_score_test", False), ("MindMap-PrimeKG (gen)", "MM_gen_test", True)):
        j = json.load(open(paths.OUT / "exp2" / "scores" / f"q35_{tag}_summary.json"))
        e = j.get("evalB" if gen else "evalA", {})
        cost.append({"method": name, "prompt_tokens_per_prompt": e.get("mean_prompt_tokens"), "prompts_per_item": 1 if gen else 5, "generated_tokens": (512 if gen else 1),
                     "gpu_seconds_per_item": round((e.get("s_per_item") or e.get("ms_per_prompt", 0) / 1000 * (1 if gen else 5)), 3), "labelled_examples": 0})
    gr = json.load(open(paths.OUT / "exp4" / "graphcare_rank_v1_result.json"))
    cost.append({"method": "GraphCare-BAT adapted", "prompt_tokens_per_prompt": None, "prompts_per_item": None, "generated_tokens": None, "gpu_seconds_per_item": "training ~1,250 s/seed x 5 seeds + 5 forward passes", "labelled_examples": 19085})
    pd.DataFrame(cost).to_csv(OUT / "cost.csv", index=False)
    print(pd.DataFrame(rows).to_string()); print(pd.DataFrame(hyb).to_string()); print(pd.DataFrame(cost).to_string())


if __name__ == "__main__":
    main()
