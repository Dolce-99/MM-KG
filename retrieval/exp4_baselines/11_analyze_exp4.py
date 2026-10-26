"""
11_analyze_exp4.py   --   env: umls_eval (numpy)   [CPU]
================================================================================
Experiment 4 tables: external comparisons vs MM-KG with paired, patient-clustered bootstraps
and Holm correction across the prespecified baselines.
  ranking (primary)  : cells in outputs/restricted/exp2/scores/q35/<tag>_evalA.jsonl  (LLM methods)
                       + outputs/restricted/exp4/graphcare_rank_v1/scores_test.jsonl (GraphCare-adapted)
                       + MM_gen parsed rankings (evalB); Hits@1 / MRR, one item per patient
  binary (secondary) : score arrays in outputs/restricted/exp1/scores/q35/<tag>.npy; AUROC / bal-acc,
                       patient-clustered; frozen arm B restricted to the test split
Outputs: outputs/exp4/analysis/{ranking.csv, binary.csv, contrasts.json, report.md}
"""
from __future__ import annotations
import csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, stats

OUT = paths.OUT / "exp4" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
SC2 = paths.RESTRICTED / "exp2" / "scores" / "q35"; SC1 = paths.RESTRICTED / "exp1" / "scores" / "q35"; CTX1 = paths.RESTRICTED / "exp1" / "contexts"
RANK = {"MM-KG (D-R4)": "ranking_v1_d12_test_DR4_none_bnatural", "Patient-only LLM (D-R0)": "ranking_v1_d12_test_DR0_none_bnatural",
        "Flat patient+KG prompt (D-R3)": "ranking_v1_d12_test_DR3_none_bnatural", "Flat patient+KG prompt (D-R1)": "ranking_v1_d12_test_DR1_none_bnatural",
        "MindMap-PrimeKG (score)": "MM_score_test"}
BIN = {"MM-KG (R4)": "R4_test_bnatural", "Flat patient+KG prompt (R3, 8K)": "R3_test_bnatural_rr", "Flat patient+KG prompt (R3, token-matched)": "R3_test_bmatch_r4_rr",
       "MindMap-PrimeKG (score)": "MM_score_test_bnatural", "MindMap-PrimeKG (gen, parsed)": "MM_gen_test_bnatural"}


def ranks_from_scores(rows):
    by = defaultdict(list)
    for r in rows: by[r["idx"]].append(r)
    return {i: next(k for k, r in enumerate(sorted(rs, key=lambda r: (-r["score"], r["cand_name"]))) if r["positive"]) for i, rs in by.items()}


def ranks_from_gen(path, items):
    out = {}
    for r in map(json.loads, open(path)):
        it = items[r["idx"]]; gold = it["gold"]["selected_name"]; cands = r["candidates"]
        m = re.search(r"\{.*?\}", r["text"], re.S); rk = None
        try:
            js = json.loads(m.group(0)) if m else None; rk = js.get("ranking") if js else None
        except Exception: rk = None
        norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
        if isinstance(rk, list) and rk:
            order = [norm(x) for x in rk] + [norm(c) for c in cands if norm(c) not in {norm(x) for x in rk}]
            out[r["idx"]] = order.index(norm(gold)) if norm(gold) in order else len(cands) - 1
        else:
            sel = js.get("selected_drug") if isinstance(js, dict) else None
            out[r["idx"]] = 0 if (sel and norm(sel) == norm(gold)) else 2      # unparsable ranking: selected-only credit, else mid-rank
    return out


def main():
    items = {it["idx"]: it for it in map(json.loads, open(paths.RESTRICTED / "exp2" / "ranking_v1_d12" / "items.jsonl"))}
    R = {}
    for name, tag in RANK.items():
        f = SC2 / f"{tag}_evalA.jsonl"
        if f.exists(): R[name] = ranks_from_scores([json.loads(l) for l in open(f)])
    g = paths.RESTRICTED / "exp4" / "graphcare_rank_v1" / "scores_test.jsonl"
    if g.exists(): R["GraphCare-BAT adapted (5-seed mean)"] = ranks_from_scores([json.loads(l) for l in open(g)])
    gb = SC2 / "MM_gen_test_evalB.jsonl"
    if gb.exists(): R["MindMap-PrimeKG (gen, parsed ranking)"] = ranks_from_gen(gb, items)
    # label-blind shortcut baseline: candidate-identity prior P(positive | drug) estimated on the training bank only
    from collections import Counter
    tr = [json.loads(l) for l in open(paths.RESTRICTED / "exp2" / "ranking_v1_d12_trainbank" / "items.jsonl")]
    pos, seen = Counter(), Counter()
    for it in tr:
        for c in it["candidates"]: seen[c["pkg"]] += 1; pos[c["pkg"]] += c["positive"]
    prior = {p: (pos[p] + 0.5) / (seen[p] + 1) for p in seen}
    R["Candidate-identity prior (train bank)"] = ranks_from_scores([{"idx": it["idx"], "cand_name": c["name"], "positive": c["positive"], "score": prior.get(c["pkg"], 0.1)}
                                                                   for it in items.values() if it["split"] == "test" for c in it["candidates"]])
    hard = {it["idx"] for it in items.values() if it["split"] == "test" and not max(it["candidates"], key=lambda c: prior.get(c["pkg"], 0.1))["positive"]}
    common = sorted(set.intersection(*[set(v) for v in R.values()])); rows = []; con = {}; pv = {}
    ref = "MM-KG (D-R4)"
    for name, rk in R.items():
        r = np.array([rk[i] for i in common]); h1 = (r == 0).astype(float); mrr = 1 / (r + 1)
        ci = stats.boot(np.ones(len(r)), {"h": h1}, lambda yy, c: float(c["h"].mean()), n=2000, require_both_classes=False)["ci"]
        hm = np.array([i in hard for i in common])
        rows.append({"method": name, "n": len(common), "hits@1": round(float(h1.mean()), 4), "hits@1_ci": ci, "hits@3": round(float((r < 3).mean()), 4), "mrr": round(float(mrr.mean()), 4),
                     "hits@1_identity_hard_subset": round(float(h1[hm].mean()), 4), "n_hard": int(hm.sum()), "hits@1_identity_easy_subset": round(float(h1[~hm].mean()), 4)})
        if name != ref and ref in R:
            r0 = np.array([R[ref][i] for i in common]); h0 = (r0 == 0).astype(float)
            d = stats.boot(np.ones(len(r)), {"a": h1, "b": h0}, lambda yy, c: float(c["a"].mean() - c["b"].mean()), n=2000, require_both_classes=False)
            dm = stats.boot(np.ones(len(r)), {"a": mrr, "b": 1 / (r0 + 1)}, lambda yy, c: float(c["a"].mean() - c["b"].mean()), n=2000, require_both_classes=False)
            con[name] = {"d_hits@1_vs_MMKG": {"estimate": round(float(h1.mean() - h0.mean()), 4)} | d, "d_mrr_vs_MMKG": dm}; pv[name] = d["p"]
    holm = stats.holm(pv)
    for k in con: con[k]["p_holm_hits@1"] = holm[k]
    pd.DataFrame(rows).to_csv(OUT / "ranking.csv", index=False)
    # ---- binary ----
    bench = paths.load_benchmark(verify=False); split = paths.load_split(); test = split["test"]
    idxp = pd.read_parquet(paths.FROZEN["icu_index"][0], columns=["hadm_id", "subject_id"]); h2s = dict(zip(idxp.hadm_id.astype(int), idxp.subject_id.astype(int)))
    y = np.array([1 if bench[i]["answer"] == "yes" else 0 for i in test]); clusters = np.array([h2s[int(bench[i]["hadm_id"])] for i in test])
    B = {}
    for name, tag in BIN.items():
        f = SC1 / f"{tag}.npy"
        if f.exists():
            rows_ = [json.loads(l)["idx"] for l in open(CTX1 / tag / "contexts.jsonl")]; s = np.load(f); pos = {i: k for k, i in enumerate(rows_)}
            B[name] = np.array([s[pos[i]] for i in test])
    fb = paths.P3 / "mm" / "cw_widev2_q35_B_mm.npy"
    if fb.exists(): B["Patient-only LLM (frozen arm B)"] = np.load(fb)[test]
    brows = []; bcon = {}; bpv = {}; ref = "MM-KG (R4)"
    for name, s in B.items():
        m = stats.boot_metric(y, s, stats.auroc, n=2000)
        brows.append({"method": name, "n": len(y), "auroc": m["estimate"], "auroc_ci": m["ci"], "bal_acc@0.5": round(stats.balanced_acc(y, s >= 0.5), 4), "ap": round(stats.average_precision(y, s), 4)})
        if name != ref and ref in B:
            d = stats.boot_diff(y, s, B[ref], stats.auroc, n=2000, seed=0); dc = stats.boot_diff(y, s, B[ref], stats.auroc, n=2000, seed=0, clusters=clusters)
            bcon[name] = {"d_auroc_vs_MMKG_item": d, "d_auroc_vs_MMKG_patient_cluster": dc}; bpv[name] = d["p"]
    holm = stats.holm(bpv)
    for k in bcon: bcon[k]["p_holm"] = holm[k]
    pd.DataFrame(brows).to_csv(OUT / "binary.csv", index=False)
    json.dump({"ranking": con, "binary": bcon}, open(OUT / "contrasts.json", "w"), indent=2)
    print(pd.DataFrame(rows).to_string()); print(json.dumps(con, indent=1)); print(pd.DataFrame(brows).to_string()); print(json.dumps(bcon, indent=1))


if __name__ == "__main__":
    main()
