"""
06_analyze.py   --   env: umls_eval (CPU)
================================================================================
Aggregates Experiment 1: retrieval metrics (from the context stats) and end-task
metrics (from the score files) per policy/budget, with paired bootstrap against
the reference policy (R4, the frozen MM-KG prompt), a patient-clustered
sensitivity bootstrap, Holm adjustment across the family of contrasts, the
efficiency table, budget curves, and a failure analysis separating retrieval
failure (gold path absent) from reasoning failure (gold present, wrong answer).

  python 06_analyze.py --model q35 --split test --tags R0_test_bnatural R1_test_b8192 ... --ref R4_test_bnatural \\
                       --threshold_from dev            # thresholds selected on the dev-split scores of the same tag family

Outputs (outputs/exp1/analysis/<model>_<split>/):  end_task.csv, retrieval.csv, efficiency.csv,
  contrasts.json, failure_analysis.csv, budget_curve.csv, report.md   -- aggregate only
Item-level merged table -> outputs/restricted/exp1/analysis/<model>_<split>_items.csv (RESTRICTED)
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, stats, runlog

CTX = paths.RESTRICTED / "exp1" / "contexts"
SC = paths.RESTRICTED / "exp1" / "scores"
FAMS = ("contra_labs", "contra_report", "drug_effect", "mechanism")


def load_tag(model, tag):
    rows = [json.loads(l) for l in open(CTX / tag / "contexts.jsonl")]
    sc = np.load(SC / model / f"{tag}.npy")
    val = [json.loads(l)["valid"] for l in open(SC / model / f"{tag}.jsonl")]
    assert len(sc) == len(rows)
    idx = np.array([r["idx"] for r in rows]); y = np.array([1 if r["stats"]["answer"] == "yes" else 0 for r in rows])
    return {"idx": idx, "y": y, "s": sc, "valid": np.array(val), "stats": [r["stats"] for r in rows]}


def dev_tag(tag: str) -> str:
    return tag.replace("_test_", "_dev_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35"); ap.add_argument("--split", default="test")
    ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--threshold_from", default="dev", choices=["dev", "fixed0.5"])
    ap.add_argument("--n_boot", type=int, default=2000); ap.add_argument("--subset", default=None, help="e.g. drug_balanced")
    args = ap.parse_args()
    run = runlog.Run("exp1", f"analyze_{args.model}_{args.split}", vars(args))
    out = paths.OUT / "exp1" / "analysis" / (f"{args.model}_{args.split}" + ("" if not args.subset else f"_{args.subset}"))
    out.mkdir(parents=True, exist_ok=True)
    (paths.RESTRICTED / "exp1" / "analysis").mkdir(parents=True, exist_ok=True)
    bench = paths.load_benchmark(verify=False)
    import pandas as pd
    idxp = pd.read_parquet(paths.FROZEN["icu_index"][0], columns=["hadm_id", "subject_id"])
    h2s = dict(zip(idxp.hadm_id.astype(int), idxp.subject_id.astype(int)))
    keep = None
    if args.subset == "drug_balanced":
        keep = set(paths.load_subset()["item_index"])

    data = {t: load_tag(args.model, t) for t in args.tags + [args.ref]}
    ref = data[args.ref]
    common = ref["idx"]
    for t, d in data.items():
        assert np.array_equal(d["idx"], common), f"{t} item order differs from {args.ref}"
    mask = np.ones(len(common), bool) if keep is None else np.array([i in keep for i in common])
    y = ref["y"][mask]; clusters = np.array([h2s[int(bench[i]["hadm_id"])] for i in common])[mask]
    fam = np.array([bench[i]["family"] for i in common])[mask]

    # thresholds: selected on the dev-split scores of the same policy/budget, never on the protected split
    thr_of = {}
    for t, d in data.items():
        if args.threshold_from == "dev" and args.split == "test":
            dd = load_tag(args.model, dev_tag(t)); thr_of[t] = stats.select_threshold(dd["y"], dd["s"])
        else:
            thr_of[t] = 0.5
    end_rows, ret_rows, eff_rows, contrasts, fail_rows = [], [], [], {}, []
    pv = {}
    for t, d in data.items():
        s = d["s"][mask]; valid = d["valid"][mask]; thr = thr_of[t]
        at = stats.at_threshold(y, s, thr)
        e = {"tag": t, "n": int(mask.sum()), "auroc": stats.boot_metric(y, s, stats.auroc, n=args.n_boot)["estimate"],
             "auroc_ci": stats.boot_metric(y, s, stats.auroc, n=args.n_boot)["ci"],
             "ap": round(stats.average_precision(y, s), 4), "threshold": round(thr, 4),
             "bal_acc": round(at["balanced_accuracy"], 4), "sensitivity": round(at["sensitivity"], 4),
             "specificity": round(at["specificity"], 4), "invalid_rate": round(1 - float(valid.mean()), 4)}
        for f in FAMS:
            m = fam == f
            e[f"auroc_{f}"] = round(stats.auroc(y[m], s[m]), 4) if m.any() else None
        end_rows.append(e)
        if t != args.ref:
            d_item = stats.boot_diff(y, s, ref["s"][mask], stats.auroc, n=args.n_boot, seed=0)
            d_clu = stats.boot_diff(y, s, ref["s"][mask], stats.auroc, n=args.n_boot, seed=0, clusters=clusters)
            d_ba = stats.boot_diff(y, (s >= thr).astype(int), (ref["s"][mask] >= thr_of[args.ref]).astype(int),
                                   stats.balanced_acc, n=args.n_boot, seed=0)
            contrasts[t] = {"vs": args.ref, "d_auroc_item": d_item, "d_auroc_patient_cluster": d_clu, "d_balacc_at_thr": d_ba}
            pv[t] = d_item["p"]
        st = [x for x, k in zip(d["stats"], mask) if k]
        pos = [x for x in st if x["answer"] == "yes"]
        r = {"tag": t, "recall_patient": _mean(pos, "recall_patient"), "recall_kg_edge": _mean(pos, "recall_kg_edge"),
             "recall_path": _mean(pos, "recall_path"), "precision": _mean(pos, "precision"),
             "n_patient_units": _mean(st, "n_patient_units"), "n_kg_nodes": _mean(st, "n_kg_nodes"),
             "n_kg_edges": _mean(st, "n_kg_edges"), "prompt_tokens": _mean(st, "prompt_tokens"),
             "untruncated_tokens": _mean(st, "untruncated_tokens"), "frac_truncated": _mean(st, "truncated"),
             "retrieval_ms": _mean(st, "retrieval_ms")}
        for f in FAMS:
            r[f"recall_path_{f}"] = _mean([x for x in pos if x["family"] == f], "recall_path")
        ret_rows.append(r)
        summ = json.load(open(paths.OUT / "exp1" / "scores" / f"{args.model}_{t}_summary.json"))
        eff_rows.append({"tag": t, "prompt_tokens": r["prompt_tokens"], "untruncated_tokens": r["untruncated_tokens"],
                         "frac_truncated": r["frac_truncated"], "retrieval_ms": r["retrieval_ms"],
                         "llm_ms_per_item": summ.get("ms_per_item"), "end_to_end_ms": round((r["retrieval_ms"] or 0) + (summ.get("ms_per_item") or 0), 1)})
        # failure analysis on positives: retrieval failure vs reasoning failure
        for f in FAMS + ("all",):
            sub = [(x, sc) for x, sc in zip(st, s) if x["answer"] == "yes" and (f == "all" or x["family"] == f)]
            if not sub:
                continue
            n = len(sub); ret_fail = sum(1 for x, _ in sub if not x["recall_path"])
            reas_fail = sum(1 for x, sc in sub if x["recall_path"] and sc < thr)
            fail_rows.append({"tag": t, "family": f, "n_pos": n, "retrieval_failure": ret_fail, "reasoning_failure": reas_fail,
                              "correct": n - ret_fail - reas_fail, "acc_given_path": round((n - ret_fail - reas_fail) / max(1, n - ret_fail), 4)})
    holm = stats.holm(pv) if pv else {}
    for t in contrasts:
        contrasts[t]["p_holm_item"] = holm[t]
    _csv(out / "end_task.csv", end_rows); _csv(out / "retrieval.csv", ret_rows); _csv(out / "efficiency.csv", eff_rows)
    _csv(out / "failure_analysis.csv", fail_rows)
    json.dump(contrasts, open(out / "contrasts.json", "w"), indent=2)
    # item-level merged table (restricted)
    with open(paths.RESTRICTED / "exp1" / "analysis" / f"{args.model}_{args.split}_items.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "family", "answer"] + [f"score_{t}" for t in data] + [f"path_{t}" for t in data])
        for k, i in enumerate(common[mask]):
            w.writerow([i, fam[k], y[k]] + [round(float(data[t]["s"][mask][k]), 4) for t in data] + [data[t]["stats"][int(np.flatnonzero(common == i)[0])]["recall_path"] for t in data])
    for p in out.glob("*"):
        run.add_output(p)
    run.finish(metrics={e["tag"]: {"auroc": e["auroc"], "bal_acc": e["bal_acc"]} for e in end_rows})
    print(json.dumps({"end_task": end_rows, "retrieval": ret_rows, "contrasts": contrasts}, indent=1))


def _mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(v)), 4) if v else None


def _csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
