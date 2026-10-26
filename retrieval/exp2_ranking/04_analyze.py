"""
04_analyze.py   --   env: umls_eval (CPU)
================================================================================
Experiment 2 analysis.
  eval A  Hits@1/3, MRR, NDCG@5, candidate AUROC/AP per cell; paired item bootstrap of
          Hits@1 and MRR against a reference cell (default D R4 none); Holm adjustment
  eval B  valid-JSON rate, selected-drug accuracy, evidence-citation precision/recall
          (tags -> unit ids vs gold), condition accuracy (normalised string match to a
          gold condition name), KG-edge validity (each cited triple exists in PrimeKG for
          that drug), path exact match / precision / recall vs gold edges, hallucinated
          node and edge rates, selected-drug accuracy conditional on a valid path
No LLM judge is used anywhere.

  python 04_analyze.py --version ranking_v1_d12 --model q35 --split dev --tags <tag...> --ref <tag>
Outputs  outputs/exp2/analysis/<model>_<split>/{evalA.csv, evalB.csv, contrasts.json}   aggregate only
"""
from __future__ import annotations
import argparse, csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, stats, runlog, frozen as fz

SC = paths.RESTRICTED / "exp2" / "scores"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def per_item_ranks(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["idx"]].append(r)
    out = {}
    for idx, rs in by.items():
        rs = sorted(rs, key=lambda r: (-r["score"], r["cand_name"]))
        out[idx] = next(i for i, r in enumerate(rs) if r["positive"])
    return out


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="ranking_v1_d12"); ap.add_argument("--model", default="q35"); ap.add_argument("--split", default="dev")
    ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--ref", default=None); ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    run = runlog.Run("exp2", f"analyze_{args.model}_{args.split}", vars(args))
    out = paths.OUT / "exp2" / "analysis" / f"{args.model}_{args.split}"; out.mkdir(parents=True, exist_ok=True)
    items = {it["idx"]: it for it in map(json.loads, open(paths.RESTRICTED / "exp2" / args.version / "items.jsonl"))}
    rel = fz.load_rel_cache(verify=False); name = rel["name"]
    valid_edges = defaultdict(set)                      # drug name (norm) -> {(relation, condition norm)}
    for p, s in rel["contra"].items():
        for c in s: valid_edges[norm(name.get(p, p))].add(("contraindication", norm(name.get(c, c))))
    for p, s in rel["effect"].items():
        for c in s: valid_edges[norm(name.get(p, p))].add(("drug_effect", norm(name.get(c, c))))
    all_names = {norm(v) for v in name.values()}

    A_rows, B_rows, contrasts, pv = [], [], {}, {}
    ranks_of = {}
    for tag in args.tags + ([args.ref] if args.ref and args.ref not in args.tags else []):
        fa = SC / args.model / f"{tag}_evalA.jsonl"
        if fa.exists():
            rows = [json.loads(l) for l in open(fa)]
            ranks = per_item_ranks(rows); ranks_of[tag] = ranks
            idxs = sorted(ranks); r = np.array([ranks[i] for i in idxs])
            y = np.array([int(x["positive"]) for x in rows]); s = np.array([x["score"] for x in rows])
            A_rows.append({"tag": tag, "n": len(idxs), "hits@1": round(float((r == 0).mean()), 4), "hits@3": round(float((r < 3).mean()), 4),
                           "mrr": round(float((1 / (r + 1)).mean()), 4), "ndcg@5": round(float((1 / np.log2(r + 2)).mean()), 4),
                           "cand_auroc": round(stats.auroc(y, s), 4), "cand_ap": round(stats.average_precision(y, s), 4),
                           "hits@1_ci": stats.boot(np.ones(len(r)), {"h": (r == 0).astype(float)}, lambda yy, c: float(c["h"].mean()), n=args.n_boot, require_both_classes=False)["ci"]})
        fb = SC / args.model / f"{tag}_evalB.jsonl"
        if fb.exists():
            rows = [json.loads(l) for l in open(fb)]
            m = defaultdict(list)
            for x in rows:
                it = items[x["idx"]]; g = it["gold"]; js = parse_json(x["text"])
                m["json_valid"].append(js is not None)
                if js is None:
                    for k in ("selected_ok", "cond_ok", "path_exact", "path_valid"): m[k].append(False)
                    continue
                sel = norm(js.get("selected_drug", "")); sel_ok = sel == norm(g["selected_name"]); m["selected_ok"].append(sel_ok)
                cited = [str(t).strip("[] ").upper() for t in (js.get("patient_evidence_ids") or [])]
                cited_uids = {x["tagmap"].get(t) for t in cited if x["tagmap"].get(t)}
                gold_uids = set(g["patient_uids"])
                if cited:
                    m["evid_precision"].append(len(cited_uids & gold_uids) / len(cited))
                m["evid_recall"].append(len(cited_uids & gold_uids) / max(1, len(gold_uids)))
                m["evid_unknown_tag_rate"].append(sum(1 for t in cited if t not in x["tagmap"]) / max(1, len(cited)))
                cond = norm(js.get("normalized_condition", "")); gold_conds = {norm(c) for c in g["condition_names"]}
                m["cond_ok"].append(cond in gold_conds)
                path = js.get("kg_path") or []
                trip = []
                for e in path:
                    if isinstance(e, (list, tuple)) and len(e) == 3:
                        trip.append((norm(e[0]), norm(e[1]).replace("adverse effect", "drug_effect").replace("has adverse effect", "drug_effect").replace("contraindicated in", "contraindication"), norm(e[2])))
                gold_e = {(norm(g["selected_name"]), e[1], norm(name.get(e[2], str(e[2])))) for e in g["kg_edges"]}
                valid = [t for t in trip if (t[1], t[2]) in valid_edges.get(t[0], set())]
                m["n_path_edges"].append(len(trip))
                if trip:
                    m["edge_validity"].append(len(valid) / len(trip))
                    m["hallucinated_edge_rate"].append(1 - len(valid) / len(trip))
                    nodes = {t[0] for t in trip} | {t[2] for t in trip}
                    m["hallucinated_node_rate"].append(sum(1 for nd in nodes if nd not in all_names) / len(nodes))
                    m["path_precision"].append(len(set(trip) & gold_e) / len(trip))
                m["path_recall"].append(len(set(trip) & gold_e) / max(1, len(gold_e)))
                m["path_exact"].append(bool(trip) and set(trip) == gold_e)
                m["path_valid"].append(bool(trip) and len(valid) == len(trip))
                m["selected_ok_given_valid_path"].append(sel_ok if (trip and len(valid) == len(trip)) else None)
            B_rows.append({"tag": tag, "n": len(rows)} | {k: round(float(np.mean([v for v in vs if v is not None])), 4) if any(v is not None for v in vs) else None for k, vs in m.items()})
    if args.ref and args.ref in ranks_of:
        ref = ranks_of[args.ref]
        for tag, ranks in ranks_of.items():
            if tag == args.ref:
                continue
            idxs = sorted(set(ranks) & set(ref)); a = np.array([ranks[i] for i in idxs]); b = np.array([ref[i] for i in idxs])
            dh = stats.boot(np.ones(len(a)), {"a": (a == 0).astype(float), "b": (b == 0).astype(float)}, lambda yy, c: float(c["a"].mean() - c["b"].mean()), n=args.n_boot, require_both_classes=False)
            dm = stats.boot(np.ones(len(a)), {"a": 1 / (a + 1.0), "b": 1 / (b + 1.0)}, lambda yy, c: float(c["a"].mean() - c["b"].mean()), n=args.n_boot, require_both_classes=False)
            contrasts[tag] = {"vs": args.ref, "d_hits@1": {"estimate": round(float((a == 0).mean() - (b == 0).mean()), 4)} | (dh or {}),
                              "d_mrr": {"estimate": round(float((1 / (a + 1)).mean() - (1 / (b + 1)).mean()), 4)} | (dm or {})}
            pv[tag] = contrasts[tag]["d_hits@1"].get("p", 1.0)
        holm = stats.holm(pv)
        for t in contrasts:
            contrasts[t]["p_holm_hits@1"] = holm[t]
    for fn, rows in (("evalA.csv", A_rows), ("evalB.csv", B_rows)):
        if rows:
            keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "tag", k))
            with open(out / fn, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
            run.add_output(out / fn)
    json.dump(contrasts, open(out / "contrasts.json", "w"), indent=2); run.add_output(out / "contrasts.json")
    run.finish(metrics={r["tag"]: {"hits@1": r["hits@1"], "mrr": r["mrr"]} for r in A_rows})
    print(json.dumps({"evalA": A_rows, "evalB": B_rows, "contrasts": contrasts}, indent=1))


if __name__ == "__main__":
    main()
