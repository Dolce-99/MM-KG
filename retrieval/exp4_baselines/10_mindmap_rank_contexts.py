"""
10_mindmap_rank_contexts.py   --   env: llm_qwen (tokenizer; CPU)
================================================================================
MindMap adapted to the five-candidate ranking benchmark (d12).  Evidence graphs are mined
per candidate with the official recipe (shortest paths candidate -> patient entities, plus
one-hop neighbourhoods), rendered in the paper's quoted-triple format, and the patient
record is shown once.  Two variants:
  score : per-candidate yes/no question (Experiment 2's eval-A question) with the frozen
          yes-token recipe -> ranking by P(yes)     (evalA.jsonl, scored by exp2 03_score_ranking.py)
  gen   : the MindMap graph-of-thought prompt (Output1 = selected drug + ranking, Output2 inference
          chain, Output3 mind map), greedy generation -> parsed ranking   (evalB.jsonl-like)
Outputs: outputs/restricted/exp2/contexts/MM_{score|gen}_<split>/{evalA.jsonl | evalB.jsonl}
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths, tokens as tk, runlog, frozen as fz
import importlib.util
_m = importlib.util.spec_from_file_location("mm", Path(__file__).resolve().parent / "05_mindmap_contexts.py"); mm = importlib.util.module_from_spec(_m); _m.loader.exec_module(mm)
_c = importlib.util.spec_from_file_location("c2", Path(__file__).resolve().parents[1] / "exp2_ranking" / "02_build_contexts.py"); c2 = importlib.util.module_from_spec(_c); _c.loader.exec_module(c2)

PUB = paths.OUT / "exp2"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="ranking_v1_d12"); ap.add_argument("--split", default="dev"); ap.add_argument("--variant", default="score", choices=["score", "gen"])
    ap.add_argument("--max_ctx", type=int, default=8192); ap.add_argument("--reserve", type=int, default=16); ap.add_argument("--gen_reserve", type=int, default=768)
    ap.add_argument("--control", default="none", choices=["none", "shuffled", "gold_edge_removed"])
    a = ap.parse_args(); tag = f"MM_{a.variant}_{a.split}" + ("" if a.control == "none" else f"_{a.control}")
    crng = np.random.default_rng(20260919)
    run = runlog.Run("exp4", f"mindmap_rank_{tag}", vars(a))
    tok = tk.load_tokenizer(); A, rel, name = mm.load_kg()
    D = paths.RESTRICTED / "exp2" / a.version; items = [it for it in map(json.loads, open(D / "items.jsonl")) if it["split"] == a.split]
    units_by_h = {u["hadm"]: u["units"] for u in map(json.loads, open(D / "patient_units.jsonl"))}
    relc = fz.load_rel_cache(verify=False); ihm = fz.load_ihm(verify=False); f2p, d2p, rep_by_study, img_by_study = fz.load_alignment_and_findings(verify=False); cui_name = fz.load_cui_names()
    outdir = paths.RESTRICTED / "exp2" / "contexts" / tag; outdir.mkdir(parents=True, exist_ok=True); (PUB / "contexts").mkdir(exist_ok=True)
    budget = a.max_ctx - a.reserve - (a.gen_reserve if a.variant == "gen" else 0); agg = defaultdict(list)
    fa = open(outdir / ("evalA.jsonl" if a.variant == "score" else "evalB.jsonl"), "w")
    for n_done, it in enumerate(items):
        if n_done % 100 == 0: print(f"  {tag}: {n_done}/{len(items)}", file=sys.stderr, flush=True)
        h = it["hadm"]; rec = ihm["recs"][h]; units = units_by_h[h]; t0 = time.time()
        blocks = fz.make_blocks(rec, h, "contra_labs", it["candidates"][0]["pkg"], it["candidates"][0]["mimic_name"], ihm, (rep_by_study, img_by_study), cui_name, relc)
        patient = "\n\n".join(blocks[b] for b in c2.PATIENT_ORDER if blocks.get(b)) if hasattr(c2, "PATIENT_ORDER") else "\n\n".join(blocks[b] for b in ("demo", "codes", "meds", "labs", "report", "imaging") if blocks.get(b))
        pat_ents = []
        for u in units:
            if u["channel"] in ("codes", "report", "imaging"):
                for p in u["pkg"]:
                    if p not in pat_ents: pat_ents.append(p)
        cnames = [c["name"] for c in it["candidates"]]; cand_line = "Candidate drugs received by this patient: " + "; ".join(f"{i + 1}. {n}" for i, n in enumerate(cnames)) + "."
        ev_per_cand = []; gold_edges_shown = 0
        for c in it["candidates"]:
            q = c["pkg"]; sp_ = mm.shortest_paths(A, q, [p for p in pat_ents if p != q], mm.MAX_HOPS)
            paths_ = sorted(sp_.values(), key=lambda p: (len(p), p))[: mm.MAX_PATHS // len(it["candidates"]) + 2]
            plines = [f"Path-based Evidence ({c['name']}) {k + 1}: " + mm.render_path(p, rel, name) for k, p in enumerate(paths_)]
            groups = defaultdict(list)
            for v in A.indices[A.indptr[q]:A.indptr[q + 1]]:
                v = int(v); r = rel.get((q, v), "related_to")
                if len(groups[r]) < mm.NB_PER_ENTITY: groups[r].append(v)
            nlines = [f"Neighbor-based Evidence ({c['name']}) {k + 1}: '{c['name']}'->'{r.replace('_', ' ')}'->" + ",".join(f"'{name.get(v, v)}'" for v in vs[: mm.NB_PER_ENTITY]) for k, (r, vs) in enumerate(sorted(groups.items()))]
            shown = {(p[j], rel.get((p[j], p[j + 1]), rel.get((p[j + 1], p[j]), "")), p[j + 1]) for p in paths_ for j in range(len(p) - 1)} | {(q, r, v) for r, vs in groups.items() for v in vs[: mm.NB_PER_ENTITY]}
            ev_per_cand.append((c, plines, nlines, shown))
        gold_e = {(it["gold"]["selected_pkg"], e[1], e[2]) for e in it["gold"]["kg_edges"]}
        if a.control == "gold_edge_removed":            # drop every evidence line that renders an answer-bearing edge of the positive
            gname = {name.get(e[2], str(e[2])) for e in gold_e}
            ev_per_cand = [(c, [l for l in pl if not any(f"'{g}'" in l for g in gname)] if c["positive"] else pl,
                            [l for l in nl if not any(f"'{g}'" in l for g in gname)] if c["positive"] else nl, s) for c, pl, nl, s in ev_per_cand]
        elif a.control == "shuffled":                    # evidence blocks re-attached to the wrong candidate names
            perm = crng.permutation(len(ev_per_cand)); names_ = [c["name"] for c, _, _, _ in ev_per_cand]
            ev_per_cand = [(ev_per_cand[i][0], [l.replace(f"({names_[i]})", f"({names_[j]})").replace(f"'{names_[i]}'", f"'{names_[j]}'") for l in ev_per_cand[i][1]],
                            [l.replace(f"({names_[i]})", f"({names_[j]})").replace(f"'{names_[i]}'", f"'{names_[j]}'") for l in ev_per_cand[i][2]], ev_per_cand[i][3]) for j, i in enumerate(perm)]
            ev_per_cand = [(it["candidates"][j], pl, nl, s) for j, (_, pl, nl, s) in enumerate(ev_per_cand)]
        gold_edges_shown = int(any(bool(gold_e & s) for _, _, _, s in ev_per_cand))
        # round-robin across candidates, paths before neighbours, until the budget is used
        def evidence_text(k_path, k_nb):
            parts = []
            for c, pl, nl, _ in ev_per_cand:
                parts += pl[:k_path] + nl[:k_nb]
            return "You have some medical knowledge information in the following:\n\n###" + "\n".join(parts)
        maxp = max(len(pl) for _, pl, _, _ in ev_per_cand); maxn = max(len(nl) for _, _, nl, _ in ev_per_cand)
        sysm = mm.SYS_GEN if a.variant == "gen" else paths.SYS_PROMPT
        def build(kp, kn, cand=None):
            if a.variant == "score":
                return patient + "\n\n" + cand_line + "\n\n" + evidence_text(kp, kn) + "\n\n" + c2.Q_A.format(d=cand)
            return ("Patient input:" + patient + "\n\n" + cand_line + "\n\n" + evidence_text(kp, kn) + "\n\n" + "Exactly one of the candidate drugs has a knowledge-graph relation (contraindication or adverse effect) to a condition supported by this patient's documented evidence. Think step by step.\n\n"
                    "Output1: The selected drug, then the full ranking of the five candidate drugs from most to least likely, as JSON: {\"selected_drug\": ..., \"ranking\": [...]}.\n\n"
                    "Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighbor-based Evidence, and in the end infer what result. Transport the inference process into the following format: Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name').\n\n"
                    "Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.")
        kp, kn = maxp, maxn
        probe = build(kp, kn, cnames[0]); n_tok = tk.n_tokens(tok, tk.render_chat(tok, sysm, probe))
        while n_tok > budget and (kp > 0 or kn > 0):
            if kn > 1: kn -= 1
            elif kp > 1: kp -= 1
            else: kn = 0 if kn else kn; kp = 0 if kp and not kn else kp
            probe = build(kp, kn, cnames[0]); n_tok = tk.n_tokens(tok, tk.render_chat(tok, sysm, probe))
            if kp == 0 and kn == 0: break
        if a.variant == "score":
            for c in it["candidates"]:
                fa.write(json.dumps({"idx": it["idx"], "cand_pkg": c["pkg"], "cand_name": c["name"], "positive": c["positive"], "user": build(kp, kn, c["name"])}) + "\n")
        else:
            fa.write(json.dumps({"idx": it["idx"], "user": build(kp, kn), "system": sysm, "candidates": cnames, "tagmap": {}}) + "\n")
        st = {"prompt_tokens": n_tok, "gold_edge_shown": gold_edges_shown, "k_path": kp, "k_nb": kn, "retrieval_ms": round((time.time() - t0) * 1000, 1)}
        for k, v in st.items(): agg[k].append(float(v))
    fa.close()
    summ = {"tag": tag, "n_items": len(items), "means": {k: round(float(np.mean(v)), 4) for k, v in agg.items()}, "max_prompt_tokens": int(max(agg["prompt_tokens"]))}
    json.dump(summ, open(PUB / "contexts" / f"{tag}_summary.json", "w"), indent=2); run.finish(metrics=summ["means"]); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
