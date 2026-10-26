"""
05_mindmap_contexts.py   --   env: llm_qwen (tokenizer; CPU)
================================================================================
Faithful local re-implementation of MindMap (Wen et al., 2024; arXiv 2308.09729) retrieval and
prompting on the frozen join benchmark, with the same local model, 8K context and scoring as
every other policy (Experiment 1).  MindMap = entity extraction -> path-based evidence
(shortest KG paths between question entities) + neighbour-based evidence (one-hop
neighbourhoods) -> "evidence graph" prompt -> LLM answer with an inference chain and a
"mind map" (decision tree).

Substitutions (all documented): entity extraction by GPT-3.5 in the paper -> the graph's own
patient-concept links plus the question drug (the same entity set every policy uses);
evidence paraphrase by GPT-3.5 -> deterministic rendering in the paper's
'entity'->'relation'->'entity' format; Neo4j shortest paths -> BFS on the clinical PrimeKG
(<= 3 hops, undirected); GPT-4 -> the frozen local Qwen model.  Two variants:
  score : MindMap evidence + the frozen yes/no question and yes-token scoring (retrieval-only comparison)
  gen   : the MindMap final prompt (Output1 answer, Output2 inference chain, Output3 mind map),
          greedy generation, answer parsed from Output1 (scored by 06_score_mindmap_gen.py)
Outputs: outputs/restricted/exp1/contexts/MM_{score|gen}_<split>_bnatural/contexts.jsonl (+ summary)
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp1_retrieval"))
from common import paths, tokens as tk, runlog
import importlib.util
_s = importlib.util.spec_from_file_location("ctx1", Path(__file__).resolve().parents[1] / "exp1_retrieval" / "03_build_contexts.py")
ctx1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ctx1)

EV = paths.RESTRICTED / "exp1" / "evidence_v1"; PUB = paths.OUT / "exp1"
MAX_HOPS = 3; PATHS_PER_PAIR = 5; MAX_PATHS = 40; NB_PER_ENTITY = 20; MAX_NB_ENTITIES = 30
SYS_GEN = "You are an excellent AI doctor, and you can answer clinical questions about a patient based on the patient record and medical knowledge."


def load_kg():
    ce = pd.read_parquet(paths.OUT / "exp4" / "clinical_edges.parquet", columns=["x_index", "relation", "y_index", "x_name", "y_name"])
    n = int(max(ce.x_index.max(), ce.y_index.max())) + 1
    order = np.lexsort((ce.y_index.values, ce.x_index.values)); xs = ce.x_index.values[order]; ys = ce.y_index.values[order]
    class CSR:                                   # minimal CSR (no scipy in llm_qwen): indptr / indices like scipy
        pass
    A = CSR(); A.indices = ys.astype(np.int64); A.indptr = np.searchsorted(xs, np.arange(n + 1), side="left")
    rel = dict(zip(zip(ce.x_index.values.tolist(), ce.y_index.values.tolist()), ce.relation.values))
    name = dict(zip(ce.x_index.values.tolist(), ce.x_name.values)); name.update(dict(zip(ce.y_index.values.tolist(), ce.y_name.values)))
    return A, rel, name


def shortest_paths(A, src, targets, max_hops):
    """BFS from src; one shortest path per reachable target (deterministic parent = lowest index)."""
    parent = {src: None}; depth = {src: 0}; q = deque([src]); targets = set(targets)
    while q:
        u = q.popleft()
        if depth[u] >= max_hops: continue
        for v in A.indices[A.indptr[u]:A.indptr[u + 1]]:
            v = int(v)
            if v not in parent:
                parent[v] = u; depth[v] = depth[u] + 1; q.append(v)
    out = {}
    for t in targets:
        if t in parent and t != src:
            p = [t]
            while parent[p[-1]] is not None: p.append(parent[p[-1]])
            out[t] = p[::-1]
    return out


def render_path(p, rel, name):
    s = f"'{name.get(p[0], p[0])}'"
    for a, b in zip(p[:-1], p[1:]):
        s += f"->'{rel.get((a, b), rel.get((b, a), 'related_to')).replace('_', ' ')}'->'{name.get(b, b)}'"
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="dev"); ap.add_argument("--variant", default="score", choices=["score", "gen"])
    ap.add_argument("--max_ctx", type=int, default=8192); ap.add_argument("--reserve", type=int, default=16); ap.add_argument("--gen_reserve", type=int, default=768)
    a = ap.parse_args(); tag = f"MM_{a.variant}_{a.split}_bnatural"
    run = runlog.Run("exp4", f"mindmap_{tag}", vars(a) | {"max_hops": MAX_HOPS, "paths_per_pair": PATHS_PER_PAIR, "max_paths": MAX_PATHS, "nb_per_entity": NB_PER_ENTITY})
    tok = tk.load_tokenizer(); bench = paths.load_benchmark(verify=False); split = paths.load_split(); idxs = split[a.split]
    items = {j["idx"]: j for j in map(json.loads, open(EV / "items.jsonl"))}; gold = {g["idx"]: g for g in map(json.loads, open(EV / "gold.jsonl"))}
    units_by_h = {u["hadm"]: u["units"] for u in map(json.loads, open(EV / "patient_units.jsonl"))}
    A, rel, name = load_kg()
    outdir = paths.RESTRICTED / "exp1" / "contexts" / tag; outdir.mkdir(parents=True, exist_ok=True); (PUB / "contexts").mkdir(exist_ok=True)
    agg = defaultdict(list)
    budget = a.max_ctx - a.reserve - (a.gen_reserve if a.variant == "gen" else 0)
    with open(outdir / "contexts.jsonl", "w") as fo:
        for n_done, i in enumerate(idxs):
            if n_done % 200 == 0: print(f"  {tag}: {n_done}/{len(idxs)}", file=sys.stderr, flush=True)
            it, itf, g = items[i], bench[i], gold[i]; units = units_by_h[it["hadm"]]; t0 = time.time()
            q = it["drug_pkg"]
            pat_ents = []
            for u in units:
                if u["channel"] in ("codes", "report", "imaging"):
                    for p in u["pkg"]:
                        if p not in pat_ents and p != q: pat_ents.append(p)
            # path-based evidence: shortest paths from the question drug to each patient entity
            sp_ = shortest_paths(A, q, pat_ents, MAX_HOPS)
            paths_ = sorted(sp_.values(), key=lambda p: (len(p), p))[:MAX_PATHS]
            path_edges = {(p[j], rel.get((p[j], p[j + 1]), rel.get((p[j + 1], p[j]), "related_to")), p[j + 1]) for p in paths_ for j in range(len(p) - 1)}
            # neighbour-based evidence: one-hop neighbourhoods of the question drug and the (first) patient entities, grouped by relation
            nb_lines, nb_edges = [], set()
            for e in [q] + pat_ents[:MAX_NB_ENTITIES]:
                groups = defaultdict(list)
                for v in A.indices[A.indptr[e]:A.indptr[e + 1]]:          # all neighbours, grouped by relation (each group capped below)
                    v = int(v); r = rel.get((e, v), "related_to")
                    if len(groups[r]) < NB_PER_ENTITY: groups[r].append(v)
                for r, vs in sorted(groups.items()):
                    vs = vs[:NB_PER_ENTITY]
                    nb_lines.append(f"'{name.get(e, e)}'->'{r.replace('_', ' ')}'->" + ",".join(f"'{name.get(v, v)}'" for v in vs))
                    nb_edges |= {(e, r, v) for v in vs}
            path_lines = [f"Path-based Evidence {k + 1}: " + render_path(p, rel, name) for k, p in enumerate(paths_)]
            nb_lines = [f"Neighbor-based Evidence {k + 1}: " + ln for k, ln in enumerate(nb_lines)]
            patient = ctx1.patient_part(itf)
            fixed = tk.n_tokens(tok, tk.render_chat(tok, SYS_GEN if a.variant == "gen" else paths.SYS_PROMPT, itf["question"])) + tk.n_tokens(tok, patient + "\n\n") + 120
            pk = tk.pack(tok, path_lines + nb_lines, budget - fixed)
            kept = pk["kept"]; kp = [path_lines[k] for k in kept if k < len(path_lines)]; kn = [nb_lines[k - len(path_lines)] for k in kept if k >= len(path_lines)]
            TASK_GEN = (" Think step by step.\n\nOutput1: The answer, a single word: yes or no, followed by one sentence.\n\n"
                        "Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighbor-based Evidence, "
                        "and in the end infer what result. Transport the inference process into the following format: Path-based Evidence number('entity name'->'relation name'->...)"
                        "->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name').\n\n"
                        "Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, "
                        "which is followed by the entity in parentheses.")
            def build_user(kp, kn):
                evidence = "You have some medical knowledge information in the following:\n\n###" + "\n".join(kp) + "\n\n###" + "\n".join(kn)
                if a.variant == "score":
                    return patient + "\n\n" + evidence + "\n\n" + itf["question"]
                return "Patient input:" + patient + "\n\n" + evidence + "\n\n" + itf["question"].replace("Question: ", "") + TASK_GEN
            system = SYS_GEN if a.variant == "gen" else paths.SYS_PROMPT
            user = build_user(kp, kn); n_tok = tk.n_tokens(tok, tk.render_chat(tok, system, user))
            while n_tok > budget and (kp or kn):
                if kn: kn.pop()
                else: kp.pop()
                user = build_user(kp, kn); n_tok = tk.n_tokens(tok, tk.render_chat(tok, system, user))
            gold_e = {tuple(e) for e in g["kg_edges"]}; shown = path_edges | nb_edges
            rp = gold_e and bool(gold_e & shown)
            path_ok = any(all(tuple(e) in shown for e in p["edges"]) for p in g["paths"]) if g["paths"] else None
            st = {"family": it["family"], "answer": g["answer"], "budget": budget, "prompt_tokens": n_tok, "untruncated_tokens": fixed + pk["untruncated_tokens"],
                  "truncated": pk["truncated"], "n_patient_units": len(ctx1.visible_uids(units)), "n_kg_edges": len(shown), "n_kg_nodes": len({n for e in shown for n in (e[0], e[2])}),
                  "n_paths": len(kp), "recall_patient": (set(g["patient_uids"]).issubset(set(ctx1.visible_uids(units))) if g["patient_uids"] else None), "recall_kg_edge": (bool(rp) if gold_e else None), "recall_path": ((bool(set(g["patient_uids"]).issubset(set(ctx1.visible_uids(units)))) and bool(path_ok)) if g["answer"] == "yes" else None),
                  "precision": (len(gold_e & shown) / len(shown) if (shown and gold_e) else None), "retrieval_ms": round((time.time() - t0) * 1000, 1)}
            fo.write(json.dumps({"idx": i, "user": user, "system": system, "stats": st}) + "\n")
            for k, v in st.items():
                if isinstance(v, (int, float, bool)) and k != "budget": agg[k].append(float(v)); agg[f"{k}|{it['family']}"].append(float(v))
    summ = {"tag": tag, "policy": "MindMap", "split": a.split, "n_items": len(idxs), "budget": "natural",
            "means": {k: round(float(np.mean(v)), 4) for k, v in agg.items() if v},
            "prompt_tokens_quantiles": {q: int(np.quantile(agg["prompt_tokens"], q)) for q in (0.5, 0.9, 0.99, 1.0)},
            "untruncated_tokens_quantiles": {q: int(np.quantile(agg["untruncated_tokens"], q)) for q in (0.5, 0.9, 0.99, 1.0)}}
    json.dump(summ, open(PUB / "contexts" / f"{tag}_summary.json", "w"), indent=2); run.add_output(outdir / "contexts.jsonl"); run.finish(metrics=summ["means"])
    print(json.dumps({k: summ["means"].get(k) for k in ("prompt_tokens", "truncated", "n_paths", "n_kg_edges", "recall_kg_edge", "recall_path", "retrieval_ms")}))


if __name__ == "__main__":
    main()
