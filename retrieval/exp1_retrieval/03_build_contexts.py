"""
03_build_contexts.py   --   env: llm_qwen (tokenizer only; CPU, no GPU)
================================================================================
Materialises the prompt context of one retrieval policy for every benchmark item
of a split, with retrieval metrics computed against the gold annotations AFTER
the context is built (the gold file is never consulted during selection, except
by the explicit oracle policy R5).

Policies
  R0  closed book: question only
  R1  full admissible context: all patient units + all typed one-hop triples of
      the linked entities + node descriptions + protein->disease bridge, in a
      deterministic label-blind order; truncated deterministically at the budget
  R2  vector RAG: patient-unit and KG-triple chunks ranked by cosine similarity
      to (question + candidate drug) with a local biomedical encoder; requires
      the chunk embeddings written by 04_embed_chunks.py
  R3  unfiltered typed one-hop graph retrieval from the linked entities (no
      relation whitelist, no descriptions), deterministic truncation
  R4  the frozen MM-KG prompt (arm D_both of 19_composite_wide_llm.py), verbatim
  R5  gold-path oracle: patient units + the answer-bearing edges (positives) or a
      size-matched sample of the drug's true edges (negatives)

Budget
  --budget natural      no truncation below --max_ctx (default 8192, the frozen setting)
  --budget <int>        total prompt tokens (system + template + evidence + question)
  --budget match_r4     per-item budget = token count of the R4 prompt (token-matched)

Outputs
  outputs/restricted/exp1/contexts/<tag>/contexts.jsonl   {"idx", "user", "stats"}   RESTRICTED
  outputs/exp1/contexts/<tag>_summary.json                 aggregate only
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, tokens as tk, runlog, frozen as fz

import os
EV = paths.RESTRICTED / "exp1" / os.environ.get("MMKG_EVIDENCE_VERSION", "evidence_v1")
KG_VERSION = os.environ.get("MMKG_KG_ONEHOP_VERSION", "kg_onehop_v1")
PUB = paths.OUT / "exp1"
PATIENT_ORDER = ["demo", "codes", "meds", "labs", "report", "imaging"]
TYPE_LABEL = {"drug": "Drug", "disease": "Disease", "effect/phenotype": "Phenotype", "gene/protein": "Protein",
              "exposure": "Exposure", "biological_process": "BiologicalProcess", "molecular_function": "MolecularFunction",
              "cellular_component": "CellularComponent", "pathway": "Pathway", "anatomy": "Anatomy"}
FAMILY_REL = {"contra_labs": "contraindication", "contra_report": "contraindication",
              "drug_effect": "drug_effect", "mechanism": None}


def triple_line(xt, xn, rel, yt, yn) -> str:
    return (f'(:{TYPE_LABEL.get(xt, xt)} {{name: "{xn}"}})-[:{rel.upper().replace(" ", "_").replace("-", "_")}]->'
            f'(:{TYPE_LABEL.get(yt, yt)} {{name: "{yn}"}})')


class KG:
    def __init__(self, order: str):
        df = pd.read_parquet(PUB / f"{KG_VERSION}.parquet")
        if order == "relation_alpha":
            df = df.sort_values(["x_index", "relation", "y_name", "y_index"], kind="mergesort")
        elif order == "round_robin":                     # interleave relation types: no relation is privileged
            df = df.sort_values(["x_index", "relation", "y_name", "y_index"], kind="mergesort")
            df["_r"] = df.groupby(["x_index", "relation"]).cumcount()
            df = df.sort_values(["x_index", "_r", "relation"], kind="mergesort").drop(columns="_r")
        else:                                            # graph_file: kg.csv order (stable)
            df = df.sort_values(["x_index"], kind="mergesort")
        self.by_x = {int(x): g for x, g in df.groupby("x_index", sort=False)}
        self._memo = {}
        t = pd.read_parquet(PUB / f"{KG_VERSION.replace('kg_onehop', 'kg_node_text')}.parquet")
        self.text = {int(r.node_index): r for r in t.itertuples()}
        self.name = {}
        for g in self.by_x.values():
            self.name[int(g.x_index.iloc[0])] = str(g.x_name.iloc[0])
            self.name.update({int(a): str(b) for a, b in zip(g.y_index, g.y_name)})

    def triples(self, x: int, role: str = "linked") -> list[dict]:
        key = (x, role)
        if key not in self._memo:
            g = self.by_x.get(x)
            if g is None:
                self._memo[key] = []
            else:
                g = g[g.hop_role == role] if role else g
                self._memo[key] = [{"x": int(r.x_index), "rel": r.relation, "y": int(r.y_index),
                                    "line": triple_line(r.x_type, r.x_name, r.relation, r.y_type, r.y_name)} for r in g.itertuples()]
        return self._memo[key]

    def description(self, x: int) -> str:
        r = self.text.get(x)
        if r is None or not r.text:
            return ""
        return f'Description of "{self.name.get(x, x)}": {r.text}'


def patient_part(it_frozen) -> str:
    return "\n\n".join(it_frozen["blocks"][b] for b in PATIENT_ORDER if it_frozen["blocks"].get(b))


def visible_uids(units) -> list[str]:
    """Units that survive the caps of the frozen block renderer (codes 25, meds 40, report 20)."""
    codes = set(sorted({u["text"] for u in units if u["channel"] == "codes"})[:25])
    meds = set(sorted(dict.fromkeys(u["text"] for u in units if u["channel"] == "meds"))[:40])
    reps = set(sorted({u["text"] for u in units if u["channel"] == "report"})[:20])
    out = []
    for u in units:
        ch = u["channel"]
        if (ch == "codes" and u["text"] not in codes) or (ch == "meds" and u["text"] not in meds) or (ch == "report" and u["text"] not in reps):
            continue
        out.append(u["uid"])
    return out


def build(policy, it, itf, units, kg, gold, args, rng):
    """Return (kg_lines, kg_edges, header, extra_notes)."""
    q = it["drug_pkg"]
    if policy in ("R0", "R4"):
        return [], [], "", {}
    if policy == "R5":
        if gold["answer"] == "yes":
            edges = [tuple(e) for e in gold["kg_edges"]]
        else:                                   # negatives: same edge count as the paired positive (same admission,
            fam = it["family"]                  # same family), drawn from the drug's TRUE edges of that relation layer
            n_target = args.r5_pair_edges.get((it["hadm"], fam), 1)
            relname = FAMILY_REL[fam] or "drug_protein"
            pool = [t for t in kg.triples(q) if t["rel"] == relname]
            prng = np.random.default_rng(20260911 + it["idx"])
            order = prng.permutation(len(pool)) if pool else []
            edges = []
            for i in order:
                t = pool[i]; edges.append((t["x"], t["rel"], t["y"]))
                if fam == "mechanism":          # mimic the 2-hop path shape: protein -> one of its diseases
                    br = kg.triples(t["y"], role="protein_bridge")
                    if br:
                        b = br[int(prng.integers(len(br)))]; edges.append((b["x"], b["rel"], b["y"]))
                if len(edges) >= n_target:
                    break
            edges = edges[:n_target] if n_target else edges
        lines = [triple_line(_t(kg, x), kg.name.get(x, x), r, _t(kg, y), kg.name.get(y, y)) for x, r, y in edges]
        return lines, [tuple(e) for e in edges], "Facts from a curated biomedical knowledge graph:", {}
    # entity set for R1 / R3
    ents = [q]
    seen = {q}
    for u in units:
        if u["channel"] in ("codes", "report", "imaging") or (args.r3_include_meds and u["channel"] == "meds"):
            for p in u["pkg"]:
                if p not in seen and p != q:
                    ents.append(p); seen.add(p)
    lines, edges = [], []
    if policy == "R1":
        for e in ents:
            d = kg.description(e)
            if d:
                lines.append(d); edges.append(None)
            trip = kg.triples(e)
            if e == q:
                # the queried drug: one group per relation type plus the two-hop protein->disease bridge
                # (bridge ordered by the drug_protein order so that a protein and its diseases survive together);
                # groups are interleaved round-robin when --kg_order round_robin, else concatenated
                groups, order = {}, []
                for t in trip:
                    groups.setdefault(t["rel"], []).append((t["line"], (t["x"], t["rel"], t["y"])))
                    if t["rel"] == "drug_protein":
                        for b in kg.triples(t["y"], role="protein_bridge"):
                            groups.setdefault("protein_bridge", []).append((b["line"], (b["x"], b["rel"], b["y"])))
                seq = _interleave(list(groups.values())) if args.kg_order == "round_robin" else [x for g in groups.values() for x in g]
                for ln, ed in seq:
                    lines.append(ln); edges.append(ed)
            else:
                for t in trip:
                    lines.append(t["line"]); edges.append((t["x"], t["rel"], t["y"]))
        hdr = (f"Facts from a curated biomedical knowledge graph (all relations of the {len(ents)} linked entities; "
               f"{sum(e is not None for e in edges)} triples):")
        return lines, edges, hdr, {"n_entities": len(ents)}
    if policy == "R3":
        for e in ents:
            for t in kg.triples(e):
                lines.append(t["line"]); edges.append((t["x"], t["rel"], t["y"]))
        hdr = f"Knowledge-graph retrieval results (typed one-hop neighbourhood of {len(ents)} linked entities; {len(lines)} triples):"
        return lines, edges, hdr, {"n_entities": len(ents)}
    raise ValueError(policy)


def _interleave(groups):
    out, i = [], 0
    while any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g):
                out.append(g[i])
        i += 1
    return out


def _t(kg, x):
    g = kg.by_x.get(x)
    if g is not None:
        return str(g.x_type.iloc[0])
    return "Node"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["R0", "R1", "R2", "R3", "R4", "R5"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--budget", default="natural")
    ap.add_argument("--max_ctx", type=int, default=8192)
    ap.add_argument("--reserve", type=int, default=16, help="tokens kept free for the answer token and tokenizer boundary effects (vLLM requires prompt + 1 <= max_model_len)")
    ap.add_argument("--kg_order", default="relation_alpha", choices=["relation_alpha", "round_robin", "graph_file"])
    ap.add_argument("--r3_include_meds", type=int, default=0)
    ap.add_argument("--emb_dir", default=None, help="R2: directory from 04_embed_chunks.py")
    ap.add_argument("--r2_mode", default="all", choices=["all", "kg_only", "multiq"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model", default="q35")
    args = ap.parse_args()
    tag = args.tag or f"{args.policy}_{args.split}_b{args.budget}"
    cfg = vars(args) | {"tag": tag, "system_prompt": paths.SYS_PROMPT}
    run = runlog.Run("exp1", f"contexts_{tag}", cfg)

    tok = tk.load_tokenizer(paths.MODELS[args.model])
    bench = paths.load_benchmark(verify=False)
    split = paths.load_split()
    idxs = sorted(split["dev"] + split["test"]) if args.split == "all" else split[args.split]
    items = {j["idx"]: j for j in map(json.loads, open(EV / "items.jsonl"))}
    GOLD_EV = paths.RESTRICTED / "exp1" / os.environ.get("MMKG_GOLD_VERSION", "evidence_v1")   # gold stays frozen (v1) under alignment overrides
    gold = {g["idx"]: g for g in map(json.loads, open(GOLD_EV / "gold.jsonl"))}
    units_by_h = {u["hadm"]: u["units"] for u in map(json.loads, open(EV / "patient_units.jsonl"))}
    kg = KG(args.kg_order)
    rel = fz.load_rel_cache(verify=False); name = rel["name"]
    r2 = None
    if args.policy == "R2":
        from r2_vector import R2Selector                   # local module (04/05)
        r2 = R2Selector(args.emb_dir, tok, mode=args.r2_mode)
    rng = np.random.default_rng(20260911)
    # R5 negatives are size-matched to their paired positive (same admission and family)
    args.r5_pair_edges = {(items[j]["hadm"], items[j]["family"]): len(gold[j]["kg_edges"])
                          for j in gold if gold[j]["answer"] == "yes"}

    outdir = paths.RESTRICTED / "exp1" / "contexts" / tag; outdir.mkdir(parents=True, exist_ok=True)
    (PUB / "contexts").mkdir(parents=True, exist_ok=True)
    agg = defaultdict(list)
    with open(outdir / "contexts.jsonl", "w") as fo:
        for n_done, i in enumerate(idxs):
            if n_done % 100 == 0:
                print(f"  {tag}: {n_done}/{len(idxs)}", file=sys.stderr, flush=True)
            it, itf, g = items[i], bench[i], gold[i]
            units = units_by_h[it["hadm"]]
            t0 = time.time()
            fixed = tk.n_tokens(tok, tk.render_chat(tok, paths.SYS_PROMPT, itf["question"]))
            if args.budget == "natural":
                budget = args.max_ctx - args.reserve
            elif args.budget == "match_r4":
                budget = tk.n_tokens(tok, tk.render_chat(tok, paths.SYS_PROMPT, patient_part(itf) + "\n\n" + itf["blocks"]["kg"] + "\n\n" + itf["question"]))
            else:
                budget = int(args.budget)
            pp = "" if args.policy == "R0" else patient_part(itf)
            if args.policy == "R4":
                kg_text = itf["blocks"]["kg"]; lines = []; hdr = ""
                relname = FAMILY_REL[it["family"]] or "drug_protein+disease_protein"
                kg_edges = [(it["drug_pkg"], relname, c) for c in sorted(rel["layer"][it["family"]].get(it["drug_pkg"], ()))]
                pat_units_shown = visible_uids(units)
                ppack = {"truncated": False}
            elif args.policy == "R2":
                sel = r2.select(it, itf, units, kg, budget - fixed, patient_text=patient_part(itf))
                pp, kg_text, lines, kg_edges = sel["patient_text"], sel["kg_text"], sel["kg_lines"], sel["kg_edges"]
                pat_units_shown = sel["patient_uids"] if sel["patient_uids"] is not None else visible_uids(units)
                ppack = {"truncated": sel["truncated"], "untruncated_tokens": sel["untruncated_tokens"]}; hdr = ""
            else:
                lines, kg_edges, hdr, extra = build(args.policy, it, itf, units, kg, g, args, rng)
                pat_units_shown = [] if args.policy == "R0" else visible_uids(units)
                pat_tokens = tk.n_tokens(tok, pp + "\n\n") if pp else 0
                avail = budget - fixed - pat_tokens - (tk.n_tokens(tok, hdr + "\n") if hdr else 0)
                ppack = tk.pack(tok, lines, max(avail, 0)) if lines else {"kept": [], "tokens": 0, "untruncated_tokens": 0, "truncated": False}
                kept = ppack["kept"]
                lines = [lines[k] for k in kept]; kg_edges = [kg_edges[k] for k in kept]
                kg_text = (hdr + "\n" + "\n".join(lines)) if lines else ""
            def _assemble(pp_, kg_):
                u = "\n\n".join(x for x in (pp_, kg_) if x) + ("\n\n" if (pp_ or kg_) else "") + itf["question"]
                return u, tk.n_tokens(tok, tk.render_chat(tok, paths.SYS_PROMPT, u))
            user, n_tok = _assemble(pp, kg_text)
            if args.policy != "R4":                       # headers / joint-tokenisation slack: enforce the budget exactly
                while n_tok > budget and lines:
                    lines = lines[:-1]; kg_edges = kg_edges[:len(lines)]
                    kg_text = (kg_text.split("\n", 1)[0] + "\n" + "\n".join(lines)) if lines else ""
                    user, n_tok = _assemble(pp, kg_text); ppack["truncated"] = True
            # ---- retrieval metrics (gold consulted only here) ----
            gold_p = set(g["patient_uids"]); gold_e = {tuple(e) for e in g["kg_edges"]}
            edge_set = {e for e in kg_edges if e is not None}
            rp = gold_p.issubset(set(pat_units_shown)) if gold_p else None
            if args.policy == "R4":
                conds = {e[2] for e in edge_set}
                any_edge = bool(set(g["condition_pkg"]) & conds) if gold_e else None
                path_ok = any_edge                       # the prose list collapses drug->protein->disease into one entry
                n_edge_hit = len(set(g["condition_pkg"]) & conds)
            else:
                any_edge = bool(gold_e & edge_set) if gold_e else None
                path_ok = any(all(tuple(e) in edge_set for e in p["edges"]) for p in g["paths"]) if g["paths"] else None
                n_edge_hit = len(gold_e & edge_set)
            complete = (bool(rp) and bool(path_ok)) if g["answer"] == "yes" else None
            n_hit = (len(gold_p & set(pat_units_shown)) + n_edge_hit) if g["answer"] == "yes" else None
            n_ctx = len(pat_units_shown) + len(edge_set)
            st = {"family": it["family"], "answer": g["answer"], "budget": budget, "prompt_tokens": n_tok,
                  "untruncated_tokens": (n_tok if not ppack.get("truncated", False) else
                                         fixed + ppack.get("untruncated_tokens", 0) + (0 if args.policy == "R2" else
                                         (tk.n_tokens(tok, pp + "\n\n") if pp else 0) + (tk.n_tokens(tok, hdr + "\n") if hdr else 0))),
                  "truncated": bool(ppack.get("truncated", False)) or n_tok > args.max_ctx - args.reserve,
                  "n_patient_units": len(pat_units_shown), "n_kg_edges": len(edge_set),
                  "n_kg_nodes": len({n for e in edge_set for n in (e[0], e[2])}),
                  "recall_patient": rp, "recall_kg_edge": any_edge, "recall_path": complete,
                  "precision": (n_hit / n_ctx if (n_ctx and n_hit is not None) else None),
                  "retrieval_ms": round((time.time() - t0) * 1000, 1)}
            fo.write(json.dumps({"idx": i, "user": user, "stats": st}) + "\n")
            for k, v in st.items():
                if isinstance(v, (int, float, bool)) and k not in ("budget",):
                    agg[k].append(float(v)); agg[f"{k}|{it['family']}"].append(float(v))
            agg["over_max_ctx"].append(float(n_tok + 1 > args.max_ctx))
    summ = {"tag": tag, "policy": args.policy, "split": args.split, "n_items": len(idxs), "budget": args.budget,
            "means": {k: round(float(np.mean(v)), 4) for k, v in agg.items() if v},
            "prompt_tokens_quantiles": {q: int(np.quantile(agg["prompt_tokens"], q)) for q in (0.5, 0.9, 0.99, 1.0)},
            "untruncated_tokens_quantiles": {q: int(np.quantile(agg["untruncated_tokens"], q)) for q in (0.5, 0.9, 0.99, 1.0)}}
    json.dump(summ, open(PUB / "contexts" / f"{tag}_summary.json", "w"), indent=2)
    run.add_output(outdir / "contexts.jsonl"); run.add_output(PUB / "contexts" / f"{tag}_summary.json")
    run.finish(counts={"n_items": len(idxs)}, metrics=summ["means"])
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
