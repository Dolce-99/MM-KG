"""
02_build_contexts.py   --   env: llm_qwen (tokenizer only; CPU)
================================================================================
Materialises Experiment 2 prompts for one (evidence arm, retrieval policy, control)
cell on one split of a ranking benchmark version, for both evaluations:

  eval A  objective ranking: one yes/no prompt per candidate (same context, the
          question names one candidate); scored by P(yes) with the frozen recipe
  eval B  structured explanation: one generation prompt per item that must return
          JSON {selected_drug, ranking, patient_evidence_ids, normalized_condition,
          kg_path, explanation}; patient units carry local tags [P1], [P2], ... so
          evidence can be cited without any internal identifier

Evidence arms   A demo+codes+meds | B +labs+report+imaging | C A+KG packets | D B+KG packets
KG policies     R0 none | R4 complete contraindication + adverse-effect packet per candidate (frozen
                prose style) | R3 typed one-hop triples per candidate, round-robin, budgeted |
                R1 R3 + node descriptions | R2 vector RAG over the candidates' KG chunks (kg_only)
Controls        none | shuffled_packets (packets permuted across candidates) |
                matched_wrong_packets (each packet replaced by a packet-length-matched other drug's) |
                gold_edge_removed (the positive's supported conditions deleted from its packet) |
                no_provenance (channel headers and tags stripped) | no_<channel> (drop one patient channel)
Budget          --budget natural (max_ctx - reserve) or an integer; KG lines are truncated
                round-robin across candidates so that no candidate is privileged.

Outputs  outputs/restricted/exp2/contexts/<tag>/{evalA.jsonl, evalB.jsonl}   RESTRICTED
         outputs/exp2/contexts/<tag>_summary.json                              aggregate only
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, tokens as tk, runlog, frozen as fz
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp1_retrieval"))
import importlib.util
_s = importlib.util.spec_from_file_location("ctx1", Path(__file__).resolve().parents[1] / "exp1_retrieval" / "03_build_contexts.py")
ctx1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ctx1)
from r2_vector import R2Selector, unit_chunk

PUB = paths.OUT / "exp2"
ARMS = {"A": ["demo", "codes", "meds"], "B": ["demo", "codes", "meds", "labs", "report", "imaging"],
        "C": ["demo", "codes", "meds"], "D": ["demo", "codes", "meds", "labs", "report", "imaging"]}
SYS_A = paths.SYS_PROMPT
SYS_B = ("You are a clinical pharmacology expert reviewing an intensive-care medication list. "
         "Answer only with a single JSON object and nothing else.")
Q_A = ("Question: Does this patient have a documented condition in which {d} is contraindicated, or a documented "
       "finding that is a listed adverse effect of {d}? Answer yes or no.")
Q_B = ("Task: Exactly one of the candidate drugs has a knowledge-graph relation (contraindication or adverse effect) to a "
       "condition supported by this patient's documented evidence. Identify it and justify it with the evidence and the "
       "knowledge-graph path. Return one JSON object with these keys:\n"
       '  "selected_drug": the candidate name,\n'
       '  "ranking": all candidate names from most to least likely,\n'
       '  "patient_evidence_ids": list of evidence tags such as ["P3"],\n'
       '  "normalized_condition": the condition name as written in the knowledge-graph facts,\n'
       '  "kg_path": list of [drug, relation, condition] triples as written in the knowledge-graph facts,\n'
       '  "explanation": one sentence.\n'
       "Candidates: {cands}.")


def packet_lines(p, rel, name):
    out = []
    for r, phrase in (("contraindication", "is contraindicated in the following conditions"),
                      ("drug_effect", "has the following documented adverse effects")):
        lst = sorted(name.get(x, str(x)) for x in rel["effect" if r == "drug_effect" else "contra"].get(p, ()))
        if lst:
            out.append((r, phrase, lst))
    return out


def render_packet(cname, groups, keep):
    """Frozen prose style; `keep` = number of entries kept per relation list (round-robin truncation)."""
    parts = []
    for (r, phrase, lst), k in zip(groups, keep):
        if k <= 0:
            continue
        shown = lst[:k]; tail = "" if k >= len(lst) else f" (first {k} of {len(lst)} entries)"
        parts.append(f"Facts from a curated biomedical knowledge graph. {cname} {phrase} (complete list, {len(lst)} entries){tail}:\n"
                     + "\n".join("  " + x for x in shown))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="ranking_v1_d12"); ap.add_argument("--split", default="dev")
    ap.add_argument("--arm", default="D", choices=list(ARMS)); ap.add_argument("--policy", default="R4", choices=["R0", "R1", "R2", "R3", "R4", "R4q"])
    ap.add_argument("--control", default="none"); ap.add_argument("--budget", default="natural"); ap.add_argument("--max_ctx", type=int, default=8192)
    ap.add_argument("--reserve", type=int, default=16); ap.add_argument("--emb_dir", default=None); ap.add_argument("--r2_mode", default="kg_only")
    ap.add_argument("--model", default="q35"); ap.add_argument("--tag", default=None); ap.add_argument("--gen_reserve", type=int, default=512)
    args = ap.parse_args()
    tag = args.tag or f"{args.version}_{args.split}_{args.arm}{args.policy}_{args.control}_b{args.budget}"
    run = runlog.Run("exp2", f"contexts_{tag}", vars(args) | {"tag": tag})
    tok = tk.load_tokenizer(paths.MODELS[args.model])
    D = paths.RESTRICTED / "exp2" / args.version
    items = [json.loads(l) for l in open(D / "items.jsonl")]
    split = json.load(open(PUB / f"{args.version}_split.json"))
    keep_idx = set(split[args.split]) if args.split != "all" else {it["idx"] for it in items}
    items = [it for it in items if it["idx"] in keep_idx]
    units_by_h = {u["hadm"]: u["units"] for u in map(json.loads, open(D / "patient_units.jsonl"))}
    rel = fz.load_rel_cache(verify=False); name = rel["name"]
    f2p, d2p, rep_by_study, img_by_study = fz.load_alignment_and_findings(verify=False)
    findings = (rep_by_study, img_by_study); ihm = fz.load_ihm(verify=False); cui_name = fz.load_cui_names()
    kg = ctx1.KG("round_robin") if args.policy in ("R1", "R3") else None
    r2 = R2Selector(args.emb_dir, tok, mode=args.r2_mode) if args.policy == "R2" else None
    rng = np.random.default_rng(20260914)
    # packet-length-matched replacement pool for the matched_wrong_packets control (label-blind: applied to every candidate)
    all_drugs = sorted({c["pkg"] for it in items for c in it["candidates"]})
    plen = {p: sum(len(l) for _, _, l in packet_lines(p, rel, name)) for p in all_drugs}

    outdir = paths.RESTRICTED / "exp2" / "contexts" / tag; outdir.mkdir(parents=True, exist_ok=True); (PUB / "contexts").mkdir(exist_ok=True)
    agg = defaultdict(list)
    with open(outdir / "evalA.jsonl", "w") as fa, open(outdir / "evalB.jsonl", "w") as fb:
        for n_done, it in enumerate(items):
            if n_done % 100 == 0:
                print(f"  {tag}: {n_done}/{len(items)}", file=sys.stderr, flush=True)
            h = it["hadm"]; rec = ihm["recs"][h]; units = units_by_h[h]
            cands = it["candidates"]; cnames = [c["name"] for c in cands]
            blocks = fz.make_blocks(rec, h, "contra_labs", cands[0]["pkg"], cands[0]["mimic_name"], ihm, findings, cui_name, rel)
            chans = [c for c in ARMS[args.arm] if not (args.control.startswith("no_") and args.control[3:] == c)]
            pat_blocks = [blocks[b] for b in chans if blocks.get(b)]
            if args.control == "no_provenance":
                pat_blocks = [b.split(": ", 1)[1] if ": " in b else b for b in pat_blocks]
            patient_text = "\n\n".join(pat_blocks)
            # tagged rendering for eval B (local tags, no identifiers)
            vis = set(ctx1.visible_uids(units)); tagged, tagmap = [], {}
            for u in units:
                if u["channel"] not in chans or u["uid"] not in vis:
                    continue
                t = f"P{len(tagged) + 1}"; tagmap[t] = u["uid"]
                tagged.append((f"[{t}] " if args.control != "no_provenance" else "") + (unit_chunk(u) if args.control != "no_provenance" else u["text"]))
            patient_text_B = "Patient evidence:\n" + "\n".join(tagged)
            cand_line = "Candidate drugs received by this patient: " + "; ".join(f"{i + 1}. {n}" for i, n in enumerate(cnames)) + "."
            # ---- KG side ----
            kg_groups = []                       # per candidate: list of (line, edge) in deterministic order
            if args.arm in ("C", "D") and args.policy != "R0":
                for c in cands:
                    p = c["pkg"]
                    if args.control == "matched_wrong_packets":
                        pool = [q for q in all_drugs if q != p and q not in {x["pkg"] for x in cands}]
                        p = min(pool, key=lambda q: (abs(plen[q] - plen[c["pkg"]]), q))
                    if args.policy in ("R4", "R4q"):
                        groups = packet_lines(p, rel, name)
                        if args.control == "gold_edge_removed" and c["positive"]:
                            gold = set(it["gold"]["condition_names"])
                            groups = [(r, ph, [x for x in lst if x not in gold]) for r, ph, lst in groups]
                        kg_groups.append({"kind": "packet", "cname": c["name"], "groups": groups,
                                          "edges": [(c["pkg"], r, x) for r, _, lst in groups for x in lst]})
                    elif args.policy in ("R1", "R3"):
                        lines = []
                        if args.policy == "R1":
                            d = kg.description(p)
                            if d: lines.append((d, None))
                        for t in kg.triples(p):
                            lines.append((t["line"], (c["pkg"], t["rel"], name.get(t["y"], str(t["y"])))))
                        kg_groups.append({"kind": "triples", "cname": c["name"], "lines": lines})
                    elif args.policy == "R2":
                        fake_it = {"drug_pkg": p, "drug_pkg_name": c["name"]}
                        chunks = [(t["line"], "kg", (c["pkg"], t["rel"], name.get(t["y"], str(t["y"])))) for t in r2_triples(r2, p, kg or ctx1.KG("round_robin"))]
                        kg_groups.append({"kind": "r2", "cname": c["name"], "chunks": chunks, "query": Q_A.format(d=c["name"])})
                if args.control == "shuffled_packets":
                    perm = rng.permutation(len(kg_groups)); names_ = [g["cname"] for g in kg_groups]
                    kg_groups = [kg_groups[i] for i in perm]
                    for g, nm in zip(kg_groups, names_):
                        g["cname"] = nm          # packets now attached to the wrong candidate names
            budget = (args.max_ctx - args.reserve) if args.budget == "natural" else int(args.budget)
            fixedA = max(tk.n_tokens(tok, tk.render_chat(tok, SYS_A, Q_A.format(d=n))) for n in cnames) + tk.n_tokens(tok, cand_line + "\n\n")
            fixedB = tk.n_tokens(tok, tk.render_chat(tok, SYS_B, Q_B.format(cands="; ".join(cnames)))) + tk.n_tokens(tok, cand_line + "\n\n") + args.gen_reserve
            pat_tokA = tk.n_tokens(tok, patient_text + "\n\n"); pat_tokB = tk.n_tokens(tok, patient_text_B + "\n\n")
            kg_text, kg_edges_shown, trunc, untr = build_kg_text(kg_groups, budget - max(fixedA + pat_tokA, fixedB + pat_tokB), tok, args)
            # ---- eval A prompts ----
            recs = []
            for c in cands:
                if args.policy == "R4q":            # query-conditioned per candidate: only the queried candidate's packet (the binary-task R4 form)
                    gq = next((g for g in kg_groups if g["cname"] == c["name"]), None)
                    kg_text = render_packet(gq["cname"], gq["groups"], [len(l) for _, _, l in gq["groups"]]) if gq else ""
                user = "\n\n".join(x for x in (patient_text, cand_line, kg_text) if x) + "\n\n" + Q_A.format(d=c["name"])
                recs.append({"idx": it["idx"], "cand_pkg": c["pkg"], "cand_name": c["name"], "positive": c["positive"], "user": user})
            nA = tk.n_tokens(tok, tk.render_chat(tok, SYS_A, recs[0]["user"]))
            for r in recs:
                fa.write(json.dumps(r) + "\n")
            userB = "\n\n".join(x for x in (patient_text_B, cand_line, kg_text) if x) + "\n\n" + Q_B.format(cands="; ".join(cnames))
            nB = tk.n_tokens(tok, tk.render_chat(tok, SYS_B, userB))
            fb.write(json.dumps({"idx": it["idx"], "user": userB, "tagmap": tagmap, "candidates": cnames}) + "\n")
            gold_e = {(it["gold"]["selected_pkg"], e[1], name.get(e[2], str(e[2]))) for e in it["gold"]["kg_edges"]}
            st = {"prompt_tokens_A": nA, "prompt_tokens_B": nB, "truncated": trunc, "untruncated_kg_tokens": untr,
                  "n_kg_edges": len(kg_edges_shown), "recall_gold_edge": (bool(gold_e & kg_edges_shown) if kg_groups else 0.0),
                  "over_ctx": int(max(nA, nB + args.gen_reserve) > args.max_ctx)}
            for k, v in st.items():
                agg[k].append(float(v))
    summ = {"tag": tag, "n_items": len(items), "means": {k: round(float(np.mean(v)), 4) for k, v in agg.items()},
            "max_prompt_tokens_A": int(max(agg["prompt_tokens_A"])), "max_prompt_tokens_B": int(max(agg["prompt_tokens_B"]))}
    json.dump(summ, open(PUB / "contexts" / f"{tag}_summary.json", "w"), indent=2)
    run.add_output(outdir / "evalA.jsonl"); run.add_output(outdir / "evalB.jsonl"); run.finish(counts=summ)
    print(json.dumps(summ, indent=1))


def r2_triples(r2, p, kg):
    return kg.triples(p)


def build_kg_text(kg_groups, avail, tok, args):
    """Round-robin truncation across candidates (and across relation lists within a packet)."""
    if not kg_groups:
        return "", set(), False, 0
    if kg_groups[0]["kind"] == "packet":
        # entries per (candidate, relation list); allocate round-robin until the budget is used
        lists = [(gi, li, len(lst)) for gi, g in enumerate(kg_groups) for li, (_, _, lst) in enumerate(g["groups"])]
        keep = {(gi, li): 0 for gi, li, _ in lists}
        def render():
            return "\n\n".join(render_packet(g["cname"], g["groups"], [keep[(gi, li)] for li in range(len(g["groups"]))]) for gi, g in enumerate(kg_groups))
        full = {(gi, li): n for gi, li, n in lists}
        keep = dict(full); text = render(); total = tk.n_tokens(tok, text)
        untr = total; trunc = False
        if total > avail:
            trunc = True
            # binary search on a common per-list cap k
            lo, hi = 0, max(full.values())
            while lo < hi:
                mid = (lo + hi + 1) // 2
                keep = {k_: min(v, mid) for k_, v in full.items()}
                if tk.n_tokens(tok, render()) <= avail:
                    lo = mid
                else:
                    hi = mid - 1
            keep = {k_: min(v, lo) for k_, v in full.items()}; text = render()
        shown = set()
        for gi, g in enumerate(kg_groups):
            for li, (r, _, lst) in enumerate(g["groups"]):
                pk = next(e[0] for e in g["edges"]) if g["edges"] else None
                shown |= {(pk, r, x) for x in lst[:keep[(gi, li)]]}
        return text, shown, trunc, untr
    # triples / r2: interleave candidates' lines round-robin, then cut by budget
    seqs = []
    for g in kg_groups:
        if g["kind"] == "triples":
            seqs.append([(f"{g['cname']}: " + ln, e) for ln, e in g["lines"]])
        else:
            seqs.append([(f"{g['cname']}: " + ln, e) for ln, _, e in g["chunks"]])
    merged = ctx1._interleave(seqs)
    lines = [m[0] for m in merged]
    hdr = "Knowledge-graph retrieval results (typed one-hop neighbourhood of each candidate):"
    pk = tk.pack(tok, lines, avail - tk.n_tokens(tok, hdr + "\n"))
    kept = [merged[k] for k in pk["kept"]]
    text = (hdr + "\n" + "\n".join(m[0] for m in kept)) if kept else ""
    return text, {m[1] for m in kept if m[1] is not None}, pk["truncated"], pk["untruncated_tokens"]


if __name__ == "__main__":
    main()
