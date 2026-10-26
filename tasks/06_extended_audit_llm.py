"""
06_extended_audit_llm.py   --   env: llm_qwen   [GPU cluster]
================================================================================
Run the extended medication-audit benchmark: hop ladder + the control arms that
establish the model is genuinely reading the graph rather than benefiting from
extra context.

ARMS
  A_flat        question only
  B_concept     + entity names as graph nodes (no edges)
  C_true        + the real retrieved neighbourhood
  C_shuffled    + a neighbourhood belonging to a DIFFERENT item. Same format, same
                approximate token count, wrong facts. This is the strict control for
                "does any extra context help?" -- if C_true > C_shuffled the gain is
                about content, not length.
  C_corrupted   + the real edge STRUCTURE with entity names replaced by plausible
                wrong ones. If the model scores well here it is pattern-matching
                the format, not reading the facts.
  C_only_<rel>  single-relation ablations -- which layer carries the gain.

--cite additionally requires the model to quote the edge it used, and scores
whether that edge was actually present in the context. Citation validity is NOT
tautological: a model can cite a real edge that does not support its answer, and
this catches it.

Usage:
  python 06_extended_audit_llm.py --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
  python 06_extended_audit_llm.py --model Qwen/Qwen3-8B --tag 8b
  python 06_extended_audit_llm.py --cite --arms A_flat C_true --tag cite
  python 06_extended_audit_llm.py --dry_run
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT  = HERE / "outputs" / "audit"

SYS = ('You are a clinical pharmacology expert. Answer with ONLY a JSON object: '
       '{"answer": "yes"|"no"}. Answer "yes" only if you are confident.')
SYS_CITE = ('You are a clinical pharmacology expert. Answer with ONLY a JSON object: '
            '{"answer": "yes"|"no", "evidence": "<the single line of provided evidence you '
            'relied on, copied exactly, or empty if none>"}.')


def balanced_acc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    se = p[y == 1].mean() if (y == 1).any() else np.nan
    sp = 1 - p[y == 0].mean() if (y == 0).any() else np.nan
    return float((se + sp) / 2)


def paired(y, pa, pb, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        a, b = balanced_acc(y[i], pa[i]), balanced_acc(y[i], pb[i])
        if not (np.isnan(a) or np.isnan(b)): d.append(b - a)
    if not d: return None
    d = np.asarray(d); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"mean": round(float(d.mean()), 4),
            "ci": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p": round(float(min(p, 1.0)), 4)}


def edges_for(it, arm):
    if arm == "C_true":      return it.get("edges_true", [])
    if arm == "C_shuffled":  return it.get("edges_shuffled", [])
    if arm == "C_corrupted": return it.get("edges_corrupted", [])
    if arm.startswith("C_only_"): return it.get("edges_only_" + arm[len("C_only_"):], [])
    return []


def build(it, arm):
    q = it["question"]
    if arm == "A_flat":
        return q
    if arm == "B_concept":
        return ("The following are entities in a curated medical knowledge graph: "
                + ", ".join(it["entity_names"][:16]) + ".\n\n" + q)
    e = edges_for(it, arm)
    blk = ("Facts retrieved from a curated medical knowledge graph:\n" + "\n".join(f"  {x}" for x in e)) \
        if e else "The knowledge graph returned no facts linking these entities."
    return f"{blk}\n\n{q}"


def parse(t, cite=False):
    body = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL)
    for raw in reversed(re.findall(r"\{[^{}]*\}", body, re.DOTALL)):
        try: o = json.loads(raw)
        except json.JSONDecodeError: continue
        a = str(o.get("answer", "")).strip().lower()
        if a.startswith(("y", "n")):
            return (1 if a.startswith("y") else 0), True, str(o.get("evidence", "") or "")
    m = re.search(r"\b(yes|no)\b", body, re.I)
    if m: return (1 if m.group(1).lower() == "yes" else 0), True, ""
    return 0, False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    ap.add_argument("--arms", nargs="+",
                    default=["A_flat", "B_concept", "C_true", "C_shuffled", "C_corrupted"])
    ap.add_argument("--ablate", action="store_true", help="also run per-relation ablation arms")
    ap.add_argument("--cite", action="store_true")
    ap.add_argument("--benchmark", default="extended_benchmark.jsonl",
                    help="benchmark file in outputs/audit/ to run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=4096)  # raise for complete retrieval
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""

    items = [json.loads(l) for l in open(OUT / args.benchmark)]
    arms = list(args.arms)
    if args.ablate:
        rels = sorted({r for it in items for r in it["relations"] if r != "atc"})
        arms += [f"C_only_{r}" for r in rels]
    print(f"{len(items)} items | arms: {arms} | cite={args.cite} | model={args.model}", flush=True)

    gen = None
    if not args.dry_run:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, max_model_len=args.max_model_len, trust_remote_code=True)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        tok = llm.get_tokenizer()
        sysmsg = SYS_CITE if args.cite else SYS

        def gen(ps):
            txt = []
            for p in ps:
                msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": p}]
                try:
                    txt.append(tok.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False))
                except TypeError:
                    txt.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            return [o.outputs[0].text for o in llm.generate(txt, sp)]

    preds, cites, bad = {}, {}, {}
    for arm in arms:
        ps = [build(it, arm) for it in items]
        if args.dry_run:
            raw = [json.dumps({"answer": it["answer"] if arm == "C_true" else "no",
                               "evidence": (it["edges_true"][0] if it["edges_true"] else "")}) for it in items]
        else:
            raw = gen(ps)
        pr, cv, nb = [], [], 0
        with open(OUT / f"ext_preds_{arm}{sfx}.jsonl", "w") as w:
            for it, t in zip(items, raw):
                v, ok, ev = parse(t, args.cite)
                if not ok: nb += 1
                pr.append(v)
                present = bool(ev) and any(ev.strip()[:40] in e for e in edges_for(it, arm)) if ev else None
                cv.append(present)
                w.write(json.dumps({"family": it["family"], "hops": it["hops"],
                                    "stratum": it["prevalence_stratum"], "answer": it["answer"],
                                    "pred": v, "arm": arm, "evidence": ev[:200],
                                    "cite_present": present, "raw": t[:300]}) + "\n")
        preds[arm] = pr; cites[arm] = cv; bad[arm] = nb
        print(f"  [{arm}] done ({nb} unparsed)", flush=True)

    y = [1 if it["answer"] == "yes" else 0 for it in items]
    fams = list(dict.fromkeys(it["family"] for it in items))
    res = {"model": args.model, "cite": args.cite, "n_items": len(items), "unparsed": bad,
           "arms": arms, "families": {}, "by_hop": {}, "by_stratum": {}}

    def block(mask):
        idx = [i for i, keep in enumerate(mask) if keep]
        if len(idx) < 10: return None
        yy = [y[i] for i in idx]
        if len(set(yy)) < 2: return None
        o = {a: round(balanced_acc(yy, [preds[a][i] for i in idx]), 4) for a in arms}
        for a in arms:
            if a != "A_flat":
                o[f"{a}_minus_A"] = paired(yy, [preds["A_flat"][i] for i in idx],
                                           [preds[a][i] for i in idx])
        if "C_true" in arms and "C_shuffled" in arms:
            o["C_true_minus_shuffled"] = paired(yy, [preds["C_shuffled"][i] for i in idx],
                                                [preds["C_true"][i] for i in idx])
        o["n"] = len(idx)
        return o

    for fam in fams: res["families"][fam] = block([it["family"] == fam for it in items])
    for h in sorted({it["hops"] for it in items}):
        res["by_hop"][str(h)] = block([it["hops"] == h for it in items])
    for s in ("common", "uncommon", "rare"):
        res["by_stratum"][s] = block([it["prevalence_stratum"] == s for it in items])
    if args.cite:
        res["citation"] = {a: {"rate_with_evidence": round(float(np.mean([c is not None for c in cites[a]])), 3),
                               "valid_when_cited": round(float(np.mean([c for c in cites[a] if c is not None])), 3)
                               if any(c is not None for c in cites[a]) else None} for a in arms}
    json.dump(res, open(OUT / f"extended_results{sfx}.json", "w"), indent=2)

    print(f"\n=== EXTENDED AUDIT  ({args.model}) ===")
    print("balanced accuracy; chance = 0.500\n")
    hdr = f"{'family':24s} {'hops':>4} " + " ".join(f"{a[:11]:>11}" for a in arms) + f"  {'C_true-A':>10}"
    print(hdr)
    for fam in fams:
        r = res["families"][fam]
        if not r: continue
        h = next(it["hops"] for it in items if it["family"] == fam)
        d = r.get("C_true_minus_A")
        ds = f"{d['mean']:+.3f}" if d else "-"
        print(f"{fam:24s} {h:>4} " + " ".join(f"{r[a]:>11.3f}" for a in arms) + f"  {ds:>10}")
    print("\n--- HOP LADDER (does the KG advantage grow with reasoning depth?) ---")
    for h, r in res["by_hop"].items():
        if not r: continue
        d = r.get("C_true_minus_A"); sh = r.get("C_true_minus_shuffled")
        if not d: continue
        line = f"  {h}-hop  n={r['n']:4d}  A {r['A_flat']:.3f} -> C {r['C_true']:.3f}   C-A {d['mean']:+.3f} p={d['p']}"
        if sh: line += f"   C-shuffled {sh['mean']:+.3f} p={sh['p']}"
        print(line)
    print("\n--- PREVALENCE STRATA (is the gain largest where the prior is thinnest?) ---")
    for s, r in res["by_stratum"].items():
        if not r: continue
        d = r.get("C_true_minus_A")
        print(f"  {s:9s} n={r['n']:4d}  A {r['A_flat']:.3f} -> C {r['C_true']:.3f}  C-A {d['mean']:+.3f} p={d['p']}")
    if args.cite:
        print("\n--- CITATION VALIDITY ---")
        for a, c in res["citation"].items():
            print(f"  {a:14s} cited {c['rate_with_evidence']:.2f} | valid when cited {c['valid_when_cited']}")
    print(f"\nwrote -> {OUT}/extended_results{sfx}.json")


if __name__ == "__main__":
    main()
