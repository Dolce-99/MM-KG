"""
19_composite_wide_llm.py   --   env: llm_qwen   [GPU cluster]
================================================================================
Two experiments over the wide composite benchmark (4 families, 3 PrimeKG
relation layers, 3,200 balanced items).

EXPERIMENT A -- the 2x2
                      no KG        KG
     no modality      A_none       C_kg
     modality         B_mm         D_both
  Headline: the interaction (D-C) - (B-A). Neither a KG-only nor a
  multimodal-only method can produce this quantity.

EXPERIMENT B -- leave-one-out with the KG ALWAYS ON
  Starting from D_both, drop one patient channel at a time. This answers a
  different question from the 2x2: given that the knowledge graph is available,
  WHICH patient channel actually carries the signal? The prediction is that each
  family responds to its own occult-evidence modality and not the others --
  the multimodal analogue of the per-relation ablation in the audit benchmark.

  contribution = balanced_acc(D_both) - balanced_acc(D_no_X)

Both experiments read the same nine score files, so one job produces both.

Usage:
  python 19_composite_wide_llm.py --arms A_none B_mm C_kg D_both D_no_labs ...
  python 19_composite_wide_llm.py --analyze_only
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT  = HERE / "outputs" / "mm"

CONV = ["demo", "codes", "meds"]            # channels a conventional system has
MM   = ["labs", "report", "imaging"]        # the patient modalities
FULL = CONV + MM + ["kg"]

ARMS = {
    "A_none":       CONV,
    "B_mm":         CONV + MM,
    "C_kg":         CONV + ["kg"],
    "D_both":       FULL,
    "D_no_labs":    [b for b in FULL if b != "labs"],
    "D_no_report":  [b for b in FULL if b != "report"],
    "D_no_imaging": [b for b in FULL if b != "imaging"],
    "D_no_codes":   [b for b in FULL if b != "codes"],
    "D_no_meds":    [b for b in FULL if b != "meds"],
    # alternative KG rendering (typed graph-RAG triples; requires the v2
    # benchmark's kg_alt block). Same facts, different syntax -- tests whether
    # the 2x2 is an artifact of one prompt format.
    "C_alt":        CONV + ["kg_alt"],
    "D_alt":        CONV + MM + ["kg_alt"],
}
LOO = ["D_no_labs", "D_no_report", "D_no_imaging", "D_no_codes", "D_no_meds"]

SYS = ("You are a clinical pharmacology expert reviewing an intensive-care "
       "medication list. Answer the question with a single word: yes or no.")


def balanced_acc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    se = p[y == 1].mean() if (y == 1).any() else np.nan
    sp = 1 - p[y == 0].mean() if (y == 0).any() else np.nan
    return float((se + sp) / 2)


def _rank_avg(s):
    n = len(s); o = np.argsort(s, kind="mergesort"); sv = s[o]
    st = np.empty(n, bool); st[0] = True
    np.not_equal(sv[1:], sv[:-1], out=st[1:])
    g = np.cumsum(st) - 1; gs = np.flatnonzero(st); gc = np.diff(np.append(gs, n))
    r = np.empty(n, float); r[o] = (gs + (gc - 1) / 2.0 + 1.0)[g]
    return r


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    return float((_rank_avg(s)[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def boot(y, cells, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        x = fn(y[i], {k: np.asarray(c)[i] for k, c in cells.items()})
        if not math.isnan(x):
            v.append(x)
    if not v:
        return None
    v = np.asarray(v); p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return {"mean": round(float(v.mean()), 4),
            "ci": [round(float(np.percentile(v, 2.5)), 4),
                   round(float(np.percentile(v, 97.5)), 4)],
            "p": round(float(min(p, 1.0)), 4)}


def build(it, arm):
    parts = [it["blocks"][b] for b in ARMS[arm] if it["blocks"].get(b)]
    return "\n\n".join(parts) + "\n\n" + it["question"]


def yes_prob(lp, tok):
    py = pn = 0.0
    for tid, o in lp.items():
        s = (o.decoded_token if hasattr(o, "decoded_token") else tok.decode([tid])).strip().lower()
        pr = math.exp(o.logprob if hasattr(o, "logprob") else o)
        if s.startswith("yes"):
            py += pr
        elif s.startswith("no"):
            pn += pr
    return (py / (py + pn), True) if py + pn > 0 else (0.5, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    ap.add_argument("--benchmark", default="composite_benchmark_wide.jsonl")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tag", default="wide")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(OUT / args.benchmark)]
    y = np.array([1 if it["answer"] == "yes" else 0 for it in items])
    fams = list(dict.fromkeys(it["family"] for it in items))
    print(f"{len(items)} items | yes {int(y.sum())} ({y.mean():.1%}) | families {fams}", flush=True)

    todo = [a for a in args.arms if not (OUT / f"cw_{args.tag}_{a}.npy").exists()]
    if not args.analyze_only and not todo:
        print("all requested arms already scored; skipping model load", flush=True)
        args.analyze_only = True

    if not args.analyze_only:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, max_model_len=args.max_model_len, trust_remote_code=True)
        tok = llm.get_tokenizer()
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        for arm in todo:
            prompts = []
            for it in items:
                msgs = [{"role": "system", "content": SYS},
                        {"role": "user", "content": build(it, arm)}]
                try:
                    prompts.append(tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
                except TypeError:
                    prompts.append(tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True))
            outs = llm.generate(prompts, sp)
            sc, bad = [], 0
            for o in outs:
                lps = o.outputs[0].logprobs
                if not lps:
                    sc.append(0.5); bad += 1; continue
                v, ok = yes_prob(lps[0], tok)
                sc.append(v); bad += (not ok)
            np.save(OUT / f"cw_{args.tag}_{arm}.npy", np.array(sc))
            print(f"  [{arm}] done, {bad} without a yes/no token", flush=True)

    S = {a: np.load(OUT / f"cw_{args.tag}_{a}.npy")
         for a in ARMS if (OUT / f"cw_{args.tag}_{a}.npy").exists()}
    need = {"A_none", "B_mm", "C_kg", "D_both"}
    if not need <= set(S):
        print(f"\nhave {sorted(S)}; need at least {sorted(need)}"); return

    res = {"model": args.model, "n": len(items), "arms": sorted(S), "families": {}}
    groups = [("ALL", np.ones(len(items), bool))] + \
             [(f, np.array([it["family"] == f for it in items])) for f in fams]

    for gname, mask in groups:
        yy = y[mask]
        cells = {a: (S[a][mask] > 0.5).astype(int) for a in S}
        blk = {"n": int(mask.sum()),
               "cells": {a: {"bal_acc": round(balanced_acc(yy, cells[a]), 4),
                             "auroc": round(auroc(yy, S[a][mask]), 4),
                             "sens": round(float(cells[a][yy == 1].mean()), 4),
                             "spec": round(float(1 - cells[a][yy == 0].mean()), 4)}
                         for a in S}}
        blk["interaction"] = boot(yy, cells, lambda yb, c:
            (balanced_acc(yb, c["D_both"]) - balanced_acc(yb, c["C_kg"]))
            - (balanced_acc(yb, c["B_mm"]) - balanced_acc(yb, c["A_none"])))
        if "C_alt" in cells and "D_alt" in cells:
            blk["interaction_alt"] = boot(yy, cells, lambda yb, c:
                (balanced_acc(yb, c["D_alt"]) - balanced_acc(yb, c["C_alt"]))
                - (balanced_acc(yb, c["B_mm"]) - balanced_acc(yb, c["A_none"])))
        blk["gains"] = {
            "mm_alone": boot(yy, cells, lambda yb, c: balanced_acc(yb, c["B_mm"]) - balanced_acc(yb, c["A_none"])),
            "kg_alone": boot(yy, cells, lambda yb, c: balanced_acc(yb, c["C_kg"]) - balanced_acc(yb, c["A_none"])),
            "both":     boot(yy, cells, lambda yb, c: balanced_acc(yb, c["D_both"]) - balanced_acc(yb, c["A_none"])),
            "mm_given_kg": boot(yy, cells, lambda yb, c: balanced_acc(yb, c["D_both"]) - balanced_acc(yb, c["C_kg"])),
        }
        blk["leave_one_out"] = {
            a: boot(yy, cells, (lambda arm: lambda yb, c:
                    balanced_acc(yb, c["D_both"]) - balanced_acc(yb, c[arm]))(a))
            for a in LOO if a in S}
        res["families"][gname] = blk

    dest = OUT / f"composite_wide_results_{args.tag}.json"
    json.dump(res, open(dest, "w"), indent=2)

    # ------------------------------------------------------------ printout
    print("\n" + "=" * 78)
    print("EXPERIMENT A -- the 2x2   (balanced accuracy, chance 0.500)")
    print("=" * 78)
    print(f"{'family':16s}{'A none':>9}{'B mm':>9}{'C kg':>9}{'D both':>9}"
          f"{'INTERACTION':>26}{'p':>8}")
    for g, b in res["families"].items():
        c = b["cells"]; i = b["interaction"]
        ci = f"{i['mean']:+.4f} [{i['ci'][0]:+.4f},{i['ci'][1]:+.4f}]" if i else "-"
        print(f"{g:16s}{c['A_none']['bal_acc']:9.3f}{c['B_mm']['bal_acc']:9.3f}"
              f"{c['C_kg']['bal_acc']:9.3f}{c['D_both']['bal_acc']:9.3f}"
              f"{ci:>26}{i['p'] if i else '-':>8}")
        ia = b.get("interaction_alt")
        if ia and "C_alt" in c:
            print(f"{'  alt-KG render':16s}{'':9s}{'':9s}{c['C_alt']['bal_acc']:9.3f}"
                  f"{c['D_alt']['bal_acc']:9.3f}"
                  f"{ia['mean']:+.4f} [{ia['ci'][0]:+.4f},{ia['ci'][1]:+.4f}]".rjust(26)
                  + f"{ia['p']:>8}")

    print("\n  gains vs A_none:")
    for g, b in res["families"].items():
        gg = b["gains"]
        print(f"    {g:14s} mm {gg['mm_alone']['mean']:+.3f}(p={gg['mm_alone']['p']})  "
              f"kg {gg['kg_alone']['mean']:+.3f}(p={gg['kg_alone']['p']})  "
              f"both {gg['both']['mean']:+.3f}  |  mm GIVEN kg {gg['mm_given_kg']['mean']:+.3f}"
              f"(p={gg['mm_given_kg']['p']})")

    if any(a in S for a in LOO):
        print("\n" + "=" * 78)
        print("EXPERIMENT B -- leave-one-out, PrimeKG always on")
        print("  contribution = balacc(D_both) - balacc(arm); positive => channel helps")
        print("=" * 78)
        cols = [a for a in LOO if a in S]
        print(f"{'family':16s}" + "".join(f"{a.replace('D_no_',''):>13}" for a in cols))
        for g, b in res["families"].items():
            row = ""
            for a in cols:
                d = b["leave_one_out"].get(a)
                row += f"{d['mean']:+9.3f}{'*' if d and d['p'] < 0.05 else ' ':<4}" if d else f"{'-':>13}"
            print(f"{g:16s}{row}")
        print("  * p < 0.05")
    print(f"\nwrote -> {dest}")


if __name__ == "__main__":
    main()
