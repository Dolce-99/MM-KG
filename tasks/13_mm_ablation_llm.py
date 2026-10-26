"""
13_mm_ablation_llm.py   --   env: llm_qwen   [GPU cluster]
================================================================================
Does MULTIMODAL patient data help an LLM?

Leave-one-out modality ablation on ICU in-hospital mortality, mirroring the
phase-2 GNN ablation so the two consumers are compared on the SAME task, the
SAME cohort and the SAME labels.  The label is an observed outcome, independent
of every KG edge -- so nothing here is circular.

ARMS
  full         every modality
  no_imaging   drop automated chest-film findings
  no_report    drop radiology-report findings
  no_labs      drop laboratory values
  no_history   drop prior-visit conditions and drugs
  no_meds      drop current medications
  no_kg        drop the PrimeKG facts
  demo_only    demographics alone (floor)

SCORING -- deliberately not a self-reported risk percentage.  Asking the model
for a number yields ~21 distinct values with ~49% ties (measured in the earlier
medication-safety runs), which caps AUROC and would hide effects the size we
expect here (the GNN's imaging effect is +0.020).  Instead we read the
probability mass on the "yes" token at the first generated position, giving a
continuous, tie-free score.  These AUROCs are therefore NOT comparable to the
earlier medsafety numbers.

REPORTING -- metrics are computed twice from the same run:
  (a) ENTIRE cohort   -- every ICU stay
  (b) all-modality stratum -- stays where imaging AND report are both present
Ablating a modality a patient does not have is a no-op, so (a) mechanically
dilutes every effect toward zero; (b) is where the question is well posed.
Both are reported.  Nothing is subsampled.

Usage:
  python 13_mm_ablation_llm.py --arms full no_imaging        # one or more arms
  python 13_mm_ablation_llm.py --analyze_only                # metrics from saved scores
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT  = HERE / "outputs" / "mm"

ORDER = ["demo", "history", "meds", "labs", "imaging", "report", "kg"]
# Arm names must NOT begin with "-": argparse would read them as option flags
# rather than values for --arms.
ARMS = {
    "full":       ORDER,
    "no_imaging": [m for m in ORDER if m != "imaging"],
    "no_report":  [m for m in ORDER if m != "report"],
    "no_labs":    [m for m in ORDER if m != "labs"],
    "no_history": [m for m in ORDER if m != "history"],
    "no_meds":    [m for m in ORDER if m != "meds"],
    "no_kg":      [m for m in ORDER if m != "kg"],
    "demo_only":  ["demo"],
    # single-channel arms for the "imaging vs full patient data, given the KG"
    # comparison. demo stays in so the model always knows age/sex/admission type;
    # what varies is which evidence channel accompanies it.
    "img_only":   ["demo", "imaging"],
    "img_kg":     ["demo", "imaging", "kg"],
    "kg_only":    ["demo", "kg"],
}

SYS = ("You are an experienced intensive-care physician. You will be shown "
       "information about a patient in intensive care. Answer the question with "
       "a single word: yes or no.")
QUESTION = ("Question: Will this patient die during this hospital admission? "
            "Answer yes or no.")


# ------------------------------------------------------------------- metrics
def _rank_avg(s):
    """Tie-averaged ranks, fully vectorised (a Python tie-loop here costs ~800M
    interpreted iterations across the bootstrap and dominates runtime)."""
    n = len(s)
    o = np.argsort(s, kind="mergesort")
    sv = s[o]
    start = np.empty(n, dtype=bool)
    start[0] = True
    np.not_equal(sv[1:], sv[:-1], out=start[1:])
    grp = np.cumsum(start) - 1
    gstart = np.flatnonzero(start)
    gcount = np.diff(np.append(gstart, n))
    avg = gstart + (gcount - 1) / 2.0 + 1.0
    r = np.empty(n, dtype=float)
    r[o] = avg[grp]
    return r


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = _rank_avg(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def ap(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if (y == 1).sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def paired(y, sa, sb, n=2000, seed=0):
    """Bootstrap over admissions of AUROC(b) - AUROC(a)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); sa = np.asarray(sa); sb = np.asarray(sb)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a, b = auroc(y[i], sa[i]), auroc(y[i], sb[i])
        if not (math.isnan(a) or math.isnan(b)):
            d.append(b - a)
    if not d:
        return None
    d = np.asarray(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"mean": round(float(d.mean()), 4),
            "ci": [round(float(np.percentile(d, 2.5)), 4),
                   round(float(np.percentile(d, 97.5)), 4)],
            "p": round(float(min(p, 1.0)), 4)}


# ------------------------------------------------------------------- prompts
def build(bundle, arm):
    parts = [bundle["blocks"][m] for m in ARMS[arm] if bundle["blocks"].get(m)]
    return "\n\n".join(parts) + "\n\n" + QUESTION


def yes_prob(logprob_dict, tok):
    """P(yes) / (P(yes)+P(no)) from the first-position logprobs."""
    py = pn = 0.0
    for tid, lp in logprob_dict.items():
        s = (lp.decoded_token if hasattr(lp, "decoded_token") else tok.decode([tid]))
        s = s.strip().lower()
        p = math.exp(lp.logprob if hasattr(lp, "logprob") else lp)
        if s.startswith("yes"):
            py += p
        elif s.startswith("no"):
            pn += p
    if py + pn == 0:
        return 0.5, False
    return py / (py + pn), True


# ---------------------------------------------------------------------- main
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    ap_.add_argument("--bundles", default="mm_bundles_ihm.jsonl")
    ap_.add_argument("--arms", nargs="+", default=list(ARMS))
    ap_.add_argument("--max_model_len", type=int, default=8192)
    ap_.add_argument("--tag", default="ihm")
    ap_.add_argument("--analyze_only", action="store_true")
    args = ap_.parse_args()

    items = [json.loads(l) for l in open(OUT / args.bundles)]
    y = np.array([it["label"] for it in items])
    print(f"{len(items)} admissions | pos {int(y.sum())} ({y.mean():.3%})", flush=True)

    todo = [a for a in args.arms
            if not (OUT / f"mm_scores_{args.tag}_{a}.npy").exists()]
    if not args.analyze_only and not todo:
        print(f"all requested arms already scored ({', '.join(args.arms)}); "
              f"skipping model load", flush=True)
        args.analyze_only = True

    if not args.analyze_only:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, max_model_len=args.max_model_len,
                  trust_remote_code=True)
        tok = llm.get_tokenizer()
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

        for arm in todo:
            dest = OUT / f"mm_scores_{args.tag}_{arm}.npy"
            prompts = []
            for it in items:
                msgs = [{"role": "system", "content": SYS},
                        {"role": "user", "content": build(it, arm)}]
                try:
                    prompts.append(tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False))
                except TypeError:
                    prompts.append(tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True))
            outs = llm.generate(prompts, sp)
            scores, nbad = [], 0
            for o in outs:
                lps = o.outputs[0].logprobs
                if not lps:
                    scores.append(0.5); nbad += 1; continue
                s, ok = yes_prob(lps[0], tok)
                scores.append(s); nbad += (not ok)
            np.save(dest, np.array(scores))
            print(f"  [{arm}] done, {nbad} without a yes/no token -> {dest.name}",
                  flush=True)

    # ------------------------------------------------------------- analysis
    S = {}
    for arm in ARMS:
        f = OUT / f"mm_scores_{args.tag}_{arm}.npy"
        if f.exists():
            S[arm] = np.load(f)
    if "full" not in S:
        print("\n'full' arm not yet run -- no comparison possible."); return

    strata = {
        "entire_cohort": np.ones(len(items), bool),
        "all_modality":  np.array([it["avail"]["imaging"] and it["avail"]["report"]
                                   for it in items]),
        "imaging_present": np.array([it["avail"]["imaging"] for it in items]),
    }
    res = {"model": args.model, "task": args.tag, "n": len(items),
           "n_pos": int(y.sum()), "arms": list(S), "strata": {}}

    for sname, mask in strata.items():
        yy = y[mask]
        blk = {"n": int(mask.sum()), "n_pos": int(yy.sum()), "arms": {}}
        for arm, sc in S.items():
            e = {"auroc": round(auroc(yy, sc[mask]), 4),
                 "ap": round(ap(yy, sc[mask]), 4),
                 "n_distinct_scores": int(len(np.unique(sc[mask])))}
            if arm != "full":
                # contribution of the dropped modality = AUROC(full) - AUROC(arm).
                # paired(a=arm, b=full) returns exactly that, already signed so
                # that POSITIVE means the modality helped. Do not negate on print.
                e["contribution"] = paired(yy, sc[mask], S["full"][mask])
            blk["arms"][arm] = e
        pairs = [("full_vs_kg_only", "kg_only", "full"),
                 ("img_kg_vs_kg_only", "kg_only", "img_kg"),
                 ("full_vs_img_kg", "img_kg", "full"),
                 ("img_kg_vs_img_only", "img_only", "img_kg")]
        cmp_ = {}
        for name_, a_, b_ in pairs:
            if a_ in S and b_ in S:
                cmp_[name_] = paired(yy, S[a_][mask], S[b_][mask])
        if cmp_:
            blk["comparisons"] = cmp_
        res["strata"][sname] = blk

    # ---------------------------------------------------------- noise floor
    # Where a patient LACKS modality m, the "full" and "no_m" prompts are
    # byte-identical (build() skips empty blocks), so any AUROC gap on that
    # subset is pure vLLM run-to-run nondeterminism -- batch composition changes
    # the floating-point reduction order. This is a free, assumption-free null
    # control: a contribution smaller than this is not distinguishable from noise.
    res["noise_floor"] = {}
    for arm, mod in [("no_imaging", "imaging"), ("no_report", "report"),
                     ("no_history", "history"), ("no_labs", "labs"),
                     ("no_meds", "meds"), ("no_kg", "kg")]:
        if arm not in S:
            continue
        absent = np.array([not it["avail"][mod] for it in items])
        if absent.sum() < 200 or len(np.unique(y[absent])) < 2:
            continue
        # cheap guard: confirm the prompts really are identical on a sample
        sample = [i for i, k in enumerate(absent) if k][:200]
        identical = all(build(items[i], "full") == build(items[i], arm)
                        for i in sample)
        gap = auroc(y[absent], S["full"][absent]) - auroc(y[absent], S[arm][absent])
        res["noise_floor"][mod] = {
            "n_absent": int(absent.sum()), "prompts_identical": bool(identical),
            "auroc_gap": round(float(gap), 4),
            "median_abs_score_diff": float(np.median(
                np.abs(S["full"][absent] - S[arm][absent]))),
        }

    dest = OUT / f"mm_ablation_results_{args.tag}.json"
    json.dump(res, open(dest, "w"), indent=2)

    for sname, blk in res["strata"].items():
        print(f"\n=== {sname}  n={blk['n']}  pos={blk['n_pos']} ===")
        print(f"{'arm':12s}{'AUROC':>8}{'AP':>8}   "
              f"{'contribution (95% CI)':>30}{'p':>8}")
        for arm, e in blk["arms"].items():
            d = e.get("contribution")
            ds = (f"{d['mean']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]"
                  if d else "")
            print(f"{arm:12s}{e['auroc']:8.4f}{e['ap']:8.4f}   {ds:>30}"
                  f"{d['p'] if d else '':>8}")
        print("  contribution = AUROC(full) - AUROC(arm), i.e. what the DROPPED")
        print("  modality was worth.  POSITIVE => it helped.  Negative => it hurt.")
        for nm, d in blk.get("comparisons", {}).items():
            print(f"    {nm:22s} {d['mean']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] p={d['p']}")

    if res["noise_floor"]:
        print("\n=== NOISE FLOOR (identical prompts; pure run-to-run variation) ===")
        for mod, d in res["noise_floor"].items():
            flag = "" if d["prompts_identical"] else "  [!! prompts NOT identical]"
            print(f"  {mod:9s} n={d['n_absent']:6d}  AUROC gap {d['auroc_gap']:+.4f}"
                  f"   median |dscore| {d['median_abs_score_diff']:.2e}{flag}")
        big = max(abs(d["auroc_gap"]) for d in res["noise_floor"].values())
        print(f"  => treat any contribution below ~{big:.4f} AUROC as indistinguishable")
        print("     from run-to-run noise, not as a real effect.")
    print(f"\nwrote -> {dest}")


if __name__ == "__main__":
    main()
