"""
22_spatial_ladder_llm.py   --   env: llm_qwen   [GPU cluster]
================================================================================
STAGE 2: the spatial-information LADDER.

Does the chest film contain task-relevant information BEYOND finding labels --
and if so, which attribute class carries it? Every rung re-describes the SAME
film using the radiologist's own extraction (stage 1), so extraction quality is
never the confound; this is the information ceiling a vision model would then
have to reach.

  R0  finding names only            "pulmonary edema, pleural effusion"
  R1  + severity/extent             "mild pulmonary edema, small pleural effusion"
  R2  + laterality + location       "small left pleural effusion at the base"
  R3  + interval change + devices   "worsening mild pulmonary edema ...;
                                     endotracheal tube, tip 4 cm above carina"
  R4  the verbatim FINDINGS text

Each rung is evaluated in TWO contexts:
  iso_*  demographics + the rung alone     (raw signal in the representation)
  ctx_*  demographics + history + meds + labs + the rung
         (incremental value given the rest of the record -- the deployment case)
plus ctx_none (no imaging information at all) as the in-context baseline.

Task, cohort, scoring identical to 13_mm_ablation_llm.py: ICU in-hospital
mortality, P(yes) token mass, AUROC with paired bootstrap.

Usage:
  python 22_spatial_ladder_llm.py                 # run all arms + analysis
  python 22_spatial_ladder_llm.py --analyze_only
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT  = HERE / "outputs" / "mm"

RUNGS = ["R0", "R1", "R2", "R3", "R4"]
ARMS  = ["iso_" + r for r in RUNGS] + ["ctx_none"] + ["ctx_" + r for r in RUNGS]
CTX_BLOCKS = ["demo", "history", "meds", "labs"]

SYS = ("You are an experienced intensive-care physician. You will be shown "
       "information about a patient in intensive care. Answer the question with "
       "a single word: yes or no.")
QUESTION = ("Question: Will this patient die during this hospital admission? "
            "Answer yes or no.")


# ---------------------------------------------------------------- rendering
def render_rung(parsed, findings_text, rung):
    if rung == "R4":
        return "Radiology report, findings section: " + findings_text.strip()
    fs = parsed.get("findings") or []
    outs = []
    for f in fs:
        name = str(f.get("name") or "").strip()
        if not name:
            continue
        sev = str(f.get("severity") or "unspecified")
        lat = str(f.get("laterality") or "unspecified")
        loc = f.get("location")
        chg = str(f.get("change") or "unspecified")
        d = name.lower()
        if rung in ("R1", "R2", "R3") and sev != "unspecified":
            d = f"{sev} {d}"
        if rung in ("R2", "R3"):
            if lat != "unspecified":
                d = f"{d}, {lat}"
            if loc:
                d = f"{d}, at the {str(loc).lower()}"
        if rung == "R3" and chg != "unspecified":
            d = f"{d} ({chg})"
        outs.append(d)
    txt = "Chest radiograph shows: " + "; ".join(outs) + "." if outs else \
          "Chest radiograph shows no acute abnormality."
    if rung == "R3":
        dev = []
        for v in parsed.get("devices") or []:
            nm = str(v.get("name") or "").strip()
            if not nm:
                continue
            pos = v.get("position")
            dev.append(f"{nm} ({pos})" if pos else nm)
        if dev:
            txt += " Support devices: " + "; ".join(dev) + "."
    return txt


def build(it, arm):
    blocks = [it["blocks"][b] for b in (CTX_BLOCKS if arm.startswith("ctx") else ["demo"])
              if it["blocks"].get(b)]
    if not arm.endswith("none"):
        blocks.append(it["rungs"][arm.split("_")[1]])
    return "\n\n".join(blocks) + "\n\n" + QUESTION


# ---------------------------------------------------------------- metrics
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


def paired(y, sa, sb, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); sa = np.asarray(sa); sb = np.asarray(sb); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a, b = auroc(y[i], sa[i]), auroc(y[i], sb[i])
        if not (math.isnan(a) or math.isnan(b)):
            d.append(b - a)
    if not d:
        return None
    d = np.asarray(d); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"mean": round(float(d.mean()), 4),
            "ci": [round(float(np.percentile(d, 2.5)), 4),
                   round(float(np.percentile(d, 97.5)), 4)],
            "p": round(float(min(p, 1.0)), 4)}


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


def load_items():
    """One parsed study per ICU stay; blocks from the existing bundles."""
    parses = {}
    for line in open(OUT / "report_spatial_ihm.jsonl"):
        r = json.loads(line)
        if r["parse_ok"] and (r["parsed"].get("findings") or r["parsed"].get("devices")):
            parses.setdefault(r["hadm_id"], r)      # first (nearest-gap not tracked; one per stay)
    items = []
    for line in open(OUT / "mm_bundles_ihm.jsonl"):
        b = json.loads(line)
        r = parses.get(b["hadm_id"])
        if not r:
            continue
        items.append({"hadm_id": b["hadm_id"], "label": b["label"],
                      "blocks": b["blocks"],
                      "rungs": {k: render_rung(r["parsed"], r["findings_text"], k)
                                for k in RUNGS}})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tag", default="ladder")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    items = load_items()
    y = np.array([it["label"] for it in items])
    print(f"{len(items)} admissions with a parsed report | pos {int(y.sum())} "
          f"({y.mean():.2%})", flush=True)

    todo = [a for a in ARMS if not (OUT / f"lad_{args.tag}_{a}.npy").exists()]
    if not args.analyze_only and not todo:
        args.analyze_only = True
        print("all arms cached; analysis only", flush=True)

    if not args.analyze_only:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, max_model_len=args.max_model_len,
                  trust_remote_code=True)
        tok = llm.get_tokenizer()
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        for arm in todo:
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
            sc, bad = [], 0
            for o in outs:
                lps = o.outputs[0].logprobs
                if not lps:
                    sc.append(0.5); bad += 1; continue
                v, ok = yes_prob(lps[0], tok)
                sc.append(v); bad += (not ok)
            np.save(OUT / f"lad_{args.tag}_{arm}.npy", np.array(sc))
            print(f"  [{arm}] done, {bad} without a yes/no token", flush=True)

    S = {a: np.load(OUT / f"lad_{args.tag}_{a}.npy")
         for a in ARMS if (OUT / f"lad_{args.tag}_{a}.npy").exists()}
    res = {"model": args.model, "n": len(items), "n_pos": int(y.sum()),
           "arms": {a: {"auroc": round(auroc(y, s), 4)} for a, s in S.items()},
           "deltas": {}}
    def dd(name, a, b):
        if a in S and b in S:
            res["deltas"][name] = paired(y, S[a], S[b])
    for pre in ("iso", "ctx"):
        for lo, hi in zip(RUNGS[:-1], RUNGS[1:]):
            dd(f"{pre}_{hi}_vs_{lo}", f"{pre}_{lo}", f"{pre}_{hi}")
        for r in RUNGS[1:]:
            dd(f"{pre}_{r}_vs_R0", f"{pre}_R0", f"{pre}_{r}")
    for r in RUNGS:
        dd(f"ctx_{r}_vs_none", "ctx_none", f"ctx_{r}")

    json.dump(res, open(OUT / f"spatial_ladder_results_{args.tag}.json", "w"), indent=2)

    print("\n=== SPATIAL LADDER  (AUROC; task: ICU in-hospital mortality) ===")
    print(f"{'rung':6s}{'isolated':>10}{'in-context':>12}")
    for r in RUNGS:
        i_ = res["arms"].get(f"iso_{r}", {}).get("auroc", float("nan"))
        c_ = res["arms"].get(f"ctx_{r}", {}).get("auroc", float("nan"))
        print(f"  {r:4s}{i_:10.4f}{c_:12.4f}")
    if "ctx_none" in res["arms"]:
        print(f"  {'none':4s}{'':10s}{res['arms']['ctx_none']['auroc']:12.4f}   (no imaging info)")
    print("\n--- paired deltas ---")
    for k, v in res["deltas"].items():
        if v:
            print(f"  {k:22s} {v['mean']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']}")
    print(f"\nwrote -> {OUT}/spatial_ladder_results_{args.tag}.json")


if __name__ == "__main__":
    main()
