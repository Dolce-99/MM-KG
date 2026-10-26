"""
10_retrieval_completeness.py   --   env: llm_qwen   [GPU cluster]
================================================================================
Does TRUNCATED retrieval explain every case where the knowledge graph hurt?

FINDING THIS TESTS
In the whole-cohort audit, three question families got WORSE with the graph. All
three show one signature: sensitivity collapses while specificity goes to ~1.0.

  family              sens no-KG -> with-KG      spec no-KG -> with-KG
  side effects          0.567 -> 0.208             0.883 -> 0.992
  protein/disease       0.375 -> 0.100             0.817 -> 0.950
  three-step chain      0.500 -> 0.242             0.692 -> 0.933

The five families that improved show the opposite (sensitivity rises). The losing
families are exactly those whose true neighbourhood is far larger than what we showed:
PrimeKG attaches a MEDIAN OF 43 side effects per drug (max 338) and we retrieved 12,
so only 22.5% of "yes" items had the queried fact anywhere in their evidence.

HYPOTHESIS: the model treats the retrieved list as COMPLETE and answers "no" to
anything absent from it. That is correct reasoning about the evidence and wrong about
the world -- an open-world graph read under a closed-world assumption.

ARMS
  C_truncated  the current retrieval (<=12 per relation)            [reproduces the harm]
  C_complete   the FULL neighbourhood, no truncation                [does completeness fix it?]
  C_warned     truncated, but the prompt states the list is partial [does saying so fix it?]
  A_flat       question only                                        [reference]

If C_complete recovers the three families, the "two-hop ceiling" and "noisy side-effect
layer" readings are both wrong and the real result is about retrieval completeness.
If C_warned also recovers them, that is a one-line prompt fix with general application.

Usage:
  python 10_retrieval_completeness.py --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
  python 10_retrieval_completeness.py --dry_run
"""
from __future__ import annotations
import os
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
HERE    = Path(__file__).resolve().parent
OUT     = HERE / "outputs" / "audit"

# the three families whose neighbourhoods we truncated hardest
TARGET = {"drug_effect", "hop2_mechanism", "hop3_chain"}
TRUNC  = 12
SYS = ('You are a clinical pharmacology expert. Answer with ONLY a JSON object: '
       '{"answer": "yes"|"no"}. Answer "yes" only if you are confident.')
WARN = ("The list below is a PARTIAL extract from a larger knowledge graph and may be "
        "incomplete. If a fact is not shown, that does not mean it is false — fall back "
        "on your own pharmacological knowledge.\n\n")


def balanced(y, p):
    y = np.asarray(y); p = np.asarray(p)
    se = p[y == 1].mean() if (y == 1).any() else np.nan
    sp = 1 - p[y == 0].mean() if (y == 0).any() else np.nan
    return float((se + sp) / 2), float(se), float(sp)


def paired(y, pa, pb, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        a, _, _ = balanced(y[i], pa[i]); b, _, _ = balanced(y[i], pb[i])
        if not (np.isnan(a) or np.isnan(b)): d.append(b - a)
    if not d: return None
    d = np.asarray(d); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"mean": round(float(d.mean()), 4),
            "ci": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p": round(float(min(p, 1.0)), 4)}


def parse(t):
    body = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL)
    for raw in reversed(re.findall(r"\{[^{}]*\}", body, re.DOTALL)):
        try: o = json.loads(raw)
        except json.JSONDecodeError: continue
        a = str(o.get("answer", "")).strip().lower()
        if a.startswith(("y", "n")): return (1 if a.startswith("y") else 0), True
    m = re.search(r"\b(yes|no)\b", body, re.I)
    if m: return (1 if m.group(1).lower() == "yes" else 0), True
    return 0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=16384)   # complete lists are long
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    idx2name, idx2type = {}, {}
    with open(PRIMEKG / "nodes.tab") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["node_index"]); idx2name[i] = r["node_name"]; idx2type[i] = r["node_type"]

    # full neighbourhoods, untruncated
    deff, dprot, disprot, dis_dis = (defaultdict(set) for _ in range(4))
    pname = {}
    for ch in pd.read_csv(PRIMEKG / "kg.csv",
                          usecols=["relation", "x_type", "x_index", "x_name", "y_type", "y_index", "y_name"],
                          dtype={"x_index": "int32", "y_index": "int32"}, chunksize=2_000_000):
        ch = ch[ch.relation.isin({"drug_effect", "drug_protein", "disease_protein", "disease_disease"})]
        for rel, xt, x, xn, y, yn in zip(ch.relation, ch.x_type, ch.x_index, ch.x_name, ch.y_index, ch.y_name):
            x, y = int(x), int(y)
            if rel == "drug_effect":
                d, e, en = (x, y, str(yn)) if xt == "drug" else (y, x, str(xn))
                deff[d].add(e); idx2name.setdefault(e, en)
            elif rel == "drug_protein":
                d, p, pn = (x, y, str(yn)) if xt == "drug" else (y, x, str(xn))
                dprot[d].add(p); pname[p] = pn
            elif rel == "disease_protein":
                d, p, pn = (x, y, str(yn)) if xt == "disease" else (y, x, str(xn))
                disprot[d].add(p); pname[p] = pn
            elif rel == "disease_disease":
                dis_dis[x].add(y); dis_dis[y].add(x)

    items = [i for i in map(json.loads, open(OUT / "extended_benchmark.jsonl"))
             if i["family"] in TARGET]
    y = [1 if i["answer"] == "yes" else 0 for i in items]
    print(f"items: {len(items)} across {sorted(TARGET)} | {sum(y)} positive", flush=True)

    def neigh(it, cap):
        out = []
        for d in it["drugs"][:6]:
            if deff.get(d):
                v = sorted(deff[d], key=lambda e: idx2name.get(e, ""))
                out.append(f"{idx2name[d]} --causes--> " +
                           "; ".join(idx2name.get(e, str(e)) for e in (v[:cap] if cap else v)))
            if dprot.get(d):
                v = sorted(dprot[d], key=lambda p: pname.get(p, ""))
                out.append(f"{idx2name[d]} --targets--> " +
                           ", ".join(pname.get(p, str(p)) for p in (v[:cap] if cap else v)))
        for c in it["conds"][:4]:
            if disprot.get(c):
                v = sorted(disprot[c], key=lambda p: pname.get(p, ""))
                out.append(f"{idx2name[c]} --implicated_proteins--> " +
                           ", ".join(pname.get(p, str(p)) for p in (v[:cap] if cap else v)))
            if dis_dis.get(c):
                v = sorted(dis_dis[c], key=lambda x: idx2name.get(x, ""))
                out.append(f"{idx2name[c]} --related_diseases--> " +
                           "; ".join(idx2name.get(x, str(x)) for x in (v[:cap] if cap else v)))
        return out

    def build(it, arm):
        if arm == "A_flat": return it["question"]
        cap = None if arm == "C_complete" else TRUNC
        e = neigh(it, cap)
        blk = "Facts retrieved from a curated medical knowledge graph:\n" + "\n".join(f"  {x}" for x in e) \
            if e else "The knowledge graph returned no facts for these entities."
        pre = WARN if arm == "C_warned" else ""
        return f"{pre}{blk}\n\n{it['question']}"

    # how much more evidence does the complete arm actually contain?
    tr = [len("\n".join(neigh(i, TRUNC))) for i in items]
    co = [len("\n".join(neigh(i, None))) for i in items]
    print(f"evidence size: truncated {np.mean(tr):.0f} chars -> complete {np.mean(co):.0f} chars "
          f"({np.mean(co)/max(np.mean(tr),1):.1f}x)", flush=True)

    arms = ["A_flat", "C_truncated", "C_complete", "C_warned"]
    gen = None
    if not args.dry_run:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, max_model_len=args.max_model_len, trust_remote_code=True)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        tok = llm.get_tokenizer()

        def gen(ps):
            txt = []
            for p in ps:
                m = [{"role": "system", "content": SYS}, {"role": "user", "content": p}]
                try: txt.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False))
                except TypeError: txt.append(tok.apply_chat_template(m, tokenize=False,
                                                                     add_generation_prompt=True))
            return [o.outputs[0].text for o in llm.generate(txt, sp)]

    preds, res = {}, {"model": args.model, "n": len(items), "families": {}, "overall": {}}
    for arm in arms:
        ps = [build(it, arm) for it in items]
        raw = ([json.dumps({"answer": it["answer"] if arm == "C_complete" else "no"}) for it in items]
               if args.dry_run else gen(ps))
        pr, bad = [], 0
        with open(OUT / f"completeness_preds_{arm}.jsonl", "w") as w:
            for it, t in zip(items, raw):
                v, ok = parse(t)
                if not ok: bad += 1
                pr.append(v)
                w.write(json.dumps({"family": it["family"], "answer": it["answer"],
                                    "pred": v, "arm": arm}) + "\n")
        preds[arm] = pr
        b, se_, sp_ = balanced(y, pr)       # NB: not `sp` -- that name holds SamplingParams
        res["overall"][arm] = {"bal_acc": round(b, 4), "sens": round(se_, 3), "spec": round(sp_, 3),
                               "unparsed": bad}
        print(f"  [{arm}] bal {b:.3f} sens {se_:.3f} spec {sp_:.3f} ({bad} unparsed)", flush=True)

    for fam in sorted(TARGET):
        m = [i for i, it in enumerate(items) if it["family"] == fam]
        yy = [y[i] for i in m]
        row = {}
        for arm in arms:
            b, se_, sp_ = balanced(yy, [preds[arm][i] for i in m])
            row[arm] = {"bal_acc": round(b, 4), "sens": round(se_, 3), "spec": round(sp_, 3)}
        row["complete_vs_truncated"] = paired(yy, [preds["C_truncated"][i] for i in m],
                                              [preds["C_complete"][i] for i in m])
        row["warned_vs_truncated"] = paired(yy, [preds["C_truncated"][i] for i in m],
                                            [preds["C_warned"][i] for i in m])
        row["complete_vs_A"] = paired(yy, [preds["A_flat"][i] for i in m],
                                      [preds["C_complete"][i] for i in m])
        res["families"][fam] = row
    json.dump(res, open(OUT / "completeness_results.json", "w"), indent=2)

    print(f"\n=== RETRIEVAL COMPLETENESS ({args.model}) ===")
    print("balanced accuracy / sensitivity — the hypothesis is that truncation kills sensitivity\n")
    print(f"{'family':18s} " + " ".join(f"{a:>22}" for a in arms))
    for fam, r in res["families"].items():
        print(f"{fam:18s} " + " ".join(f"{r[a]['bal_acc']:>10.3f} (se {r[a]['sens']:.2f})" for a in arms))
    print("\n--- did completeness fix it? (vs the truncated arm) ---")
    for fam, r in res["families"].items():
        d = r["complete_vs_truncated"]; w = r["warned_vs_truncated"]; a = r["complete_vs_A"]
        print(f"  {fam:18s} complete {d['mean']:+.3f} p={d['p']:<6} | warned {w['mean']:+.3f} "
              f"p={w['p']:<6} | complete vs no-KG {a['mean']:+.3f} p={a['p']}")
    print(f"\nwrote -> {OUT}/completeness_results.json")


if __name__ == "__main__":
    main()
