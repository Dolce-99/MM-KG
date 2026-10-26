"""
03_score_ranking.py   --   env: llm_qwen   [GPU, L40; one vLLM engine per job]
================================================================================
Scores Experiment 2 context cells with the frozen inference recipe.

  eval A  yes/no per candidate: temperature 0, max_tokens 1, top-20 logprobs, P(yes)/(P(yes)+P(no))
  eval B  structured JSON explanation: temperature 0, max_tokens 512, greedy; raw text stored

  python 03_score_ranking.py --model q35 --max_model_len 8192 --tags <tag> [<tag> ...] [--evals A B]

Outputs (RESTRICTED)  outputs/restricted/exp2/scores/<model>/<tag>_evalA.jsonl  {"idx","cand_pkg","score","valid"}
                      outputs/restricted/exp2/scores/<model>/<tag>_evalB.jsonl  {"idx","text","prompt_tokens","out_tokens"}
Aggregate             outputs/exp2/scores/<model>_<tag>_summary.json  (ranking metrics for A; JSON-valid rate for B)
"""
from __future__ import annotations
import argparse, json, math, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, runlog, stats

CTX = paths.RESTRICTED / "exp2" / "contexts"
SYS_A = paths.SYS_PROMPT
SYS_B = ("You are a clinical pharmacology expert reviewing an intensive-care medication list. "
         "Answer only with a single JSON object and nothing else.")


def yes_prob(lp, tok):
    py = pn = 0.0
    for tid, o in lp.items():
        s = (o.decoded_token if hasattr(o, "decoded_token") else tok.decode([tid])).strip().lower()
        pr = math.exp(o.logprob if hasattr(o, "logprob") else o)
        if s.startswith("yes"): py += pr
        elif s.startswith("no"): pn += pr
    return (py / (py + pn), True) if py + pn > 0 else (0.5, False)


def ranking_metrics(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["idx"]].append(r)
    h1 = h3 = mrr = ndcg = 0.0; y = []; s = []
    for idx, rs in by.items():
        rs = sorted(rs, key=lambda r: (-r["score"], r["cand_name"]))
        pos = next(i for i, r in enumerate(rs) if r["positive"])
        h1 += pos == 0; h3 += pos < 3; mrr += 1 / (pos + 1); ndcg += 1 / math.log2(pos + 2)
        y += [int(r["positive"]) for r in rs]; s += [r["score"] for r in rs]
    n = len(by)
    return {"n_items": n, "hits@1": round(h1 / n, 4), "hits@3": round(h3 / n, 4), "mrr": round(mrr / n, 4), "ndcg@5": round(ndcg / n, 4),
            "cand_auroc": round(stats.auroc(np.array(y), np.array(s)), 4), "cand_ap": round(stats.average_precision(np.array(y), np.array(s)), 4)}


def chat(tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35", choices=list(paths.MODELS)); ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--evals", nargs="+", default=["A", "B"])
    ap.add_argument("--gen_tokens", type=int, default=512); ap.add_argument("--gpu_mem", type=float, default=0.90)
    args = ap.parse_args()
    run = runlog.Run("exp2", f"score_{args.model}_" + "+".join(args.tags)[:50], vars(args) | {"hf_id": paths.MODELS[args.model]})
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.MODELS[args.model])
    llm = LLM(model=paths.MODELS[args.model], max_model_len=args.max_model_len, trust_remote_code=True, gpu_memory_utilization=args.gpu_mem)
    spA = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    spB = SamplingParams(temperature=0.0, max_tokens=args.gen_tokens)
    outdir = paths.RESTRICTED / "exp2" / "scores" / args.model; outdir.mkdir(parents=True, exist_ok=True)
    (paths.OUT / "exp2" / "scores").mkdir(parents=True, exist_ok=True)
    for tag in args.tags:
        summ = {"tag": tag, "model": args.model}
        if "A" in args.evals and (CTX / tag / "evalA.jsonl").exists():
            rows = [json.loads(l) for l in open(CTX / tag / "evalA.jsonl")]; run.add_input(CTX / tag / "evalA.jsonl")
            prompts = [chat(tok, SYS_A, r["user"]) for r in rows]
            lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
            assert max(lens) + 1 <= args.max_model_len, f"{tag} evalA: prompt of {max(lens)} tokens exceeds the engine"
            t0 = time.time(); outs = llm.generate(prompts, spA); wall = time.time() - t0
            with open(outdir / f"{tag}_evalA.jsonl", "w") as f:
                for r, o in zip(rows, outs):
                    p, ok = yes_prob(o.outputs[0].logprobs[0], tok)
                    r.update({"score": round(p, 6), "valid": ok}); r.pop("user")
                    f.write(json.dumps(r) + "\n")
            summ["evalA"] = ranking_metrics(rows) | {"wall_seconds": round(wall, 1), "ms_per_prompt": round(1000 * wall / len(rows), 1),
                                                    "invalid_rate": round(1 - float(np.mean([r["valid"] for r in rows])), 4),
                                                    "mean_prompt_tokens": round(float(np.mean(lens)), 1)}
            run.add_output(outdir / f"{tag}_evalA.jsonl")
        if "B" in args.evals and (CTX / tag / "evalB.jsonl").exists():
            rows = [json.loads(l) for l in open(CTX / tag / "evalB.jsonl")]; run.add_input(CTX / tag / "evalB.jsonl")
            prompts = [chat(tok, r.get("system", SYS_B), r["user"]) for r in rows]     # MindMap rows carry their own system prompt
            lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
            assert max(lens) + args.gen_tokens <= args.max_model_len, f"{tag} evalB: prompt of {max(lens)} + {args.gen_tokens} exceeds the engine"
            t0 = time.time(); outs = llm.generate(prompts, spB); wall = time.time() - t0
            nvalid = 0
            with open(outdir / f"{tag}_evalB.jsonl", "w") as f:
                for r, o, n in zip(rows, outs, lens):
                    text = o.outputs[0].text
                    m = re.search(r"\{.*\}", text, re.S)
                    try:
                        json.loads(m.group(0)) if m else None; ok = bool(m)
                    except Exception:
                        ok = False
                    nvalid += ok
                    f.write(json.dumps({"idx": r["idx"], "text": text, "prompt_tokens": n, "out_tokens": len(o.outputs[0].token_ids),
                                        "tagmap": r["tagmap"], "candidates": r["candidates"]}) + "\n")
            summ["evalB"] = {"n_items": len(rows), "json_parse_rate": round(nvalid / len(rows), 4), "wall_seconds": round(wall, 1),
                             "s_per_item": round(wall / len(rows), 2), "mean_prompt_tokens": round(float(np.mean(lens)), 1)}
            run.add_output(outdir / f"{tag}_evalB.jsonl")
        json.dump(summ, open(paths.OUT / "exp2" / "scores" / f"{args.model}_{tag}_summary.json", "w"), indent=2)
        print(json.dumps(summ), flush=True)
    run.finish()


if __name__ == "__main__":
    main()
