"""
05_score_contexts.py   --   env: llm_qwen   [GPU, L40; one vLLM engine per job]
================================================================================
Scores one or more materialised context sets (03_build_contexts.py) with the
frozen inference recipe of phase3/19_composite_wide_llm.py: same system prompt,
chat template with thinking disabled, temperature 0, max_tokens 1, top-20
first-token log-probabilities, score = P(yes) / (P(yes) + P(no)).

  python 05_score_contexts.py --model q35 --max_model_len 8192 --tags R0_dev_bnatural R4_dev_bnatural ...

Outputs (per tag)
  outputs/restricted/exp1/scores/<model>/<tag>.npy         yes-scores aligned with contexts.jsonl order
  outputs/restricted/exp1/scores/<model>/<tag>.jsonl        {"idx", "score", "valid", "top"}   RESTRICTED
  outputs/exp1/scores/<model>_<tag>_summary.json            aggregate only
Every tag scored in the same job shares one engine instance, as the frozen arms did.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, runlog, stats

CTX = paths.RESTRICTED / "exp1" / "contexts"


def yes_prob(lp, tok):
    py = pn = 0.0; top = None
    for tid, o in lp.items():
        s = (o.decoded_token if hasattr(o, "decoded_token") else tok.decode([tid])).strip().lower()
        pr = math.exp(o.logprob if hasattr(o, "logprob") else o)
        if top is None or pr > top[1]:
            top = (s, pr)
        if s.startswith("yes"):
            py += pr
        elif s.startswith("no"):
            pn += pr
    return ((py / (py + pn), True) if py + pn > 0 else (0.5, False)), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35", choices=list(paths.MODELS))
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    args = ap.parse_args()
    run = runlog.Run("exp1", f"score_{args.model}_" + "+".join(args.tags)[:60],
                     vars(args) | {"hf_id": paths.MODELS[args.model], "decoding": {"temperature": 0.0, "max_tokens": 1, "logprobs": 20},
                                   "system_prompt": paths.SYS_PROMPT})
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.MODELS[args.model])
    llm = LLM(model=paths.MODELS[args.model], max_model_len=args.max_model_len, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    outdir = paths.RESTRICTED / "exp1" / "scores" / args.model; outdir.mkdir(parents=True, exist_ok=True)
    (paths.OUT / "exp1" / "scores").mkdir(parents=True, exist_ok=True)
    for tag in args.tags:
        rows = [json.loads(l) for l in open(CTX / tag / "contexts.jsonl")]
        run.add_input(CTX / tag / "contexts.jsonl")
        prompts = []
        for r in rows:
            msgs = [{"role": "system", "content": paths.SYS_PROMPT}, {"role": "user", "content": r["user"]}]
            try:
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
        over = [r["idx"] for r, n in zip(rows, lens) if n + 1 > args.max_model_len]
        if over:
            raise RuntimeError(f"{tag}: {len(over)} prompts exceed max_model_len-1 ({args.max_model_len - 1}); "
                               f"rebuild the contexts with a larger --reserve (first offenders: {over[:5]})")
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        wall = time.time() - t0
        sc, valid, recs = [], [], []
        for r, o in zip(rows, outs):
            (p, ok), top = yes_prob(o.outputs[0].logprobs[0], tok)
            sc.append(p); valid.append(ok)
            recs.append({"idx": r["idx"], "score": round(p, 6), "valid": ok, "top": top[0] if top else None,
                         "prompt_tokens": len(o.prompt_token_ids)})
        np.save(outdir / f"{tag}.npy", np.array(sc))
        with open(outdir / f"{tag}.jsonl", "w") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
        y = np.array([1 if r["stats"]["answer"] == "yes" else 0 for r in rows])
        summ = {"tag": tag, "model": args.model, "n": len(rows), "wall_seconds": round(wall, 1),
                "ms_per_item": round(1000 * wall / len(rows), 1), "invalid_rate": round(1 - float(np.mean(valid)), 4),
                "auroc": round(stats.auroc(y, np.array(sc)), 4), "bal_acc_at_0.5": round(stats.balanced_acc(y, np.array(sc) >= 0.5), 4),
                "mean_prompt_tokens": round(float(np.mean([x["prompt_tokens"] for x in recs])), 1)}
        json.dump(summ, open(paths.OUT / "exp1" / "scores" / f"{args.model}_{tag}_summary.json", "w"), indent=2)
        run.add_output(outdir / f"{tag}.npy"); run.add_output(outdir / f"{tag}.jsonl")
        print(json.dumps(summ), flush=True)
    run.finish()


if __name__ == "__main__":
    main()
