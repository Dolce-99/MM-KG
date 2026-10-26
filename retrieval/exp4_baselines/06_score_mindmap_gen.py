"""
06_score_mindmap_gen.py   --   env: llm_qwen   [GPU]
================================================================================
Scores the MindMap generation variant: greedy decoding (temperature 0, up to --gen_tokens),
answer parsed from "Output1" (first yes/no token after it; fallback: first yes/no in the text);
invalid = no parsable answer.  Also stores the full generation (inference chain + mind map)
in the restricted output.  Reports AUROC on the parsed hard answer (0/1; ties at 0.5 for
invalid) and balanced accuracy, comparable to the threshold metrics of the other policies.
  python 06_score_mindmap_gen.py --model q35 --max_model_len 8192 --tags MM_gen_dev_bnatural
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, runlog, stats

CTX = paths.RESTRICTED / "exp1" / "contexts"


def parse(text):
    m = re.search(r"Output\s*1\s*[:：]?\s*(.*)", text, re.S | re.I)
    seg = m.group(1)[:200] if m else text[:200]
    w = re.search(r"\b(yes|no)\b", seg, re.I) or re.search(r"\b(yes|no)\b", text, re.I)
    return (w.group(1).lower(), True) if w else (None, False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="q35"); ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--gen_tokens", type=int, default=768); ap.add_argument("--gpu_mem", type=float, default=0.90)
    a = ap.parse_args(); run = runlog.Run("exp4", f"mindmap_gen_{a.model}", vars(a))
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.MODELS[a.model])
    llm = LLM(model=paths.MODELS[a.model], max_model_len=a.max_model_len, trust_remote_code=True, gpu_memory_utilization=a.gpu_mem)
    sp = SamplingParams(temperature=0.0, max_tokens=a.gen_tokens)
    outdir = paths.RESTRICTED / "exp1" / "scores" / a.model; outdir.mkdir(parents=True, exist_ok=True)
    for tag in a.tags:
        rows = [json.loads(l) for l in open(CTX / tag / "contexts.jsonl")]
        prompts = []
        for r in rows:
            msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
            try: prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError: prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
        assert max(lens) + a.gen_tokens <= a.max_model_len, f"{tag}: prompt {max(lens)} + {a.gen_tokens} exceeds engine"
        t0 = time.time(); outs = llm.generate(prompts, sp); wall = time.time() - t0
        sc, valid = [], []
        with open(outdir / f"{tag}.jsonl", "w") as f:
            for r, o in zip(rows, outs):
                text = o.outputs[0].text; ans, ok = parse(text)
                s = 1.0 if ans == "yes" else 0.0 if ans == "no" else 0.5; sc.append(s); valid.append(ok)
                f.write(json.dumps({"idx": r["idx"], "score": s, "valid": ok, "answer": ans, "text": text, "prompt_tokens": len(o.prompt_token_ids), "out_tokens": len(o.outputs[0].token_ids)}) + "\n")
        np.save(outdir / f"{tag}.npy", np.array(sc))
        y = np.array([1 if r["stats"]["answer"] == "yes" else 0 for r in rows]); s = np.array(sc)
        summ = {"tag": tag, "model": a.model, "n": len(rows), "wall_seconds": round(wall, 1), "ms_per_item": round(1000 * wall / len(rows), 1),
                "invalid_rate": round(1 - float(np.mean(valid)), 4), "auroc": round(stats.auroc(y, s), 4), "bal_acc_at_0.5": round(stats.balanced_acc(y, s >= 0.5), 4),
                "mean_prompt_tokens": round(float(np.mean(lens)), 1), "mean_out_tokens": round(float(np.mean([len(o.outputs[0].token_ids) for o in outs])), 1)}
        json.dump(summ, open(paths.OUT / "exp1" / "scores" / f"{a.model}_{tag}_summary.json", "w"), indent=2); print(json.dumps(summ), flush=True)
    run.finish()


if __name__ == "__main__":
    main()
