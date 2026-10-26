"""
04_embed_chunks.py   --   env: llm_qwen   [GPU preferred; CPU works for small pools]
================================================================================
Embeds every unique chunk text and every query text that R2 can encounter on the
requested splits with a LOCAL biomedical encoder, once, so that selection
(r2_vector.R2Selector inside 03_build_contexts.py --policy R2) runs on CPU.

Encoders (all from the local HF cache; nothing leaves the machine):
  sapbert     cambridgeltl/SapBERT-from-PubMedBERT-fulltext   (CLS pooling)   -- cached
  biomedbert  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract (mean pooling) -- cached
  medcpt      ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder (CLS)     -- only if cached

Outputs: outputs/restricted/exp1/emb/<encoder>/{chunk_texts.json, chunk_emb.npy, query_texts.json, query_emb.npy}
(chunk texts include patient evidence strings -> RESTRICTED)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, runlog
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
ctx = importlib.import_module("03_build_contexts")
from r2_vector import R2Selector

ENC = {"sapbert": ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "cls"),
       "biomedbert": ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "mean"),
       "medcpt": ("ncbi/MedCPT-Article-Encoder", "cls")}
ENC_Q = {"medcpt": "ncbi/MedCPT-Query-Encoder"}


def embed(texts, model_id, pooling, batch=256, max_len=128):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_id, local_files_only=True).to(dev).eval()
    out = np.zeros((len(texts), mdl.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            h = mdl(**b).last_hidden_state
            if pooling == "cls":
                v = h[:, 0]
            else:
                m = b["attention_mask"].unsqueeze(-1).float(); v = (h * m).sum(1) / m.sum(1)
            out[i:i + batch] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
            if i % (batch * 200) == 0:
                print(f"  {i}/{len(texts)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="sapbert", choices=list(ENC))
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--include_meds_neighbours", type=int, default=0)
    args = ap.parse_args()
    run = runlog.Run("exp1", f"embed_{args.encoder}", vars(args) | {"model": ENC[args.encoder][0]})
    EV = paths.RESTRICTED / "exp1" / "evidence_v1"
    split = paths.load_split(); idxs = sorted(sum((split[s] for s in args.splits), []))
    bench = paths.load_benchmark(verify=False)
    items = {j["idx"]: j for j in map(json.loads, open(EV / "items.jsonl"))}
    units_by_h = {u["hadm"]: u["units"] for u in map(json.loads, open(EV / "patient_units.jsonl"))}
    kg = ctx.KG("relation_alpha")
    sel = R2Selector.__new__(R2Selector); sel.include_descriptions = True; sel.include_meds = bool(args.include_meds_neighbours)
    texts, queries = {}, {}
    for i in idxs:
        it = items[i]
        for c in sel.pool(it, units_by_h[it["hadm"]], kg):
            texts.setdefault(c[0], len(texts))
        queries.setdefault(bench[i]["question"] + " Candidate: " + it["drug_pkg_name"], len(queries))
    print(f"{len(texts)} unique chunks, {len(queries)} queries", flush=True)
    d = paths.RESTRICTED / "exp1" / "emb" / args.encoder; d.mkdir(parents=True, exist_ok=True)
    tl = list(texts); ql = list(queries)
    json.dump(tl, open(d / "chunk_texts.json", "w")); json.dump(ql, open(d / "query_texts.json", "w"))
    np.save(d / "chunk_emb.npy", embed(tl, ENC[args.encoder][0], ENC[args.encoder][1]))
    np.save(d / "query_emb.npy", embed(ql, ENC_Q.get(args.encoder, ENC[args.encoder][0]), ENC[args.encoder][1]))
    for p in ("chunk_texts.json", "chunk_emb.npy", "query_texts.json", "query_emb.npy"):
        run.add_output(d / p)
    run.finish(counts={"chunks": len(tl), "queries": len(ql)})


if __name__ == "__main__":
    main()
