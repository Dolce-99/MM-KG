"""
02_biomedclip_findings.py   --   env: torch_gpu   (CPU; run as a SLURM array)
=============================================================================
BioMedCLIP zero-shot CheXpert-13 over ALL whole-cohort studies (~204k). Sharded
for a SLURM array: each task handles a strided slice and writes its own jsonl, so
there is no write contention. Resumable: a study already in this shard's jsonl is
skipped. Merge later with 02b_merge.py.

  WG_NSHARDS         total number of shards (set in the sbatch script)
  SLURM_ARRAY_TASK_ID this task's shard index (0 .. NSHARDS-1)

Output:  outputs/imaging/shard_<id>.jsonl
           {study_id, image_path, scores:{label:prob}, present:[...], status}
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HOME", os.path.expandvars("$MMKG_HOME/.cache/huggingface"))

import json, sys, time
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

from imaging_findings import CHEXPERT_FINDINGS, LABEL_PHRASE, PRESENT_THRESHOLD

BIOMEDCLIP_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CONTEXT_LEN = 256
HERE = Path(__file__).resolve().parent
STUDIES = HERE / "outputs" / "whole_cohort_studies.parquet"
OUTDIR = HERE / "outputs" / "imaging"

NSHARDS = int(os.environ.get("WG_NSHARDS", "1"))
SHARD = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))


class BioMedCLIP:
    def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ref = f"hf-hub:{BIOMEDCLIP_REPO}"
        self.model, self.preprocess = open_clip.create_model_from_pretrained(ref)
        self.tokenizer = open_clip.get_tokenizer(ref)
        self.model.eval().to(self.device)
        prompts = []
        for lab in CHEXPERT_FINDINGS:
            p = LABEL_PHRASE[lab]
            prompts += [f"this is a chest x-ray showing {p}", f"this is a chest x-ray with no {p}"]
        toks = self.tokenizer(prompts, context_length=CONTEXT_LEN).to(self.device)
        with torch.no_grad():
            self.text_features = F.normalize(self.model.encode_text(toks), dim=-1)
        self.logit_scale = self.model.logit_scale.exp()

    @torch.no_grad()
    def classify(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = F.normalize(self.model.encode_image(x), dim=-1)
        logits = (self.logit_scale * feat @ self.text_features.t()).squeeze(0)
        scores, present = {}, []
        for i, lab in enumerate(CHEXPERT_FINDINGS):
            p = torch.softmax(logits[2 * i:2 * i + 2], dim=-1)[0].item()
            scores[lab] = round(p, 4)
            if p >= PRESENT_THRESHOLD:
                present.append(lab)
        return scores, present


def load_done(path):
    done = set()
    if path.exists():
        for line in open(path):
            try:
                done.add(str(json.loads(line)["study_id"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(STUDIES)
    shard = df.iloc[SHARD::NSHARDS]                       # strided slice
    out_path = OUTDIR / f"shard_{SHARD:03d}.jsonl"
    done = load_done(out_path)
    todo = [r for r in shard.itertuples(index=False) if str(r.study_id) not in done]
    print(f"shard {SHARD}/{NSHARDS}: {len(shard)} studies, {len(done)} done, {len(todo)} to run "
          f"| threads={torch.get_num_threads()}", flush=True)
    if not todo:
        print("nothing to do", flush=True); return

    clf = BioMedCLIP(); clf.load()
    print(f"    device={clf.device}", flush=True)
    logf = open(out_path, "a")
    n_ok = n_err = 0
    for i, r in enumerate(todo, 1):
        sid, path = str(r.study_id), r.image_path
        try:
            scores, present = clf.classify(path); status = "ok"; n_ok += 1
        except Exception as e:
            scores, present, status = {}, [], f"error: {type(e).__name__}: {e}"; n_err += 1
        logf.write(json.dumps({"study_id": sid, "image_path": path,
                               "scores": scores, "present": present, "status": status}) + "\n")
        if i % 200 == 0:
            logf.flush(); print(f"  [{i}/{len(todo)}] ok={n_ok} err={n_err}", flush=True)
    logf.close()
    print(f"=== shard {SHARD} done: {n_ok} ok, {n_err} err -> {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
