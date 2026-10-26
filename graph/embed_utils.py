"""
embed_utils.py   --   env: torch_gpu (transformers, torch)
==========================================================
Shared text-embedding helpers for the node-feature enrichment (scripts 06-08).
Mean-pool over tokens (attention-masked) + L2-normalise -> one 768-d vector per
text. All three encoders live in the local HF cache (offline).
"""
from __future__ import annotations
import os
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODELS = {
    "sapbert":     "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "pubmedbert":  "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "bioclinical": "emilyalsentzer/Bio_ClinicalBERT",
}


def _threads() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))


def load_encoder(name: str):
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(_threads())
    mid = MODELS[name]
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModel.from_pretrained(mid).eval()
    return tok, mdl


@torch.no_grad()
def embed_texts(tok, mdl, texts, max_len=512, batch_size=64, log_every=0):
    """texts -> (len(texts), 768) float32, mean-pooled + L2-normalised."""
    out = np.zeros((len(texts), mdl.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        chunk = [t if isinstance(t, str) and t.strip() else "[UNK]" for t in texts[i:i + batch_size]]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        h = mdl(**enc).last_hidden_state
        m = enc.attention_mask.unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        v = torch.nn.functional.normalize(v, dim=-1)
        out[i:i + len(chunk)] = v.cpu().numpy()
        if log_every and (i // batch_size) % log_every == 0:
            print(f"  embedded {i + len(chunk)}/{len(texts)}", flush=True)
    return out
