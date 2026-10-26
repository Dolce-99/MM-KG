"""Vector-RAG (R2) selection on CPU from pre-computed chunk embeddings (04_embed_chunks.py).

Chunks of an item = its admissible pool: every patient unit (rendered with a
channel prefix), every typed one-hop triple of the linked entities, the
protein->disease bridge of the queried drug, and one description chunk per
linked node.  The query is the item question plus the candidate drug's PrimeKG
name.  Chunks are ranked by cosine similarity and taken in rank order until the
token budget is used.  Exact search over <= a few thousand vectors per item is
identical to a flat inner-product FAISS index; no approximate index is needed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import tokens as tk

CHANNEL_PREFIX = {"demo": "Patient", "codes": "Coded diagnosis on record", "meds": "Current medication",
                  "labs": "Laboratory value, first 24 hours", "report": "Radiology report finding",
                  "imaging": "Chest radiograph, automated image analysis"}


def unit_chunk(u) -> str:
    return u["text"] if u["channel"] == "demo" else f"{CHANNEL_PREFIX[u['channel']]}: {u['text']}"


class R2Selector:
    """Three label-blind vector-RAG variants, selected on dev only:
      all      pool = patient units + KG chunks; query = question + candidate name (plain RAG)
      kg_only  patient record always shown (frozen blocks); RAG over KG chunks with the same query
      multiq   patient record always shown; KG chunks scored by max similarity to the question OR
               to any patient unit (multi-query max-sim; the patient's own concept strings act as queries)
    """
    def __init__(self, emb_dir: str, tok, mode: str = "all", include_descriptions: bool = True,
                 include_meds_neighbours: bool = False):
        d = Path(emb_dir)
        self.emb = np.load(d / "chunk_emb.npy", mmap_mode="r")
        self.index = {t: i for i, t in enumerate(json.load(open(d / "chunk_texts.json")))}
        self.qemb = np.load(d / "query_emb.npy", mmap_mode="r")
        self.qindex = {t: i for i, t in enumerate(json.load(open(d / "query_texts.json")))}
        self.tok = tok; self.mode = mode
        self.include_descriptions = include_descriptions; self.include_meds = include_meds_neighbours

    def pool(self, it, units, kg):
        chunks = []                                              # (text, kind, ref)
        for u in units:
            chunks.append((unit_chunk(u), "patient", u["uid"]))
        ents = [it["drug_pkg"]]; seen = {it["drug_pkg"]}
        for u in units:
            if u["channel"] in ("codes", "report", "imaging") or (self.include_meds and u["channel"] == "meds"):
                for p in u["pkg"]:
                    if p not in seen:
                        ents.append(p); seen.add(p)
        for e in ents:
            if self.include_descriptions:
                dsc = kg.description(e)
                if dsc:
                    chunks.append((dsc, "desc", None))
            for t in kg.triples(e):
                chunks.append((t["line"], "kg", (t["x"], t["rel"], t["y"])))
                if e == it["drug_pkg"] and t["rel"] == "drug_protein":
                    for b in kg.triples(t["y"], role="protein_bridge"):
                        chunks.append((b["line"], "kg", (b["x"], b["rel"], b["y"])))
        return chunks

    def _vecs(self, texts):
        return np.stack([self.emb[self.index[t]] for t in texts]).astype(np.float32)

    def select(self, it, itf, units, kg, budget_tokens: int, patient_text: str = "") -> dict:
        chunks = self.pool(it, units, kg)
        q = itf["question"] + " Candidate: " + it["drug_pkg_name"]
        qv = np.asarray(self.qemb[self.qindex[q]], dtype=np.float32)
        if self.mode != "all":
            chunks = [c for c in chunks if c[1] != "patient"]
        if not chunks:
            return {"patient_text": patient_text, "kg_text": "", "kg_lines": [], "kg_edges": [], "patient_uids": [],
                    "truncated": False, "untruncated_tokens": 0, "n_pool": 0}
        M = self._vecs([c[0] for c in chunks])
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        sims = Mn @ (qv / (np.linalg.norm(qv) + 1e-9))
        if self.mode == "multiq" and units:
            U = self._vecs([unit_chunk(u) for u in units])
            Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
            sims = np.maximum(sims, (Mn @ Un.T).max(axis=1))
        order = np.argsort(-sims, kind="mergesort")
        ranked = [chunks[i][0] for i in order]
        pt = patient_text if self.mode != "all" else ""
        fixed = tk.n_tokens(self.tok, pt + "\n\n") if pt else 0
        pk = tk.pack(self.tok, ranked, max(budget_tokens - fixed, 0))
        kept = [chunks[order[k]] for k in pk["kept"]]
        pat = [c for c in kept if c[1] == "patient"]; kgc = [c for c in kept if c[1] != "patient"]
        out = {"kg_lines": [c[0] for c in kgc], "kg_edges": [c[2] if c[1] == "kg" else None for c in kgc],   # aligned with kg_lines
               "truncated": pk["truncated"], "untruncated_tokens": pk["untruncated_tokens"] + fixed, "n_pool": len(chunks)}
        if self.mode == "all":
            out["patient_text"] = "Retrieved patient evidence:\n" + "\n".join(c[0] for c in pat) if pat else ""
            out["patient_uids"] = [c[2] for c in pat]
        else:
            out["patient_text"] = pt; out["patient_uids"] = None          # caller fills the visible frozen units
        out["kg_text"] = ("Retrieved knowledge-graph facts (ranked by similarity; " + str(len(kgc)) + " of " +
                          str(sum(c[1] != "patient" for c in chunks)) + "):\n" + "\n".join(c[0] for c in kgc)) if kgc else ""
        return out
