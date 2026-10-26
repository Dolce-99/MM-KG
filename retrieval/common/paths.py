"""Canonical paths and frozen-input digests for the phase-4 extensions.

All experiment scripts import their inputs from here so that a frozen file is
never addressed by an ad-hoc path, and so that the digest of every frozen input
is checked before use.  Nothing in this module reads patient data.
"""
from __future__ import annotations
import os
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent            # retrieval/common
P4 = HERE.parent                                   # retrieval
SCRIPTS = P4.parent                                # repository root (graph/, tasks/, retrieval/)
REPO = SCRIPTS                                     # repository root
P2 = SCRIPTS / "graph" / "outputs"
P3 = SCRIPTS / "tasks" / "outputs"
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))   # read-only, outside the repo
HF_HOME = Path(os.environ.get("HF_HOME", os.path.expandvars("$MMKG_HOME/.cache/huggingface")))

OUT = P4 / "outputs"                               # git-ignored
RESTRICTED = OUT / "restricted"                    # item-level / patient-level material, never tracked
STAGE0 = OUT / "stage0"

# ---- frozen inputs (digest checked on load) ---------------------------------
FROZEN = {
    "bench_v2":   (P3 / "mm" / "composite_benchmark_wide_v2.jsonl",
                   "bd9f36e5a7bcc6fe78db66632df55f57427776f1ef41133767fec44e9f7b10ca"),
    "rel_cache":  (P3 / "mm" / "primekg_rel_cache.parquet",
                   "3ef7bb647f7621b5427bd5c72e9f6b10ea1e04159117ae6b2531103871356b85"),
    "graph_edges": (P2 / "patient_graph" / "edges.parquet",
                    "e0818254cb3100d0c3db5ac0813a2967b9a3927de40fb47072ecbf13c6e484ac"),
    "graph_nodes": (P2 / "patient_graph" / "nodes.parquet",
                    "ca21b9f6cf597802a4a0132b429ad423574e96aca8723e002f49e691ed6790d6"),
    "icu_index":  (P2 / "ihm" / "icu_index.parquet",
                   "651841cd3713b97869c77f09f3307e642539989703a9621ba7c1940faef09439"),
    "bridge_v2_icd":  (P3 / "bridge_v2" / "icd2primekg_v2.tsv",
                       "4a85da6b0c0fbbdf02b97c17234fc83d6be7780a5137baff52af15dca4605eb2"),
    "bridge_v2_drug": (P3 / "bridge_v2" / "drug2primekg_v2.tsv",
                       "a838052b1dda6d8e3b6858c27a94af315a965acc6ace75531af502b21941195f"),
}
ICU_MEDS = P2 / "ihm" / "icu_early_meds.parquet"
PRIOR_DX = P2 / "ihm" / "prior_dx.parquet"
ICU_CXR = P2 / "ihm" / "icu_early_cxr.parquet"
CUI2NAME = P3 / "mm" / "cui2name_full.tsv"
SPLIT_V1 = STAGE0 / "split_v1.json"
SUBSET_V2 = STAGE0 / "drug_balanced_subset_v2.json"
Q35_SCORES = P3 / "mm"                              # cw_widev2_q35_<arm>.npy

MODELS = {
    "q35": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    "q9":  "Qwen/Qwen3.5-9B",
    "m3":  "meta-llama/Llama-3.1-8B-Instruct",
}
SYS_PROMPT = ("You are a clinical pharmacology expert reviewing an intensive-care "
              "medication list. Answer the question with a single word: yes or no.")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def frozen(name: str, verify: bool = True) -> Path:
    """Return the path of a frozen input, asserting its digest when verify=True."""
    path, digest = FROZEN[name]
    if verify:
        got = sha256(path)
        if got != digest:
            raise RuntimeError(f"frozen input {name} changed: {path}\n expected {digest}\n got {got}")
    return path


def load_benchmark(verify: bool = True) -> list[dict]:
    path = frozen("bench_v2", verify)
    return [json.loads(l) for l in open(path)]


def load_split() -> dict:
    return json.load(open(SPLIT_V1))


def load_subset() -> dict:
    return json.load(open(SUBSET_V2))
