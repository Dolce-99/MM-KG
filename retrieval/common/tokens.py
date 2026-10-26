"""Tokenizer-based accounting and deterministic truncation (CPU, offline).

The Qwen tokenizer loads on the login node with local_files_only=True, so prompt
sizes, budget curves and truncation decisions are computed without a GPU.
"""
from __future__ import annotations
import os
from . import paths

os.environ.setdefault("HF_HOME", str(paths.HF_HOME))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_CACHE = {}


def load_tokenizer(model_id: str = paths.MODELS["q35"]):
    if model_id not in _CACHE:
        from transformers import AutoTokenizer
        _CACHE[model_id] = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    return _CACHE[model_id]


def render_chat(tok, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:                                  # tokenizers without the thinking switch
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


_LEN_CACHE: dict = {}


def unit_lengths(tok, units: list[str], sep: str = "\n") -> list[int]:
    """Token length of every unit (+sep), batch-tokenised, cached across calls."""
    todo = [u for u in dict.fromkeys(units) if (u + sep) not in _LEN_CACHE]
    for i in range(0, len(todo), 4096):
        chunk = todo[i:i + 4096]
        enc = tok([u + sep for u in chunk], add_special_tokens=False)
        for u, ids in zip(chunk, enc["input_ids"]):
            _LEN_CACHE[u + sep] = len(ids)
    return [_LEN_CACHE[u + sep] for u in units]


def pack(tok, units: list[str], budget: int, sep: str = "\n", fixed: int = 0) -> dict:
    """Keep the longest prefix of `units` (in the given order) whose joined text
    fits within `budget - fixed` tokens.  Deterministic; label-blind by
    construction because the caller fixes the order.

    Returns kept indices, the exact token count of the kept text, the
    untruncated size (sum of per-unit token counts; equals the joint count up to
    boundary effects of <1%), and whether truncation happened.
    """
    if not units:
        return {"kept": [], "tokens": 0, "untruncated_tokens": 0, "truncated": False}
    limit = budget - fixed
    per = unit_lengths(tok, units, sep)
    total = sum(per)
    k = 0; acc = 0
    while k < len(units) and acc + per[k] <= limit:
        acc += per[k]; k += 1
    while k > 0 and n_tokens(tok, sep.join(units[:k])) > limit:   # boundary effects of joint tokenisation
        k -= 1
    exact = n_tokens(tok, sep.join(units[:k])) if k else 0
    return {"kept": list(range(k)), "tokens": exact,
            "untruncated_tokens": total if k < len(units) else exact, "truncated": k < len(units)}
