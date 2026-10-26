"""
nlp_cui.py   --   env: umls_eval
=================================
Reusable finding-CUI extractor, factored from phase-1 02.evaluate_cui.py so the
phase-2 assembler can turn report/VLM text into asserted finding-CUIs with the
same QuickUMLS + medspaCy ConText pipeline (negation-aware, finding semtypes,
section restriction for sectioned reports).

    nlp = build_nlp()
    cuis = extract_cuis(nlp, text)        # -> {cui: matched_term}
    text = findings_sections(report)      # keep FINDINGS/IMPRESSION of a report
"""
from __future__ import annotations
import os
import re

QUICKUMLS_INDEX = os.environ.get("QUICKUMLS_INDEX", os.path.expandvars("$MMKG_HOME/.cache/quickumls_index"))
THRESHOLD = 0.8
SIMILARITY = "jaccard"

FINDING_SEMTYPES = {
    "T047", "T046", "T033", "T184", "T190", "T020", "T019", "T037", "T191",
    "T074", "T203",
}
EXCLUDE_FLAGS = ("is_negated", "is_hypothetical", "is_family")
STOPTERMS = {
    "unchanged", "change", "changed", "changes", "new", "old", "prior", "previous",
    "interval", "increase", "increased", "decrease", "decreased", "improved",
    "worsened", "stable", "similar", "again", "overlying", "underlying", "persistent",
    "redemonstrated", "redemonstration", "residual", "evidence", "possible", "probable",
    "status", "history", "finding", "findings", "comparison", "chest", "lung", "lungs",
    "thorax", "thoracic", "male", "female", "patient", "present", "seen", "normal",
}
_FINDINGS_SECTIONS = {"FINDINGS", "IMPRESSION", "CONCLUSION"}
_HEADER_RE = re.compile(r"\b([A-Z][A-Z /]{2,}):")


def findings_sections(text: str) -> str:
    heads = list(_HEADER_RE.finditer(text or ""))
    if not heads:
        return text or ""
    keep = []
    for i, h in enumerate(heads):
        name = h.group(1).strip()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        if any(name.startswith(s) for s in _FINDINGS_SECTIONS):
            keep.append(text[h.end():end])
    return "  ".join(keep) if keep else (text or "")


def build_nlp():
    from loguru import logger
    logger.remove()
    import spacy, medspacy            # noqa: F401  (registers factories)
    import quickumls.spacy_component  # noqa: F401
    nlp = spacy.blank("en")
    nlp.add_pipe("medspacy_pyrush")
    nlp.add_pipe("medspacy_quickumls", config={
        "quickumls_fp": QUICKUMLS_INDEX, "threshold": THRESHOLD,
        "similarity_name": SIMILARITY, "result_type": "ents", "best_match": True,
    })
    nlp.add_pipe("medspacy_context")
    return nlp


def extract_cuis(nlp, text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not text or not text.strip():
        return out
    for ent in nlp(text).ents:
        cui = ent.label_
        if not cui or not cui.startswith("C"):
            continue
        semtypes = ent._.semtypes if isinstance(ent._.semtypes, (set, list, tuple)) else set()
        if not (set(semtypes) & FINDING_SEMTYPES):
            continue
        if any(getattr(ent._, flag, False) for flag in EXCLUDE_FLAGS):
            continue
        if ent.text.lower().strip() in STOPTERMS:
            continue
        out.setdefault(cui, ent.text.lower())
    return out
