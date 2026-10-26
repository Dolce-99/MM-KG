"""
23_prepare_v2_benchmark.py   --   env: umls_eval   [CPU]
================================================================================
Two revisions to the wide composite benchmark, producing
composite_benchmark_wide_v2.jsonl for the multi-model runs:

(1) REPHRASE drug_effect. The permissive wording "Could any of this patient's
    documented findings be a known adverse effect of D?" saturates the decision
    threshold (sensitivity 1.00 / specificity 0.06 at arm D) while AUROC (0.77)
    shows the ranking is intact. The v2 wording demands an identified match, so
    "yes" is no longer the safe default. Items, labels, blocks are UNCHANGED --
    only the drug_effect question string differs.

(2) ADD AN ALTERNATIVE KG RENDERING (kg_alt) for every item: the same complete
    fact list re-rendered in the typed-triple style used by published
    graph-RAG systems, with node types and relation names, instead of our
    prose-list style. Arms C_alt/D_alt then test whether the near-chance C and
    the interaction are properties of the channel or artifacts of one prompt
    format. Content is identical by construction: same facts, different syntax.
"""
from __future__ import annotations
import json, re
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs" / "mm"

V2_Q = ("Question: Is at least one of this patient's documented findings a "
        "listed adverse effect of {d}? Answer yes only if you can identify a "
        "specific finding that appears in the adverse-effect list; otherwise "
        "answer no. Answer yes or no.")

REL = {"contra": "contraindicated_in", "drug_effect": "has_adverse_effect",
       "mechanism": "targets_protein_implicated_in"}


def kg_alt_block(item):
    """Same facts as blocks['kg'], re-rendered as typed graph-RAG triples."""
    kg = item["blocks"]["kg"]
    if not kg:
        return ""
    header, _, body = kg.partition(":\n")
    m = re.search(r"^Facts from a curated biomedical knowledge graph\.\s+(\S.*?)\s+"
                  r"(is contraindicated in|has the following documented adverse effects|"
                  r"acts on proteins implicated in)", header)
    drug = m.group(1) if m else item["drug"]
    phrase = m.group(2) if m else ""
    rel = ("contraindicated_in" if "contraindicated" in phrase else
           "has_adverse_effect" if "adverse" in phrase else
           "targets_protein_implicated_in")
    ents = [ln.strip() for ln in body.split("\n") if ln.strip()]
    lines = [f"(:Drug {{name: \"{drug}\"}})-[:{rel.upper()}]->"
             f"(:Condition {{name: \"{e}\"}})" for e in ents]
    return ("Knowledge-graph retrieval results (complete; "
            f"{len(ents)} triples):\n" + "\n".join(lines))


def main():
    src = OUT / "composite_benchmark_wide.jsonl"
    dst = OUT / "composite_benchmark_wide_v2.jsonl"
    n = n_q = 0
    with open(dst, "w") as w:
        for line in open(src):
            it = json.loads(line); n += 1
            if it["family"] == "drug_effect":
                it["question"] = V2_Q.format(d=it["drug"]); n_q += 1
            it["blocks"]["kg_alt"] = kg_alt_block(it)
            w.write(json.dumps(it) + "\n")
    print(f"{n} items -> {dst} | drug_effect questions rephrased: {n_q}")


if __name__ == "__main__":
    main()
