"""R4 contexts must equal the frozen D_both prompt of 19_composite_wide_llm.py byte-for-byte;
R5 negatives must carry edges but no gold; R0 must carry no patient text.  (env: umls_eval)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths

ARMS_D = ["demo", "codes", "meds", "labs", "report", "imaging", "kg"]


def frozen_build(it):                       # verbatim 19_composite_wide_llm.build(it, "D_both")
    parts = [it["blocks"][b] for b in ARMS_D if it["blocks"].get(b)]
    return "\n\n".join(parts) + "\n\n" + it["question"]


def test_r4_matches_frozen_prompt():
    bench = paths.load_benchmark(verify=False)
    rows = [json.loads(l) for l in open(paths.RESTRICTED / "exp1" / "contexts" / "R4_dev_bnatural" / "contexts.jsonl")]
    assert len(rows) == 802
    bad = [r["idx"] for r in rows if r["user"] != frozen_build(bench[r["idx"]])]
    assert not bad, f"{len(bad)} R4 prompts differ from the frozen D_both prompt, e.g. {bad[:3]}"


def test_r0_and_r5_shape():
    bench = paths.load_benchmark(verify=False)
    gold = {g["idx"]: g for g in map(json.loads, open(paths.RESTRICTED / "exp1" / "evidence_v1" / "gold.jsonl"))}
    r0 = [json.loads(l) for l in open(paths.RESTRICTED / "exp1" / "contexts" / "R0_dev_bnatural" / "contexts.jsonl")]
    assert all(r["user"] == bench[r["idx"]]["question"] for r in r0)
    r5 = [json.loads(l) for l in open(paths.RESTRICTED / "exp1" / "contexts" / "R5_dev_bnatural" / "contexts.jsonl")]
    neg = [r for r in r5 if gold[r["idx"]]["answer"] == "no"]; pos = [r for r in r5 if gold[r["idx"]]["answer"] == "yes"]
    assert all(r["stats"]["n_kg_edges"] >= 1 for r in neg), "R5 negatives must carry size-matched true edges"
    assert all(r["stats"]["recall_path"] for r in pos), "R5 positives must contain the complete gold path"
    assert abs(sum(r["stats"]["n_kg_edges"] for r in neg) / len(neg) - sum(r["stats"]["n_kg_edges"] for r in pos) / len(pos)) < 0.5


if __name__ == "__main__":
    test_r4_matches_frozen_prompt(); print("ok test_r4_matches_frozen_prompt")
    test_r0_and_r5_shape(); print("ok test_r0_and_r5_shape")
