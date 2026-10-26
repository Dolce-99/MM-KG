"""Print a markdown table of every context summary in outputs/exp1/contexts (aggregate only)."""
import glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths

rows = []
for f in sorted(glob.glob(str(paths.OUT / "exp1" / "contexts" / "*_summary.json"))):
    j = json.load(open(f)); m = j["means"]; q = j["prompt_tokens_quantiles"]; u = j["untruncated_tokens_quantiles"]
    fam = " / ".join(f"{m.get(f'recall_path|{k}', 0) or 0:.2f}" for k in ("contra_labs", "contra_report", "drug_effect", "mechanism"))
    rows.append((j["tag"], j["n_items"], int(q["0.5"]), int(q["1.0"]), int(u["0.5"]), int(u["1.0"]), m.get("truncated", 0),
                 m.get("n_kg_edges", 0), m.get("recall_patient"), m.get("recall_kg_edge"), m.get("recall_path"), fam, m.get("retrieval_ms")))
print("| tag | n | prompt tok med | max | untruncated med | max | truncated | KG edges | R_patient | R_edge | R_path | R_path by family (labs/report/effect/mech) | retrieval ms |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(("%.2f" % x if isinstance(x, float) else str(x)) for x in r) + " |")
