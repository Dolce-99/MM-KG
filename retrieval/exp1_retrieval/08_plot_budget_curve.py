"""AUROC vs prompt tokens per policy (aggregate only).  env: torch_gpu (matplotlib).
   python 08_plot_budget_curve.py --sub q35_test"""
import argparse, csv, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths

ap = argparse.ArgumentParser(); ap.add_argument("--sub", default="q35_test"); a = ap.parse_args()
d = paths.OUT / "exp1" / "analysis" / a.sub
end = {r["tag"]: r for r in csv.DictReader(open(d / "end_task.csv"))}
ret = {r["tag"]: r for r in csv.DictReader(open(d / "retrieval.csv"))}
split = "test" if "test" in a.sub else "dev"
series = {"R1 full context (round-robin)": [f"R1_{split}_bmatch_r4_rr", f"R1_{split}_b2048_rr", f"R1_{split}_b4096_rr", f"R1_{split}_bnatural_rr", f"R1_{split}_bnatural_ctx32k_rr"],
          "R3 one-hop (round-robin)": [f"R3_{split}_bmatch_r4_rr", f"R3_{split}_b2048_rr", f"R3_{split}_b4096_rr", f"R3_{split}_bnatural_rr", f"R3_{split}_bnatural_ctx32k_rr"],
          "R2 vector RAG (SapBERT, KG-only)": [f"R2sap_kg_only_{split}_bmatch_r4", f"R2sap_kg_only_{split}_bnatural"]}
points = {"R4 frozen MM-KG prompt": f"R4_{split}_bnatural", "R5 gold-path oracle": f"R5_{split}_bnatural", "R0 closed book": f"R0_{split}_bnatural"}
fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
for lab, tags in series.items():
    tags = [t for t in tags if t in end]
    x = [float(ret[t]["prompt_tokens"]) for t in tags]; y = [float(end[t]["auroc"]) for t in tags]
    lo = [float(json.loads(end[t]["auroc_ci"])[0]) for t in tags]; hi = [float(json.loads(end[t]["auroc_ci"])[1]) for t in tags]
    ax.errorbar(x, y, yerr=[[yy - l for yy, l in zip(y, lo)], [h - yy for yy, h in zip(y, hi)]], marker="o", ms=4, lw=1.4, capsize=2, label=lab)
mk = {"R4 frozen MM-KG prompt": ("*", 11), "R5 gold-path oracle": ("^", 8), "R0 closed book": ("s", 6)}
for lab, t in points.items():
    if t in end:
        ci = json.loads(end[t]["auroc_ci"]); y = float(end[t]["auroc"])
        ax.errorbar([float(ret[t]["prompt_tokens"])], [y], yerr=[[y - ci[0]], [ci[1] - y]], marker=mk[lab][0], ms=mk[lab][1], ls="none", capsize=2, label=lab, color="k" if "R4" in lab else None)
ax.set_xscale("log"); ax.set_xlabel("prompt tokens (log scale)"); ax.set_ylabel("AUROC"); ax.set_ylim(0.45, 0.95)
ax.set_title(f"Join task, {split} split, Qwen3.5-35B: accuracy vs context cost"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(d / "budget_curve.png"); fig.savefig(d / "budget_curve.pdf"); print("wrote", d / "budget_curve.png")
