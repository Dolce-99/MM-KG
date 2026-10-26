"""
12_merge_results.py   --   env: any (stdlib only)
=================================================
Combine the per-config ablation JSONs written by 10 (mortality) and 11 (readmit)
into one results file + printed table per task. Safe to run repeatedly while the
arrays are still finishing (reports whatever configs are done).
"""
from __future__ import annotations
import json, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent

TASKS = [
    ("WHOLE-COHORT EARLY (48h) MORTALITY  [frozen emb, all admissions]", HERE / "outputs" / "mortality" / "results",
     HERE / "outputs" / "mortality" / "mortality_whole_ablation_results.json",
     ["Full", "-KG", "-Imaging", "-Report", "-Labs"], "LabsLR"),
    ("EARLY (48h) MORTALITY  [FINE-TUNED emb, IMAGING-PRESENT subset]", HERE / "outputs" / "mortality" / "results_ft_img",
     HERE / "outputs" / "mortality" / "mortality_ft_img_results.json",
     ["Full", "-KG", "-Imaging", "-Report", "-Labs"], "LabsLR"),
    ("WHOLE-COHORT 30-DAY READMISSION", HERE / "outputs" / "readmit" / "results",
     HERE / "outputs" / "readmit" / "readmit_whole_ablation_results.json",
     ["Full", "-KG", "-Imaging", "-History"], "LR"),
    ("WHOLE-COHORT ICU IN-HOSPITAL MORTALITY (IHM)  [FINE-TUNED emb]", HERE / "outputs" / "ihm" / "results_ft",
     HERE / "outputs" / "ihm" / "ihm_whole_ablation_results.json",
     ["Full", "-History", "-Imaging", "-Report", "-KG"], "LabsLR"),
]


def main():
    for title, resdir, outfile, order, lrname in TASKS:
        files = glob.glob(str(resdir / "cfg_*.json"))
        if not files:
            print(f"\n{title}: (no results yet)")
            continue
        merged = {}
        for fp in files:
            merged.update(json.loads(Path(fp).read_text()))
        outfile.write_text(json.dumps(merged, indent=2))

        lr = merged.get(lrname, {})
        base = lr.get("base"); ntest = lr.get("n"); pos = lr.get("pos")
        print(f"\n===== {title} =====")
        if ntest:
            print(f"test: {ntest} adm, base {base:.3f} (pos {pos})   [done: {len(merged)} configs]")
        for name in order + [k for k in merged if k not in order and k != lrname]:
            r = merged.get(name)
            if not r:
                continue
            print(f"  {name:10s} AUROC {r['auroc_mean']:.3f}±{r['auroc_std']:.3f}  "
                  f"AP {r['ap_mean']:.3f}  ens {r['ensemble_auroc']:.3f} "
                  f"CI[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  (test pos {r.get('test_pos','?')})")
        if lr:
            print(f"  {lrname:10s} AUROC {lr['auroc']:.3f}          AP {lr['ap']:.3f}  "
                  f"CI[{lr['ci95'][0]:.3f},{lr['ci95'][1]:.3f}]")


if __name__ == "__main__":
    main()
