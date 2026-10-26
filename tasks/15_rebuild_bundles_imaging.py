"""
15_rebuild_bundles_imaging.py   --   env: umls_eval   [CPU]
================================================================================
Swap the imaging block of the existing bundles for the recalibrated channel from
14_recalibrate_imaging.py. Everything else -- demographics, history, meds, labs,
report, KG -- is copied verbatim, so the only thing that changes between the v1
and v2 runs is how the chest film is described.

Two variants, because thresholding and confidence are separate questions:
  v2a  hard list, calibrated and restricted to detectable findings
  v2b  the same findings with a confidence word and probability, letting the
       model discount a weak detection instead of reading it as fact

Note the "no_imaging" arm needs no re-run: its prompts contain no imaging block
at all and are byte-identical across versions. Only "full" has to be rescored.

Output: outputs/mm/mm_bundles_ihm_v2a.jsonl, mm_bundles_ihm_v2b.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P2   = HERE.parent / "graph" / "outputs"
OUT  = HERE / "outputs" / "mm"


def main():
    v2 = pd.read_parquet(OUT / "imaging_v2.parquet")
    hard = {int(s): list(p) for s, p in zip(v2.study_id, v2.present_v2)}
    soft = {int(s): list(g) for s, g in zip(v2.study_id, v2.graded_v2)}

    cxr = pd.read_parquet(P2 / "ihm" / "icu_early_cxr.parquet")
    study_of = dict(zip(cxr.hadm_id.astype(int), cxr.study_id.astype(int)))

    src = OUT / "mm_bundles_ihm.jsonl"
    wa = open(OUT / "mm_bundles_ihm_v2a.jsonl", "w")
    wb = open(OUT / "mm_bundles_ihm_v2b.jsonl", "w")
    n = na = nb = n_old = 0

    for line in open(src):
        it = json.loads(line)
        n += 1
        n_old += bool(it["blocks"]["imaging"])
        st = study_of.get(it["hadm_id"])

        f_hard = hard.get(st, []) if st is not None else []
        f_soft = soft.get(st, []) if st is not None else []

        a_txt = ("Chest radiograph, automated image analysis reports: "
                 + ", ".join(f_hard) + ".") if f_hard else ""
        b_txt = ("Chest radiograph, automated image analysis (confidence shown; "
                 "this detector is imperfect): " + "; ".join(f_soft) + ".") \
            if f_soft else ""

        for txt, w, cnt in ((a_txt, wa, "a"), (b_txt, wb, "b")):
            o = {**it, "blocks": {**it["blocks"], "imaging": txt},
                 "avail": {**it["avail"], "imaging": bool(txt)}}
            w.write(json.dumps(o) + "\n")
        na += bool(a_txt); nb += bool(b_txt)

    wa.close(); wb.close()
    print(f"{n} admissions")
    print(f"  imaging block present   OLD {n_old:6d} ({n_old/n:.1%})")
    print(f"                          v2a {na:6d} ({na/n:.1%})")
    print(f"                          v2b {nb:6d} ({nb/n:.1%})")
    print(f"wrote -> {OUT}/mm_bundles_ihm_v2a.jsonl , _v2b.jsonl")


if __name__ == "__main__":
    main()
