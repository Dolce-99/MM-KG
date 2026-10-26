"""
14_recalibrate_imaging.py   --   env: umls_eval   [CPU]
================================================================================
Rebuild the chest-imaging channel from the SAVED BiomedCLIP scores. No GPU and
no re-inference: 02_biomedclip_findings.py already stored a probability per
finding, and only the thresholding was wrong.

WHY
  The shipped channel used a flat PRESENT_THRESHOLD = 0.5 on a binary softmax.
  Against MIMIC-CXR's own CheXpert labels that fires "Pleural Other" 21.8x more
  often than the finding actually occurs, "Fracture" 8.1x, "Enlarged
  Cardiomediastinum" 5.9x -- a median of 5 of 13 findings per study. An LLM
  reading that list conditions on findings that are mostly false.

HOW
  Calibrate against mimic-cxr-2.0.0-chexpert.csv.gz -- the reference standard
  shipped with MIMIC-CXR, and INDEPENDENT of the mortality outcome we later
  evaluate on. Thresholds are chosen on a random half of studies and reported on
  the held-out half. Nothing here ever looks at mortality; tuning an input
  channel against the evaluation label would be test-set optimisation.

  1. Drop findings BiomedCLIP cannot detect (AUROC < AUROC_MIN against CheXpert).
     Emitting a coin-flip finding adds noise the reader cannot discount.
  2. For the survivors pick the lowest threshold reaching TARGET_PRECISION; if
     that is unreachable, fall back to the max-F1 threshold and record the fact.

TWO REPRESENTATIONS ARE WRITTEN, because thresholding destroys information the
reader could otherwise use:
  present_v2  a hard list, as before but calibrated and restricted
  graded_v2   findings with a confidence word and the probability, so the model
              can discount a weak detection instead of treating it as fact

Output: outputs/mm/imaging_v2.parquet   (study_id, present_v2, graded_v2)
        outputs/mm/imaging_v2_calibration.json
"""
from __future__ import annotations
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE     = Path(__file__).resolve().parent
P2       = HERE.parent / "graph" / "outputs"
OUT      = HERE / "outputs" / "mm"
CHEXPERT = Path(os.environ.get("MIMIC_CXR_DIR", os.path.expandvars("$MMKG_HOME/data/mimic-cxr-jpg/2.1.0"))) / "mimic-cxr-2.0.0-chexpert.csv.gz"

AUROC_MIN        = 0.70   # below this the detector is not usable
TARGET_PRECISION = 0.50   # of emitted findings, at least half should be right
MIN_RECALL       = 0.20   # a threshold so strict it never fires is not useful
PRECISION_FLOOR  = 0.25   # emitting a finding wrong 3 times in 4 misleads the reader
SPLIT_SEED       = 42

OUT.mkdir(parents=True, exist_ok=True)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def main():
    img = pd.read_parquet(P2 / "imaging" / "biomedclip_findings.parquet")
    img["study_id"] = img.study_id.astype("int64")
    score_cols = [c for c in img.columns if c.startswith("score_")]

    cx = pd.read_csv(CHEXPERT)
    m = img.merge(cx, on="study_id", how="inner")
    print(f"{len(img)} scored studies | {len(m)} with CheXpert reference labels",
          flush=True)

    rng = np.random.default_rng(SPLIT_SEED)
    cal = rng.random(len(m)) < 0.5

    calib, kept = {}, []
    print(f"\n{'finding':28s}{'AUROC':>7}{'thr':>7}{'HOprec':>8}{'HOrec':>7}"
          f"{'fire%':>7}  rule")
    for c in score_cols:
        lab = c[len("score_"):].replace("_", " ")
        if lab not in cx.columns:
            continue
        y = (m[lab] == 1.0).astype(int).values      # NaN/-1 -> absent (U-zeros)
        s = m[c].values
        a = auroc(y, s)
        if not (a >= AUROC_MIN):
            print(f"  {lab:26s}{a:7.3f}{'--':>7}{'':>8}{'':>7}{'':>7}  "
                  f"DROP (AUROC < {AUROC_MIN})")
            calib[lab] = {"auroc": round(a, 4), "kept": False}
            continue

        grid = np.quantile(s[cal], np.linspace(0.50, 0.9995, 400))
        npos_cal = max(int((y[cal] == 1).sum()), 1)
        thr, rule = None, ""
        for t in grid:            # lowest thr meeting BOTH precision and recall
            pred = s[cal] >= t
            if pred.sum() < 50:
                continue
            tp = int((pred & (y[cal] == 1)).sum())
            if (tp / pred.sum() >= TARGET_PRECISION
                    and tp / npos_cal >= MIN_RECALL):
                thr, rule = float(t), f"precision>={TARGET_PRECISION}, recall>={MIN_RECALL}"
                break
        if thr is None:                             # fall back to max F1
            best = (-1.0, 0.5)
            for t in grid:
                pred = s[cal] >= t
                tp = (pred & (y[cal] == 1)).sum()
                if tp == 0 or pred.sum() < 50:
                    continue
                pr = tp / pred.sum(); rc = tp / max((y[cal] == 1).sum(), 1)
                f1 = 2 * pr * rc / (pr + rc)
                if f1 > best[0]:
                    best = (f1, float(t))
            thr, rule = best[1], "max-F1 (target precision unreachable)"

        ho = ~cal
        pred = s[ho] >= thr
        tp = int((pred & (y[ho] == 1)).sum())
        pr = tp / max(int(pred.sum()), 1)
        rc = tp / max(int((y[ho] == 1).sum()), 1)
        if pr < PRECISION_FLOOR:
            print(f"  {lab:26s}{a:7.3f}{thr:7.3f}{pr:8.3f}{rc:7.3f}"
                  f"{pred.mean():7.3f}  DROP (precision < {PRECISION_FLOOR})")
            calib[lab] = {"auroc": round(a, 4), "kept": False,
                          "heldout_precision": round(pr, 4),
                          "reason": f"best achievable precision {pr:.3f} "
                                    f"< floor {PRECISION_FLOOR}"}
            continue
        calib[lab] = {"auroc": round(a, 4), "kept": True, "threshold": round(thr, 4),
                      "heldout_precision": round(pr, 4), "heldout_recall": round(rc, 4),
                      "fire_rate": round(float(pred.mean()), 4),
                      "chexpert_prevalence": round(float(y.mean()), 4), "rule": rule}
        kept.append((lab, c, thr, pr))
        print(f"  {lab:26s}{a:7.3f}{thr:7.3f}{pr:8.3f}{rc:7.3f}"
              f"{pred.mean():7.3f}  {rule}")

    print(f"\nkept {len(kept)} of {len(score_cols)} findings: "
          f"{sorted(k[0] for k in kept)}")

    # ---- emit both representations for EVERY scored study ------------------
    # The graded form deliberately does NOT print the raw BiomedCLIP softmax
    # score: it is not a calibrated probability, so "p=0.06" on a finding whose
    # threshold is 0.03 reads as near-certain absence when it is in fact a
    # detection, and would push the reader the wrong way. We report instead the
    # measured held-out precision of that finding -- how often a detection of
    # this kind is actually right -- plus a within-finding strength word.
    def strength(p, thr):
        return "strong" if p >= thr + (1 - thr) * 0.5 else "weak"

    present, graded = [], []
    for row in img[["study_id"] + [k[1] for k in kept]].itertuples(index=False):
        hard, soft = [], []
        for (lab, _c, thr, pr), p in zip(kept, row[1:]):
            if p >= thr:
                hard.append(lab)
                soft.append(f"{lab} ({strength(p, thr)} signal; detections of "
                            f"this finding are correct about {pr*100:.0f}% of the time)")
        present.append(hard); graded.append(soft)

    out = pd.DataFrame({"study_id": img.study_id.values,
                        "present_v2": present, "graded_v2": graded})
    out.to_parquet(OUT / "imaging_v2.parquet", index=False)

    n_f = out.present_v2.map(len)
    old = pd.read_parquet(P2 / "imaging" / "biomedclip_findings.parquet",
                          columns=["present"]).present.map(len)
    print(f"\nfindings per study   OLD median {np.median(old):.0f} mean {old.mean():.2f}"
          f"  |  NEW median {np.median(n_f):.0f} mean {n_f.mean():.2f}")
    print(f"studies with >=1 finding  OLD {(old > 0).mean():.1%}  NEW {(n_f > 0).mean():.1%}")

    json.dump({"auroc_min": AUROC_MIN, "target_precision": TARGET_PRECISION,
               "split_seed": SPLIT_SEED, "reference": CHEXPERT.name,
               "note": "calibrated against CheXpert labels only; the mortality "
                       "outcome was never consulted",
               "findings": calib}, open(OUT / "imaging_v2_calibration.json", "w"),
              indent=2)
    print(f"\nwrote -> {OUT/'imaging_v2.parquet'} and imaging_v2_calibration.json")


if __name__ == "__main__":
    main()
