"""
bridge_v2_validate.py   --   env: umls_eval   [CPU, fast]
================================================================================
Verify bridge v2 against the SPECIFIC defects the audits identified. Coverage
going up is necessary but not sufficient -- a bridge can gain coverage by adding
wrong links, so every named error gets an explicit regression test here.

Checks:
  A. the billing-CUI dead ends now resolve (25000, I4891, 41401, 486, 49390, ...)
  B. the known WRONG mappings are gone (septic shock -> toxic shock syndrome;
     depression -> anxiety disorder; E7800 -> familial hypercholesterolemia;
     E8339 -> hypophosphatasia)
  C. HPO twins are attached, so the drug_effect layer is reachable for conditions
  D. informative-edge mass per mapped row, v1 vs v2
  E. drug side: vehicles gone, aspirin/insulin/pip-tazo now mapped, no d->Aspartic acid
"""
from __future__ import annotations
import os
import csv, json
from collections import defaultdict
from pathlib import Path
import pandas as pd

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
MIMIC   = Path(os.environ.get("MIMIC_IV_DIR", os.path.expandvars("$MMKG_HOME/data/mimiciv/3.1"))) / "hosp"
HERE    = Path(__file__).resolve().parent
V2      = HERE / "outputs" / "bridge_v2"

INFORMATIVE = {"disease_phenotype_positive", "disease_protein", "indication",
               "contraindication", "off-label use", "drug_effect"}


def load_nodes():
    idx2name, idx2type = {}, {}
    with open(PRIMEKG / "nodes.tab") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["node_index"]); idx2name[i] = r["node_name"]; idx2type[i] = r["node_type"]
    return idx2name, idx2type


def main():
    idx2name, idx2type = load_nodes()

    v2 = {}
    for line in open(V2 / "icd2primekg_v2.tsv"):
        code, ver, idxs, level, nrows = line.rstrip("\n").split("\t")
        v2[code] = ([int(i) for i in idxs.split("|")], level, int(nrows))
    v1 = {}
    for line in open(PRIMEKG / "bridge" / "icd2primekg.tsv"):
        c, i = line.rstrip("\n").split("\t")
        v1[c] = [int(x) for x in i.split("|")]

    def names(idxs, k=4):
        return ", ".join(f"{idx2name.get(i,'?')}[{idx2type.get(i,'?')[:4]}]" for i in idxs[:k])

    print("=" * 78)
    print("A. BILLING-CUI DEAD ENDS  (v1 reached nothing; do they resolve now?)")
    print("=" * 78)
    for code, label in [("25000", "type 2 diabetes"), ("I4891", "atrial fibrillation"),
                        ("41401", "coronary atherosclerosis"), ("486", "pneumonia"),
                        ("49390", "asthma"), ("2724", "hyperlipidemia"), ("496", "COPD"),
                        ("J9601", "acute resp failure w/ hypoxia"), ("2761", "hyponatremia")]:
        a = "YES" if code in v1 else "no "
        b, lvl = (("YES", v2[code][1]) if code in v2 else ("NO ", "-"))
        got = names(v2[code][0]) if code in v2 else ""
        print(f"  {code:8s} {label:30s} v1={a} v2={b} [{lvl:11s}] {got[:70]}")

    print()
    print("=" * 78)
    print("B. KNOWN WRONG MAPPINGS  (must be ABSENT in v2)")
    print("=" * 78)
    bad = [("78552", "septic shock", "toxic shock"),
           ("311", "depressive disorder", "anxiety"),
           ("F329", "major depressive disorder", "anxiety"),
           ("E7800", "pure hypercholesterolemia", "familial"),
           ("E8339", "phosphorus metabolism disorder", "hypophosphatasia")]
    for code, label, forbidden in bad:
        v1n = names(v1.get(code, []), 6).lower()
        v2n = names(v2[code][0], 6).lower() if code in v2 else ""
        was = "PRESENT" if forbidden in v1n else "absent"
        now = "STILL PRESENT ***" if forbidden in v2n else "fixed"
        print(f"  {code:7s} {label:32s} '{forbidden}' in v1: {was:8s} -> v2: {now}")
        if code in v2:
            print(f"          v2 now -> {names(v2[code][0], 4)[:80]}")

    print()
    print("=" * 78)
    print("C. HPO TWIN ATTACHED?  (drug_effect lives ONLY on HPO nodes)")
    print("=" * 78)
    for code, label in [("5849", "acute kidney failure"), ("N179", "acute kidney failure"),
                        ("4280", "congestive heart failure"), ("2767", "hyperkalemia"),
                        ("E875", "hyperkalemia"), ("51881", "acute respiratory failure")]:
        if code not in v2:
            print(f"  {code:7s} {label:28s} UNMAPPED"); continue
        idxs = v2[code][0]
        n_hpo = sum(1 for i in idxs if idx2type.get(i) == "effect/phenotype")
        n_dis = sum(1 for i in idxs if idx2type.get(i) == "disease")
        v1_hpo = sum(1 for i in v1.get(code, []) if idx2type.get(i) == "effect/phenotype")
        print(f"  {code:7s} {label:28s} v1 HPO={v1_hpo}  ->  v2 HPO={n_hpo}, disease={n_dis}")

    print()
    print("=" * 78)
    print("D. INFORMATIVE EDGE MASS reachable from the diagnosis layer")
    print("=" * 78)
    deg = defaultdict(int)
    for ch in pd.read_csv(PRIMEKG / "kg.csv", usecols=["relation", "x_index", "y_index"],
                          dtype={"x_index": "int32", "y_index": "int32"}, chunksize=2_000_000):
        ch = ch[ch.relation.isin(INFORMATIVE)]
        for x, y in zip(ch.x_index, ch.y_index):
            deg[int(x)] += 1; deg[int(y)] += 1
    dx = pd.read_csv(MIMIC / "diagnoses_icd.csv.gz", usecols=["icd_code"])
    rows = dx.icd_code.astype(str).str.replace(".", "", regex=False).value_counts().to_dict()
    m1 = sum(rows.get(c, 0) * sum(deg.get(i, 0) for i in idxs) for c, idxs in v1.items())
    m2 = sum(rows.get(c, 0) * sum(deg.get(i, 0) for i in idxs) for c, (idxs, _, _) in v2.items())
    zero1 = sum(rows.get(c, 0) for c, idxs in v1.items() if sum(deg.get(i, 0) for i in idxs) == 0)
    zero2 = sum(rows.get(c, 0) for c, (idxs, _, _) in v2.items() if sum(deg.get(i, 0) for i in idxs) == 0)
    tot = sum(rows.values())
    print(f"  informative-edge mass  v1 {m1:,}  ->  v2 {m2:,}   ({m2/max(m1,1):.2f}x)")
    print(f"  rows landing on 0-informative nodes: v1 {zero1/tot:.1%} -> v2 {zero2/tot:.1%}")

    # ---- E. drug side -------------------------------------------------------
    p = V2 / "drug2primekg_v2.tsv"
    if p.exists():
        print()
        print("=" * 78)
        print("E. DRUG BRIDGE v2")
        print("=" * 78)
        d2 = {}
        for line in open(p):
            d, i, r, n = line.rstrip("\n").split("\t"); d2[d] = (int(i), r, int(n))
        d1 = {}
        for line in open(PRIMEKG / "bridge" / "drug2primekg.tsv"):
            n_, i = line.rstrip("\n").split("\t"); d1[n_] = int(i)
        for probe in ["aspirin", "insulin", "piperacillin-tazobactam", "torsemide",
                      "albuterol", "phytonadione", "sulfameth/trimethoprim ds"]:
            hits = [k for k in d2 if k.startswith(probe[:12])]
            if hits:
                k = hits[0]; i, r, n = d2[k]
                print(f"  {probe:26s} v2 -> {idx2name.get(i,'?'):28s} [{r}] {n:,} rx")
            else:
                print(f"  {probe:26s} still unmapped")
        veh = [l.split('\t')[0] for l in open(V2 / "drug_vehicles.tsv")]
        print(f"  vehicles excluded: {len(veh):,}  e.g. {veh[:3]}")
        bad_d = [k for k in d2 if idx2name.get(d2[k][0], "").lower() == "aspartic acid"]
        print(f"  'd5*' -> Aspartic acid bug: {len(bad_d)} remaining (was 19,211 rx)")


if __name__ == "__main__":
    main()
