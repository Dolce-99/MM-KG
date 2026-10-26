"""
09a_build_audit_cohort.py   --   env: umls_eval   [CPU, ~3 min]
================================================================================
Build the audit patient base DIRECTLY FROM THE WHOLE GRAPH.

WHY THIS EXISTS
The medication-audit benchmark was built on top of `medsafety_dataset_aki_v2.jsonl`,
which inherited two eligibility rules from the ADE PREDICTION pipeline:

  * admission observed for >= 48h        (needed to define an outcome window)
  * AKI-assessable: baseline AND outcome creatinine present   (needed for the label)

Neither has anything to do with a fact-checking question such as "is this drug
contraindicated in this condition?". They cost us 66% of admissions and 56% of
patients for no methodological reason:

  whole cohort                  219,720 adm / 51,844 patients
  after >=48h filter            132,949 adm / 40,713 patients
  after AKI-assessable filter    94,783 adm / 32,692 patients
  after conditions+drugs          74,348 adm / 22,771 patients   <- what we used

This script applies ONLY the filter the task actually requires -- an admission must
have at least one drug and at least one condition that map into PrimeKG -- reading
prescriptions and diagnoses straight from the whole-cohort patient graph and mapping
them with bridge v2.

Output: outputs/audit/audit_cohort.jsonl
        one row per admission: hadm_id, subject_id, drug node ids, condition node ids,
        drug display names. Drop-in replacement for the `pat` list in 06a/08a.
"""
from __future__ import annotations
import os
import csv, json
from collections import defaultdict
from pathlib import Path
import pandas as pd

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
MIMIC   = Path(os.environ.get("MIMIC_IV_DIR", os.path.expandvars("$MMKG_HOME/data/mimiciv/3.1"))) / "hosp"
WG      = Path(__file__).resolve().parent.parent / "graph"
HERE    = Path(__file__).resolve().parent
OUT     = HERE / "outputs" / "audit"
V2      = HERE / "outputs" / "bridge_v2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx2name, idx2type = {}, {}
    with open(PRIMEKG / "nodes.tab") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["node_index"]); idx2name[i] = r["node_name"]; idx2type[i] = r["node_type"]

    drug2pkg, vehicles = {}, set()
    for line in open(V2 / "drug2primekg_v2.tsv"):
        d, i, r, n = line.rstrip("\n").split("\t"); drug2pkg[d] = int(i)
    for line in open(V2 / "drug_vehicles.tsv"):
        vehicles.add(line.split("\t")[0])
    icd2pkg = {}
    for line in open(V2 / "icd2primekg_v2.tsv"):
        code, ver, idxs, lvl, n = line.rstrip("\n").split("\t")
        icd2pkg[(int(ver), code)] = [int(i) for i in idxs.split("|")]
    print(f"bridge v2: {len(drug2pkg):,} drugs ({len(vehicles)} vehicles excluded), "
          f"{len(icd2pkg):,} icd codes")

    # ---- ALL admissions in the whole graph ---------------------------------
    nodes = pd.read_parquet(WG / "outputs" / "patient_graph" / "nodes.parquet",
                            columns=["key", "ntype"])
    all_adm = {int(k.split(":", 1)[1]) for k in nodes.loc[nodes.ntype == "Admission", "key"]}
    all_pt = int((nodes.ntype == "Patient").sum())
    print(f"whole cohort: {len(all_adm):,} admissions / {all_pt:,} patients")

    edges = pd.read_parquet(WG / "outputs" / "patient_graph" / "edges.parquet",
                            columns=["src", "dst", "etype"])
    pres = edges[edges.etype == "PRESCRIBED"]
    adm_drugs = defaultdict(list)
    for s, d in zip(pres.src, pres.dst):
        name = d.split(":", 1)[1]
        if name in vehicles: continue                     # saline flushes, line-locks etc.
        adm_drugs[int(s.split(":", 1)[1])].append(name)
    print(f"admissions with >=1 non-vehicle drug: {len(adm_drugs):,}")

    # conditions: the admission's own coded diagnoses, mapped through bridge v2
    dx = pd.read_csv(MIMIC / "diagnoses_icd.csv.gz",
                     usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    dx = dx[dx.hadm_id.isin(all_adm)]
    dx["c"] = dx.icd_code.astype(str).str.replace(".", "", regex=False).str.strip()
    adm_conds, adm_subj = defaultdict(set), {}
    for r in dx.itertuples(index=False):
        adm_subj[int(r.hadm_id)] = int(r.subject_id)
        for i in icd2pkg.get((int(r.icd_version), r.c), []):
            if idx2type.get(i) == "disease": adm_conds[int(r.hadm_id)].add(i)
    print(f"admissions with >=1 mapped condition: {len(adm_conds):,}")

    rows = []
    for h in sorted(all_adm):
        names = adm_drugs.get(h, [])
        dn = sorted({drug2pkg[m] for m in names if m in drug2pkg})
        cn = sorted(adm_conds.get(h, ()))
        if not dn or not cn: continue                     # the ONLY task-required filter
        rows.append({"hadm_id": h, "subject_id": adm_subj.get(h),
                     "drug_nodes": dn, "cond_nodes": cn,
                     "drug_names": [m for m in names if m in drug2pkg][:40]})
    with open(OUT / "audit_cohort.jsonl", "w") as w:
        for r in rows: w.write(json.dumps(r) + "\n")

    pts = len({r["subject_id"] for r in rows})
    print(f"\n=== AUDIT COHORT (whole graph, task-required filter only) ===")
    print(f"  admissions {len(rows):,} / {len(all_adm):,} ({len(rows)/len(all_adm):.0%})")
    print(f"  patients   {pts:,} / {all_pt:,} ({pts/all_pt:.0%})")
    print(f"  previously (inherited prediction filters): 74,348 adm / 22,771 patients")
    print(f"  gain: {len(rows)/74348:.2f}x admissions, {pts/22771:.2f}x patients")
    print(f"  mean drugs/admission {sum(len(r['drug_nodes']) for r in rows)/len(rows):.1f} | "
          f"mean conditions {sum(len(r['cond_nodes']) for r in rows)/len(rows):.1f}")
    print(f"\nwrote -> {OUT}/audit_cohort.jsonl")


if __name__ == "__main__":
    main()
