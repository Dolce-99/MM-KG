"""
00_mrconso_subset.py   --   env: umls_eval   [CPU, a few minutes; streams MRCONSO once]
================================================================================
Deduplicated CUI / SAB / CODE projection of the UMLS MRCONSO.RRF restricted to
the vocabularies the bridge needs. No language, suppression or term-status
filter is applied.

Output: $PRIMEKG_DIR/mrconso_subset.tsv   (read by build_crosswalk.py,
        04_patient_tables.py and 13_icu_ihm_features.py)
"""
import os
from pathlib import Path

META = Path(os.environ.get("UMLS_META_DIR", os.path.expandvars("$MMKG_HOME/data/umls/2026AA/META")))
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
SABS = {"SNOMEDCT_US", "MSH", "RXNORM", "NCI", "OMIM", "ICD10CM", "MDR", "ICD9CM", "HPO"}


def main():
    rows = set()
    with open(META / "MRCONSO.RRF", encoding="utf-8") as f:
        for line in f:
            p = line.split("|")
            if p[11] in SABS:
                rows.add((p[0], p[11], p[13]))
    dest = PRIMEKG / "mrconso_subset.tsv"
    with open(dest, "w", encoding="utf-8") as w:
        for cui, sab, code in sorted(rows):
            w.write(f"{cui}\t{sab}\t{code}\n")
    print(f"{len(rows)} rows -> {dest}")


if __name__ == "__main__":
    main()
