"""
build_crosswalk.py   --   env: umls_eval (or any python3; stdlib only)
======================================================================
Build the patient-finding -> PrimeKG bridge crosswalks (the ALIGNS_TO_CUI
backbone) and report coverage. We map to PrimeKG node_index via two concept
families:
  * phenotypes -> HPO  (HPO is a UMLS source vocab, so CUI->HPO is direct)
  * diseases   -> MONDO (via mondo.sssom.tsv, using MULTIPLE xref routes to
                  maximize coverage: UMLS, MedGen, SNOMED, OMIM, MeSH, NCIt,
                  ICD10CM -- not just UMLS)

Produces (in <primekg>/bridge/):
  cui2primekg.tsv    cui            -> node_index(s)   (pipe-joined)
  icd2primekg.tsv    icd_code(nodot)-> node_index(s)
  drug2primekg.tsv   drug_lower     -> node_index      (PrimeKG DrugBank, name match)
and prints coverage (canonical CXR findings, phase-1 CUIs, weighted MIMIC dx rows).

Inputs assumed present (created earlier this session):
  <primekg>/nodes.tab, mondo.sssom.tsv, mrconso_subset.tsv  (CUI<TAB>SAB<TAB>CODE)
"""
from __future__ import annotations
import os
import csv, gzip, json
from collections import defaultdict
from pathlib import Path

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
MIMIC   = Path(os.environ.get("MIMIC_IV_DIR", os.path.expandvars("$MMKG_HOME/data/mimiciv/3.1"))) / "hosp"
PHASE1_TERMS = Path(os.environ.get("MMKG_EXTRA_CUI_TERMS", "/nonexistent"))  # optional JSON list of CUIs for the coverage report
OUT = PRIMEKG / "bridge"

# sssom object prefix -> MRCONSO SAB that shares that code space
SSSOM_TO_SAB = {"SCTID": "SNOMEDCT_US", "OMIM": "OMIM", "mesh": "MSH", "NCIT": "NCI"}


def norm(code: str) -> str:
    return code.replace(".", "").strip()


def load_primekg_nodes():
    hpo2idx, mondo2idx, drug2idx, info = {}, {}, {}, {}
    with open(PRIMEKG / "nodes.tab") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            idx = int(r["node_index"]); src = r["node_source"]
            info[idx] = (r["node_type"], r["node_name"])
            if src == "HPO":
                try: hpo2idx[int(r["node_id"])] = idx
                except ValueError: pass
            elif src in ("MONDO", "MONDO_grouped"):
                for part in str(r["node_id"]).split("_"):
                    try: mondo2idx[int(part)] = idx
                    except ValueError: pass
            elif src == "DrugBank":
                drug2idx[r["node_name"].lower()] = idx
    return hpo2idx, mondo2idx, drug2idx, info


def load_mondo_xrefs():
    """vocab -> {code -> set(mondo_int)}, exactMatch only."""
    by_vocab: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    with open(PRIMEKG / "mondo.sssom.tsv") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 4 or c[0] == "subject_id" or c[2] != "skos:exactMatch":
                continue
            if not c[0].startswith("MONDO:"):
                continue
            mid = int(c[0].split(":")[1])
            pref, _, val = c[3].partition(":")
            by_vocab[pref][val].add(mid)
    return by_vocab


def load_mrconso_subset():
    """CUI -> {SAB -> set(code)}."""
    cui2codes: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    with open(PRIMEKG / "mrconso_subset.tsv") as f:
        for line in f:
            cui, sab, code = line.rstrip("\n").split("\t")
            cui2codes[cui][sab].add(code)
    return cui2codes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading PrimeKG nodes, MONDO xrefs, MRCONSO subset ...", flush=True)
    hpo2idx, mondo2idx, drug2idx, info = load_primekg_nodes()
    xref = load_mondo_xrefs()
    cui2codes = load_mrconso_subset()

    # --- CUI -> set(MONDO int) via every available route ---------------------
    umls_direct = xref.get("UMLS", {})
    medgen = xref.get("MEDGEN", {})       # MedGen ids are CUIs
    icd10_mondo = {norm(k): v for k, v in xref.get("ICD10CM", {}).items()}

    def cui_to_mondo(cui: str) -> set[int]:
        out = set(umls_direct.get(cui, ())) | set(medgen.get(cui, ()))
        codes = cui2codes.get(cui, {})
        for pref, sab in SSSOM_TO_SAB.items():
            for code in codes.get(sab, ()):
                out |= xref[pref].get(code, set())
        return out

    # --- CUI -> set(HPO int) -------------------------------------------------
    def cui_to_hpo(cui: str) -> set[int]:
        return {int(c.replace("HP:", "").lstrip("0") or 0)
                for c in cui2codes.get(cui, {}).get("HPO", ())}

    def cui_to_idx(cui: str) -> set[int]:
        idxs = {hpo2idx[h] for h in cui_to_hpo(cui) if h in hpo2idx}
        idxs |= {mondo2idx[m] for m in cui_to_mondo(cui) if m in mondo2idx}
        return idxs

    # ===== write cui2primekg over all CUIs we can reach ======================
    all_cuis = set(cui2codes) | set(umls_direct) | set(medgen)
    cui_map = {}
    for cui in all_cuis:
        idxs = cui_to_idx(cui)
        if idxs:
            cui_map[cui] = idxs
    with open(OUT / "cui2primekg.tsv", "w") as f:
        for cui, idxs in cui_map.items():
            f.write(f"{cui}\t{'|'.join(map(str, sorted(idxs)))}\n")
    print(f"cui2primekg: {len(cui_map)} CUIs map to >=1 PrimeKG node", flush=True)

    # ===== ICD -> PrimeKG (direct sssom ICD10CM + via CUI) ===================
    icd2cui = defaultdict(set)
    for cui, sabd in cui2codes.items():
        for sab in ("ICD10CM", "ICD9CM"):
            for code in sabd.get(sab, ()):
                icd2cui[norm(code)].add(cui)

    def icd_to_idx(code: str) -> set[int]:
        code = norm(code)
        idxs = {mondo2idx[m] for m in icd10_mondo.get(code, ()) if m in mondo2idx}
        for cui in icd2cui.get(code, ()):
            idxs |= cui_map.get(cui, set())
        return idxs

    # ===== drug -> PrimeKG (DrugBank node name match) ========================
    # build once for coverage; full med handling can add RxNorm/NDC later
    with open(OUT / "drug2primekg.tsv", "w") as f:
        for name, idx in drug2idx.items():
            f.write(f"{name}\t{idx}\n")

    # ===================== COVERAGE REPORT ===================================
    print("\n===== bridge coverage =====")
    chex = {"C0004144": "atelectasis", "C0018800": "cardiomegaly", "C0521530": "consolidation",
            "C0034063": "pulmonary edema", "C0032227": "pleural effusion", "C0032285": "pneumonia",
            "C0032326": "pneumothorax", "C0016658": "fracture", "C2073563": "lung opacity"}
    hit = sum(bool(cui_to_idx(c)) for c in chex)
    print(f"canonical CXR findings: {hit}/{len(chex)}  " +
          ", ".join(n for c, n in chex.items() if not cui_to_idx(c)) + " still unmapped")
    if PHASE1_TERMS.exists():
        terms = json.loads(PHASE1_TERMS.read_text())
        h = sum(bool(cui_to_idx(c)) for c in terms)
        print(f"phase-1 extracted CUIs: {h}/{len(terms)} reach PrimeKG")

    # weighted MIMIC diagnosis coverage
    from collections import Counter
    cnt = Counter()
    with gzip.open(MIMIC / "diagnoses_icd.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            cnt[r["icd_code"]] += 1
    cov = {code: bool(icd_to_idx(code)) for code in cnt}
    rows_cov = sum(c for code, c in cnt.items() if cov[code])
    tot = sum(cnt.values())
    print(f"MIMIC diagnoses: WEIGHTED {rows_cov}/{tot} = {rows_cov/tot:.1%} of rows | "
          f"DISTINCT {sum(cov.values())}/{len(cov)} = {sum(cov.values())/len(cov):.1%} of codes")

    # write icd2primekg for the codes that map (for the graph builder)
    with open(OUT / "icd2primekg.tsv", "w") as f:
        for code in cnt:
            idxs = icd_to_idx(code)
            if idxs:
                f.write(f"{code}\t{'|'.join(map(str, sorted(idxs)))}\n")
    print(f"\nwrote -> {OUT}/cui2primekg.tsv, icd2primekg.tsv, drug2primekg.tsv")


if __name__ == "__main__":
    main()
