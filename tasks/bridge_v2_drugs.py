"""
bridge_v2_drugs.py   --   env: umls_eval   [CPU, ~25-40 min, streams MRCONSO+MRSAT+MRREL]
================================================================================
CORRECTED MIMIC drug -> PrimeKG bridge (v2).

v1 matched MIMIC drug strings against DrugBank *names* and reached 73.1% of
prescription volume, flat at every frequency stratum -- a naming-convention failure.
Key structural fact the audit established and we verified: **PrimeKG's drug
`node_id` IS the DrugBank accession** (e.g. node 14012 = "DB09130" = Copper), so
CUI -> DrugBank -> PrimeKG is a direct join with no string matching at all.

Routes, in precedence order (each tagged in the output):
  R0 name        exact/normalized DrugBank name match          (v1 behaviour)
  R1 umls_syn    drug string -> any UMLS drug-vocab string -> CUI -> DrugBank
                 recovers aspirin (PrimeKG: "Acetylsalicylic acid"), insulin,
                 torsemide ("Torasemide"), albuterol ("Salbutamol"), phytonadione
  R2 ndc         MIMIC ndc -> MRSAT(ATN=NDC) -> RxNorm CUI -> MRREL ingredient
                 -> CUI -> DrugBank. The ONLY route that resolves combination
                 products: piperacillin-tazobactam, TMP-SMX, oxycodone-APAP
  ATC            same join also yields ATC class codes -- emitted separately so
                 downstream can use drug CLASS (the one evidence unit that beat
                 pill-counting: 0.6101 vs 0.5966)

Correctness guards (v1 defects found in audit):
  VEHICLE   saline/dextrose flushes, sterile water, and heparin LINE-LOCKS are
            excluded. Vehicles are 12% of mapped volume; `sodium chloride 0.9%
            flush` is the single most frequent "drug", and Heparin absorbs
            190,982 records of which >=43,501 are catheter locks -- a patient who
            only got saline flushes otherwise reads as "on heparin", which is
            close to a label leak for the bleeding task.
  MINLEN    candidate tokens shorter than 3 chars are rejected: `d5w` -> leading
            token "d" -> UMLS "D" = Aspartic acid (19,211 records).
  SALT      reject collapsing a salt/compound onto a bare ELEMENT name
            (potassium chloride -> "Potassium", iron dextran -> "Iron").

Outputs (outputs/bridge_v2/):
  drug2primekg_v2.tsv   drug_string <TAB> node_idx <TAB> route <TAB> n_rx
  drug2atc_v2.tsv       drug_string <TAB> atc_codes(|) <TAB> atc_level3_names
  drug_vehicles.tsv     strings excluded as vehicle/flush, with volume
  bridge_v2_drug_stats.json
"""
from __future__ import annotations
import os
import json, re
from collections import defaultdict
from pathlib import Path

META    = Path(os.environ.get("UMLS_META_DIR", os.path.expandvars("$MMKG_HOME/data/umls/2026AA/META")))
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
MIMIC   = Path(os.environ.get("MIMIC_IV_DIR", os.path.expandvars("$MMKG_HOME/data/mimiciv/3.1"))) / "hosp"
WG      = Path(__file__).resolve().parent.parent / "graph"
HERE    = Path(__file__).resolve().parent
OUT     = HERE / "outputs" / "bridge_v2"

DRUG_SABS = {"DRUGBANK", "RXNORM", "MSH", "ATC", "MMSL", "VANDF", "GS", "NDDF", "MTHSPL", "SNOMEDCT_US"}

# --- vehicle / diluent / line-lock exclusion ------------------------------
VEHICLE_PAT = re.compile(
    r"\b(flush|dwell|lock|line lock|picc|port|cath(eter)?\s*(flush|lock)?)\b", re.I)
VEHICLE_BASE = re.compile(
    r"^(0\.9%?\s*)?(sodium chloride|nacl|ns|1/2 ns|1/4 ns|normal saline|sterile water|"
    r"free water|d5w|d5 ?ns|d5 ?1/2 ?ns|d5 ?lr|d10w|dextrose( \d+%)?|iso-?osmotic dextrose|"
    r"lactated ringers|lr|ringers|sw|swfi|heparin flush.*|bag|vial|syringe|"
    r"soln|solution|water)\b", re.I)
ELEMENTS = {"potassium", "sodium", "calcium", "magnesium", "iron", "zinc", "copper",
            "phosphorus", "chloride", "lithium", "selenium", "chromium", "manganese"}
_DOSE = re.compile(r"\b(\d+(\.\d+)?)\s*(mg|ml|mcg|gm?|g|units?|meq|%|iu|mmol)\b")
_FORM = re.compile(r"\b(tab|tablet|cap|capsule|soln|solution|inj|injection|oral|iv|po|sc|im|"
                   r"syringe|vial|bag|drops?|ointment|cream|patch|er|xr|sr|cr|ec|dr|hcl|hbr|premix|"
                   r"flush|neb|nebu|supp|lozenge|elixir|susp|suspension|conc|kit|pca|human|immediate|"
                   r"release|iso-?osmotic|preservative|free)\b")


def is_vehicle(s: str) -> bool:
    low = s.lower().strip()
    if VEHICLE_PAT.search(low):
        return True
    return bool(VEHICLE_BASE.match(low))


def candidates(s: str):
    """Ingredient-name candidates, with the v1 defects fixed."""
    out = []
    paren = re.findall(r"\(([^)]+)\)", s); base = re.sub(r"\([^)]*\)", " ", s)
    for x in [base] + paren:
        x = _DOSE.sub(" ", x.lower()); x = re.sub(r"[0-9.%]", " ", x); x = _FORM.sub(" ", x)
        x = re.sub(r"[^a-z\s/-]", " ", x); x = re.sub(r"\s+", " ", x).strip()
        if len(x) >= 3:
            out.append(x)
            toks = x.split()
            if toks and len(toks[0]) >= 3:            # MINLEN guard: kills "d" -> Aspartic acid
                out.append(toks[0])
    # longest first so the full string always wins over the leading token
    return sorted(set(out), key=len, reverse=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd, csv

    # ---- PrimeKG drug nodes: node_id IS the DrugBank accession ------------
    db2idx, idx2name = {}, {}
    with open(PRIMEKG / "nodes.tab") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["node_type"] == "drug":
                i = int(r["node_index"]); idx2name[i] = r["node_name"]
                db2idx[str(r["node_id"]).strip().upper()] = i
    name2idx = {v.lower(): k for k, v in idx2name.items()}
    print(f"PrimeKG drug nodes {len(idx2name):,} | DrugBank-id keyed {len(db2idx):,}")

    # ---- cohort drug strings + volumes ------------------------------------
    nodes = pd.read_parquet(WG / "outputs" / "patient_graph" / "nodes.parquet",
                            columns=["ntype", "name"])
    drugs = [str(x) for x in nodes.loc[nodes.ntype == "Drug", "name"]]
    rx = pd.read_csv(MIMIC / "prescriptions.csv.gz", usecols=["drug", "ndc"],
                     dtype={"ndc": "string"})
    rx["d"] = rx.drug.astype(str).str.strip().str.lower()
    vol = rx.d.value_counts().to_dict()
    ndc_by_drug = (rx.dropna(subset=["ndc"]).groupby("d")["ndc"]
                     .agg(lambda s: s.value_counts().index[0]).to_dict())
    total_vol = sum(vol.get(d, 0) for d in drugs)
    print(f"cohort drug strings {len(drugs):,} | total rx volume {total_vol:,}")

    # ---- vehicles ----------------------------------------------------------
    vehicles = [d for d in drugs if is_vehicle(d)]
    veh_vol = sum(vol.get(d, 0) for d in vehicles)
    with open(OUT / "drug_vehicles.tsv", "w") as w:
        for d in sorted(vehicles, key=lambda x: -vol.get(x, 0)):
            w.write(f"{d}\t{vol.get(d,0)}\n")
    print(f"vehicles/flushes excluded: {len(vehicles):,} strings, {veh_vol:,} rx "
          f"({veh_vol/max(total_vol,1):.1%} of volume)")
    real = [d for d in drugs if not is_vehicle(d)]
    real_vol = sum(vol.get(d, 0) for d in real)

    # ---- MRCONSO pass: drug strings -> CUI -> DrugBank; + ATC; + NDC seeds -
    print("streaming MRCONSO for drug vocabularies ...", flush=True)
    str2cui = defaultdict(set); cui2db = defaultdict(set); cui2atc = defaultdict(set)
    rxcui_codes = {}
    with open(META / "MRCONSO.RRF", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split("|")
            if len(p) < 15: continue
            cui, lat, sab, code, s = p[0], p[1], p[11], p[13], p[14]
            if sab not in DRUG_SABS: continue
            if sab == "DRUGBANK":
                cui2db[cui].add(code.strip().upper())
            elif sab == "ATC":
                cui2atc[cui].add(code.strip())
            if lat == "ENG":
                key = re.sub(r"\s+", " ", s.lower().strip())
                if 3 <= len(key) <= 80:
                    str2cui[key].add(cui)
            if sab == "RXNORM":
                rxcui_codes[code.strip()] = cui
    print(f"  UMLS drug strings {len(str2cui):,} | CUIs with DrugBank id {len(cui2db):,} "
          f"| CUIs with ATC {len(cui2atc):,}")

    def db_nodes(cuis):
        out = set()
        for c in cuis:
            for db in cui2db.get(c, ()):
                if db in db2idx: out.add(db2idx[db])
        return out

    mapped = {}; route = {}; atc_of = {}

    # ---- R0: DrugBank name match (v1) --------------------------------------
    for d in real:
        for cand in candidates(d):
            if cand in name2idx:
                base_tok = cand.split()[0]
                if cand in ELEMENTS and len(d.split()) > 1:
                    continue                                  # SALT guard
                mapped[d] = name2idx[cand]; route[d] = "R0_name"; break

    # ---- R1: UMLS synonym -> CUI -> DrugBank -------------------------------
    for d in real:
        if d in mapped: continue
        for cand in candidates(d):
            if cand in ELEMENTS and len(d.split()) > 1:
                continue
            cuis = str2cui.get(cand)
            if not cuis: continue
            nodes_ = db_nodes(cuis)
            if nodes_:
                mapped[d] = sorted(nodes_)[0]; route[d] = "R1_umls_syn"
                for c in cuis:
                    if cui2atc.get(c): atc_of.setdefault(d, set()).update(cui2atc[c])
                break

    # ---- R2: NDC -> RxNorm -> ingredient -> DrugBank ------------------------
    need_ndc = {d for d in real if d not in mapped and d in ndc_by_drug}
    ndc_want = {ndc_by_drug[d] for d in need_ndc}
    print(f"streaming MRSAT for {len(ndc_want):,} NDCs ...", flush=True)
    ndc2cui = {}
    with open(META / "MRSAT.RRF", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split("|")
            if len(p) < 11 or p[8] != "NDC": continue
            v = p[10].strip()
            if v in ndc_want and p[0]:
                ndc2cui.setdefault(v, p[0])
    print(f"  resolved {len(ndc2cui):,} NDCs to a CUI")
    seed = set(ndc2cui.values())
    ingr = defaultdict(set)
    if seed:
        print("streaming MRREL for RxNorm ingredient links ...", flush=True)
        with open(META / "MRREL.RRF", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.split("|")
                if len(p) < 12 or p[10] != "RXNORM": continue
                if p[7] not in ("has_ingredient", "ingredient_of", "tradename_of",
                                "consists_of", "form_of", "precise_ingredient_of"): continue
                c1, c2 = p[0], p[4]
                if c1 in seed: ingr[c1].add(c2)
                if c2 in seed: ingr[c2].add(c1)
    for d in need_ndc:
        cui = ndc2cui.get(ndc_by_drug[d])
        if not cui: continue
        cands = {cui} | ingr.get(cui, set())
        nodes_ = db_nodes(cands)
        if nodes_:
            mapped[d] = sorted(nodes_)[0]; route[d] = "R2_ndc"
        for c in cands:
            if cui2atc.get(c): atc_of.setdefault(d, set()).update(cui2atc[c])
    # ATC for already-mapped drugs too
    for d in real:
        if d in atc_of: continue
        for cand in candidates(d):
            for c in str2cui.get(cand, ()):
                if cui2atc.get(c): atc_of.setdefault(d, set()).update(cui2atc[c])
            if d in atc_of: break

    # ---- write -------------------------------------------------------------
    with open(OUT / "drug2primekg_v2.tsv", "w") as w:
        for d, i in sorted(mapped.items()):
            w.write(f"{d}\t{i}\t{route[d]}\t{vol.get(d,0)}\n")
    with open(OUT / "drug2atc_v2.tsv", "w") as w:
        for d, codes in sorted(atc_of.items()):
            lvl3 = sorted({c[:4] for c in codes if len(c) >= 4})
            w.write(f"{d}\t{'|'.join(sorted(codes))}\t{'|'.join(lvl3)}\n")

    mv = sum(vol.get(d, 0) for d in mapped)
    by_route = defaultdict(int); by_route_vol = defaultdict(int)
    for d, r in route.items():
        by_route[r] += 1; by_route_vol[r] += vol.get(d, 0)
    st = {"cohort_strings": len(drugs), "vehicle_strings": len(vehicles),
          "vehicle_rx": veh_vol, "real_strings": len(real), "real_rx": real_vol,
          "mapped_strings": len(mapped), "mapped_rx": mv,
          "coverage_distinct_of_real": round(len(mapped) / max(len(real), 1), 4),
          "coverage_weighted_of_real": round(mv / max(real_vol, 1), 4),
          "by_route": dict(by_route), "by_route_rx": dict(by_route_vol),
          "with_atc": len(atc_of),
          "atc_weighted": round(sum(vol.get(d, 0) for d in atc_of) / max(real_vol, 1), 4)}
    json.dump(st, open(OUT / "bridge_v2_drug_stats.json", "w"), indent=2)

    print("\n=== BRIDGE v2 (drugs) ===")
    print(f"  vehicles/flushes removed : {len(vehicles):,} strings / {veh_vol:,} rx "
          f"({veh_vol/max(total_vol,1):.1%})")
    print(f"  real drug strings        : {len(real):,} / {real_vol:,} rx")
    for r in ("R0_name", "R1_umls_syn", "R2_ndc"):
        print(f"    {r:12s} {by_route[r]:5,} strings  {by_route_vol[r]:9,} rx "
              f"({by_route_vol[r]/max(real_vol,1):.1%})")
    print(f"  COVERAGE: {len(mapped)/max(len(real),1):.1%} distinct | "
          f"{mv/max(real_vol,1):.1%} weighted   (v1: 44.0% / 73.1%)")
    print(f"  ATC classes for {st['atc_weighted']:.1%} of volume ({len(atc_of):,} strings)")
    print(f"\nwrote -> {OUT}/drug2primekg_v2.tsv, drug2atc_v2.tsv, drug_vehicles.tsv")


if __name__ == "__main__":
    main()
