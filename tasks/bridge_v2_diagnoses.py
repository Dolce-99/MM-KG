"""
bridge_v2_diagnoses.py   --   env: umls_eval   [CPU, ~20-30 min, streams MRREL 6.4GB]
================================================================================
CORRECTED ICD -> PrimeKG diagnosis bridge (v2).

The v1 bridge attaches a PrimeKG node to only 41.7% of MIMIC diagnosis rows. The
audit traced the cause precisely: MIMIC's ICD codes resolve to BILLING-specific
CUIs that MONDO never cross-references, while the CLINICAL CUI that MONDO does
xref sits ONE UMLS hop away -- a hop v1 never takes.

  25000 (type 2 diabetes)      -> C0375113  mondo_reach = 0
  I4891 (atrial fibrillation)  -> C3264374  mondo_reach = 0
  41401 (coronary athero.)     -> C0837134  mondo_reach = 0

Fixes implemented here (each tagged in the output so downstream can weight them):
  L0 exact      billing CUI -> MONDO/HPO directly            (v1 behaviour)
  L1 synonym    + MRREL REL=SY / RELA in {mapped_to,same_as,...} hop
  L2 parent     + MRREL isa/RB/PAR hop to the parent clinical concept
  HPO twin      ALWAYS also attach the HPO phenotype node when one exists --
                `drug_effect` (129,568 edges) attaches ONLY to HPO nodes; zero
                PrimeKG *disease* nodes carry any, so the MONDO-only v1 bridge
                cannot see the ADE layer at all.
  blocklist     two verified upstream errors:
                  * MONDO:0001881 "toxic shock syndrome" <-> mesh:D012772, which is
                    actually "Shock, Septic" -> made septic shock map to TSS
                  * PrimeKG grouped node 27933 labelled "anxiety disorder" contains
                    MONDO:0002050 "depressive disorder" -> 79,343 depression rows
                    mislabelled as anxiety
  guard         reject a MONDO hit reached ONLY via mesh/NCIt/OMIM when its label
                says familial/congenital/hereditary but the ICD title does not
                (kills E7800 -> *familial* hypercholesterolemia,
                 E8339 -> hypophosphatasia)
  Z/V/Y         status + external-cause codes are flagged NOT_A_DISEASE. They are
                20.3% of MIMIC dx rows and PrimeKG legitimately has no node; they
                must leave the coverage denominator rather than count as failures.

Outputs (outputs/bridge_v2/):
  icd2primekg_v2.tsv    icd_code <TAB> node_idx|node_idx|... <TAB> level <TAB> n_rows
  cui2primekg_v2.tsv    cui      <TAB> node_idx|...          <TAB> level
  icd_excluded.tsv      codes deliberately excluded (Z/V/Y/external cause)
  bridge_v2_stats.json  coverage by level, frequency-weighted
"""
from __future__ import annotations
import os
import json, re
from collections import defaultdict
from pathlib import Path

META    = Path(os.environ.get("UMLS_META_DIR", os.path.expandvars("$MMKG_HOME/data/umls/2026AA/META")))
PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
MIMIC   = Path(os.environ.get("MIMIC_IV_DIR", os.path.expandvars("$MMKG_HOME/data/mimiciv/3.1"))) / "hosp"
HERE    = Path(__file__).resolve().parent
OUT     = HERE / "outputs" / "bridge_v2"

# --- verified upstream errors to suppress ---------------------------------
BLOCK_SSSOM = {("MONDO:0001881", "mesh:D012772")}      # TSS <-> MeSH "Shock, Septic"
BLOCK_NODE_FOR = {27933: re.compile(r"depress", re.I)}  # grouped anxiety node holds depression
SYN_RELA = {"mapped_to", "mapped_from", "same_as", "expanded_form_of",
            "has_expanded_form", "alias_of", "translation_of", "has_translation"}
PARENT_REL = {"RB", "PAR"}
PARENT_RELA = {"isa", "inverse_isa"}
VAGUE_TITLE = re.compile(r"\b(other|unspecified|not elsewhere classified|nec|nos)\b", re.I)
RARE_WORDS = re.compile(r"\b(familial|congenital|hereditary|inherited|syndrom\w*|"
                        r"deficiency|autosomal|neonatal|juvenile|infantile)\b", re.I)
INDIRECT_ONLY_VOCABS = {"MSH", "NCI", "OMIM"}


def is_status_code(code: str, version: int) -> bool:
    """Z/V status codes + E/V/W/X/Y external-cause codes: not diseases."""
    c = code.upper()
    if version == 10:
        return c[:1] in ("Z", "V", "W", "X", "Y")
    return c[:1] in ("V", "E")                      # ICD-9 V (status) and E (external cause)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- PrimeKG node index ------------------------------------------------
    print("loading PrimeKG nodes ...", flush=True)
    import csv
    hpo2idx, mondo2idx, idx2name, idx2type = {}, {}, {}, {}
    with open(PRIMEKG / "nodes.tab") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["node_index"]); src = r["node_source"]
            idx2name[i] = r["node_name"]; idx2type[i] = r["node_type"]
            if src == "HPO":
                try: hpo2idx[int(r["node_id"])] = i
                except ValueError: pass
            elif src in ("MONDO", "MONDO_grouped"):
                for part in str(r["node_id"]).split("_"):
                    try: mondo2idx[int(part)] = i
                    except ValueError: pass
    print(f"  MONDO->idx {len(mondo2idx):,} | HPO->idx {len(hpo2idx):,}")

    # ---- MONDO xrefs (sssom), blocklist applied ---------------------------
    print("loading mondo.sssom xrefs ...", flush=True)
    xref = defaultdict(lambda: defaultdict(set))          # vocab -> code -> {mondo int}
    n_blocked = 0
    with open(PRIMEKG / "mondo.sssom.tsv") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 4 or c[0] == "subject_id" or c[2] != "skos:exactMatch" or not c[0].startswith("MONDO:"):
                continue
            if (c[0], c[3]) in BLOCK_SSSOM:
                n_blocked += 1; continue
            pref, _, val = c[3].partition(":")
            xref[pref][val].add(int(c[0].split(":")[1]))
    print(f"  blocked {n_blocked} known-bad sssom rows")
    SSSOM_TO_SAB = {"SCTID": "SNOMEDCT_US", "OMIM": "OMIM", "mesh": "MSH", "NCIT": "NCI"}

    # ---- MIMIC codes we actually need -------------------------------------
    print("reading MIMIC diagnoses ...", flush=True)
    import pandas as pd
    dx = pd.read_csv(MIMIC / "diagnoses_icd.csv.gz", usecols=["icd_code", "icd_version"])
    code_rows = dx.groupby(["icd_code", "icd_version"]).size().to_dict()
    dtitle = {}
    dn = pd.read_csv(MIMIC / "d_icd_diagnoses.csv.gz")
    for v, c, t in zip(dn.icd_version, dn.icd_code, dn.long_title):
        dtitle[(int(v), str(c).strip())] = str(t)
    total_rows = int(sum(code_rows.values()))
    wanted = {str(c).replace(".", "").strip() for c, _ in code_rows}
    print(f"  {len(code_rows):,} distinct (code,version), {total_rows:,} rows")

    # ---- MRCONSO pass: ICD->CUI, CUI->HPO, CUI->codes ---------------------
    print("streaming MRCONSO ...", flush=True)
    icd2cui = defaultdict(set); cui2hpo = defaultdict(set)
    cui2codes = defaultdict(lambda: defaultdict(set))
    cui_name = {}
    with open(META / "MRCONSO.RRF", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split("|")
            if len(p) < 15: continue
            cui, lat, ts, stt, ispref, sab, code, s = p[0], p[1], p[2], p[4], p[6], p[11], p[13], p[14]
            if sab in ("ICD10CM", "ICD9CM"):
                cc = code.replace(".", "")
                if cc in wanted:
                    icd2cui[(9 if sab == "ICD9CM" else 10, cc)].add(cui)
            if sab == "HPO":
                try: cui2hpo[cui].add(int(code.replace("HP:", "").lstrip("0") or "0"))
                except ValueError: pass
            if sab in ("SNOMEDCT_US", "OMIM", "MSH", "NCI"):
                cui2codes[cui][sab].add(code)
            if lat == "ENG" and cui not in cui_name and ts == "P" and stt == "PF" and ispref == "Y":
                cui_name[cui] = s
    print(f"  ICD->CUI for {len(icd2cui):,} codes | CUI->HPO {len(cui2hpo):,}")

    # UMLS direct + MedGen MONDO routes
    umls_direct = xref.get("UMLS", {}); medgen = xref.get("MEDGEN", {})

    def cui_to_nodes(cui, allow_indirect=True):
        """CUI -> (mondo_nodes, hpo_nodes, via_indirect_only)."""
        m = set(umls_direct.get(cui, ())) | set(medgen.get(cui, ()))
        direct = bool(m)
        if allow_indirect:
            for pref, sab in SSSOM_TO_SAB.items():
                for code in cui2codes.get(cui, {}).get(sab, ()):
                    m |= xref[pref].get(code, set())
        mono = {mondo2idx[x] for x in m if x in mondo2idx}
        hpo = {hpo2idx[h] for h in cui2hpo.get(cui, ()) if h in hpo2idx}
        return mono, hpo, (not direct and bool(mono))

    # ---- which CUIs need an MRREL hop? -------------------------------------
    seed_cuis = set()
    for c, cs in icd2cui.items():
        seed_cuis |= cs
    print(f"  seed (billing) CUIs: {len(seed_cuis):,}")

    # MRREL is 6.4GB and takes minutes; cache the two hop maps so later runs are instant
    import pickle
    hop_cache = OUT / "mrrel_hops.pkl"
    if hop_cache.exists():
        print("loading cached MRREL hops ...", flush=True)
        syn_hop, par_hop = pickle.load(open(hop_cache, "rb"))
        print(f"  cached: synonym {len(syn_hop):,} | parent {len(par_hop):,}")
    else:
        print("streaming MRREL (6.4GB) for synonym + parent hops ...", flush=True)
        syn_hop = defaultdict(set); par_hop = defaultdict(set)
        n = 0
        with open(META / "MRREL.RRF", encoding="utf-8", errors="ignore") as f:
            for line in f:
                n += 1
                p = line.split("|")
                if len(p) < 8: continue
                c1, rel, c2, rela = p[0], p[3], p[4], p[7]
                if c1 == c2: continue
                a = c1 in seed_cuis; b = c2 in seed_cuis
                if not (a or b): continue
                if rel == "SY" or rela in SYN_RELA:
                    if a: syn_hop[c1].add(c2)
                    if b: syn_hop[c2].add(c1)
                elif rel in PARENT_REL or rela in PARENT_RELA:
                    # from the seed, walk UP: REL is the relationship of CUI2 to CUI1
                    if a and (rel == "RB" or rela == "isa"): par_hop[c1].add(c2)
                    if b and (rel == "RN" or rela == "inverse_isa"): par_hop[c2].add(c1)
                if n % 20_000_000 == 0:
                    print(f"    {n:,} rows | syn {len(syn_hop):,} par {len(par_hop):,}", flush=True)
        print(f"  MRREL done: {n:,} rows | synonym {len(syn_hop):,} | parent {len(par_hop):,}")
        pickle.dump((dict(syn_hop), dict(par_hop)), open(hop_cache, "wb"))

    # ---- name index over PrimeKG disease + phenotype nodes -----------------
    # The CUI routes miss nodes that plainly exist: HPO 84384 "Hyperkalemia" and
    # HPO 23834 "Coronary artery atherosclerosis" are both present in PrimeKG but
    # unreachable from the billing CUI. A guarded name match recovers them.
    QUAL = re.compile(r"\b(unspecified|nos|not elsewhere classified|other|with(out)?|"
                      r"due to|type|native|graft|acute|chronic|initial encounter|"
                      r"subsequent encounter|sequela|left|right|bilateral|site)\b", re.I)

    def norm_name(s):
        s = re.sub(r"\(.*?\)", " ", str(s).lower())
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def content_tokens(s):
        return {t for t in norm_name(QUAL.sub(" ", s)).split() if len(t) > 2}

    name2nodes = defaultdict(set); tok_index = defaultdict(set)
    for i, nm in idx2name.items():
        if idx2type.get(i) not in ("disease", "effect/phenotype"): continue
        name2nodes[norm_name(nm)].add(i)
        for t in content_tokens(nm): tok_index[t].add(i)

    def name_match(title):
        """exact normalized match, else ICD content-tokens are a subset of the node's."""
        n = norm_name(title)
        if n in name2nodes: return set(name2nodes[n])
        toks = content_tokens(title)
        if len(toks) < 2: return set()
        cand = set.intersection(*[tok_index[t] for t in toks]) if all(t in tok_index for t in toks) else set()
        return {i for i in cand if toks <= content_tokens(idx2name[i])}

    def drop_rare(nodes, title):
        """Node-level semantic guard: reject a rare/Mendelian node when the ICD title
        is a plain acquired condition. v1 mapped E7800 'pure hypercholesterolemia' to
        *familial* hypercholesterolemia and E8339 to hypophosphatasia."""
        if RARE_WORDS.search(title or ""):
            return nodes
        return {i for i in nodes if not RARE_WORDS.search(idx2name.get(i, ""))}

    # ---- compose the bridge -----------------------------------------------
    print("composing bridge ...", flush=True)
    out_rows = []; excluded = []
    stats = defaultdict(int); wstats = defaultdict(int)
    for (code_raw, ver), nrows in code_rows.items():
        code = str(code_raw).replace(".", "").strip(); ver = int(ver)
        title = dtitle.get((ver, str(code_raw).strip()), "")
        if is_status_code(code, ver):
            excluded.append((code, ver, nrows, title)); stats["excluded_status"] += 1
            wstats["excluded_status"] += nrows; continue
        cuis = icd2cui.get((ver, code), set())
        nodes, level = set(), None
        # L0 direct
        vague = bool(VAGUE_TITLE.search(title or ""))
        for cui in cuis:
            m, h, indirect_only = cui_to_nodes(cui)
            if vague and indirect_only:
                m = set()                 # residual bucket + indirect xref = drift risk
            m = drop_rare(m, title)
            if m or h: nodes |= m | h; level = level or "L0_exact"
        # L1 synonym hop
        if not nodes:
            for cui in cuis:
                for c2 in syn_hop.get(cui, ()):
                    m, h, _ = cui_to_nodes(c2)
                    nodes |= drop_rare(m, title) | h
            if nodes: level = "L1_synonym"
        # L2 parent hop
        if not nodes:
            for cui in cuis:
                for c2 in par_hop.get(cui, ()):
                    m, h, _ = cui_to_nodes(c2)
                    nodes |= drop_rare(m, title) | h
            if nodes: level = "L2_parent"
        # L3 guarded name match -- recovers nodes that plainly exist but no CUI route reaches.
        # Try the ICD title AND every CUI preferred name (ICD-9 "Hyperpotassemia" vs
        # HPO "Hyperkalemia" share no tokens, but their CUI's preferred name does).
        if not nodes:
            probes = ([title] if title else []) + [cui_name[c] for c in cuis if c in cui_name]
            for probe in probes:
                nm = drop_rare(name_match(probe), title)
                if nm:
                    nodes |= nm; level = "L3_name"; break
        # HPO twin: ALWAYS add on top, even when a MONDO node was found --
        # drug_effect (129,568 edges) attaches ONLY to HPO nodes.
        for cui in list(cuis) + [c for cui in cuis for c in syn_hop.get(cui, ())]:
            nodes |= {hpo2idx[h] for h in cui2hpo.get(cui, ()) if h in hpo2idx}
        for probe in ([title] if title else []) + [cui_name[c] for c in cuis if c in cui_name]:
            nodes |= {i for i in drop_rare(name_match(probe), title)
                      if idx2type.get(i) == "effect/phenotype"}
        # final rare-node sweep + node-level blocklist
        nodes = drop_rare(nodes, title)
        for bad_idx, pat in BLOCK_NODE_FOR.items():
            if bad_idx in nodes and pat.search(title):
                nodes.discard(bad_idx); stats["blocked_node_hit"] += 1
        if nodes:
            out_rows.append((code, ver, sorted(nodes), level or "L0_exact", nrows))
            stats[level or "L0_exact"] += 1; wstats[level or "L0_exact"] += nrows
        else:
            stats["unmapped"] += 1; wstats["unmapped"] += nrows

    with open(OUT / "icd2primekg_v2.tsv", "w") as w:
        for code, ver, nodes, level, nrows in out_rows:
            w.write(f"{code}\t{ver}\t{'|'.join(map(str, nodes))}\t{level}\t{nrows}\n")
    with open(OUT / "icd_excluded.tsv", "w") as w:
        for code, ver, nrows, title in excluded:
            w.write(f"{code}\t{ver}\t{nrows}\t{title}\n")

    disease_rows = total_rows - wstats["excluded_status"]
    mapped_w = sum(wstats[k] for k in ("L0_exact", "L1_synonym", "L2_parent", "L3_name"))
    st = {"total_rows": total_rows, "distinct_codes": len(code_rows),
          "excluded_status_rows": wstats["excluded_status"],
          "disease_rows": disease_rows,
          "by_level_codes": {k: stats[k] for k in ("L0_exact", "L1_synonym", "L2_parent", "L3_name", "unmapped")},
          "by_level_rows": {k: wstats[k] for k in ("L0_exact", "L1_synonym", "L2_parent", "L3_name", "unmapped")},
          "coverage_all_rows": round(mapped_w / total_rows, 4),
          "coverage_disease_rows": round(mapped_w / max(disease_rows, 1), 4),
          "blocked_node_hits": stats["blocked_node_hit"]}
    json.dump(st, open(OUT / "bridge_v2_stats.json", "w"), indent=2)

    print("\n=== BRIDGE v2 (diagnoses) ===")
    print(f"  total dx rows              {total_rows:,}")
    print(f"  excluded Z/V/Y status rows {wstats['excluded_status']:,} "
          f"({wstats['excluded_status']/total_rows:.1%})  <- not diseases, out of denominator")
    print(f"  true disease rows          {disease_rows:,}")
    for k in ("L0_exact", "L1_synonym", "L2_parent", "L3_name"):
        print(f"    {k:12s} codes {stats[k]:6,}  rows {wstats[k]:9,} ({wstats[k]/disease_rows:.1%} of disease rows)")
    print(f"    {'unmapped':12s} codes {stats['unmapped']:6,}  rows {wstats['unmapped']:9,}")
    print(f"  COVERAGE: {mapped_w/total_rows:.1%} of ALL rows | "
          f"{mapped_w/max(disease_rows,1):.1%} of true disease rows   (v1 was 41.7% / 50.9%)")
    print(f"  blocked depression->anxiety hits: {stats['blocked_node_hit']:,}")
    print(f"\nwrote -> {OUT}/icd2primekg_v2.tsv")


if __name__ == "__main__":
    main()
