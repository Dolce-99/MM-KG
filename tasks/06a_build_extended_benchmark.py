"""
06a_build_extended_benchmark.py   --   env: umls_eval   [CPU, ~5 min]
================================================================================
Extended medication-audit benchmark: a HOP LADDER plus the control variants needed
to prove the model is genuinely reading the graph.

WHY A HOP LADDER
The week-1 audit showed a large gain (+0.18 to +0.26) but every family was one edge
deep, so a reader can fairly object that we built a lookup table. Parametric memory
degrades on COMPOSITION while graph traversal does not, so if the KG advantage GROWS
with hop count that is a qualitatively stronger claim than any single-fact result.

  hop1_contraindication  drug --contraindicated_in--> condition
  hop1_indication        drug --indicated_for--> condition           (CONTROL: strong prior)
  hop2_shared_target     drug --targets--> protein <--targets-- drug
  hop2_mechanism         drug --targets--> protein <--implicated_in-- disease
  hop3_chain             drug --targets--> protein <--implicated_in-- disease
                              --related_to--> the patient's condition
  class_level            is any drug of ATC class X present  (uses the ATC layer from bridge v2)

EDGE VARIANTS PER ITEM (the controls that make a positive interpretable)
  edges_true       the real retrieved neighbourhood
  edges_shuffled   a neighbourhood taken from a DIFFERENT item -- same format, same
                   approximate token count, wrong facts. This is the strict control
                   for "does extra context help regardless of content?"
  edges_corrupted  the real neighbourhood with entity names swapped for plausible
                   wrong ones. If the model still scores well, it is not reading.
  edges_<relation> single-relation ablations, to see which layer carries the gain.

Every item also carries `prevalence_stratum` (how common the drug is in the cohort)
so the gain can be split by how thin the model's prior should be.

Output: outputs/audit/extended_benchmark.jsonl
"""
from __future__ import annotations
import os
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

PRIMEKG = Path(os.environ.get("PRIMEKG_DIR", os.path.expandvars("$MMKG_HOME/data/primekg")))
HERE    = Path(__file__).resolve().parent
OUT     = HERE / "outputs" / "audit"
MS      = HERE / "outputs" / "medsafety"
V2      = HERE / "outputs" / "bridge_v2"
NEIGH   = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_family", type=int, default=240)
    ap.add_argument("--complete", action="store_true",
                    help="emit COMPLETE neighbourhoods (no per-relation cap). Writes to "
                         "extended_benchmark_complete.jsonl so the bounded version is kept "
                         "for the retrieval-budget comparison.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cap = None if args.complete else NEIGH      # None = no truncation

    idx2name, idx2type = {}, {}
    with open(PRIMEKG / "nodes.tab") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["node_index"]); idx2name[i] = r["node_name"]; idx2type[i] = r["node_type"]

    # ---- relation layers ---------------------------------------------------
    contra, indic, deffect = defaultdict(set), defaultdict(set), defaultdict(set)
    dprot, disprot, dis_dis = defaultdict(set), defaultdict(set), defaultdict(set)
    prot_name = {}
    for ch in pd.read_csv(PRIMEKG / "kg.csv",
                          usecols=["relation", "x_type", "x_index", "x_name", "y_type", "y_index", "y_name"],
                          dtype={"x_index": "int32", "y_index": "int32"}, chunksize=2_000_000):
        ch = ch[ch.relation.isin({"contraindication", "indication", "drug_protein",
                                  "disease_protein", "disease_disease", "drug_effect"})]
        for rel, xt, x, xn, yt, y, yn in zip(ch.relation, ch.x_type, ch.x_index, ch.x_name,
                                             ch.y_type, ch.y_index, ch.y_name):
            x, y = int(x), int(y)
            if rel == "contraindication":
                contra[x if xt == "drug" else y].add(y if xt == "drug" else x)
            elif rel == "indication":
                indic[x if xt == "drug" else y].add(y if xt == "drug" else x)
            elif rel == "drug_protein":
                d, p, pn = (x, y, str(yn)) if xt == "drug" else (y, x, str(xn))
                dprot[d].add(p); prot_name[p] = pn
            elif rel == "disease_protein":
                d, p, pn = (x, y, str(yn)) if xt == "disease" else (y, x, str(xn))
                disprot[d].add(p); prot_name[p] = pn
            elif rel == "disease_disease":
                dis_dis[x].add(y); dis_dis[y].add(x)
            elif rel == "drug_effect":
                d, e, en = (x, y, str(yn)) if xt == "drug" else (y, x, str(xn))
                deffect[d].add(e); idx2name.setdefault(e, en)
    prot2dis = defaultdict(set)
    for d, ps in disprot.items():
        for p in ps: prot2dis[p].add(d)
    print(f"layers: contra {sum(len(v) for v in contra.values()):,} | indic "
          f"{sum(len(v) for v in indic.values()):,} | drug_protein {sum(len(v) for v in dprot.values()):,} "
          f"| disease_protein {sum(len(v) for v in disprot.values()):,} | disease_disease "
          f"{sum(len(v) for v in dis_dis.values()):,}")

    # ---- cohort ------------------------------------------------------------
    drug2pkg, atc = {}, {}
    for line in open(V2 / "drug2primekg_v2.tsv"):
        d, i, r, n = line.rstrip("\n").split("\t"); drug2pkg[d] = int(i)
    for line in open(V2 / "drug2atc_v2.tsv"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2]: atc[p[0]] = set(p[2].split("|"))
    cond_name2idx = {nm: i for i, nm in idx2name.items() if idx2type.get(i) == "disease"}
    # WHOLE-COHORT audit base (09a). The previous source inherited two eligibility
    # rules from the ADE PREDICTION pipeline -- >=48h observation and AKI-assessable
    # labs -- neither of which a fact-checking question needs. They cost 66% of
    # admissions for no methodological reason.
    recs = [json.loads(l) for l in open(OUT / "audit_cohort.jsonl")]
    pat, drug_freq = [], defaultdict(int)
    for r in recs:
        dn, cn = r["drug_nodes"], r["cond_nodes"]
        for d in dn: drug_freq[d] += 1
        pat.append((r["hadm_id"], dn, cn, r["drug_names"]))
    N = len(recs)
    print(f"patients usable: {len(pat):,}")

    items = []
    def add(fam, hops, q, ans, drugs, conds, rels, hadm):
        items.append({"family": fam, "hops": hops, "question": q, "answer": ans,
                      "drugs": drugs, "conds": conds, "relations": rels, "hadm_id": hadm})

    # ---- hop1: contraindication / indication -------------------------------
    cooc = [(d, c, h) for h, dn, cn, _ in pat for d in dn[:20] for c in cn[:20]]
    rng.shuffle(cooc)
    for fam, table, verb in (("hop1_contraindication", contra, "contraindicated in a patient with"),
                             ("hop1_indication", indic, "indicated for the treatment of")):
        posl = [(d, c, h) for d, c, h in cooc[:400000] if c in table.get(d, ())]
        negl = [(d, c, h) for d, c, h in cooc[:400000] if c not in table.get(d, ())]
        k = min(args.n_per_family // 2, len(posl), len(negl))
        for src, a in ((posl, "yes"), (negl, "no")):
            for i in rng.choice(len(src), k, replace=False):
                d, c, h = src[i]
                add(fam, 1, f"Is {idx2name[d]} {verb} {idx2name[c]}?", a, [d], [c],
                    ["contraindication" if "contra" in fam else "indication"], h)
        print(f"  {fam}: {k}+{k}")

    # ---- hop2: shared target ------------------------------------------------
    pp, nn = [], []
    for h, dn, cn, _ in pat:
        w = [d for d in dn if dprot.get(d)][:10]
        for i, a in enumerate(w):
            for b in w[i + 1:]:
                (pp if dprot[a] & dprot[b] else nn).append((a, b, h))
    k = min(args.n_per_family // 2, len(pp), len(nn))
    for src, a in ((pp, "yes"), (nn, "no")):
        for i in rng.choice(len(src), k, replace=False):
            x, y, h = src[i]
            add("hop2_shared_target", 2,
                f"Do {idx2name[x]} and {idx2name[y]} share a molecular target or metabolising enzyme?",
                a, [x, y], [], ["drug_protein"], h)
    print(f"  hop2_shared_target: {k}+{k}")

    # ---- hop2: mechanism (drug -> protein <- disease) ----------------------
    pp, nn = [], []
    for h, dn, cn, _ in pat:
        for d in dn[:10]:
            if not dprot.get(d): continue
            for c in cn[:10]:
                if not disprot.get(c): continue
                (pp if dprot[d] & disprot[c] else nn).append((d, c, h))
    k = min(args.n_per_family // 2, len(pp), len(nn))
    for src, a in ((pp, "yes"), (nn, "no")):
        for i in rng.choice(len(src), k, replace=False):
            d, c, h = src[i]
            add("hop2_mechanism", 2,
                f"Does {idx2name[d]} act on a protein that is implicated in {idx2name[c]}?",
                a, [d], [c], ["drug_protein", "disease_protein"], h)
    print(f"  hop2_mechanism: {k}+{k}")

    # ---- hop3: drug -> protein <- disease -> related disease ---------------
    pp, nn = [], []
    for h, dn, cn, _ in pat:
        for d in dn[:8]:
            if not dprot.get(d): continue
            reach = {dd for p in dprot[d] for dd in prot2dis.get(p, ())}
            for c in cn[:8]:
                rel = dis_dis.get(c, set())
                if not rel: continue
                (pp if (reach & rel) else nn).append((d, c, h))
    k = min(args.n_per_family // 2, len(pp), len(nn))
    for src, a in ((pp, "yes"), (nn, "no")):
        for i in rng.choice(len(src), k, replace=False):
            d, c, h = src[i]
            add("hop3_chain", 3,
                f"Does {idx2name[d]} act on a protein implicated in a disease that is "
                f"related to {idx2name[c]}?", a, [d], [c],
                ["drug_protein", "disease_protein", "disease_disease"], h)
    print(f"  hop3_chain: {k}+{k}")

    # ---- class level (ATC, from bridge v2) ---------------------------------
    inv = defaultdict(set)
    for m, cs in atc.items():
        for c in cs: inv[c].add(m)
    classes = [c for c, ms in inv.items() if 20 <= len(ms) <= 400]
    pp, nn = [], []
    for h, dn, cn, names in pat:
        got = set().union(*[atc.get(m, set()) for m in names]) if names else set()
        if not got: continue
        cl_yes = list(got & set(classes)); cl_no = [c for c in classes if c not in got]
        if cl_yes: pp.append((h, names, cl_yes[0]))
        if cl_no:  nn.append((h, names, cl_no[int(rng.integers(0, len(cl_no)))]))
    k = min(args.n_per_family // 2, len(pp), len(nn))
    for src, a in ((pp, "yes"), (nn, "no")):
        for i in rng.choice(len(src), k, replace=False):
            h, names, cl = src[i]
            add("class_level", 1,
                "A patient is prescribed: " + ", ".join(
                    idx2name.get(drug2pkg[m], m) for m in names[:13]) +
                f". Does this list include any drug of ATC class {cl}?", a,
                [drug2pkg[m] for m in names[:13]], [], ["atc"], h)
    print(f"  class_level: {k}+{k}")

    # ---- drug_effect family (the HPO-side layer, 129,568 edges, never queried before) ----
    all_eff = sorted({e for es in deffect.values() for e in es})
    pp, nn = [], []
    for h, dn, cn, _ in pat:
        for d in dn[:8]:
            es = deffect.get(d)
            if not es: continue
            pp.append((d, sorted(es)[int(rng.integers(0, len(es)))], h))
            for _ in range(3):
                cand = all_eff[int(rng.integers(0, len(all_eff)))]
                if cand not in es: nn.append((d, cand, h)); break
    k = min(args.n_per_family // 2, len(pp), len(nn))
    for src, a in ((pp, "yes"), (nn, "no")):
        for i in rng.choice(len(src), k, replace=False):
            d, e, h = src[i]
            add("drug_effect", 1,
                f"Is {idx2name.get(e, e)} a documented adverse effect of {idx2name[d]}?",
                a, [d], [], ["drug_effect"], h)
    print(f"  drug_effect: {k}+{k}")

    # ---- comorbidity family (disease_disease as the primary relation) ------
    pp, nn = [], []
    for h, dn, cn, _ in pat:
        for i, a_ in enumerate(cn[:10]):
            for b_ in cn[i + 1:10]:
                (pp if b_ in dis_dis.get(a_, ()) else nn).append((a_, b_, h))
    k = min(args.n_per_family // 2, len(pp), len(nn))
    for src, a in ((pp, "yes"), (nn, "no")):
        for i in rng.choice(len(src), k, replace=False):
            x, y, h = src[i]
            add("comorbidity", 1,
                f"Are {idx2name[x]} and {idx2name[y]} recognised as related or commonly co-occurring conditions?",
                a, [], [x, y], ["disease_disease"], h)
    print(f"  comorbidity: {k}+{k}")

    # ---- attach edge variants ----------------------------------------------
    def neigh_edges(it, rels=None):
        out = []
        rels = rels or set(it["relations"]) | {"contraindication", "indication", "drug_protein"}
        for d in it["drugs"][:6]:
            if "contraindication" in rels and contra.get(d):
                out.append(f"{idx2name[d]} --contraindicated_in--> " +
                           "; ".join(idx2name.get(c, str(c)) for c in (sorted(contra[d])[:cap] if cap else sorted(contra[d]))))
            if "indication" in rels and indic.get(d):
                out.append(f"{idx2name[d]} --indicated_for--> " +
                           "; ".join(idx2name.get(c, str(c)) for c in (sorted(indic[d])[:cap] if cap else sorted(indic[d]))))
            if "drug_protein" in rels and dprot.get(d):
                out.append(f"{idx2name[d]} --targets--> " +
                           ", ".join(prot_name.get(p, str(p)) for p in (sorted(dprot[d])[:8] if cap else sorted(dprot[d]))))
            if "drug_effect" in rels and deffect.get(d):
                out.append(f"{idx2name[d]} --causes--> " +
                           "; ".join(idx2name.get(e, str(e)) for e in (sorted(deffect[d])[:cap] if cap else sorted(deffect[d]))))
        for c in it["conds"][:4]:
            if "disease_protein" in rels and disprot.get(c):
                out.append(f"{idx2name[c]} --implicated_proteins--> " +
                           ", ".join(prot_name.get(p, str(p)) for p in (sorted(disprot[c])[:8] if cap else sorted(disprot[c]))))
            if "disease_disease" in rels and dis_dis.get(c):
                out.append(f"{idx2name[c]} --related_diseases--> " +
                           "; ".join(idx2name.get(x, str(x)) for x in (sorted(dis_dis[c])[:cap] if cap else sorted(dis_dis[c]))))
        return out if cap is None else out[:30]

    # ATC membership lines for the class family
    atc_lines = defaultdict(list)
    for m, cs in atc.items():
        for c in sorted(cs)[:4]:
            atc_lines[drug2pkg.get(m)].append(c)

    for it in items:
        it["edges_true"] = neigh_edges(it)
        if it["family"] == "class_level":
            it["edges_true"] = [f"{idx2name.get(d, d)} --atc_class--> {', '.join(atc_lines.get(d, [])[:3])}"
                                for d in it["drugs"][:13] if atc_lines.get(d)][:20]
        for rel in set(it["relations"]):
            if rel != "atc":
                it[f"edges_only_{rel}"] = neigh_edges(it, {rel})
        it["entity_names"] = [idx2name.get(e, str(e)) for e in it["drugs"] + it["conds"]][:16]
        # items with no drug (comorbidity family) have no drug prevalence -- mark them
        # n/a rather than letting them default into the "rare" stratum and distort it
        if not it["drugs"]:
            it["drug_prevalence"] = None; it["prevalence_stratum"] = "n/a"
        else:
            fr = float(np.mean([drug_freq.get(d, 0) / N for d in it["drugs"]]))
            it["drug_prevalence"] = round(fr, 4)
            it["prevalence_stratum"] = "common" if fr > 0.05 else "uncommon" if fr > 0.01 else "rare"

    # shuffled (length-matched, wrong facts) + corrupted (entity-swapped)
    pool = [it["edges_true"] for it in items if it["edges_true"]]
    allnames = [idx2name[i] for i in list(idx2name)[:60000] if idx2type.get(i) in ("disease", "drug")]
    for it in items:
        src = pool[int(rng.integers(0, len(pool)))] if pool else []
        it["edges_shuffled"] = src[:len(it["edges_true"])] or src[:3]
        corr = []
        for e in it["edges_true"]:
            head, _, tail = e.partition("--> ")
            fake = "; ".join(allnames[int(rng.integers(0, len(allnames)))] for _ in range(3))
            corr.append(head + "--> " + fake)
        it["edges_corrupted"] = corr

    rng.shuffle(items)
    outfile = "extended_benchmark_complete.jsonl" if args.complete else "extended_benchmark.jsonl"
    with open(OUT / outfile, "w") as w:
        for it in items: w.write(json.dumps(it) + "\n")

    print(f"\n=== extended benchmark: {len(items)} items ===")
    print(f"{'family':24s} {'hops':>5} {'n':>5} {'yes':>5} {'edges':>6}  strata(common/uncommon/rare)")
    for fam in dict.fromkeys(i["family"] for i in items):
        s = [i for i in items if i["family"] == fam]
        st = [sum(1 for i in s if i["prevalence_stratum"] == k) for k in ("common", "uncommon", "rare")]
        print(f"{fam:24s} {s[0]['hops']:>5} {len(s):>5} {np.mean([i['answer']=='yes' for i in s]):>5.2f} "
              f"{np.mean([len(i['edges_true']) for i in s]):>6.1f}  {st}")
    sc = np.mean([len(i["edges_true"]) > 0 for i in items if i["answer"] == "yes"]) - \
         np.mean([len(i["edges_true"]) > 0 for i in items if i["answer"] == "no"])
    print(f"\nshortcut check -- P(edges|yes) - P(edges|no) = {sc:+.3f}  (near 0 = no leak)")
    print(f"wrote -> {OUT}/{outfile}")


if __name__ == "__main__":
    main()
