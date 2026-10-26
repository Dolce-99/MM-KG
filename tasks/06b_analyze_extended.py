"""
06b_analyze_extended.py   --   env: any python3 (stdlib + numpy)   [CPU, instant]
================================================================================
Merge every extended-audit run into the evidence tables for the write-up.

Reads outputs/audit/extended_results_{main,ablate,cite,small}.json (whichever exist)
plus the week-1 medication_audit_results.json, and emits:

  1. HOP LADDER          does the KG advantage grow with reasoning depth?
  2. CONTENT vs LENGTH   C_true vs C_shuffled -- is the gain about facts or tokens?
  3. READING vs PATTERN  C_true vs C_corrupted -- does swapping entity names break it?
  4. PER-FAMILY          every question family, every arm
  5. PREVALENCE STRATA   is the gain largest where the parametric prior is thinnest?
  6. RELATION ABLATION   which KG layer carries the gain
  7. MODEL SIZE          9B vs 35B
  8. CITATION VALIDITY   when the model quotes an edge, was it really in the context?

Writes outputs/audit/EVIDENCE_TABLES.md (+ prints it).
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT  = HERE / "outputs" / "audit"


def load(tag):
    p = OUT / (f"extended_results_{tag}.json" if tag else "extended_results.json")
    return json.loads(p.read_text()) if p.exists() else None


def fmt(d):
    if not d: return "-"
    star = "**" if (d["ci"][0] > 0 or d["ci"][1] < 0) else ""
    return f"{star}{d['mean']:+.3f}{star} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] p={d['p']}"


def main():
    main_r, abl, cite, small = load("main"), load("ablate"), load("cite"), load("small")
    wk1 = json.loads((OUT / "medication_audit_results.json").read_text()) \
        if (OUT / "medication_audit_results.json").exists() else None
    if not any([main_r, abl, cite, small, wk1]):
        raise SystemExit("no results found -- submit 06_run_extended.sh first")

    L = ["# Extended medication audit — evidence tables", "",
         "Balanced accuracy; chance = 0.500. CIs from a paired bootstrap over items;",
         "**bold** marks a CI excluding zero.", ""]

    if main_r:
        L += ["## 1. Hop ladder — does the advantage grow with reasoning depth?", "",
              "| Hops | n | A_flat | C_true | C_true − A | C_true − shuffled |", "|---|---|---|---|---|---|"]
        for h in sorted(main_r["by_hop"]):
            r = main_r["by_hop"][h]
            if not r: continue
            L.append(f"| {h} | {r['n']} | {r['A_flat']:.3f} | {r['C_true']:.3f} | "
                     f"{fmt(r.get('C_true_minus_A'))} | {fmt(r.get('C_true_minus_shuffled'))} |")
        L += ["", "*If C − A rises with hop count, the graph is supporting composition, not lookup.*", ""]

        L += ["## 2 & 3. Controls — content vs length, reading vs pattern-matching", "",
              "| Family | A_flat | C_true | C_shuffled | C_corrupted | C_true − shuffled |",
              "|---|---|---|---|---|---|"]
        for fam, r in main_r["families"].items():
            if not r: continue
            L.append(f"| {fam} | {r.get('A_flat',0):.3f} | {r.get('C_true',0):.3f} | "
                     f"{r.get('C_shuffled',0):.3f} | {r.get('C_corrupted',0):.3f} | "
                     f"{fmt(r.get('C_true_minus_shuffled'))} |")
        L += ["", "*C_shuffled is length-matched with wrong facts; C_corrupted keeps the edge "
                  "structure but swaps entity names. Both should sit near A_flat.*", ""]

        L += ["## 5. Prevalence strata — is the gain largest where the prior is thinnest?", "",
              "| Drug prevalence | n | A_flat | C_true | C_true − A |", "|---|---|---|---|---|"]
        for s in ("common", "uncommon", "rare"):
            r = main_r["by_stratum"].get(s)
            if not r: continue
            L.append(f"| {s} | {r['n']} | {r['A_flat']:.3f} | {r['C_true']:.3f} | {fmt(r.get('C_true_minus_A'))} |")
        L.append("")

    if abl:
        arms = [a for a in abl["arms"] if a.startswith("C_only_")]
        if arms:
            L += ["## 6. Relation ablation — which KG layer carries the gain?", "",
                  "| Family | A_flat | C_true | " + " | ".join(a.replace("C_only_", "") for a in arms) + " |",
                  "|---" * (3 + len(arms)) + "|"]
            for fam, r in abl["families"].items():
                if not r: continue
                cells = " | ".join(f"{r[a]:.3f}" if a in r else "-" for a in arms)
                L.append(f"| {fam} | {r.get('A_flat',0):.3f} | {r.get('C_true',0):.3f} | {cells} |")
            L.append("")

    if small and main_r:
        L += ["## 7. Model size — does the effect survive a smaller model?", "",
              "| Hops | 9B: A → C (Δ) | 35B: A → C (Δ) |", "|---|---|---|"]
        for h in sorted(main_r["by_hop"]):
            a, b = small["by_hop"].get(h), main_r["by_hop"].get(h)
            if not (a and b): continue
            da, db = a.get("C_true_minus_A"), b.get("C_true_minus_A")
            L.append(f"| {h} | {a['A_flat']:.3f} → {a['C_true']:.3f} "
                     f"({da['mean']:+.3f}) | {b['A_flat']:.3f} → {b['C_true']:.3f} ({db['mean']:+.3f}) |")
        L += ["", "*A larger gain for the smaller model supports the thin-prior account.*", ""]

    if cite and cite.get("citation"):
        L += ["## 8. Citation validity — is the quoted evidence really in the context?", "",
              "| Arm | quoted an edge | quote actually present |", "|---|---|---|"]
        for a, c in cite["citation"].items():
            L.append(f"| {a} | {c['rate_with_evidence']:.2f} | {c['valid_when_cited']} |")
        L += ["", "*Not tautological: the model can quote a real edge that does not support its "
                  "answer, or invent one. This measures auditable grounding.*", ""]

    if wk1:
        L += ["## Reference — week-1 audit (single-hop, same design)", "",
              "| Family | A_flat | B_concept | C_graph | C − A |", "|---|---|---|---|---|"]
        for fam, r in wk1["families"].items():
            d = r.get("C_minus_A")
            L.append(f"| {fam} | {r['A_flat']['bal_acc']:.3f} | {r['B_concept']['bal_acc']:.3f} | "
                     f"{r['C_graph']['bal_acc']:.3f} | {fmt(d)} |")
        L.append("")

    txt = "\n".join(L)
    (OUT / "EVIDENCE_TABLES.md").write_text(txt + "\n")
    print(txt)
    print(f"\nwrote -> {OUT}/EVIDENCE_TABLES.md")


if __name__ == "__main__":
    main()
