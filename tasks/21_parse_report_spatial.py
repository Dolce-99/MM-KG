"""
21_parse_report_spatial.py   --   env: llm_qwen   [GPU cluster]
================================================================================
STAGE 1 of the spatial-information ladder.

For every ICU stay's early chest film, parse the matched radiology-report text
into a structured SPATIAL schema:

  per finding : severity, laterality, anatomic location, interval change
  per device  : name and described position

The note<->study matching replicates 04_patient_tables.py exactly (nearest note
within +/-24h by subject, merge_asof on study datetime) so the report we parse
here is the same one whose CUIs became the graph's REPORTED_FINDING edges.

Only the FINDINGS section of the report is parsed. The IMPRESSION often echoes
the clinical indication ("concern for CHF"), which would leak non-image context
into what is supposed to be an image-only information ceiling.

Output: outputs/mm/report_spatial_ihm.jsonl
  {hadm_id, study_id, note_id, findings_text, parsed:{findings:[...],devices:[...]},
   parse_ok}
"""
from __future__ import annotations
import os
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P2   = HERE.parent / "graph" / "outputs"
CXR  = Path(os.environ.get("MIMIC_CXR_DIR", os.path.expandvars("$MMKG_HOME/data/mimic-cxr-jpg/2.1.0")))
OUT  = HERE / "outputs" / "mm"

SYS = ("You are a radiologist structuring a chest x-ray report. Output ONLY a JSON "
       "object, no other text, with this exact schema:\n"
       '{"findings": [{"name": "<short finding name>", '
       '"severity": "trace|small|mild|moderate|severe|large|massive|unspecified", '
       '"laterality": "left|right|bilateral|unspecified", '
       '"location": "<anatomic location or null>", '
       '"change": "new|worsened|improved|resolved|unchanged|unspecified"}], '
       '"devices": [{"name": "<device>", "position": "<described position or null>"}]}\n'
       "List only ABNORMAL findings that are present (skip explicit negations like "
       "'no pneumothorax'). List every support device mentioned, with its position "
       "if the report describes one.")


def findings_section(text: str) -> str:
    """FINDINGS section only, excluding EXAMINATION / INDICATION / HISTORY /
    COMPARISON / TECHNIQUE and IMPRESSION.

    The indication carries clinical context ("cirrhosis ... airway compromise")
    that would leak non-image information into the image-only ceiling, so it must
    never survive. But many reports have no FINDINGS header at all -- their
    narrative sits inside the body of the last administrative header (usually
    COMPARISON). For those we take that last body and drop only its first
    sentence (the actual comparison phrase), which recovers the narrative
    without the indication.
    """
    t = re.sub(r"IMPRESSION\s*:.*$", "", text or "", flags=re.S | re.I)
    parts = re.split(r"([A-Z][A-Z ()/]{3,}:)", t)
    # parts = [pre, HDR1, body1, HDR2, body2, ...]
    headers = [parts[i].strip(": ").upper() for i in range(1, len(parts), 2)]
    bodies  = [parts[i + 1] for i in range(1, len(parts), 2)]
    for hdr, body in zip(headers, bodies):
        if hdr.startswith("FINDING") and len(body.strip()) > 40:
            return body.strip()
    if not bodies:
        return t.strip()
    ADMIN = ("EXAMINATION", "INDICATION", "HISTORY", "CLINICAL", "REASON",
             "COMPARISON", "TECHNIQUE", "STUDY", "WET READ", "NOTIFICATION")
    last = bodies[-1].strip()
    if headers and any(headers[-1].startswith(a) for a in ADMIN):
        # drop the header's own first sentence, keep the trailing narrative
        m = re.search(r"[.!?]\s+", last)
        last = last[m.end():] if m else ""
    return last.strip()


def study_dt(date, time):
    d = pd.to_datetime(date.astype(str).str.split(".").str[0], format="%Y%m%d", errors="coerce")
    t = time.astype(str).str.split(".").str[0].str.zfill(6).str[:6]
    hh = pd.to_numeric(t.str[:2], errors="coerce"); mm = pd.to_numeric(t.str[2:4], errors="coerce")
    ss = pd.to_numeric(t.str[4:6], errors="coerce")
    ok = (hh < 24) & (mm < 60) & (ss < 60)
    delta = pd.to_timedelta(np.where(ok, hh * 3600 + mm * 60 + ss, 12 * 3600).astype("float"), unit="s")
    return d + delta


def build_table():
    cxr = pd.read_parquet(P2 / "ihm" / "icu_early_cxr.parquet")
    notes = pd.read_parquet(P2 / "reports" / "cohort_chest_notes.parquet")
    md = pd.read_csv(CXR / "mimic-cxr-2.0.0-metadata.csv.gz",
                     usecols=["subject_id", "study_id", "StudyDate", "StudyTime"])
    md = md.drop_duplicates("study_id")
    md["sdt"] = study_dt(md.StudyDate, md.StudyTime)
    stt = md[md.study_id.isin(set(cxr.study_id))][["subject_id", "study_id", "sdt"]] \
            .dropna(subset=["sdt"]).sort_values("sdt")
    rep = notes.copy(); rep["ct"] = pd.to_datetime(rep.charttime, errors="coerce")
    rep = rep.dropna(subset=["ct"]).sort_values("ct")
    m = pd.merge_asof(rep[["subject_id", "note_id", "ct", "text"]], stt,
                      left_on="ct", right_on="sdt", by="subject_id",
                      direction="nearest", tolerance=pd.Timedelta(hours=24))
    m = m.dropna(subset=["study_id"])
    # one note per study: the nearest in time
    m["gap"] = (m.ct - m.sdt).abs()
    m = m.sort_values("gap").drop_duplicates("study_id")
    tab = cxr.merge(m[["study_id", "note_id", "text"]], on="study_id", how="inner")
    tab["findings_text"] = tab.text.map(findings_section)
    tab = tab[tab.findings_text.str.len() > 40]
    return tab[["hadm_id", "study_id", "note_id", "findings_text"]]


def parse_json(t: str):
    body = re.sub(r"<think>.*?</think>", " ", t, flags=re.S)
    m = re.search(r"\{.*\}", body, re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(o, dict) or "findings" not in o:
        return None
    o.setdefault("devices", [])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N only")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tab = build_table()
    if args.limit:
        tab = tab.head(args.limit)
    print(f"{len(tab)} ICU studies with a matched FINDINGS section", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=args.max_model_len, trust_remote_code=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=700)

    prompts = []
    for txt in tab.findings_text:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": txt[:3000]}]
        try:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True,
                                                   enable_thinking=False))
        except TypeError:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True))
    outs = llm.generate(prompts, sp)

    n_ok = 0
    with open(OUT / "report_spatial_ihm.jsonl", "w") as w:
        for row, o in zip(tab.itertuples(index=False), outs):
            parsed = parse_json(o.outputs[0].text)
            ok = parsed is not None
            n_ok += ok
            w.write(json.dumps({"hadm_id": int(row.hadm_id), "study_id": int(row.study_id),
                                "note_id": str(row.note_id),
                                "findings_text": row.findings_text,
                                "parsed": parsed if ok else None,
                                "parse_ok": ok}) + "\n")
    print(f"parsed OK {n_ok}/{len(tab)} ({n_ok/len(tab):.1%}) -> "
          f"{OUT/'report_spatial_ihm.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
