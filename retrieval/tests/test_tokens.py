"""Tokenizer packing test (env: llm_qwen, CPU, offline)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import tokens, paths


def test_pack():
    tok = tokens.load_tokenizer()
    units = [f"unit {i}: creatinine {i}.{i} mg/dL measured in the intensive care unit." for i in range(40)]
    full = tokens.pack(tok, units, budget=10_000)
    assert not full["truncated"] and full["kept"] == list(range(40))
    cut = tokens.pack(tok, units, budget=120, fixed=20)
    assert cut["truncated"] and 0 < len(cut["kept"]) < 40 and cut["tokens"] <= 100
    assert abs(cut["untruncated_tokens"] - full["tokens"]) <= max(1, 0.01 * full["tokens"])   # per-unit sum vs joint count
    p = tokens.render_chat(tok, paths.SYS_PROMPT, "Question: yes or no?")
    assert tokens.n_tokens(tok, p) > 10


if __name__ == "__main__":
    test_pack(); print("ok test_pack")
