"""Minimal YAML writer (no PyYAML in the analysis environments).

Supports dict / list / str / int / float / bool / None / pathlib.Path, nested to
any depth.  Output is valid YAML 1.2 and round-trips through PyYAML where it is
available.  A JSON twin is always written alongside for machine reading.
"""
from __future__ import annotations
import json, re
from pathlib import Path

_SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-./+@ ]*$")
_RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "~", ""}


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v):
            return ".nan"
        return repr(v)
    if isinstance(v, Path):
        v = str(v)
    s = str(v)
    if (_SAFE.match(s) and s.lower() not in _RESERVED and not s.startswith(" ")
            and not s.endswith(" ") and not re.match(r"^[-+]?[0-9.]", s)):
        return s
    return json.dumps(s, ensure_ascii=False)


def dumps(obj, indent: int = 0) -> str:
    pad = "  " * indent
    out = []
    if isinstance(obj, dict):
        if not obj:
            return pad + "{}\n"
        for k, v in obj.items():
            key = _scalar(str(k))
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{key}:\n{dumps(v, indent + 1)}")
            elif isinstance(v, (dict, list)):
                out.append(f"{pad}{key}: {'{}' if isinstance(v, dict) else '[]'}\n")
            else:
                out.append(f"{pad}{key}: {_scalar(v)}\n")
        return "".join(out)
    if isinstance(obj, list):
        if not obj:
            return pad + "[]\n"
        for v in obj:
            if isinstance(v, (dict, list)) and v:
                body = dumps(v, indent + 1)
                first, rest = body.split("\n", 1)
                out.append(f"{pad}- {first.lstrip()}\n{rest}" if rest.strip() else f"{pad}- {first.lstrip()}\n")
            else:
                out.append(f"{pad}- {_scalar(v) if not isinstance(v, (dict, list)) else ('{}' if isinstance(v, dict) else '[]')}\n")
        return "".join(out)
    return pad + _scalar(obj) + "\n"


def dump(obj, path: Path | str, json_twin: bool = True) -> None:
    path = Path(path)
    path.write_text(dumps(obj))
    if json_twin:
        path.with_suffix(".json").write_text(json.dumps(obj, indent=2, default=str))
