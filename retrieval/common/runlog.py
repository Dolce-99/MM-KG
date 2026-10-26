"""Per-run provenance: config, command, environment, input/output digests, timings.

    run = Run("exp1", "r2_dev", config)          # creates outputs/exp1/runs/r2_dev_<stamp>/
    run.add_input(path); ... ; run.add_output(path)
    run.finish(metrics={...}, counts={...})      # writes manifest.json

Item-level predictions must be written under RESTRICTED (git-ignored) and only
referenced from the manifest by path and digest.
"""
from __future__ import annotations
import os
import json, os, platform, socket, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from . import paths, yamlw


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(paths.REPO), *args],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unavailable"


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "executable": sys.executable,
            "host": socket.gethostname(), "platform": platform.platform(),
            "cwd": os.getcwd(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_head": _git("rev-parse", "HEAD"),
            "git_dirty_files": len(_git("status", "--porcelain").splitlines()) if _git("status", "--porcelain") != "unavailable" else None}
    for m in ["numpy", "pandas", "torch", "transformers", "vllm", "torch_geometric", "sklearn", "faiss"]:
        try:
            mod = __import__(m); info[m] = getattr(mod, "__version__", "?")
        except Exception:
            info[m] = None
    return info


class Run:
    def __init__(self, exp: str, name: str, config: dict, out_root: Path | None = None):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = (out_root or paths.OUT) / exp / "runs" / f"{name}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.t0 = time.time(); self.exp = exp; self.name = name
        self.config = config; self.inputs = []; self.outputs = []
        yamlw.dump(config, self.dir / "config.yaml")
        (self.dir / "command.txt").write_text(" ".join(map(_q, sys.argv)) + "\n")
        yamlw.dump(env_info(), self.dir / "env.yaml")

    def _rec(self, path):
        p = Path(path)
        return {"path": str(p), "sha256": paths.sha256(p) if p.exists() else None,
                "bytes": p.stat().st_size if p.exists() else None}

    def add_input(self, path):  self.inputs.append(self._rec(path))
    def add_output(self, path): self.outputs.append(self._rec(path))

    def finish(self, metrics: dict | None = None, counts: dict | None = None, notes: str = "") -> Path:
        man = {"exp": self.exp, "name": self.name, "run_dir": str(self.dir),
               "started": datetime.fromtimestamp(self.t0).isoformat(timespec="seconds"),
               "finished": datetime.now().isoformat(timespec="seconds"),
               "wall_seconds": round(time.time() - self.t0, 1),
               "config": self.config, "inputs": self.inputs, "outputs": self.outputs,
               "counts": counts or {}, "metrics": metrics or {}, "notes": notes}
        (self.dir / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
        return self.dir / "manifest.json"


def _q(s: str) -> str:
    return s if all(c.isalnum() or c in "-_./=:," for c in s) else json.dumps(s)
