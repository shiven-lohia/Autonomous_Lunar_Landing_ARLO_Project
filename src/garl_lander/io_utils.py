"""Atomic metadata, immutable run paths, provenance and tabular logs."""

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd
import psutil


def write_json(path: str | Path, data: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def new_run(root: Path, name: str, cfg: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{name}_{config_hash(cfg)}_{stamp}_{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def provenance(cfg: dict) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"config": cfg, "config_hash": config_hash(cfg), "git_commit": commit,
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
            "packages": {d.metadata["Name"]: d.version for d in distributions()},
            "created_utc": datetime.now(timezone.utc).isoformat()}
