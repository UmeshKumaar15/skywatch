"""Configuration loading: defaults <- user YAML <- environment overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
SCENARIO_DIR = CONFIG_DIR / "scenarios"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _env_overrides(cfg: dict) -> dict:
    """SKYWATCH__SERVER__PORT=9000 -> cfg['server']['port'] = 9000."""
    for key, val in os.environ.items():
        if not key.startswith("SKYWATCH__"):
            continue
        path = [p.lower() for p in key.split("__")[1:]]
        node = cfg
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = yaml.safe_load(val)
    return cfg


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return _env_overrides(cfg)


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def list_scenarios() -> list[dict[str, Any]]:
    out = []
    for f in sorted(SCENARIO_DIR.glob("*.yaml")):
        with open(f) as fh:
            s = yaml.safe_load(fh)
        out.append({"name": s.get("name", f.stem), "description": s.get("description", ""), "seed": s.get("seed", 0),
                    "duration": s.get("duration")})
    return out


def load_scenario(name_or_path: str, seed: Optional[int] = None, **overrides: Any) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.exists():
        p = SCENARIO_DIR / f"{name_or_path}.yaml"
    with open(p) as f:
        s = yaml.safe_load(f)
    if seed is not None:
        s["seed"] = int(seed)
    s.update(overrides)
    return s
