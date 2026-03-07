from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return a dict.

    Resolves the path and returns a dict. Keeps contents as-is; callers can
    provide defaults where needed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    # Normalize common keys
    if "data" not in cfg:
        cfg.setdefault("data", {})
    return cfg
