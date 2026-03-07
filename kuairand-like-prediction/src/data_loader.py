import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import os


def _find_sample_source(cfg: Dict) -> Optional[Path]:
    """Try to find existing sample/rob data in common locations.

    Returns a Path to a directory containing CSVs or None.
    """
    candidates = []
    data_cfg = cfg.get("data", {})
    # explicit sample_dir in config
    if data_cfg.get("sample_dir"):
        candidates.append(Path(data_cfg.get("sample_dir")))
    # common places
    candidates.extend([Path("data/sample"), Path("data"), Path("rob/data"), Path("../rob/data")])
    for p in candidates:
        if p.exists() and p.is_dir():
            # check for at least interactions.csv
            for cand in ["interactions_small.csv", "interactions.csv"]:
                f = p / cand
                if f.exists():
                    # prefer small files: accept only if small (<=1k rows) to avoid picking full datasets
                    try:
                        import pandas as _pd

                        n = _pd.read_csv(f, nrows=1001).shape[0]
                    except Exception:
                        n = 1001
                    if n <= 1000:
                        return p
                    # if interactions_small exists but is large, skip; continue searching
    return None


def _ensure_sample_generated(sample_dir: Path):
    """Create synthetic sample files in sample_dir if they don't exist.

    This keeps generation idempotent and lightweight.
    """
    from .sample_data import generate_synthetic_sample

    sample_dir.mkdir(parents=True, exist_ok=True)
    # only generate if interactions not present
    if not (sample_dir / "interactions.csv").exists():
        generate_synthetic_sample(sample_dir)


def load_csv(path: Path, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """Load a CSV from `path` into a DataFrame.

    Args:
        path: Path to CSV file.
        parse_dates: list of columns to parse as datetime (optional).

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    if parse_dates is None:
        parse_dates = ["timestamp"]
    try:
        return pd.read_csv(path, parse_dates=[c for c in parse_dates if c in pd.read_csv(path, nrows=0).columns], low_memory=False)
    except Exception:
        # Last resort: read without date parsing
        return pd.read_csv(path, low_memory=False)


def load_tables(config: Dict) -> Dict[str, pd.DataFrame]:
    """Load the set of expected tables from the `data` section of a config dict.

    Expected keys (optional):
      - interactions: path
      - users: path
      - videos: path (basic)
      - video_stats: path (aggregated stats)

    The returned dict maps table names to DataFrames (only for files that exist).
    """
    data_cfg = config.get("data", {})
    data_dir = Path(data_cfg.get("dir", "data"))
    mode = config.get("mode") or data_cfg.get("mode") or config.get("run_mode") or "full"
    # Prefer an explicit `data.dir` from config. Only fall back to a top-level
    # `real_data/` folder when no explicit data.dir is provided and not in sample mode.
    real_path = Path("real_data")
    if not data_cfg.get("dir") and real_path.exists() and real_path.is_dir() and str(mode).lower() != "sample":
        data_dir = real_path

    # If sample mode requested, prefer existing sample source or generate synthetic
    if str(mode).lower() == "sample":
        sample_src = _find_sample_source(config)
        if sample_src:
            # if sample source is the workspace data dir containing large files, look for small variants
            data_dir = sample_src
        else:
            # generate synthetic sample under data/sample
            sample_dir = data_dir / "sample"
            _ensure_sample_generated(sample_dir)
            data_dir = sample_dir
    tables = {}
    mapping = {
        # prefer explicit paths in config; fall back to common KuaiRand filenames
        # treat empty strings as not-provided
        "interactions": data_cfg.get("interactions") or None,
        "users": data_cfg.get("users") or None,
        "videos": data_cfg.get("videos") or None,
        "video_stats": data_cfg.get("video_stats") or None,
    }
    # helper to find kuairand-style files in data_dir
    files_in_dir = {p.name: p for p in data_dir.glob("*") if p.is_file()}

    # If interactions not provided, try common log files (prefer random log)
    if not mapping.get("interactions"):
        for cand in ["log_random_4_22_to_5_08_1k.csv", "log_random_4_22_to_5_08.csv"]:
            if (data_dir / cand).exists():
                mapping["interactions"] = cand
                break
    if not mapping.get("interactions"):
        # pick any file starting with log_
        for fname in files_in_dir:
            if fname.startswith("log_"):
                mapping["interactions"] = fname
                break

    # if users not provided, try user_features or user_clusters
    if not mapping.get("users"):
        for cand in ["user_features_1k.csv", "user_features.csv", "user_clusters.csv"]:
            if (data_dir / cand).exists():
                mapping["users"] = cand
                break

    # if videos not provided, try video features
    if not mapping.get("videos"):
        for cand in ["video_features_basic_1k.csv", "video_features_basic.csv", "video_clusters.csv"]:
            if (data_dir / cand).exists():
                mapping["videos"] = cand
                break
    # also accept a nested video_clusters/video_clusters.csv produced by unzipping
    if not mapping.get("videos") and (data_dir / "video_clusters").exists():
        nested = data_dir / "video_clusters" / "video_clusters.csv"
        if nested.exists():
            mapping["videos"] = str(Path("video_clusters") / "video_clusters.csv")

    # if video_stats not provided, try the statistics file
    if not mapping.get("video_stats"):
        for cand in ["video_features_statistic_1k.csv", "video_features_statistic.csv"]:
            if (data_dir / cand).exists():
                mapping["video_stats"] = cand
                break

    for name, rel in mapping.items():
        if rel is None:
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = data_dir / rel
        if path.exists():
            try:
                tables[name] = load_csv(path)
            except Exception as e:
                # skip problematic files but surface the issue
                raise
    return tables

