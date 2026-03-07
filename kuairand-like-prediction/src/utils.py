import os
import joblib
import torch


def save_sklearn(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def save_torch(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_df(df, path: str):
    """Save DataFrame to CSV (creates parent dir)."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def load_yaml(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    """Return short git commit hash if available, otherwise empty string."""
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        return out
    except Exception:
        return ""


def write_run_metadata(path: str, metadata: dict):
    """Write run metadata as JSON to `path` (creates parent dir).

    The metadata should contain timestamp, config, commit hash, dataset counts, and metrics.
    """
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


