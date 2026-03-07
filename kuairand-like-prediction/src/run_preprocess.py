"""Simple CLI to run preprocessing (joins tables, applies leakage policy, saves processed artifacts)."""
import argparse
import yaml
from pathlib import Path
from .data_loader import load_tables
from .preprocess import build_and_save_processed
from .utils import ensure_dir
import json


def run(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    tables = load_tables(cfg)
    processed_dir = cfg.get("processed_dir", "data/processed")
    ensure_dir(processed_dir)
    summary = build_and_save_processed(tables, cfg, processed_dir=processed_dir)
    out = Path(processed_dir) / "preprocess_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("Preprocessing complete. Summary:")
    print(json.dumps(summary, indent=2))
    # Print discovered training columns and banned columns if available
    feat_json = Path(processed_dir) / "features_summary.json"
    if feat_json.exists():
        print("Features summary:")
        print(feat_json.read_text())


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    cli()
