import sys
from pathlib import Path

# ensure repo root is on sys.path
repo_base = Path(__file__).resolve().parents[1]
# project folder inside workspace
proj_dir = repo_base / 'kuairand-like-prediction'
if proj_dir.exists():
    sys.path.insert(0, str(proj_dir))
else:
    sys.path.insert(0, str(repo_base))

from src.train_mmoe import main

if __name__ == '__main__':
    import os
    os.chdir(proj_dir)
    main()
