Quick MMoE run instructions

1) Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Run a smoke training (one epoch) using Python 3.12 via the wrapper:

```powershell
py -3.12 scripts\\run_mmoe_wrapper.py --config C:\\Users\\hamza\\DLP\\kuairand-like-prediction\\configs\\mmoe_smoke.yaml
```

4) Run tests (ensure repo root is on PYTHONPATH):

```powershell
py -3.12 -m pytest -q
```

Notes:
- If `py -3.12` isn't available, replace with an absolute python path (e.g. `C:\\Python312\\python.exe`).
- The wrapper `scripts/run_mmoe_wrapper.py` sets the working directory and `sys.path` so `src` imports resolve.
