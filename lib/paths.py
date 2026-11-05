"""Repository paths. Scripts write JSON under RESULTS and plots under FIGURES."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
RAW = DATA_RAW / "arena-human-preference-140k.parquet"
RESULTS = Path(os.environ.get("OUT_DIR", ROOT / "results")).resolve()
FIGURES = Path(os.environ.get("FIG_DIR", ROOT / "figures")).resolve()

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Compatibility aliases used by copied analysis scripts
OUT_DIR = str(RESULTS)
