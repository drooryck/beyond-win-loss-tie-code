"""
Row-bootstrap of the split-ties, deterministic and stochastic fits on all 52 competitors, keeping every draw, and of
Rao-Kupper fitted after removing both-bad outcomes and of grounded Rao-Kupper, on the same resampled rows.

Saves results/boot52.npz, with the abilities in the order of `models`. Run with `make bootstrap`.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from multiprocessing import Pool
import numpy as np, pandas as pd

from lib.data import load, counts_for, load_clean
from lib.exact_score import lbfgs, rk3, grk_fit
from lib.paths import RESULTS

BOOT = 200
MODES = ["rk", "det", "two"]
df = load_clean(load)
models = list(pd.concat([df.model_a, df.model_b]).value_counts().index); n = len(models)
D, _ = counts_for(df, models)
zhat = {m: lbfgs(D, n, m).x for m in MODES}
zhat["grk"] = grk_fit(D, n).x


def one(b):
    rng = np.random.default_rng([20261105, b])
    d, _ = counts_for(df.iloc[rng.integers(0, len(df), len(df))], models)
    return {**{m: lbfgs(d, n, m, z0=zhat[m].copy()).x for m in MODES}, "drop": rk3(d, n, False)[:n], "grk": grk_fit(d, n, z0=zhat["grk"].copy()).x}


if __name__ == "__main__":
    with Pool() as p: out = p.map(one, range(BOOT))
    np.savez(RESULTS / "boot52.npz", models=np.array(models), comparisons=pd.concat([df.model_a, df.model_b]).value_counts().reindex(models).to_numpy(),
             **{f"zhat_{m}": zhat[m] for m in MODES + ["grk"]}, **{f"draws_{m}": np.array([o[m] for o in out]) for m in MODES + ["drop", "grk"]})
    print("wrote", RESULTS / "boot52.npz")
