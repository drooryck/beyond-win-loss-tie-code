"""
Figure A.4.2: how using the both-bad outcomes changes the abilities, for the twenty most compared competitors in the
fits to all 52. Each difference is taken within one resample of boot52.py.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from lib.data import load, counts_for, load_clean
from lib.exact_score import rk3
from lib.paths import RESULTS, FIGURES

ORANGE, VIOLET, BLUE, INK = "#eb6834", "#6250d6", "#2a78d6", "#0b0b0b"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
Z = np.load(RESULTS / "boot52.npz")
models = list(Z["models"]); n = len(models)
theta = lambda z: z[..., :n] - z[..., :n].mean(axis=-1, keepdims=True)
D, _ = counts_for(load_clean(load), models)
drop = theta(rk3(D, n, False)[:n]); drop_draws = theta(Z["draws_drop"])
th = {m: theta(Z[f"zhat_{m}"]) for m in ("det", "two")}
top = np.arange(20); order = top[np.argsort(-th["two"][top])]; y = np.arange(20)[::-1]
fig, ax = plt.subplots(figsize=(6.4, 5.0))
for o, m, lab, col in [(.12, "det", "deterministic", VIOLET), (-.12, "two", "stochastic", BLUE)]:
    dd = theta(Z[f"draws_{m}"]) - drop_draws
    ax.hlines(y + o, *np.percentile(dd, [2.5, 97.5], axis=0)[:, order], color=col, lw=1.4, alpha=.5)
    ax.plot((th[m] - drop)[order], y + o, "o", ms=3, color=col, label=lab)
    print(lab, "largest absolute difference among the twenty", round(float(np.abs(th[m] - drop)[top].max()), 3), "all 52", round(float(np.abs(th[m] - drop).max()), 3))
ax.axvline(0, color=INK, lw=.8); ax.set_yticks(y); ax.set_yticklabels(np.array(models)[order], fontsize=6.8); ax.grid(axis="x", color="0.93", lw=.6)
ax.set_xlabel(r"$\hat\theta_i$ minus $\hat\theta_i$ from Rao-Kupper with both-bad removed"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIGURES / "both_bad_effect.pdf")
print("rank corr with stochastic", round(spearmanr(drop, th["two"]).statistic, 3), "with det", round(spearmanr(drop, th["det"]).statistic, 3))
