"""Figure 4.4.2: the twenty most compared competitors from the fits to all 52, with the intervals of boot52.py, in one panel.
The colors are those of Figure 4.4.1 (bothbad_vs_strength.pdf in main_fits.py)."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.paths import RESULTS, FIGURES

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
Z = np.load(RESULTS / "boot52.npz"); n = len(Z["models"])
MODES = [(.21, "rk", "Rao-Kupper, split ties", "#b45f45"), (.07, "grk", "grounded Rao-Kupper", "#d9a400"),
         (-.07, "det", "deterministic", "#3b6ea5"), (-.21, "two", "stochastic", "#5c8a4a")]
theta = lambda z: z[..., :n] - z[..., :n].mean(axis=-1, keepdims=True)
top = np.arange(20); order = top[np.argsort(-theta(Z["zhat_two"])[top])]; y = np.arange(20)[::-1]
fig, ax = plt.subplots(figsize=(6.4, 7.0))
for o, m, lab, col in MODES:
    d = theta(Z[f"draws_{m}"])[:, order]
    ax.hlines(y + o, np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0), color=col, lw=1.2, alpha=.5)
    ax.plot(theta(Z[f"zhat_{m}"])[order], y + o, "o", ms=2.6, color=col, label=lab)
ax.set_yticks(y); ax.set_yticklabels(Z["models"][order], fontsize=7); ax.set_xlabel(r"$\hat\theta_i$")
ax.grid(axis="x", color="0.93", lw=.6); ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(FIGURES / "rule_comparison_theta.pdf")
