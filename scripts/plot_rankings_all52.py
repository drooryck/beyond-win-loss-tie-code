"""Connected-lines chart of the ranks of all 52 competitors under our two models, from rankings.json."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.paths import RESULTS, FIGURES
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(RESULTS / "rankings.json"))
models = R["models"]; n = len(models)
COLS = [("stoch", "stochastic"), ("det", "deterministic")]
rank = np.array([R["rank"][k] for k, _ in COLS]).T
move = rank.max(axis=1) - rank.min(axis=1)
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
high = list(np.argsort(-move)[:4]); color = {k: HUES[s] for s, k in enumerate(high)}
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#c9c8c2"
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(5.6, 7.3))
x = np.arange(len(COLS))
for k in range(n):
    if k not in color: ax.plot(x, rank[k], color=FAINT, lw=0.7, marker="o", ms=1.6, zorder=1)
for k in high:
    ax.plot(x, rank[k], color="white", lw=3.2, zorder=2)
    ax.plot(x, rank[k], color=color[k], lw=1.6, marker="o", ms=3.4, mec="white", mew=0.5, zorder=3)
for k in range(n):
    ink = INK if k in color else MUTED; w = "bold" if k in color else "normal"
    ax.text(-0.06, rank[k][0], f"{models[k]}  {rank[k][0]}", ha="right", va="center", fontsize=4.9, color=ink, fontweight=w)
    ax.text(len(COLS) - 1 + 0.06, rank[k][-1], f"{rank[k][-1]}  {models[k]}", ha="left", va="center", fontsize=4.9, color=ink, fontweight=w)
ax.set_ylim(n + 0.8, 0.2); ax.set_xlim(-0.04, len(COLS) - 0.96)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in COLS], fontsize=6.5, color=INK)
ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=3); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
for c in x: ax.axvline(c, color="#ecebe6", lw=0.6, zorder=0)
fig.subplots_adjust(left=0.36, right=0.64, top=0.945, bottom=0.005)
fig.savefig(FIGURES / "rankings_all52.pdf")
fig.savefig(FIGURES / "rankings_all52.png")
print([(models[k], rank[k].tolist()) for k in high])
