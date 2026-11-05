"""Ranks on non-code and code prompts of the competitors in the top 20 of either stratum, from code_theta_draws.npz."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.paths import RESULTS, FIGURES
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = np.load(RESULTS / "code_theta_draws.npz")
m = [str(x) for x in Z["models"]]; tc, tn = Z["theta_code"], Z["theta_noncode"]
rc, rn = (-tc).argsort().argsort() + 1, (-tn).argsort().argsort() + 1
dd = Z["draws_code"] - Z["draws_noncode"]
claude4 = [i for i, x in enumerate(m) if x.startswith("claude") and ("opus-4" in x or "sonnet-4" in x)]
claude = [i for i, x in enumerate(m) if x.startswith("claude")]; gemini = [i for i, x in enumerate(m) if x.startswith("gemini")]
for lab, v in [("claude 4", claude4), ("claude", claude), ("gemini", gemini)]:
    print(lab, len(v), round(float((tc - tn)[v].mean()), 3), np.percentile(dd[:, v].mean(axis=1), [2.5, 97.5]).round(3))
lo, hi = np.percentile(dd, [2.5, 97.5], axis=0); print("intervals excluding zero", int(((lo > 0) | (hi < 0)).sum()))
print("same top 10", set(np.where(rc <= 10)[0]) == set(np.where(rn <= 10)[0]), "first", m[int(rc.argmin())], m[int(rn.argmin())])
for i in claude4: print(m[i], rn[i], rc[i], round(float(tc[i] - tn[i]), 2), lo[i].round(2), hi[i].round(2))
show = [i for i in range(len(m)) if rc[i] <= 20 or rn[i] <= 20] + [i for i in claude4 if rc[i] > 20 and rn[i] > 20]
print(len(show), max(max(rc[i], rn[i]) for i in show))
BLUE, ORANGE, INK, MUTED, FAINT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#c9c8c2"
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(5.6, 5.4))
colour = {i: BLUE for i in claude4}; colour.update({i: ORANGE for i in gemini})
for i in show:
    c = colour.get(i)
    if c is None: ax.plot([0, 1], [rn[i], rc[i]], color=FAINT, lw=0.8, marker="o", ms=2, zorder=1)
for i in show:
    c = colour.get(i)
    if c is not None:
        ax.plot([0, 1], [rn[i], rc[i]], color="white", lw=3, zorder=2)
        ax.plot([0, 1], [rn[i], rc[i]], color=c, lw=1.5, marker="o", ms=3.2, mec="white", mew=0.5, zorder=3)
    ink = INK if c is not None else MUTED
    ax.text(-0.03, rn[i], f"{m[i]}  {rn[i]}", ha="right", va="center", fontsize=5.3, color=ink)
    ax.text(1.03, rc[i], f"{rc[i]}  {m[i]}", ha="left", va="center", fontsize=5.3, color=ink)
top = max(max(rc[i], rn[i]) for i in show)
ax.set_ylim(top + 0.8, 0.2); ax.set_xlim(-0.02, 1.02); ax.set_xticks([0, 1]); ax.set_xticklabels(["non-code prompts", "code prompts"], fontsize=7, color=INK)
ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=3); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.plot([], [], color=BLUE, lw=1.5, label="Claude 4 models"); ax.plot([], [], color=ORANGE, lw=1.5, label="Gemini models")
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=2)
fig.subplots_adjust(left=0.37, right=0.63, top=0.95, bottom=0.05)
fig.savefig(FIGURES / "code_ranks.pdf")
