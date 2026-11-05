"""
Two candidate figures of the fitted model, on the first-order data.

latent_plane: latent performances simulated from the fit for a strong and a weak pair, under the
stochastic and deterministic models, with the fitted tie band and quality threshold. The points are
draws from the fitted model. They are not observed, since the data record only the button.

field_vs_threshold: each competitor's ability relative to the fitted quality threshold.

Run with `make figures`.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.data import load, counts_for
from lib.data import load_clean
from lib.exact_score import lbfgs, probs, sig
from lib.paths import RESULTS, FIGURES

df = load_clean(load)
models = list(pd.concat([df.model_a, df.model_b]).value_counts().index); n = len(models)
D, _ = counts_for(df, models)
zs, zd = lbfgs(D, n, "two").x, lbfgs(D, n, "det").x
fit = {}
for name, z, bw, bt in [("stochastic", zs, sig(zs[n + 1]), sig(zs[n + 2])), ("deterministic", zd, 1.0, 1.0)]:
    c = -z[:n].mean()
    fit[name] = dict(theta=z[:n] + c, c=c, tau=float(np.log(1 + np.exp(z[n]))), bw=float(bw), bt=float(bt))
order = (-fit["stochastic"]["theta"]).argsort()
strong = (order[0], order[2]); weak = (order[-1], order[-2])
BLUE, ORANGE, VIOLET, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#6250d6", "#b5b4ad", "#0b0b0b", "#52514e"
COL = {0: BLUE, 1: ORANGE, 2: VIOLET, 3: GRAY}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(7)
lim = (-3.2, 6.2)
fig, axes = plt.subplots(2, 2, figsize=(4.4, 4.5), sharex=True, sharey=True)
check = {}
for r, name in enumerate(["stochastic", "deterministic"]):
    f = fit[name]
    for col, (pi, pj) in enumerate([strong, weak]):
        ax = axes[r, col]
        N = 1500
        yi, yj = f["theta"][pi] + rng.gumbel(size=N), f["theta"][pj] + rng.gumbel(size=N)
        win_i, win_j = yi - yj > f["tau"], yj - yi > f["tau"]; tied = ~(win_i | win_j)
        quad = (yi < f["c"]) & (yj < f["c"])
        u = rng.random(N)
        bad = quad & ((tied & (u < f["bt"])) | (~tied & (u < f["bw"])))
        s = np.where(bad, 2, np.where(win_i, 0, np.where(win_j, 1, 3)))
        for k in (3, 0, 1, 2):
            ax.scatter(yi[s == k], yj[s == k], s=3.5, color=COL[k], alpha=0.65, lw=0, zorder=2 if k != 2 else 3)
        t, c = f["tau"], f["c"]
        ax.plot(lim, np.array(lim) - t, color=INK, lw=1); ax.plot(lim, np.array(lim) + t, color=INK, lw=1)
        ax.plot([c, c], [lim[0], c], color=INK, lw=1, ls="--"); ax.plot([lim[0], c], [c, c], color=INK, lw=1, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.text(c - 0.1, lim[0] + 0.15, "$c$", ha="right", color=INK, fontsize=9)
        mu = np.exp(f["theta"] - c); q = np.exp(t)
        p = np.array(probs(np.array([mu[pi]]), np.array([mu[pj]]), q, "two", f["bw"], f["bt"])).ravel()
        check[f"{name}_{col}"] = dict(formula=p.round(3).tolist(), simulated=[round(float((s == k).mean()), 3) for k in range(4)])
        ax.text(0.03, 0.97, f"$i$ wins {p[0]:.2f},  $j$ wins {p[1]:.2f}\ntie {p[3]:.2f},  both-bad {p[2]:.2f}", transform=ax.transAxes, va="top", fontsize=7, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        ax.grid(False); ax.tick_params(colors=MUTED, labelsize=7)
        if r == 0:
            ax.set_title(("Strong pair" if col == 0 else "Weak pair") + f"\n{models[pi]} ($i$)\nagainst {models[pj]} ($j$)", fontsize=7, color=INK)
        if r == 1: ax.set_xlabel("$Y_i$", color=INK)
    axes[r, 0].set_ylabel(f"{name.capitalize()} model\n$\\hat c={f['c']:.2f}$, $\\hat\\tau={f['tau']:.2f}$" + (f"\n$\\hat\\beta_w={f['bw']:.2f}$, $\\hat\\beta_t={f['bt']:.2f}$" if name == "stochastic" else "\n$\\beta_w=\\beta_t=1$") + "\n\n$Y_j$", color=INK, fontsize=8)
h = [Line2D([], [], marker="o", ls="", color=COL[k], ms=5, label=l) for k, l in [(0, "$i$ wins"), (1, "$j$ wins"), (3, "tie"), (2, "both-bad")]]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.subplots_adjust(left=0.26, right=0.97, top=0.86, bottom=0.145, wspace=0.05, hspace=0.05)
out = FIGURES
fig.savefig(FIGURES / "latent_plane.pdf"); fig.savefig(out / "latent_plane.png", dpi=170)
print(json.dumps(check, indent=0))

fig, ax = plt.subplots(figsize=(6.4, 2.5))
for y, name, colr in [(1, "stochastic", BLUE), (0, "deterministic", ORANGE)]:
    f = fit[name]; x = f["theta"] - f["c"]
    jit = np.random.default_rng(3).uniform(-0.13, 0.13, n)
    ax.scatter(x, y + jit, s=16, color=colr, alpha=0.75, lw=0.4, ec="white", zorder=3)
    ax.text(-2.62, y + 0.3, f"{name.capitalize()} model: {100 * (x > 0).mean():.0f}% of competitors above the threshold", fontsize=7.5, color=INK)
ax.axvline(0, color=INK, lw=1, ls="--"); ax.text(0.03, -0.42, "quality threshold $c$", fontsize=7.5, color=INK, va="center")
ax.set_yticks([]); ax.set_ylim(-0.55, 1.6); ax.set_xlim(-2.65, 1.4); ax.spines["left"].set_visible(False)
ax.set_xlabel(r"$\hat\theta_i-\hat c$, ability above the fitted quality threshold", color=INK); ax.tick_params(colors=MUTED, labelsize=7)
fig.tight_layout(); fig.savefig(FIGURES / "field_vs_threshold.pdf"); fig.savefig(out / "field_vs_threshold.png", dpi=170)
for name in fit: print(name, round(fit[name]["c"], 3), round(float((fit[name]["theta"] - fit[name]["c"] > 0).mean()), 3), round(float((fit[name]["theta"] - fit[name]["c"]).min()), 2), round(float((fit[name]["theta"] - fit[name]["c"]).max()), 2))
