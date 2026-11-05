"""Main Chapter 4 fits: one 80/20 split for the likelihood-ratio tests and Hessian diagnostics, and the both-bad calibration.
The held-out table averages ten splits and comes from splits.py, so the single-split scores saved here are not the table entries.

Models: Rao-Kupper with split ties (rk), deterministic (det), stochastic (stoch).
Writes results/main_fits.json and figures/bothbad_vs_strength.pdf.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2, spearmanr

from lib.data import load, counts_for, split, load_clean
from lib.simulate import probs_rule, unpack, to_theta_c, probs_completion, betas_from, sig
from lib.exact_score import fit_rule, fit_completion, hessian as hess_exact, score as score_exact, grk_fit, grk_probs
from lib.paths import FIGURES, RESULTS

OUT = RESULTS
FIG = FIGURES
line = lambda t: print("\n" + t + "\n" + "=" * len(t), flush=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
R = {}   # everything the chapter cites

def predict(zdict, name, data, n):
    z = zdict["z"]
    mu, q = unpack(z, n)
    i, j = data[:, 0].astype(int), data[:, 1].astype(int)
    if name == "rk":
        return np.column_stack(probs_rule(mu[i], mu[j], q, "rk", sig(z[n + 1])))
    if name == "det":
        return np.column_stack(probs_rule(mu[i], mu[j], q, "lmarena"))
    bw, bt = betas_from(z, n, "two")
    return np.column_stack(probs_completion(mu[i], mu[j], q, bw, bt))


def per_obs(P, data):
    out = []
    for r, p in zip(data, P):
        for s in range(4):
            k = int(round(r[2 + s]))
            if k: out.append(np.full(k, np.log(max(p[s], 1e-300))))
    return np.concatenate(out)


def fit_one(name, data, n):
    if name == "rk":
        z, ll, _ = fit_rule(data, n, "rk"); k = n + 2
    elif name == "det":
        z, ll, _ = fit_rule(data, n, "lmarena"); k = n + 1
    elif name == "one":
        z, ll, _ = fit_completion(data, n, "one"); k = n + 2
    else:
        z, ll, _ = fit_completion(data, n, "two"); k = n + 3
    return dict(z=z, ll=float(ll), k=k)


# ===================================================== B1: subsets, held out
line("B1  held-out NLL on nested subsets")
df = load_clean(load)
print("rows", len(df), flush=True)
freq = pd.concat([df.model_a, df.model_b]).value_counts()
tr_df, te_df = split(df)
R["heldout"] = {}
pobs53 = {}
for label, top in [("top10", 10), ("top20", 20), ("top40", 40), ("all53", len(freq))]:
    models = list(freq.index[:top]); n = len(models)
    train, _ = counts_for(tr_df, models)
    test, _ = counts_for(te_df, models)
    row = dict(rows=int(train[:, 2:].sum() + test[:, 2:].sum()),
               test_rows=int(test[:, 2:].sum()))
    # baselines
    gp = train[:, 2:].sum(axis=0); gp = gp / gp.sum()
    Pg = np.tile(gp, (len(test), 1))
    row["global"] = float(-per_obs(Pg, test).mean())
    tr_map = {(int(r[0]), int(r[1])): r[2:] for r in train}
    Pp = np.array([(tr_map.get((int(r[0]), int(r[1])), np.zeros(4)) + 1.0) /
                   (tr_map.get((int(r[0]), int(r[1])), np.zeros(4)).sum() + 4.0)
                   for r in test])
    row["pair"] = float(-per_obs(Pp, test).mean())
    for name in ["rk", "det", "stoch"]:
        t0 = time.time()
        f = fit_one(name, train, n)
        P = predict(f, name, test, n)
        po = per_obs(P, test)
        row[name] = float(-po.mean())
        th, c, tau = to_theta_c(f["z"], n)
        f.update(theta=th, c=float(c), tau=float(tau))
        if name == "rk": f["rho"] = float(sig(f["z"][n + 1]))
        if name == "stoch":
            bw, bt = betas_from(f["z"], n, "two")
            f.update(bw=float(bw), bt=float(bt))
        if label == "all53":
            pobs53[name] = po
            R.setdefault("fits53", {})[name] = {k2: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                for k2, v in f.items() if k2 != "z"}
            R["fits53"][name]["models"] = models if name == "rk" else None
            np.save(OUT / f"z53_{name}.npy", f["z"])
        print(f"  {label} {name:6s} NLL {row[name]:.4f}  [{time.time()-t0:.1f}s]", flush=True)
    R["heldout"][label] = row

line("B2  paired differences and rank correlations (all 53)")
R["paired"] = {}
for a, b in [("det", "rk"), ("det", "stoch"), ("rk", "stoch")]:
    d = pobs53[b] - pobs53[a]          # positive: b better
    se = d.std(ddof=1) / np.sqrt(len(d))
    R["paired"][f"{b}_minus_{a}"] = dict(mean=float(d.mean()), se=float(se),
                                         z=float(d.mean() / se))
    print(f"  NLL({a}) - NLL({b}) = {d.mean():+.5f}  se {se:.5f}  z {d.mean()/se:+.2f}")
th = {m: np.array(R["fits53"][m]["theta"]) for m in ["rk", "det", "stoch"]}
R["rankcorr"] = {f"{a}_{b}": float(spearmanr(th[a], th[b]).statistic)
                 for a, b in [("rk", "det"), ("rk", "stoch"), ("det", "stoch")]}
print(" ", R["rankcorr"])

line("B3  likelihood-ratio tests (all 53, train)")
models53 = list(freq.index); n53 = len(models53)
train53, _ = counts_for(tr_df, models53)
f_one = fit_one("one", train53, n53)
ll = {m: R["fits53"][m]["ll"] for m in ["det", "stoch"]}
ll["one"] = f_one["ll"]
R["lrt"] = dict(
    two_vs_one=dict(stat=float(2 * (ll["stoch"] - ll["one"])), dof=1),
    one_vs_det=dict(stat=float(2 * (ll["one"] - ll["det"])), dof=1, boundary=True),
    two_vs_det=dict(stat=float(2 * (ll["stoch"] - ll["det"])), dof=2, boundary=True),
    one_beta=float(sig(f_one["z"][n53 + 1])),
)
R["lrt"]["two_vs_one"]["p"] = float(1 - chi2.cdf(R["lrt"]["two_vs_one"]["stat"], 1))
print(" ", json.dumps(R["lrt"], indent=1), flush=True)

line("B4  nondegeneracy at the all-53 stochastic maximum")
R["hessian53"] = {}
for name, mode in [("stoch", "two"), ("det", "det")]:
    z53 = np.load(OUT / f"z53_{name}.npy")
    ev = np.linalg.eigvalsh(-hess_exact(z53, train53, n53, mode))
    R["hessian53"][name] = dict(min_eig=float(ev.min()), cond=float(ev.max() / ev.min()),
                                grad_sup=float(np.abs(score_exact(z53, train53, n53, mode)).max()))
print(" ", R["hessian53"], flush=True)

# ===================================================== D: both-bad vs strength
line("D  both-bad rate against pair quality (full data fits)")
full53, _ = counts_for(df, models53)
fits_full = {name: fit_one(name, full53, n53) for name in ["rk", "det", "stoch"]}
for name, f in fits_full.items():
    th2, c2, tau2 = to_theta_c(f["z"], n53)
    f.update(theta=th2, c=float(c2), tau=float(tau2), q=float(np.exp(tau2)))
    if name == "rk": f["rho"] = float(sig(f["z"][n53 + 1]))
    if name == "stoch":
        bw, bt = betas_from(f["z"], n53, "two"); f.update(bw=float(bw), bt=float(bt))
R["fits_full"] = {k: {k2: v for k2, v in f.items() if k2 in
                      ("ll", "c", "tau", "q", "rho", "bw", "bt")} for k, f in fits_full.items()}
print(" ", json.dumps(R["fits_full"], indent=1), flush=True)

i, j = full53[:, 0].astype(int), full53[:, 1].astype(int)
tot = full53[:, 2:].sum(axis=1); obs = full53[:, 4] / tot
keep = tot >= 60
idx = {m: k for k, m in enumerate(models53)}
a_ = df.model_a.map(idx).to_numpy(); b_ = df.model_b.map(idx).to_numpy(); s_ = df.s.to_numpy()
win = np.zeros(n53); dec = np.zeros(n53)
np.add.at(win, a_[s_ == 0], 1); np.add.at(win, b_[s_ == 1], 1)
np.add.at(dec, a_[s_ < 2], 1); np.add.at(dec, b_[s_ < 2], 1)
wr = win / np.maximum(dec, 1)
x = wr[i] + wr[j]
pred = {name: predict(f, name, full53, n53)[:, 2] for name, f in fits_full.items()}
phi_g = np.exp(grk_fit(full53, n53).x[:n53])
pred["grk"] = grk_probs(phi_g[i], phi_g[j], 1.0)[2]   # p(3) does not involve the tie parameter
edges = np.quantile(x[keep], np.linspace(0, 1, 9))
ctr, ob, lo, hi = [], [], [], []
pb = {c: [] for c in pred}
for a2, b2 in zip(edges[:-1], edges[1:]):
    m = keep & (x >= a2) & (x < b2) if b2 < edges[-1] else keep & (x >= a2) & (x <= b2)
    if not m.sum(): continue
    w = tot[m]; p = np.average(obs[m], weights=w); se = np.sqrt(p * (1 - p) / w.sum())
    ctr.append(0.5 * (a2 + b2)); ob.append(p); lo.append(p - 1.96 * se); hi.append(p + 1.96 * se)
    for c in pred: pb[c].append(float(np.average(pred[c][m], weights=w)))
R["bothbad_bins"] = dict(obs=[float(v) for v in ob],
                         **{c: pb[c] for c in pb})
# between-pair sd benchmark
o_sd = float(obs[keep].std())
binom = float(np.sqrt(np.mean(obs[keep] * (1 - obs[keep]) / tot[keep])))
genuine = float(np.sqrt(max(o_sd ** 2 - binom ** 2, 0)))
R["bothbad_sd"] = dict(observed=o_sd, binomial=binom, genuine=genuine,
                       **{c: float(pred[c][keep].std()) for c in pred})
print("  sd benchmark:", json.dumps(R["bothbad_sd"], indent=1), flush=True)
print("  corr(obs rate, quality):", float(np.corrcoef(x[keep], obs[keep])[0, 1]))
R["bothbad_corr"] = float(np.corrcoef(x[keep], obs[keep])[0, 1])

fig, ax = plt.subplots(figsize=(5.4, 3.3))
ax.fill_between(ctr, lo, hi, color="0.75", alpha=.55, lw=0, zorder=2,
                label="observed, 95% interval")
ax.plot(ctr, ob, "o-", ms=4, color="black", lw=1.2, zorder=4, label="observed")
for c, col, ls, lb in zip(["rk", "grk", "det", "stoch"], ["#b45f45", "#d9a400", "#3b6ea5", "#5c8a4a"],
                          ["--", ":", "-", "-."],
                          ["Rao-Kupper, split ties", "grounded Rao-Kupper", "deterministic", "stochastic"]):
    ax.plot(ctr, pb[c], ls, color=col, lw=1.5, label=lb, zorder=3)
ax.set_xlabel("sum of the two competitors' empirical win rates")
ax.set_ylabel("both-bad rate"); ax.grid(color="0.93", lw=.6)
h, l = ax.get_legend_handles_labels()
ax.legend([h[1], h[0], h[2], h[3], h[4], h[5]], [l[1], l[0], l[2], l[3], l[4], l[5]],
          frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "bothbad_vs_strength.pdf")
print("  wrote bothbad_vs_strength.pdf", flush=True)

# Simulations live in scripts/simulations.py and scripts/regularization.py.

(OUT / "main_fits.json").write_text(json.dumps(R, indent=1, default=float))
print("\nwrote", OUT / "main_fits.json", flush=True)
