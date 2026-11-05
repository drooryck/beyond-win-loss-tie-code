"""
Ridge simulation of the Regularization subsection, on the design of the other simulations. The truth is the
stochastic fit on the twenty most compared competitors. Nineteen of them meet at n = 200 per pair. The
twentieth meets n_sparse different opponents once each, drawn afresh in each replication. Each dataset is
fitted from z = 0 at every lambda. The sparse competitor is run at two true abilities, near the average and
one standard deviation above it, since a penalty toward zero flatters a competitor whose true ability is zero.

Also fits the ridge penalty to the 108,305 first-order comparisons on all 52 competitors.

Run with `make ridge`.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from lib.data import load, counts_for
from lib.simulate import probs_completion
from lib.data import load_clean
from lib.exact_score import lbfgs, sig
from lib.paths import RESULTS, FIGURES

REPS, NS, LAMBDAS, N_PAIR, ESCAPE = 100, [2, 5, 10], [0.0, 1.0, 5.0, 20.0], 200, 5.0

df = load_clean(load)
freq = list(pd.concat([df.model_a, df.model_b]).value_counts().index)
R = dict(reps=REPS, n_pair=N_PAIR, escape=ESCAPE)

# ---- the ridge penalty on the real data, all 52 competitors
D52, _ = counts_for(df, freq); n52 = len(freq)
base = None; R["real"] = []
for l in LAMBDAS:
    z = lbfgs(D52, n52, "two", ridge=l).x
    th, c = z[:n52] - z[:n52].mean(), -z[:n52].mean()
    base = th if base is None else base
    R["real"].append(dict(lam=l, spearman=float(spearmanr(th, base).statistic), max_change=float(np.abs(th - base).max()),
                          sd=float(th.std()), c=float(c), bw=float(sig(z[n52 + 1])), bt=float(sig(z[n52 + 2])), top=freq[int(th.argmax())]))
    print(R["real"][-1], flush=True)

# ---- the truth
top = freq[:20]; n = 20
D20, _ = counts_for(df[df.model_a.isin(top) & df.model_b.isin(top)], top)
f = lbfgs(D20, n, "two").x
mu, q, bw, bt = np.exp(f[:n]), 1 + np.exp(f[n]), sig(f[n + 1]), sig(f[n + 2])
th0 = f[:n] - f[:n].mean()
R["truth"] = dict(c=float(-f[:n].mean()), tau=float(np.log(q)), bw=float(bw), bt=float(bt), sd=float(th0.std()))
print(R["truth"], flush=True)
rng = np.random.default_rng(20261105)


def P(i, j):
    p = np.array(probs_completion(mu[i], mu[j], q, bw, bt), dtype=float).ravel()
    return p / p.sum()


R["sim"] = {}
for label, target in [("near_average", 0.0), ("one_sd_above", float(th0.std()))]:
    s = int(np.abs(th0 - target).argmin())
    others = [k for k in range(n) if k != s]
    rows = []
    for ns in NS:
        err = {l: [] for l in LAMBDAS}; rest = {l: [] for l in LAMBDAS}; esc = {l: 0 for l in LAMBDAS}; onesided = 0
        for r in range(REPS):
            d = [[a, b, *rng.multinomial(N_PAIR, P(a, b))] for x, a in enumerate(others) for b in others[x + 1:]]
            W = np.zeros(4)
            for o in rng.choice(others, ns, replace=False):
                x = rng.multinomial(1, P(s, o)); d.append([s, o, *x]); W += x
            onesided += int(W.sum() == W[0] or W.sum() == W[1])
            d = np.array(d, dtype=float)
            for l in LAMBDAS:
                z = lbfgs(d, n, "two", ridge=l).x
                th = z[:n] - z[:n].mean()
                e = th[s] - th0[s]; err[l].append(e)
                rest[l].append(np.sqrt(np.mean((th[others] - th0[others]) ** 2)))
                esc[l] += int(abs(e) > ESCAPE)
        for l in LAMBDAS:
            rows.append(dict(ns=ns, lam=l, rmse_sparse=float(np.sqrt(np.mean(np.square(err[l])))), median_abs_sparse=float(np.median(np.abs(err[l]))),
                             bias_sparse=float(np.mean(err[l])), rmse_rest=float(np.mean(rest[l])), escaped=esc[l], onesided=onesided))
            print(label, rows[-1], flush=True)
    R["sim"][label] = dict(sparse=top[s], theta=float(th0[s]), rows=rows)
(RESULTS / "regularization.json").write_text(json.dumps(R, indent=1))
