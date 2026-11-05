"""
The stochastic reporting model with a random threshold: c is a standard Gumbel draw for each comparison.
With beta_w = beta_t = 1 this is the fixed-coefficient grounded Rao-Kupper model of Frick et al. (2025).

    p(1) = A (1 - beta_w / (1 + Z)),   p(2) = B (1 - beta_w / (1 + Z')),
    p(3) = beta_t / (1 + phi_i + phi_j) + (beta_w - beta_t) (A / (1 + Z) + B / (1 + Z')),   p(4) = remainder.

Run with `make threshold`.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from lib.data import load, counts_for
from lib.data import load_clean
from lib.exact_score import lbfgs, grk_fit, grk_ll, loglik, sig, TINY
from lib.paths import RESULTS, FIGURES


def probs_rt(z, D, n):
    i, j = D[:, 0].astype(int), D[:, 1].astype(int)
    phi, q, bw, bt = np.exp(z[:n]), 1 + np.exp(z[n]), sig(z[n + 1]), sig(z[n + 2])
    a, b = phi[i], phi[j]
    Z, Y = a + q * b, q * a + b
    A, B = a / Z, b / Y
    C = A / (1 + Z) + B / (1 + Y)
    p1, p2 = A * (1 - bw / (1 + Z)), B * (1 - bw / (1 + Y))
    p3 = bt / (1 + a + b) + (bw - bt) * C
    return p1, p2, p3, 1 - p1 - p2 - p3


def ll_rt(z, D, n):
    p = probs_rt(z, D, n)
    return sum((D[:, 2 + k] * np.log(np.maximum(p[k], TINY))).sum() for k in range(4))


def fit_rt(D, n, z0):
    return minimize(lambda z: -ll_rt(z, D, n), z0, method="L-BFGS-B", options=dict(maxiter=20000, maxfun=2000000, ftol=1e-14))


df = load_clean(load)
models = list(pd.concat([df.model_a, df.model_b]).value_counts().index); n = len(models)
D, _ = counts_for(df, models)
g = grk_fit(D, n).x; st = lbfgs(D, n, "two").x
R = dict(ll_grk=float(grk_ll(g, D, n)), ll_stoch=float(loglik(st, D, n, "two")))
starts = [np.r_[g, 3.0, 3.0], np.r_[g, -2.0, 0.0], np.r_[g, 0.0, 0.0], np.r_[g, 0.0, 3.0], np.r_[g, 3.0, 0.0]]
fits = [fit_rt(D, n, s) for s in starts]
print("starts", [(round(-f.fun, 3), round(float(sig(f.x[n + 1])), 3), round(float(sig(f.x[n + 2])), 3)) for f in fits], flush=True)
best = min(fits, key=lambda f: f.fun); z = best.x
R.update(ll=-float(best.fun), bw=float(sig(z[n + 1])), bt=float(sig(z[n + 2])), q=float(1 + np.exp(z[n])), params=n + 3,
         lrt_vs_grk=float(2 * (-best.fun - grk_ll(g, D, n))), rank_stoch=float(spearmanr(z[:n], st[:n]).statistic), rank_grk=float(spearmanr(z[:n], g[:n]).statistic),
         share_phi_above_1=float((z[:n] > 0).mean()), start_results=[-float(f.fun) for f in fits])
rows = []
for sp in range(10):
    r = np.random.default_rng(1000 + sp); tr = np.zeros(len(df), bool)
    for s in range(4):
        m = np.where((df.s == s).to_numpy())[0]; tr[r.permutation(m)[: int(round(0.8 * len(m)))]] = True
    train, _ = counts_for(df[tr], models); test, _ = counts_for(df[~tr], models); M = test[:, 2:].sum()
    gt = grk_fit(train, n).x
    rows.append([-ll_rt(fit_rt(train, n, np.r_[gt, 0.0, 0.0]).x, test, n) / M, -loglik(lbfgs(train, n, "two").x, test, n, "two") / M, -grk_ll(gt, test, n) / M])
    print(sp, np.round(rows[-1], 5), flush=True)
a = np.array(rows)
R.update(nll=a[:, 0].mean(), nll_stoch=a[:, 1].mean(), nll_grk=a[:, 2].mean(), beats_stoch=int((a[:, 0] < a[:, 1]).sum()), beats_grk=int((a[:, 0] < a[:, 2]).sum()))
pn = probs_rt(z, D, n)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in R.items() if k != "start_results"})
(RESULTS / "random_threshold.json").write_text(json.dumps(R, indent=1))
