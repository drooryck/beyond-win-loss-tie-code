"""
Calibration of the two likelihood-ratio tests of Section 4.5, on the top-20 design with the observed
first-order pair counts. Each test is simulated under its own null hypothesis, from the fit of that null.

Run with `make lrt`.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2

from lib.data import load, counts_for
from lib.data import load_clean
from lib.exact_score import lbfgs, probs, sig
from lib.paths import RESULTS, FIGURES

REPS = 500
rng = np.random.default_rng(20261104)
df = load_clean(load)
models = list(pd.concat([df.model_a, df.model_b]).value_counts().index)[:20]; n = 20
D, _ = counts_for(df, models)
i, j = D[:, 0].astype(int), D[:, 1].astype(int); tot = D[:, 2:].sum(axis=1).astype(int)


def draw(P):
    P = P / P.sum(axis=1, keepdims=True)
    return np.column_stack([i, j, np.array([rng.multinomial(m, p) for m, p in zip(tot, P)])]).astype(float)


R = dict(reps=REPS, design=dict(n=n, pairs=len(D), rows=int(tot.sum())))
# beta_w = beta_t against two free probabilities: an interior null
f1 = lbfgs(D, n, "one").x; mu, q, b = np.exp(f1[:n]), 1 + np.exp(f1[n]), sig(f1[n + 1])
P = np.array(probs(mu[i], mu[j], q, "one", b)).T
stats = []
for r in range(REPS):
    d = draw(P)
    l0 = lbfgs(d, n, "one", z0=f1.copy())
    l1 = lbfgs(d, n, "two", z0=np.r_[l0.x, l0.x[n + 1]])
    stats.append(max(2 * (l0.fun - l1.fun), 0.0))
s = np.array(stats)
R["interior"] = dict(truth_beta=float(b), mean=float(s.mean()), q95=float(np.quantile(s, 0.95)), reject=float((s > chi2.ppf(0.95, 1)).mean()))
print(R["interior"], flush=True)
# the deterministic model against one free probability: a boundary null
f0 = lbfgs(D, n, "det").x; mu, q = np.exp(f0[:n]), 1 + np.exp(f0[n])
P = np.array(probs(mu[i], mu[j], q, "det")).T
stats = []
for r in range(REPS):
    d = draw(P)
    l0 = lbfgs(d, n, "det", z0=f0.copy())
    l1 = lbfgs(d, n, "one", z0=np.r_[l0.x, 4.0])
    stats.append(max(2 * (l0.fun - l1.fun), 0.0))
s = np.array(stats)
R["boundary"] = dict(mean=float(s.mean()), frac_zero=float((s < 1e-3).mean()), reject_mix=float((s > chi2.ppf(0.90, 1)).mean()), reject_chi2=float((s > chi2.ppf(0.95, 1)).mean()))
print(R["boundary"], flush=True)
(RESULTS / "lrt_calibration.json").write_text(json.dumps(R, indent=1))
