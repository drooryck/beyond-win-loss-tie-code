"""
Parameter recovery and misreporting simulations of the Simulation studies section, at 100 replications.
The truth is the stochastic fit on the twenty most compared competitors,
and all twenty meet in 190 pairs.

Run with `make sims`.
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
from lib.simulate import to_theta_c
from lib.simulate import sig, simulate_completion
from lib.data import load_clean
from lib.exact_score import fit_rule, fit_completion
from lib.paths import RESULTS, FIGURES

REPS = 100
rng = np.random.default_rng(20261106)
df = load_clean(load)
top = list(pd.concat([df.model_a, df.model_b]).value_counts().index)[:20]; n = 20
D, _ = counts_for(df[df.model_a.isin(top) & df.model_b.isin(top)], top)
z, _, _ = fit_completion(D, n, "two")
th_t, c_t, tau_t = to_theta_c(z, n)
bw_t, bt_t = float(sig(z[n + 1])), float(sig(z[n + 2]))
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
R = dict(reps=REPS, truth=dict(c=float(c_t), tau=float(tau_t), bw=bw_t, bt=bt_t, sd=float(th_t.std())), recovery=[], misreporting=[])
print(R["truth"], flush=True)

for npp in [25, 50, 100, 200, 400, 800]:
    a = []
    for rep in range(REPS):
        d = simulate_completion(th_t, c_t, tau_t, bw_t, bt_t, pairs, npp, rng)
        zf, _, _ = fit_completion(d, n, "two")
        thf, cf, _ = to_theta_c(zf, n)
        a.append([np.sqrt(np.mean((thf - th_t) ** 2)), cf - c_t, sig(zf[n + 1]) - bw_t, sig(zf[n + 2]) - bt_t])
    a = np.array(a)
    R["recovery"].append(dict(npp=npp, rmse_theta=float(a[:, 0].mean()), bias_c=float(a[:, 1].mean()), sd_c=float(a[:, 1].std()),
                              bias_bw=float(a[:, 2].mean()), sd_bw=float(a[:, 2].std()), bias_bt=float(a[:, 3].mean()), sd_bt=float(a[:, 3].std())))
    print(R["recovery"][-1], flush=True)

for pi in [0.0, 0.1, 0.2, 0.3, 0.5]:
    a = []
    for rep in range(REPS):
        d = simulate_completion(th_t, c_t, tau_t, 1 - pi, 1 - pi, pairs, 400, rng)
        zd, _, _ = fit_rule(d, n, "lmarena")
        thd, cd, _ = to_theta_c(zd, n)
        zs, _, _ = fit_completion(d, n, "two")
        _, cs, _ = to_theta_c(zs, n)
        a.append([cd - c_t, spearmanr(thd, th_t).statistic, cs - c_t, 0.5 * (sig(zs[n + 1]) + sig(zs[n + 2]))])
    a = np.array(a)
    R["misreporting"].append(dict(pi=pi, bias_c_det=float(a[:, 0].mean()), rank_det=float(a[:, 1].mean()), bias_c_stoch=float(a[:, 2].mean()),
                                  sd_c_stoch=float(a[:, 2].std()), beta_hat=float(a[:, 3].mean())))
    print(R["misreporting"][-1], flush=True)
(RESULTS / "simulations.json").write_text(json.dumps(R, indent=1))
