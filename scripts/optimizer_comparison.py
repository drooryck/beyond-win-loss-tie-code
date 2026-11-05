"""
Optimizer comparison and multi-start search, on the data without self-comparisons.

  1. Checks the exact score of exact_score.py against central differences, for all four modes.
  2. Fits the deterministic and stochastic models by L-BFGS-B and by Newton-Raphson from z = 0.
  3. Computes the Hessian at the maximum by central differences of the score.
  4. Stochastic model, all 52: refits from 40 dispersed starts, restarts any run that stops low, and
     solves the 3,603 equations by least squares from 20 dispersed starts.

Run with `make optimizers`.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares

from lib.data import load, counts_for
from lib.data import load_clean
from lib.exact_score import loglik, score, hessian, lbfgs, newton, size, probs, sig

from lib.paths import FIGURES, RESULTS
OUT = RESULTS; OUT.mkdir(exist_ok=True)


def params(z, n, mode):
    c = -z[:n].mean()
    bw, bt = (1.0, 1.0) if mode == "det" else (float(sig(z[n + 1])), float(sig(z[n + 2])))
    return dict(c=float(c), q=float(1 + np.exp(z[n])), bw=bw, bt=bt), z[:n] + c


df = load_clean(load)
freq = list(pd.concat([df.model_a, df.model_b]).value_counts().index)
R = dict(rows=len(df))
for label, models in [("top20", freq[:20]), ("all53", freq)]:
    D, _ = counts_for(df, models)
    n = len(models)
    print(label, "pairs", len(D), "self rows", int((D[:, 0] == D[:, 1]).sum()), flush=True)
    for mode in ["det", "one", "two", "rk"]:
        k = size(n, mode)
        zt = np.random.default_rng(1).normal(-0.3, 0.5, k)
        fd = np.array([(loglik(zt + 1e-6 * e, D, n, mode) - loglik(zt - 1e-6 * e, D, n, mode)) / 2e-6 for e in np.eye(k)])
        ex = score(zt, D, n, mode)
        print(" score check", mode, np.abs(fd - ex).max() / max(1.0, np.abs(ex).max()), flush=True)
    for mode in ["det", "two"]:
        z0 = np.zeros(size(n, mode))
        t = time.time(); L = lbfgs(D, n, mode); tL = time.time() - t
        t = time.time(); zN, llN, itN, okN = newton(z0, D, n, mode); tN = time.time() - t
        pL, thL = params(L.x, n, mode); pN, thN = params(zN, n, mode)
        ev = np.linalg.eigvalsh(-hessian(zN if llN >= -L.fun else L.x, D, n, mode))
        R[f"{label}_{mode}"] = dict(ll_lbfgs=-float(L.fun), ll_newton=float(llN), it_lbfgs=int(L.nit), it_newton=int(itN),
                                    converged=bool(okN), sec_lbfgs=tL, sec_newton=tN, max_theta_diff=float(np.abs(thL - thN).max()),
                                    min_eig=float(ev.min()), max_eig=float(ev.max()), **pL)
        print(label, mode, R[f"{label}_{mode}"], flush=True)

D, _ = counts_for(df, freq); n = len(freq)
ref = lbfgs(D, n, "two"); _, th0 = params(ref.x, n, "two")
rng = np.random.default_rng(20261103)
disp = lambda: np.r_[rng.normal(rng.uniform(-3, 3), rng.uniform(0.2, 1.5), n), rng.normal(-0.5, 2), rng.normal(0, 3), rng.normal(0, 3)]
runs = []
for r in range(40):
    f = lbfgs(D, n, "two", z0=disp())
    p, th = params(f.x, n, "two")
    row = dict(dll=-float(f.fun) + float(ref.fun), max_theta_diff=float(np.abs(th - th0).max()), psi_w=float(f.x[n + 1]), **p)
    if row["dll"] < -1e-3:
        g = score(f.x, D, n, "two")
        bw = sig(f.x[n + 1]); y = f.x.copy(); y[n + 1] = np.log(0.01 / 0.99)
        row["restart_dll"] = -float(lbfgs(D, n, "two", z0=y).fun) + float(ref.fun)
    runs.append(row); print(r, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

# slope in beta_w at beta_w = 0, holding beta_w at the boundary
idx = [k for k in range(n + 3) if k != n + 1]
def at0(y):
    z = np.zeros(n + 3); z[idx] = y; z[n + 1] = -60.0; return z
from scipy.optimize import minimize
b0 = minimize(lambda y: -loglik(at0(y), D, n, "two"), ref.x[idx], jac=lambda y: -score(at0(y), D, n, "two")[idx], method="L-BFGS-B",
              options=dict(maxiter=20000, maxfun=100000, ftol=1e-15, gtol=1e-7))
z0_ = at0(b0.x); bw0 = sig(z0_[n + 1])
R["boundary"] = dict(dll=-float(b0.fun) + float(ref.fun), slope_bw=float(score(z0_, D, n, "two")[n + 1] / (bw0 * (1 - bw0))))
print(R["boundary"], flush=True)

i, j = D[:, 0].astype(int), D[:, 1].astype(int)
def F(z):
    mu, q = np.exp(z[:n]), 1 + np.exp(z[n])
    return np.column_stack(probs(mu[i], mu[j], q, "two", sig(z[n + 1]), sig(z[n + 2])))[:, :3].ravel()
target = F(ref.x); B = []
for r in range(20):
    f = least_squares(lambda z: F(z) - target, disp(), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=300)
    B.append(dict(cost=float(f.cost), max_coord_diff=float(np.abs(f.x - ref.x).max())))
    print("LS", r, B[-1], flush=True)
R["multistart"] = dict(ref_ll=-float(ref.fun), pairs=len(D), runs=runs, least_squares=B)
(OUT / "optimizer_comparison.json").write_text(json.dumps(R, indent=1))
print("done")
