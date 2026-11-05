"""
Bootstrap intervals for the main fits, fits within the language groups of the data section, and the abilities
within code and non-code prompts.

main: the deterministic and stochastic models on all 52 competitors, 200 row-bootstrap resamples.
language: the stochastic model within English, Polish, undetermined, Russian, Chinese and other languages, on one
    shared set of competitors with at least MIN_ROWS comparisons in every group, so that c can be compared.
code: the stochastic model within code and non-code prompts, with the bootstrap draws of every ability.

Saves strata.json and code_theta_draws.npz. Run with `make strata`.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

from lib.data import load, counts_for
from lib.data import load_clean
from lib.exact_score import lbfgs, sig
from lib.paths import RAW, RESULTS, FIGURES

MIN_ROWS, BOOT = 50, 200
df = load_clean(load)
meta = pd.read_parquet(RAW, columns=["language", "is_code"]).loc[df.index]
rng = np.random.default_rng(20261107)
ORDER = list(pd.concat([df.model_a, df.model_b]).value_counts().index)


def summary(z, n, mode):
    nu = z[:n]; th = nu - nu.mean()
    s = dict(c=float(-nu.mean()), tau=float(np.log(1 + np.exp(z[n]))), q=float(1 + np.exp(z[n])), sd_theta=float(th.std()))
    if mode == "two":
        s.update(bw=float(sig(z[n + 1])), bt=float(sig(z[n + 2])))
    return s


def boot(d, models, mode, z0):
    n = len(models); out = []; th = []
    for b in range(BOOT):
        db, _ = counts_for(d.iloc[rng.integers(0, len(d), len(d))], models)
        z = lbfgs(db, n, mode, z0=z0.copy()).x
        out.append(summary(z, n, mode)); th.append(z[:n] - z[:n].mean())
    return pd.DataFrame(out), np.array(th)


def ci(B):
    return {k: [float(np.percentile(B[k], 2.5)), float(np.percentile(B[k], 97.5))] for k in B.columns}


R = dict(boot=BOOT, min_rows=MIN_ROWS, main={}, language=dict(strata={}), code={})

# ---- the main fits
D, _ = counts_for(df, ORDER); n = len(ORDER)
for mode in ["det", "two"]:
    z = lbfgs(D, n, mode).x
    B, _ = boot(df, ORDER, mode, z)
    R["main"][mode] = dict(**summary(z, n, mode), ci=ci(B))
    print(mode, R["main"][mode], flush=True)

# ---- the language groups, on a shared set of competitors
LANG = {"English": "en", "Polish": "pl", "Undetermined": "und", "Russian": "ru", "Chinese": "zh"}
group = meta.language.map({v: k for k, v in LANG.items()}).fillna("Other").to_numpy()
NAMES = list(LANG) + ["Other"]
keep = set(ORDER)
while True:
    new = set(keep)
    for g in NAMES:
        d = df[(group == g) & df.model_a.isin(keep) & df.model_b.isin(keep)]
        c = pd.concat([d.model_a, d.model_b]).value_counts(); new &= set(c[c >= MIN_ROWS].index)
    if new == keep: break
    keep = new
models = [m for m in ORDER if m in keep]; n = len(models)
inset = (df.model_a.isin(keep) & df.model_b.isin(keep)).to_numpy()
Dp, _ = counts_for(df[inset], models); zp = lbfgs(Dp, n, "two").x
R["language"].update(n=n, dropped=[m for m in ORDER if m not in keep], pooled=summary(zp, n, "two"))
boots = {}
for g in NAMES:
    d = df[inset & (group == g)]
    Dg, _ = counts_for(d, models); z = lbfgs(Dg, n, "two", z0=zp.copy()).x
    B, _ = boot(d, models, "two", z); boots[g] = B
    s = d.s.to_numpy()
    R["language"]["strata"][g] = dict(rows=len(d), both_bad=float((s == 2).mean()), tie=float((s == 3).mean()), **summary(z, n, "two"), ci=ci(B))
    print(g, R["language"]["strata"][g], flush=True)
R["language"]["diff_from_english"] = {g: {k: [float(np.percentile(boots[g][k] - boots["English"][k], 2.5)), float(np.percentile(boots[g][k] - boots["English"][k], 97.5))]
                                          for k in ("c", "tau", "bw", "bt")} for g in NAMES[1:]}
print(R["language"]["diff_from_english"], flush=True)

# ---- code and non-code, all 52 competitors, with the draws of every ability
isc = meta.is_code.astype(bool).to_numpy(); n = len(ORDER)
zp = lbfgs(D, n, "two").x; theta, draws = {}, {}
for lab, mask in [("code", isc), ("noncode", ~isc)]:
    d = df[mask]; Dg, _ = counts_for(d, ORDER); z = lbfgs(Dg, n, "two", z0=zp.copy()).x
    _, draws[lab] = boot(d, ORDER, "two", z)
    theta[lab] = z[:n] - z[:n].mean()
    R["code"][lab] = dict(rows=len(d), **summary(z, n, "two"))
np.savez((RESULTS / "code_theta_draws.npz"), models=np.array(ORDER), theta_pooled=zp[:n] - zp[:n].mean(), **{f"theta_{k}": v for k, v in theta.items()}, **{f"draws_{k}": v for k, v in draws.items()})
(RESULTS / "strata.json").write_text(json.dumps(R, indent=1))
