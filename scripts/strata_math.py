"""
The stochastic model and its comparisons, fitted separately within strata of the first-order data:
English versus other languages, code versus non-code, and math versus non-math prompts.

Within each pair of strata we keep the competitors with at least MIN_ROWS comparisons in both strata,
so that the two fits share one set of competitors, which makes c and the abilities comparable.

Saves strata_math.json. Run with `make strata-math`.
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
from lib.data import load_clean
from lib.exact_score import lbfgs, loglik, sig, grk_fit, grk_ll
from lib.paths import RAW, RESULTS, FIGURES

MIN_ROWS, SPLITS, BOOT = 100, 10, 200
df = load_clean(load)
meta = pd.read_parquet(RAW, columns=["language", "is_code", "category_tag"]).loc[df.index]
math = meta.category_tag.map(lambda d: bool(d["math_v0.1"]["math"]) if d and d.get("math_v0.1") else False)
GROUPS = {"language": ("English", "other languages", meta.language == "en"),
          "code": ("code", "non-code", meta.is_code.astype(bool)),
          "math": ("math", "non-math", math)}
rng = np.random.default_rng(20261104)


def summary(z, n):
    nu = z[:n]
    return dict(c=float(-nu.mean()), tau=float(np.log(1 + np.exp(z[n]))), q=float(1 + np.exp(z[n])), bw=float(sig(z[n + 1])), bt=float(sig(z[n + 2])))


def eligible(d, need):
    c = pd.concat([d.model_a, d.model_b]).value_counts()
    return set(c[c >= need].index)


R = dict(min_rows=MIN_ROWS, splits=SPLITS, boot=BOOT, total_rows=len(df))
for g, (la, lb, mask) in GROUPS.items():
    mask = mask.to_numpy()
    A, B = df[mask], df[~mask]
    keep = eligible(A, MIN_ROWS) & eligible(B, MIN_ROWS)
    while True:                                   # dropping competitors reduces the others' counts
        fa = A[A.model_a.isin(keep) & A.model_b.isin(keep)]; fb = B[B.model_a.isin(keep) & B.model_b.isin(keep)]
        new = eligible(fa, MIN_ROWS) & eligible(fb, MIN_ROWS)
        if new == keep: break
        keep = new
    allk = df[df.model_a.isin(keep) & df.model_b.isin(keep)]
    models = list(pd.concat([allk.model_a, allk.model_b]).value_counts().index); n = len(models)
    Dp, _ = counts_for(allk, models)
    zp = lbfgs(Dp, n, "two").x
    R[g] = dict(n=n, dropped=sorted(set(pd.concat([df.model_a, df.model_b])) - keep), pooled=summary(zp, n), strata={})
    thetas, zs, boots = {}, {}, {}
    for lab, d in [(la, fa), (lb, fb)]:
        D, _ = counts_for(d, models)
        z = lbfgs(D, n, "two", z0=zp.copy()).x
        zs[lab] = z; th = z[:n] - z[:n].mean(); thetas[lab] = th
        s = d.s.to_numpy(); order = (-th).argsort()
        row = dict(rows=len(d), pairs=len(D), share=float(len(d) / len(allk)), both_bad=float((s == 2).mean()), tie=float((s == 3).mean()), **summary(z, n),
                   sd_theta=float(th.std()), top3=[models[k] for k in order[:3]],
                   rank_with_pooled=float(spearmanr(th, zp[:n] - zp[:n].mean()).statistic), one_sided=[])
        W = np.zeros((n, 2))
        for x in D:
            i, j = int(x[0]), int(x[1]); W[i] += [x[2], x[3]]; W[j] += [x[3], x[2]]
        row["one_sided"] = [models[k] for k in range(n) if W[k].min() == 0]
        # bootstrap over rows
        B_ = []
        for b in range(BOOT):
            db, _ = counts_for(d.iloc[rng.integers(0, len(d), len(d))], models)
            B_.append(summary(lbfgs(db, n, "two", z0=z.copy()).x, n))
        B_ = pd.DataFrame(B_); boots[lab] = B_
        row["ci"] = {k: [float(np.percentile(B_[k], 2.5)), float(np.percentile(B_[k], 97.5))] for k in ("c", "tau", "bw", "bt")}
        # held-out within the stratum, ten splits
        rows = []
        for sp in range(SPLITS):
            r2 = np.random.default_rng(1000 + sp); tr = np.zeros(len(d), bool)
            for k in range(4):
                m = np.where((d.s == k).to_numpy())[0]
                tr[r2.permutation(m)[: int(round(0.8 * len(m)))]] = True
            train, _ = counts_for(d[tr], models); test, _ = counts_for(d[~tr], models); M = test[:, 2:].sum()
            rows.append([-loglik(lbfgs(train, n, m_).x, test, n, m_) / M for m_ in ("two", "det", "rk")] + [-grk_ll(grk_fit(train, n).x, test, n) / M])
        a = np.array(rows)
        row["nll"] = dict(stoch=a[:, 0].mean(), det=a[:, 1].mean(), rk=a[:, 2].mean(), grk=a[:, 3].mean(),
                          stoch_beats_det=int((a[:, 0] < a[:, 1]).sum()), stoch_beats_rk=int((a[:, 0] < a[:, 2]).sum()), stoch_beats_grk=int((a[:, 0] < a[:, 3]).sum()),
                          rk_minus_stoch=(a[:, 2] - a[:, 0]).mean(), grk_minus_stoch=(a[:, 3] - a[:, 0]).mean(), det_minus_stoch=(a[:, 1] - a[:, 0]).mean())
        R[g]["strata"][lab] = row
        print(g, lab, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("ci", "nll")}, {k: [round(x, 3) for x in v] for k, v in row["ci"].items()},
              {k: round(float(v), 4) for k, v in row["nll"].items()}, flush=True)
    R[g]["rank_between"] = float(spearmanr(thetas[la], thetas[lb]).statistic)
    dth = np.abs(thetas[la] - thetas[lb]); w = dth.argsort()[::-1][:3]
    R[g]["largest_moves"] = [[models[k], float(thetas[la][k]), float(thetas[lb][k])] for k in w]
    # difference between the strata, by bootstrap
    R[g]["diff_ci"] = {k: [float(np.percentile(boots[la][k] - boots[lb][k], 2.5)), float(np.percentile(boots[la][k] - boots[lb][k], 97.5))] for k in ("c", "tau", "bw", "bt")}
    print(g, "n", n, "rank between", round(R[g]["rank_between"], 4), "moves", [(m, round(a, 2), round(b, 2)) for m, a, b in R[g]["largest_moves"]], "diff", {k: [round(x, 3) for x in v] for k, v in R[g]["diff_ci"].items()}, flush=True)
(RESULTS / "strata_math.json").write_text(json.dumps(R, indent=1, default=float))
print("done")

# Benchmark for the rank correlation between strata: the same comparison after assigning rows to the
# two groups at random, with the same group sizes. This shows how low the correlation is from noise alone.
R["noise_benchmark"] = {}
for g, (la, lb, mask) in GROUPS.items():
    dropped = set(R[g]["dropped"])
    keep_rows = (~df.model_a.isin(dropped) & ~df.model_b.isin(dropped)).to_numpy()
    d = df[keep_rows]; share = float(mask.to_numpy()[keep_rows].mean())
    models = list(pd.concat([d.model_a, d.model_b]).value_counts().index); n = len(models)
    cors = []
    for r in range(20):
        u = rng.random(len(d)) < share
        D1, _ = counts_for(d[u], models); D2, _ = counts_for(d[~u], models)
        z1, z2 = lbfgs(D1, n, "two").x, lbfgs(D2, n, "two").x
        cors.append(float(spearmanr(z1[:n], z2[:n]).statistic))
    R["noise_benchmark"][g] = dict(mean=float(np.mean(cors)), low=float(np.min(cors)), high=float(np.max(cors)), observed=R[g]["rank_between"])
    print(g, R["noise_benchmark"][g], flush=True)
(RESULTS / "strata_math.json").write_text(json.dumps(R, indent=1, default=float))
