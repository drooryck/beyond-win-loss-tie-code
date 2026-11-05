"""
Rankings under Rao-Kupper fits that use the both-bad outcome differently, against the other models,
on the first-order data:

    rk_drop   Rao-Kupper on the comparisons that were not both-bad
    rk_merge  Rao-Kupper with every both-bad outcome counted as a tie

Saves rankings.json. Run with `make rankings`.
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
from lib.exact_score import lbfgs, grk_fit, rk3, TINY
from lib.paths import RESULTS, FIGURES


df = load_clean(load)
models = list(pd.concat([df.model_a, df.model_b]).value_counts().index); n = len(models)
D, _ = counts_for(df, models)
fits = {"rk_drop": rk3(D, n, False), "rk_merge": rk3(D, n, True), "split_ties": lbfgs(D, n, "rk").x,
        "grounded": grk_fit(D, n).x, "det": lbfgs(D, n, "det").x, "stoch": lbfgs(D, n, "two").x}
theta = {k: v[:n] - v[:n].mean() for k, v in fits.items()}
rank = {k: (-v).argsort().argsort() + 1 for k, v in theta.items()}
tot = pd.concat([df.model_a, df.model_b]).value_counts().reindex(models).to_numpy()
names = list(fits)
R = dict(models=models, comparisons=tot.tolist(), q={k: float(1 + np.exp(fits[k][n])) for k in names},
         theta={k: v.tolist() for k, v in theta.items()}, rank={k: v.tolist() for k, v in rank.items()},
         spearman={a: {b: float(spearmanr(theta[a], theta[b]).statistic) for b in names} for a in names},
         pearson={a: {b: float(np.corrcoef(theta[a], theta[b])[0, 1]) for b in names} for a in names},
         sd={k: float(v.std()) for k, v in theta.items()},
         merge_vs_split_max=float(np.abs(theta["rk_merge"] - theta["split_ties"]).max()))
print("sd", {k: round(v, 3) for k, v in R["sd"].items()}, "q", {k: round(v, 3) for k, v in R["q"].items()})
print(pd.DataFrame(R["spearman"]).round(3).to_string())
print("merge vs split ties, largest difference", R["merge_vs_split_max"])
keys = ["rk_drop", "rk_merge", "grounded", "det", "stoch"]
tab = pd.DataFrame({k: rank[k] for k in keys}, index=models)
tab["move"] = tab[keys].max(axis=1) - tab[keys].min(axis=1); tab["n"] = tot
print(tab.sort_values("stoch").head(12).to_string()); print(tab.sort_values("move", ascending=False).head(10).to_string())
print("drop vs det", {m: (int(rank["rk_drop"][k]), int(rank["det"][k]), int(tot[k])) for k, m in enumerate(models) if abs(rank["rk_drop"][k] - rank["det"][k]) >= 6})
(RESULTS / "rankings.json").write_text(json.dumps(R, indent=1))
