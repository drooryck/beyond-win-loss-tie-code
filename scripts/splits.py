"""
Table tab:heldout: held-out comparison over ten random 80/20 splits on four nested subsets of the first-order comparisons,
including the grounded Rao-Kupper baseline, with the differences between models quoted in Section 4.4.2.

Saves results/splits.json. Run with `make splits`.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import numpy as np, pandas as pd

from lib.data import load, counts_for, load_clean
from lib.exact_score import lbfgs, loglik, grk_fit, grk_ll
from lib.paths import RESULTS

SPLITS = 10
df = load_clean(load)
freq = list(pd.concat([df.model_a, df.model_b]).value_counts().index)
R = {}
for sub, top in [("top10", 10), ("top20", 20), ("top40", 40), ("all52", 52)]:
    models = freq[:top]; n = len(models)
    rows = []
    for sp in range(SPLITS):
        rng = np.random.default_rng(1000 + sp)
        tr = np.zeros(len(df), bool)
        for s in range(4):
            m = np.where((df.s == s).to_numpy())[0]
            tr[rng.permutation(m)[: int(round(0.8 * len(m)))]] = True
        train, _ = counts_for(df[tr], models); test, _ = counts_for(df[~tr], models)
        M = test[:, 2:].sum()
        nll = {}
        for mode in ["rk", "det", "two"]:
            z = lbfgs(train, n, mode).x
            nll[mode] = -loglik(z, test, n, mode) / M
        nll["grk"] = -grk_ll(grk_fit(train, n).x, test, n) / M
        gp = train[:, 2:].sum(axis=0); gp = gp / gp.sum()
        base_g = -(test[:, 2:] * np.log(gp)).sum() / M
        tr_map = {(int(r[0]), int(r[1])): r[2:] for r in train}
        Pp = np.array([(tr_map.get((int(r[0]), int(r[1])), np.zeros(4)) + 1.0) /
                       (tr_map.get((int(r[0]), int(r[1])), np.zeros(4)).sum() + 4.0) for r in test])
        base_p = -(test[:, 2:] * np.log(Pp)).sum() / M
        rows.append([nll["rk"], nll["det"], nll["two"], base_g, base_p, M, nll["det"] - nll["rk"], nll["grk"]])
    a = np.array(rows)
    d_rk, d_det = a[:, 0] - a[:, 2], a[:, 1] - a[:, 2]
    R[sub] = dict(rk=a[:, 0].mean(), det=a[:, 1].mean(), stoch=a[:, 2].mean(), glob=a[:, 3].mean(), pair=a[:, 4].mean(),
                  test_rows=a[:, 5].mean(), rows=int(train[:, 2:].sum() + test[:, 2:].sum()),
                  det_minus_rk=a[:, 6].mean(), min_det_rk=a[:, 6].min(), max_det_rk=a[:, 6].max(), min_det=d_det.min(), max_det=d_det.max(),
                  stoch_best=int(((a[:, 2] < a[:, 0]) & (a[:, 2] < a[:, 1]) & (a[:, 2] < a[:, 3]) & (a[:, 2] < a[:, 4])).sum()),
                  rk_minus_stoch=d_rk.mean(), sd_rk=d_rk.std(ddof=1), min_rk=d_rk.min(), max_rk=d_rk.max(),
                  stoch_better_than_rk=int((d_rk > 0).sum()), det_minus_stoch=d_det.mean(), stoch_better_than_det=int((d_det > 0).sum()),
                  grk=a[:, 7].mean(), grk_minus_stoch=(a[:, 7] - a[:, 2]).mean(), min_grk=(a[:, 7] - a[:, 2]).min(), max_grk=(a[:, 7] - a[:, 2]).max(),
                  stoch_better_than_grk=int((a[:, 7] > a[:, 2]).sum()), grk_better_than_rk=int((a[:, 7] < a[:, 0]).sum()),
                  grk_better_than_det=int((a[:, 7] < a[:, 1]).sum()), det_minus_grk=(a[:, 1] - a[:, 7]).mean())
    print(sub, {k: round(float(v), 5) for k, v in R[sub].items()}, flush=True)
(RESULTS / "splits.json").write_text(json.dumps(R, indent=1, default=float))
