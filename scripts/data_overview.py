"""Section 4.1 data summary and EDA figures (comparison graph, both-bad vs win rate)."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json, os
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from lib.paths import RESULTS, FIGURES, RAW

FIG = FIGURES
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
raw = pd.read_parquet(RAW,
                      columns=["model_a", "model_b", "winner", "language", "is_code", "timestamp", "evaluation_order", "conv_metadata"])
if os.environ.get("FIRST_ORDER") == "1":
    raw = raw[raw.evaluation_order == 1].copy()
print("rows at this subset", len(raw), "with more than one turn", int((raw.conv_metadata.map(lambda d: d["turns"]) > 1).sum()))
print("winner values", raw.winner.value_counts().to_dict())
self_rows = raw[raw.model_a == raw.model_b]
print("self rows", len(self_rows), self_rows.model_a.unique(), self_rows.winner.value_counts().to_dict())
drop = "qwen3-coder-480b-a35b-instruct"
print("dropped competitor rows", raw[(raw.model_a == drop) | (raw.model_b == drop)].winner.value_counts().to_dict())
df = raw[(raw.model_a != raw.model_b) & (raw.model_a != drop) & (raw.model_b != drop)].copy()
code = {"model_a": 0, "model_b": 1, "both_bad": 2, "tie (bothbad)": 2, "tie": 3}
df = df[df.winner.isin(code)].copy(); df["s"] = df.winner.map(code).astype(int)
R = dict(rows=len(df))
R["outcomes"] = {k: [int((df.s == v).sum()), round(100 * float((df.s == v).mean()), 1)] for k, v in dict(a=0, b=1, bb=2, tie=3).items()}
R["language"] = (df.language.value_counts(normalize=True).head(6) * 100).round(1).to_dict()
R["is_code"] = round(100 * float(df.is_code.mean()), 1)
R["bb_code"] = {str(k): round(100 * float(v), 1) for k, v in df.groupby("is_code").s.apply(lambda x: (x == 2).mean()).items()}
ts = pd.to_datetime(df.timestamp); R["span"] = [str(ts.min().date()), str(ts.max().date())]
mon = ts.dt.to_period("M")
R["monthly_bb"] = {str(k): round(100 * float(v), 1) for k, v in df.groupby(mon).s.apply(lambda x: (x == 2).mean()).items()}
R["monthly_tie"] = {str(k): round(100 * float(v), 1) for k, v in df.groupby(mon).s.apply(lambda x: (x == 3).mean()).items()}
freq = pd.concat([df.model_a, df.model_b]).value_counts()
models = list(freq.index); n = len(models); idx = {m: k for k, m in enumerate(models)}
a = df.model_a.map(idx).to_numpy(); b = df.model_b.map(idx).to_numpy(); s = df.s.to_numpy()
M = np.zeros((n, n)); np.add.at(M, (a, b), 1); np.add.at(M, (b, a), 1)
iu = np.triu_indices(n, 1); cnt = M[iu]; obs = cnt[cnt > 0]
R.update(n=n, possible=len(cnt), pairs=int((cnt > 0).sum()), median_pair=float(np.median(obs)), max_pair=int(obs.max()),
         pairs_ge50=int((obs >= 50).sum()), below500=[(m, int(v)) for m, v in freq.items() if v < 500])
print(json.dumps(R, indent=1), flush=True)

fig, axm = plt.subplots(figsize=(5.6, 4.9))
Mm = np.ma.masked_where(M == 0, M)
for k in range(n):
    axm.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, color="0.80", lw=0, zorder=1))
im = axm.imshow(Mm, cmap="viridis", norm=LogNorm(vmin=1, vmax=M.max()), zorder=2)
axm.set_xlim(-.6, n - .4); axm.set_ylim(n - .4, -.6)
axm.set_xticks([0, 10, 20, 30, 40, n - 1]); axm.set_yticks([0, 10, 20, 30, 40, n - 1]); axm.tick_params(labelsize=7)
axm.set_xlabel("competitor, ordered by total comparisons", fontsize=8); axm.set_ylabel("competitor", fontsize=8)
axm.legend(handles=[Patch(fc="white", ec="0.6", lw=.6, label="never compared"), Patch(fc="0.80", ec="0.6", lw=.6, label="same competitor")],
           loc="lower left", bbox_to_anchor=(0.005, 0.005), fontsize=6.5, frameon=True, framealpha=.92, borderpad=.4)
cb = fig.colorbar(im, ax=axm, fraction=.046, pad=.03); cb.set_label("comparisons in pair", fontsize=8); cb.ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(FIGURES / "comparison_graph.pdf")

win = np.zeros(n); dec = np.zeros(n)
np.add.at(win, a[s == 0], 1); np.add.at(win, b[s == 1], 1); np.add.at(dec, a[s < 2], 1); np.add.at(dec, b[s < 2], 1)
wr = win / np.maximum(dec, 1)
bb = np.zeros((n, n)); np.add.at(bb, (np.minimum(a, b), np.maximum(a, b)), (s == 2).astype(float))
bbc = bb[iu]; keep = cnt >= 60
x = (wr[iu[0]] + wr[iu[1]])[keep]; y = (bbc / np.maximum(cnt, 1))[keep]; w = cnt[keep]
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.scatter(x, y, s=np.sqrt(w) * 1.2, alpha=.28, color="#3b6ea5", lw=0, label="pair")
e = np.quantile(x, np.linspace(0, 1, 9)); cx, cy, lo, hi = [], [], [], []
for u, v in zip(e[:-1], e[1:]):
    m = (x >= u) & (x < v)
    if not m.sum(): continue
    p = np.average(y[m], weights=w[m]); se = np.sqrt(p * (1 - p) / w[m].sum())
    cx.append(np.average(x[m], weights=w[m])); cy.append(p); lo.append(p - 1.96 * se); hi.append(p + 1.96 * se)
ax.fill_between(cx, lo, hi, color="0.6", alpha=.5, lw=0, zorder=3)
ax.plot(cx, cy, "o-", color="black", ms=4, lw=1.3, zorder=4, label="binned mean")
ax.set_xlabel("sum of the two competitors' empirical win rates"); ax.set_ylabel("both-bad rate for the pair")
ax.grid(color="0.93", lw=.6); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIGURES / "bothbad_winrate.pdf")
R.update(corr=float(np.corrcoef(x, y)[0, 1]), pairs60=int(keep.sum()), bins=[round(cy[0], 3), round(cy[-1], 3)])
print({k: R[k] for k in ("corr", "pairs60", "bins")})
(RESULTS / "data_overview.json").write_text(json.dumps(R, indent=1))
