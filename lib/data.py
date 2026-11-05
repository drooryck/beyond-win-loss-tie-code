"""Load and aggregate the LMArena preference data."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from lib.paths import RAW

DROPPED = "qwen3-coder-480b-a35b-instruct"
_rng = np.random.default_rng(20260804)


def first_order(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with evaluation_order equal to 1, unless FIRST_ORDER=0 is set."""
    if os.environ.get("FIRST_ORDER", "1") != "1":
        return df
    order = pd.read_parquet(RAW, columns=["evaluation_order"]).evaluation_order
    return df[order.loc[df.index] == 1].copy()


def load() -> pd.DataFrame:
    df = pd.read_parquet(RAW, columns=["model_a", "model_b", "winner"])
    code = {"model_a": 0, "model_b": 1, "both_bad": 2, "tie (bothbad)": 2, "tie": 3}
    df = df[df.winner.isin(code)].copy()
    df["s"] = df.winner.map(code).astype(int)
    return df


def load_clean(load_fn=load) -> pd.DataFrame:
    """Remove self-comparisons and the two comparisons of the dropped competitor."""
    df = first_order(load_fn())
    df = df[df.model_a != df.model_b]
    return df[(df.model_a != DROPPED) & (df.model_b != DROPPED)].copy()


def counts_for(df: pd.DataFrame, models):
    """Aggregate to (i, j, n1, n2, n3, n4) with i < j in the model index."""
    idx = {m: k for k, m in enumerate(models)}
    d = df[df.model_a.isin(idx) & df.model_b.isin(idx)].copy()
    a = d.model_a.map(idx).to_numpy()
    b = d.model_b.map(idx).to_numpy()
    s = d.s.to_numpy()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    flip = a > b
    s2 = s.copy()
    s2[flip & (s == 0)] = 1
    s2[flip & (s == 1)] = 0
    key = lo * len(models) + hi
    out = {}
    for k, ss in zip(key, s2):
        r = out.setdefault(k, [0, 0, 0, 0])
        r[ss] += 1
    rows = [[k // len(models), k % len(models), *v] for k, v in out.items()]
    return np.array(rows, dtype=float), d


def split(df: pd.DataFrame, frac: float = 0.8):
    """Stratified 80/20 split by observed outcome."""
    tr = np.zeros(len(df), bool)
    for s in range(4):
        m = np.where((df.s == s).to_numpy())[0]
        tr[_rng.permutation(m)[: int(round(frac * len(m)))]] = True
    return df[tr], df[~tr]
