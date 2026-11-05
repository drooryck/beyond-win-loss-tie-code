"""Simulation helpers and reparameterizations used by the Chapter 4 experiments."""
from __future__ import annotations

import numpy as np

_TINY = 1e-300
sig = lambda a: 1.0 / (1.0 + np.exp(-a))


def unpack(z, n):
    nu, gamma = z[:n], z[n]
    return np.exp(nu), 1.0 + np.exp(gamma)


def to_theta_c(z, n):
    """Apply sum(theta) = 0: returns (theta, c, tau)."""
    nu = z[:n]
    c = -nu.mean()
    return nu + c, c, np.log(1.0 + np.exp(z[n]))


def probs(mu_i, mu_j, q):
    """Deterministic-reporting outcome probabilities."""
    Z, Zp = mu_i + q * mu_j, q * mu_i + mu_j
    A, B = mu_i / Z, mu_j / Zp
    E, Ep = np.exp(-Z), np.exp(-Zp)
    p1 = A * (-np.expm1(-Z))
    p2 = B * (-np.expm1(-Zp))
    p3 = np.exp(-(mu_i + mu_j))
    rk_tie = (q * q - 1.0) * mu_i * mu_j / (Z * Zp)
    p4 = rk_tie - p3 + A * E + B * Ep
    return p1, p2, p3, p4


def probs_rule(mu_i, mu_j, q, rule, rho=None):
    Z, Zp = mu_i + q * mu_j, q * mu_i + mu_j
    A, B = mu_i / Z, mu_j / Zp
    rk_tie = (q * q - 1.0) * mu_i * mu_j / (Z * Zp)
    if rule == "lmarena":
        return probs(mu_i, mu_j, q)
    if rule == "c7":
        C3, C10 = A * np.exp(-Z), B * np.exp(-Zp)
        C7 = np.exp(-(mu_i + mu_j)) - C3 - C10
        return A, B, C7, rk_tie - C7
    if rule == "rk":
        return A, B, rho * rk_tie, (1.0 - rho) * rk_tie
    raise ValueError(rule)


def probs_completion(mu_i, mu_j, q, bw, bt):
    Z, Zp = mu_i + q * mu_j, q * mu_i + mu_j
    A, B = mu_i / Z, mu_j / Zp
    E, Ep = np.exp(-Z), np.exp(-Zp)
    rk_tie = (q * q - 1.0) * mu_i * mu_j / (Z * Zp)
    C7 = np.maximum(np.exp(-(mu_i + mu_j)) - A * E - B * Ep, 0.0)
    p1 = A * (1.0 - bw * E)
    p2 = B * (1.0 - bw * Ep)
    p3 = bw * (A * E + B * Ep) + bt * C7
    p4 = rk_tie - bt * C7
    return p1, p2, p3, p4


def betas_from(z, n, mode):
    if mode == "two":
        return sig(z[n + 1]), sig(z[n + 2])
    if mode == "one":
        return sig(z[n + 1]), sig(z[n + 1])
    return mode


def simulate_completion(theta, c, tau, bw, bt, pairs, n_per, rng):
    mu, q = np.exp(np.asarray(theta) - c), np.exp(tau)
    rows = []
    for (i, j) in pairs:
        p = np.array(probs_completion(mu[i], mu[j], q, bw, bt))
        rows.append([i, j, *rng.multinomial(n_per, p / p.sum())])
    return np.array(rows, dtype=float)
