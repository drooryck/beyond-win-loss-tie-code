"""
Exact log-likelihood, score and Hessian for the models of Chapter 4, in the working coordinates.

    mode "det"   z = (nu, gamma)                 beta_w = beta_t = 1
    mode "two"   z = (nu, gamma, psi_w, psi_t)   the stochastic reporting model
    mode "one"   z = (nu, gamma, psi)            beta_w = beta_t = beta
    mode "rk"    z = (nu, gamma, psi)            Rao-Kupper with a share rho of ties reported both-bad

The derivatives are those of Appendix A.1. Data rows are (i, j, n1, n2, n3, n4).
"""
import numpy as np
from scipy.optimize import minimize

from lib.data import DROPPED, first_order, load_clean  # noqa: F401
from lib.paths import OUT_DIR, RAW  # noqa: F401

TINY = 1e-300
sig = lambda a: 1.0 / (1.0 + np.exp(-a))


def blocks(a, b, q):
    Z, Y = a + q * b, q * a + b
    A, B = a / Z, b / Y
    E, F, S = np.exp(-Z), np.exp(-Y), np.exp(-(a + b))
    T = (q * q - 1) * a * b / (Z * Y)
    return Z, Y, A, B, E, F, S, T


def probs(a, b, q, mode, r=None, s=None):
    Z, Y, A, B, E, F, S, T = blocks(a, b, q)
    if mode == "rk":
        return A, B, r * T, (1 - r) * T
    bw, bt = (1.0, 1.0) if mode == "det" else (r, r) if mode == "one" else (r, s)
    C7 = np.maximum(S - A * E - B * F, 0.0)
    return A * (1 - bw * E), B * (1 - bw * F), bw * (A * E + B * F) + bt * C7, T - bt * C7


def extras(z, n, mode):
    if mode == "det":
        return None, None
    if mode == "two":
        return sig(z[n + 1]), sig(z[n + 2])
    return sig(z[n + 1]), None


def loglik(z, D, n, mode, ridge=0.0):
    i, j = D[:, 0].astype(int), D[:, 1].astype(int)
    mu, q = np.exp(z[:n]), 1.0 + np.exp(z[n])
    r, s = extras(z, n, mode)
    p = probs(mu[i], mu[j], q, mode, r, s)
    ll = sum((D[:, 2 + k] * np.log(np.maximum(p[k], TINY))).sum() for k in range(4))
    if ridge:
        ll -= 0.5 * ridge * np.sum((z[:n] - z[:n].mean()) ** 2)
    return ll


def score(z, D, n, mode, ridge=0.0):
    i, j = D[:, 0].astype(int), D[:, 1].astype(int)
    mu, q = np.exp(z[:n]), 1.0 + np.exp(z[n])
    r, s = extras(z, n, mode)
    a, b = mu[i], mu[j]
    Z, Y, A, B, E, F, S, T = blocks(a, b, q)
    p = probs(a, b, q, mode, r, s)
    w = [D[:, 2 + k] / np.maximum(p[k], TINY) for k in range(4)]
    dA = (q * b / Z**2, -q * a / Z**2, -a * b / Z**2)
    dB = (-q * b / Y**2, q * a / Y**2, -a * b / Y**2)
    dE = (-E, -q * E, -b * E)
    dF = (-q * F, -F, -a * F)
    dS = (-S, -S, 0.0 * S)
    acc = []
    if mode == "rk":
        for k in range(3):
            dT = -(dA[k] + dB[k])
            acc.append(w[0] * dA[k] + w[1] * dB[k] + (w[2] * r + w[3] * (1 - r)) * dT)
    else:
        bw, bt = (1.0, 1.0) if mode == "det" else (r, r) if mode == "one" else (r, s)
        for k in range(3):
            d1 = (1 - bw * E) * dA[k] - bw * A * dE[k]
            d2 = (1 - bw * F) * dB[k] - bw * B * dF[k]
            d3 = bt * dS[k] + (bw - bt) * (E * dA[k] + A * dE[k] + F * dB[k] + B * dF[k])
            acc.append(w[0] * d1 + w[1] * d2 + w[2] * d3 - w[3] * (d1 + d2 + d3))
    g = np.zeros(len(z))
    np.add.at(g, i, acc[0] * a)
    np.add.at(g, j, acc[1] * b)
    g[n] = acc[2].sum() * (q - 1)
    if mode == "rk":
        g[n + 1] = ((w[2] - w[3]) * T).sum() * r * (1 - r)
    elif mode != "det":
        C = A * E + B * F
        dw = (-w[0] * A * E - w[1] * B * F + w[2] * C).sum()
        dt = ((w[2] - w[3]) * (S - C)).sum()
        if mode == "two":
            g[n + 1], g[n + 2] = dw * r * (1 - r), dt * s * (1 - s)
        else:
            g[n + 1] = (dw + dt) * r * (1 - r)
    if ridge:
        g[:n] -= ridge * (z[:n] - z[:n].mean())
    return g


def hessian(z, D, n, mode, h=1e-5):
    m = len(z); H = np.zeros((m, m))
    for k in range(m):
        e = np.zeros(m); e[k] = h
        H[:, k] = (score(z + e, D, n, mode) - score(z - e, D, n, mode)) / (2 * h)
    return 0.5 * (H + H.T)


def size(n, mode):
    return n + {"det": 1, "two": 3, "one": 2, "rk": 2}[mode]


def lbfgs(D, n, mode, ridge=0.0, z0=None):
    z0 = np.zeros(size(n, mode)) if z0 is None else z0
    return minimize(lambda z: -loglik(z, D, n, mode, ridge), z0, method="L-BFGS-B",
                    jac=lambda z: -score(z, D, n, mode, ridge),
                    options=dict(maxiter=20000, maxfun=100000, ftol=1e-15, gtol=1e-7))


def newton(z0, D, n, mode, tol=1e-6, maxit=200):
    z, ll = z0.copy(), loglik(z0, D, n, mode)
    for it in range(maxit):
        g = score(z, D, n, mode)
        if np.abs(g).max() < tol:
            return z, ll, it, True
        H = hessian(z, D, n, mode)
        lam = 0.0
        for _ in range(60):
            M = -H + lam * np.eye(len(z))
            if np.linalg.eigvalsh(M).min() > 0:
                step = np.linalg.solve(M, g)
                break
            lam = max(1e-8, lam * 10)
        t = 1.0
        for _ in range(50):
            new = z + t * step
            lln = loglik(new, D, n, mode)
            if lln > ll:
                break
            t *= 0.5
        else:
            return z, ll, it, False
        z, ll = new, lln
    return z, ll, maxit, False


def fit_rule(data, n, rule, ridge=0.0, z0=None):
    r = lbfgs(data, n, "det" if rule == "lmarena" else rule, ridge, z0)
    return r.x, -r.fun, r


def fit_completion(data, n, mode, ridge=0.0, z0=None):
    r = lbfgs(data, n, mode, ridge, z0)
    ll = -r.fun + (0.5 * ridge * np.sum((r.x[:n] - r.x[:n].mean()) ** 2) if ridge else 0.0)
    return r.x, ll, r


# ---- fixed-coefficient grounded Rao-Kupper model of Frick et al. (2025), z = (nu, gamma)
def grk_probs(phi_i, phi_j, lam):
    a = phi_i / (phi_i + lam * phi_j + 1)
    b = phi_j / (phi_j + lam * phi_i + 1)
    c = 1 / (1 + phi_i + phi_j)
    return a, b, c, 1 - a - b - c


def grk_ll(z, D, n):
    i, j = D[:, 0].astype(int), D[:, 1].astype(int)
    phi, lam = np.exp(z[:n]), 1 + np.exp(z[n])
    p = grk_probs(phi[i], phi[j], lam)
    return sum((D[:, 2 + k] * np.log(np.maximum(p[k], TINY))).sum() for k in range(4))


def grk_score(z, D, n):
    i, j = D[:, 0].astype(int), D[:, 1].astype(int)
    phi, lam = np.exp(z[:n]), 1 + np.exp(z[n])
    x, y = phi[i], phi[j]
    p = grk_probs(x, y, lam)
    w = [D[:, 2 + k] / np.maximum(p[k], TINY) for k in range(4)]
    U, V, C = x + lam * y + 1, y + lam * x + 1, 1 + x + y
    da = ((lam * y + 1) / U**2, -lam * x / U**2, -x * y / U**2)
    db = (-lam * y / V**2, (lam * x + 1) / V**2, -x * y / V**2)
    dc = (-1 / C**2, -1 / C**2, 0.0 * C)
    acc = [w[0] * da[k] + w[1] * db[k] + w[2] * dc[k] - w[3] * (da[k] + db[k] + dc[k]) for k in range(3)]
    g = np.zeros(n + 1)
    np.add.at(g, i, acc[0] * x); np.add.at(g, j, acc[1] * y)
    g[n] = acc[2].sum() * (lam - 1)
    return g


def grk_fit(D, n, z0=None):
    z0 = np.zeros(n + 1) if z0 is None else z0
    return minimize(lambda z: -grk_ll(z, D, n), z0, jac=lambda z: -grk_score(z, D, n), method="L-BFGS-B",
                    options=dict(maxiter=20000, maxfun=100000, ftol=1e-15, gtol=1e-7))


# ---- Rao-Kupper on three outcomes, with both-bad outcomes removed or merged into ties
def rk3_ll_score(z, W, L, T, i, j, n):
    mu, q = np.exp(z[:n]), 1 + np.exp(z[n])
    a, b = mu[i], mu[j]
    Z, Y = a + q * b, q * a + b
    A, B = a / Z, b / Y
    P = 1 - A - B
    ll = (W * np.log(A) + L * np.log(B) + T * np.log(np.maximum(P, TINY))).sum()
    dA = (q * b / Z**2, -q * a / Z**2, -a * b / Z**2)
    dB = (-q * b / Y**2, q * a / Y**2, -a * b / Y**2)
    acc = [W / A * dA[k] + L / B * dB[k] - T / np.maximum(P, TINY) * (dA[k] + dB[k]) for k in range(3)]
    g = np.zeros(n + 1)
    np.add.at(g, i, acc[0] * a); np.add.at(g, j, acc[1] * b)
    g[n] = acc[2].sum() * (q - 1)
    return -ll, -g


def rk3(D, n, merge):
    i, j = D[:, 0].astype(int), D[:, 1].astype(int)
    W, L, T = D[:, 2], D[:, 3], D[:, 5] + (D[:, 4] if merge else 0.0)
    f = minimize(rk3_ll_score, np.zeros(n + 1), args=(W, L, T, i, j, n), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=20000, maxfun=100000, ftol=1e-15, gtol=1e-7))
    return f.x
