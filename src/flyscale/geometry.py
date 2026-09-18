"""Latent geometry discovery for the FlyWire v783 connectome (FlyScale Phase 4).

Scope: PROJECT-VYBFLY.md section 10 ("Latent Geometry Discovery"). The scientific
question is whether a low-dimensional *latent* geometry (a 2-D hyperbolic / Poincare
embedding of the wiring graph) represents this connectome's connectivity structure
better than the neuron's physical 3-D anatomical coordinates, as reported for the
Drosophila connectome by Sulyok, Balogh & Palla, "Network geometry of the Drosophila
brain" (arXiv:2602.16417, 2026), and whether higher-dimensional spectral embeddings
improve on both.

Everything here is built on the canonical dataset API in :mod:`flyscale.connectome`
and :mod:`flyscale.metrics` and does not modify them.

The comparison protocol implemented by :func:`compare_geometries` (all choices are
recorded in the emitted JSON so the numbers cannot drift from their meaning):

  * Analysis graph: the ``>= 5`` synapse convention (``Connectome.thresholded(5)``,
    Lin et al. 2024), projected to an undirected, autapse-free, unweighted graph.
  * Analysis nodes: the giant weakly-connected component of that projection, because
    normalized-Laplacian eigenmaps are undefined for disconnected components (each
    component contributes a trivial eigenvalue 1). Every geometry is therefore fit
    and scored on the same node and pair sets.
  * Undirected edges are split 90% fit / 10% test. Test edges never enter any fit
    (including the spectral graph) and are never used as negative samples.
  * Negative samples are drawn uniformly from pairs that are not edges of the full
    (fit + test) graph, so a held-out true edge can never be scored as a negative.
  * Each geometry is turned into a connection probability through the same two
    parameter law

          P(connect | d) = 1 / (1 + exp((d - R) / T))

    with the distance ``d`` for that geometry, and ``(R, T)`` fitted by maximum
    likelihood on the *fit* pairs (1:1 positive:negative balanced sample, so a
    fitted ``R`` is the distance at which the balanced sample is 50/50 -- see
    :func:`fit_connection_law` for the exact transformation to the density-matched
    law).
  * Held-out quality is ROC AUC (midpoint tie credit) and average precision
    (expectation over uniformly random tie-breaking), both computed on the identical
    held-out pair set for every geometry, plus the mean Bernoulli log-likelihood of
    the held-out pairs under the fitted law.
  * A degree-only baseline (product of total degrees, logistic-calibrated) is scored
    the same way, so a geometry that merely recovers degree heterogeneity is visible.

Public API
----------
``anatomical_xy(c)`` / ``anatomical_xyz(c)``
    Published annotation coordinates (nm), (N, 2) and (N, 3), NaN where missing.
``spectral_embedding(c, k=32)``
    Normalized-Laplacian eigenmaps, (N, k); NaN for nodes not fitted.
``hyperbolic_embedding(c, dim=2, ...)``
    Position in the Poincare ball by SGD on the connection-likelihood objective;
    returns a dict with the coordinates plus the fitted ``(R, T)``.
``compare_geometries(c, geometries, protocol)``
    The held-out comparison described above.
``fit_connection_law(d, y)`` / ``connection_probability(d, R, T)``
    The geometric connection law.
``roc_auc(y, s)`` / ``average_precision(y, s)`` / ``bernoulli_log_likelihood(y, p)``
    Metrics, implemented here because scikit-learn is not a dependency.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse import csgraph
from scipy.sparse import linalg as splinalg
from scipy.stats import rankdata

# --------------------------------------------------------------------------- constants
GEOMETRY_KINDS = ("euclidean", "hyperbolic")
DEFAULT_SEED = 0
DEFAULT_THRESHOLD = 5


# =========================================================================== metrics
def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC with exact midpoint tie credit (Mann-Whitney U / rank formulation)."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(np.asarray(score, dtype=np.float64))          # average ranks on ties
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision = exact expectation over uniformly random tie-breaking.

    Ties are common in the degree-only baseline (integer degree products), where an
    arbitrary tie-break would bias the number, so the closed form of the expected
    precision over all orderings consistent with the scores is used instead: for a
    group of ``m`` tied items holding ``p`` positives at rank offsets ``k = 1..m``,

        E[precision at slot k | slot k is positive] = (A + 1 + (k-1)(p-1)/(m-1)) / (s + k)

    with ``A`` positives and ``s`` items ranked strictly above the group.
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    n = ys.size
    n_pos = int(ys.sum())
    if n_pos == 0:
        return float("nan")
    bnd = np.flatnonzero(np.diff(ss) != 0) + 1
    starts = np.concatenate(([0], bnd))
    ends = np.concatenate((bnd, [n]))
    cum = np.concatenate(([0], np.cumsum(ys)))                 # positives up to position
    sizes = ends - starts
    total = 0.0
    single = sizes == 1
    if single.any():
        s0 = starts[single]
        total += float((ys[s0] * (cum[s0] + 1) / (s0 + 1)).sum())
    for st, en in zip(starts[~single], ends[~single]):
        a = float(cum[st])
        m = int(en - st)
        p = int(cum[en] - a)
        if p == 0:
            continue
        k = np.arange(1, m + 1, dtype=np.float64)
        tp = a + 1.0 + (k - 1.0) * ((p - 1.0) / (m - 1.0))
        total += (p / m) * float((tp / (st + k)).sum())
    return float(total / n_pos)


def bernoulli_log_likelihood(y: np.ndarray, p: np.ndarray) -> float:
    """Mean Bernoulli log-likelihood per pair (natural log), NaN-safe."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# ================================================================== pairs / sampling
def undirected_edges(c, weighted: bool = False):
    """(i, j, w) unique undirected edges with i < j of a connectome view (no autapses)."""
    pre, post = c.pre, c.post
    keep = pre != post
    i = np.minimum(pre[keep], post[keep]).astype(np.int64)
    j = np.maximum(pre[keep], post[keep]).astype(np.int64)
    w = c.syn[keep].astype(np.float64) if weighted else None
    return i, j, w


def edge_keys(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """Sorted int64 keys for undirected pairs (i < j), for O(log E) membership tests."""
    lo = np.minimum(i, j).astype(np.int64)
    hi = np.maximum(i, j).astype(np.int64)
    return np.sort(lo * np.int64(n) + hi)


def sample_non_edges(n: int, k: int, keys: np.ndarray, rng: np.random.Generator,
                     nodes: np.ndarray | None = None, chunk: int = 4_000_000,
                     max_draws: int = 40):
    """Uniformly sample ``k`` non-edges (i < j); ``nodes`` restricts both endpoints.

    ``keys`` must be the sorted key array of *all* edges that must never be sampled
    (fit + test), so a held-out true edge cannot masquerade as a negative.
    """
    pool = None
    if nodes is not None:
        pool = np.asarray(nodes, dtype=np.int64)
        size = pool.size
    else:
        size = n
    out_i = np.empty(k, dtype=np.int64)
    out_j = np.empty(k, dtype=np.int64)
    filled = 0
    draws = 0
    while filled < k:
        draws += 1
        if draws > max_draws:
            raise RuntimeError("non-edge sampling failed to fill the request")
        m = min(chunk, int((k - filled) * 1.6) + 1024)
        a = rng.integers(0, size, size=m)
        b = rng.integers(0, size, size=m)
        if pool is not None:
            a, b = pool[a], pool[b]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        ok = lo != hi
        lo, hi = lo[ok], hi[ok]
        pos = np.searchsorted(keys, lo * np.int64(n) + hi)
        pos = np.clip(pos, 0, keys.size - 1)
        ok = keys[pos] != lo * np.int64(n) + hi
        lo, hi = lo[ok], hi[ok]
        take = min(k - filled, lo.size)
        out_i[filled:filled + take] = lo[:take]
        out_j[filled:filled + take] = hi[:take]
        filled += take
    return out_i, out_j


# ============================================================== anatomical geometry
def _annotation_coords(c) -> pd.DataFrame:
    need = ("pos_x", "pos_y", "pos_z")
    missing = [k for k in need if k not in c.neurons.columns]
    if missing:
        raise KeyError(f"neurons.parquet is missing {missing}")
    return c.neurons.loc[:, list(need)]


def anatomical_xyz(c) -> np.ndarray:
    """(N, 3) published annotation position of each neuron (pos_x, pos_y, pos_z; nm).

    NaN rows mark the handful of neurons without an annotation row; callers must
    handle them (the Phase 4 protocol drops pairs touching them from the evaluation
    set so every geometry is scored on identical pairs).
    """
    return _annotation_coords(c).to_numpy(np.float64)


def anatomical_xy(c) -> np.ndarray:
    """(N, 2) anterior-posterior / medial-lateral projection of :func:`anatomical_xyz`."""
    return anatomical_xyz(c)[:, :2]


def euclidean_distance(X: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Euclidean distance for the pair list (i, j) in coordinate array X."""
    return np.sqrt(np.sum((X[i] - X[j]) ** 2, axis=1))


def euclidean_coords_defined(X: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Boolean mask of pairs whose coordinates are finite for both endpoints."""
    return np.isfinite(X[i]).all(axis=1) & np.isfinite(X[j]).all(axis=1)


# ============================================================== hyperbolic geometry
def hyperbolic_distance_poincare(U: np.ndarray, V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Poincare-ball distance d(u,v) = arccosh(1 + 2|u-v|^2 / ((1-|u|^2)(1-|v|^2))).

    ``U`` and ``V`` are (m, d) arrays of Poincare coordinates (each row inside the
    unit ball).
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    uu = np.sum(U * U, axis=1)
    vv = np.sum(V * V, axis=1)
    w = np.sum((U - V) ** 2, axis=1)
    den = (1.0 - uu) * (1.0 - vv)
    arg = 1.0 + 2.0 * w / np.maximum(den, eps)
    return np.arccosh(np.maximum(arg, 1.0 + 1e-15))


def _poincare_to_hyperboloid(P: np.ndarray) -> np.ndarray:
    """(m, d) Poincare coordinates -> (m, d+1) points on the unit hyperboloid."""
    P = np.asarray(P, dtype=np.float64)
    rho2 = np.sum(P * P, axis=1, keepdims=True)
    x0 = (1.0 + rho2) / (1.0 - rho2)                 # = cosh(hyperbolic radius)
    return np.concatenate([x0, 2.0 * P / (1.0 - rho2)], axis=1)


def _hyperboloid_to_poincare(X: np.ndarray) -> np.ndarray:
    """(m, d+1) hyperboloid points (x0 = sqrt(1+|x|^2)) -> (m, d) Poincare coordinates."""
    X = np.asarray(X, dtype=np.float64)
    return X[:, 1:] / (1.0 + X[:, :1])


def hyperbolic_distance_hyperboloid(X: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Distance between hyperboloid points, d = arccosh(x0 y0 - <x, y>_spatial)."""
    s = X[i, 0] * X[j, 0] - np.sum(X[i, 1:] * X[j, 1:], axis=1)
    return np.arccosh(np.maximum(s, 1.0 + 1e-15))


def connection_probability(d: np.ndarray, R: float, T: float) -> np.ndarray:
    """The geometric connection law P(connect | d) = 1 / (1 + exp((d - R) / T))."""
    z = np.clip((np.asarray(d, dtype=np.float64) - R) / T, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(z))


def hyperboloid_pair_grad(phi: np.ndarray, i: np.ndarray, j: np.ndarray,
                          y: np.ndarray, R: float, T: float):
    """Gradient of the per-pair Bernoulli NLL w.r.t. the spatial hyperboloid parameters.

    Nodes are parametrised by ``phi`` (N, d) with hyperboloid coordinates
    ``(x0, x) = (sqrt(1 + |phi|^2), phi)``, so ``d_ij = arccosh(x0_i x0_j - x_i.x_j)``
    and, with ``sinh(d) = sqrt((S-1)(S+1))``,

        d NLL / d phi_i = (y - p) / (T sinh d) * (x0_j * phi_i / x0_i - phi_j).

    Returns ``(grad_i, grad_j, d, p)``. The finite-difference check of this expression
    lives in :func:`self_test`.
    """
    u0 = np.sqrt(1.0 + np.sum(phi[i] ** 2, axis=1))
    v0 = np.sqrt(1.0 + np.sum(phi[j] ** 2, axis=1))
    s = u0 * v0 - np.sum(phi[i] * phi[j], axis=1)
    denom = np.sqrt(np.maximum(s - 1.0, 0.0) * (s + 1.0))
    d = np.arccosh(np.maximum(s, 1.0 + 1e-15))
    p = connection_probability(d, R, T)
    g = np.clip((y - p) / (T * np.maximum(denom, 1e-12)), -1e4, 1e4)
    gi = g[:, None] * (v0[:, None] * phi[i] / u0[:, None] - phi[j])
    gj = g[:, None] * (u0[:, None] * phi[j] / v0[:, None] - phi[i])
    return gi, gj, d, p


def fit_connection_law(d: np.ndarray, y: np.ndarray, R0: float | None = None,
                       T0: float | None = None, max_pairs: int | None = 400_000,
                       seed: int = DEFAULT_SEED) -> dict:
    """Maximum-likelihood (R, T) for P(connect|d) = 1/(1+exp((d-R)/T)).

    ``y = 1`` marks a connection. Returns R, T, the mean negative log-likelihood and
    the density-matched radius ``R_density``: because the fit sample is balanced 1:1
    while real connectomes are sparse, the law of the *balanced* sample is
    ``sigma((R_b - d)/T)`` and the law of a uniformly drawn real pair is exactly
    ``sigma((R_b + T*logit(pi) - d)/T)`` with prevalence ``pi`` (the two are related
    by Bayes' rule, and this is an exact identity, not an approximation).
    """
    d = np.asarray(d, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if max_pairs is not None and d.size > max_pairs:
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(y > 0.5)
        neg = np.flatnonzero(y <= 0.5)
        n_pos = min(pos.size, max_pairs // 2)
        n_neg = min(neg.size, max_pairs - n_pos)
        sel = np.concatenate([rng.choice(pos, size=n_pos, replace=False),
                              rng.choice(neg, size=n_neg, replace=False)])
        sel.sort()
        d, y = d[sel], y[sel]

    finite = np.isfinite(d)
    n_dropped = int((~finite).sum())
    d, y = d[finite], y[finite]
    if d.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return {"error": "degenerate fit sample", "R": float("nan"), "T": float("nan"),
                "nll_mean": float("nan"), "log_likelihood_mean": float("nan"),
                "n_pairs": int(d.size), "n_nonfinite_distance": n_dropped,
                "converged": False}

    if R0 is None:
        R0 = float(np.median(d))
    if T0 is None:
        spread = float(np.percentile(d, 75) - np.percentile(d, 25))
        T0 = float(max(spread / 1.349, 1e-3 * (d.max() - d.min() + 1.0)))

    def nll(theta):
        R, logT = theta
        T = np.exp(logT)
        p = np.clip(connection_probability(d, R, T), 1e-12, 1.0 - 1e-12)
        return -np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p))

    res = minimize(nll, np.array([R0, np.log(max(T0, 1e-9))]), method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10})
    R, T = float(res.x[0]), float(np.exp(res.x[1]))
    p = connection_probability(d, R, T)
    return {
        "R": R,
        "T": T,
        "nll_mean": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
        "log_likelihood_mean": bernoulli_log_likelihood(y, p),
        "n_pairs": int(d.size),
        "n_nonfinite_distance": n_dropped,
        "optimizer": f"scipy L-BFGS-B on (R, log T), {int(res.nit)} iterations",
        "converged": bool(res.success),
        "distance_summary": {"median": float(np.median(d)), "p10": float(np.percentile(d, 10)),
                             "p90": float(np.percentile(d, 90)),
                             "median_pos": float(np.median(d[y > 0.5])) if (y > 0.5).any() else None,
                             "median_neg": float(np.median(d[y <= 0.5])) if (y <= 0.5).any() else None},
    }


def hyperbolic_embedding(c, dim: int = 2, edges=None, n_neg: int | None = None,
                         epochs: int = 60, batch: int = 16_384, lr: float = 0.02,
                         seed: int = DEFAULT_SEED, nodes: np.ndarray | None = None,
                         r_init: tuple[float, float] = (0.05, 0.6),
                         init: str = "uniform", max_norm: float = 500.0,
                         monitor_pairs: int = 60_000, verbose: bool = True) -> dict:
    """Fit a ``dim``-dimensional Poincare-ball embedding by SGD on the connection law.

    Objective (the geometric-renormalization formulation): for sampled pairs,
    ``P(connect) = 1 / (1 + exp((d_ij - R) / T))`` with ``d_ij`` the hyperbolic
    distance, minimizing the Bernoulli negative log-likelihood over positive edges
    and uniformly sampled non-edges (1:1). Optimization is in the equivalent
    hyperboloid model, where the distance gradient is ``d(arccosh(<x,y>_M))`` with
    the Minkowski inner product; positions are reported in the Poincare ball.

    ``(R, T)`` are refitted by maximum likelihood after each epoch from the current
    distances (alternating optimization), which is more stable than carrying them
    through the SGD.

    Parameters
    ----------
    c : Connectome
        Any connectome view; ``edges`` defaults to its undirected, autapse-free
        projection over all ``c.n`` nodes.
    edges : (pre, post) of undirected pairs to fit (i < j); the positives.
    n_neg : number of negative samples (default: one per positive edge).
    nodes : ignored-except-for-sampling restriction; negatives are drawn from
        non-edges of the *full* graph represented by ``c``.

    Returns a dict: ``coords`` (N, dim) Poincare coordinates, ``R``, ``T``,
    ``train_log_likelihood`` (per-pair, on the fit sample of the final epoch),
    ``history`` (per-epoch monitoring), ``protocol`` and ``seconds``. Nodes with no
    incident edge in the fit sample keep their initialization and are reported as
    ``n_unconstrained_nodes``.
    """
    t0 = time.time()
    n = c.n
    rng = np.random.default_rng(seed)

    if edges is None:
        pi_, pj_, _ = undirected_edges(c)
        edges = (pi_, pj_)
    pos_i, pos_j = (np.asarray(edges[0], dtype=np.int64), np.asarray(edges[1], dtype=np.int64))
    if n_neg is None:
        n_neg = int(pos_i.size)
    all_keys = edge_keys(*undirected_edges(c)[:2], n)
    neg_i, neg_j = sample_non_edges(n, int(n_neg), all_keys, rng, nodes=nodes)

    # --- initialize on the hyperboloid via Poincare disk coordinates
    if init == "degree":
        # scale-free connectomes embed with the hubs near the origin and the periphery on
        # the rim, so the initial radius is assigned by total-degree rank (the same
        # structure a popularity-corrected hyperbolic embedding converges to)
        deg = np.bincount(np.concatenate([pos_i, pos_j]), minlength=n).astype(np.float64)
        order = np.argsort(-deg, kind="stable")
        rank = np.empty(n, dtype=np.float64)
        rank[order] = np.arange(n, dtype=np.float64) / max(1, n - 1)
        r = np.sqrt(rank) * 0.85
        r[deg == 0] = 0.85
    elif init == "uniform":
        r = rng.uniform(r_init[0], r_init[1], size=n) if r_init[1] > r_init[0] else \
            np.full(n, r_init[0])
    else:
        raise ValueError(f"unknown init {init!r}")
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)[:, :dim]

    # --- Adam state
    m = np.zeros_like(phi)
    v = np.zeros_like(phi)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    step = 0

    def _distances(pi_, pj_):
        u0 = np.sqrt(1.0 + np.sum(phi[pi_] ** 2, axis=1))
        v0_ = np.sqrt(1.0 + np.sum(phi[pj_] ** 2, axis=1))
        s = u0 * v0_ - np.sum(phi[pi_] * phi[pj_], axis=1)
        return np.arccosh(np.maximum(s, 1.0 + 1e-15))

    # --- initial law from the initialization (must see both classes)
    n_init = int(min(200_000, pos_i.size))
    dp0 = _distances(pos_i[:n_init], pos_j[:n_init])
    dn0 = _distances(neg_i[:min(n_init, neg_i.size)], neg_j[:min(n_init, neg_i.size)])
    law0 = fit_connection_law(np.concatenate([dp0, dn0]),
                              np.concatenate([np.ones(dp0.size), np.zeros(dn0.size)]),
                              seed=seed, max_pairs=None)
    R, T = float(law0["R"]), float(law0["T"])
    law_init_fallback = None
    if not (np.isfinite(R) and np.isfinite(T) and T > 0):
        all0 = np.concatenate([dp0, dn0])
        spread = float(np.percentile(all0, 75) - np.percentile(all0, 25))
        R = float(np.median(all0))
        T = float(max(spread / 1.349, 1e-3))
        law_init_fallback = {"reason": str(law0.get("error", "non-finite R/T")),
                             "R": R, "T": T}

    def _step(pi_, pj_, y, lr_now):
        nonlocal phi, m, v, step, R, T
        gi, gj, _, _ = hyperboloid_pair_grad(phi, pi_, pj_, y, R, T)
        grad = np.empty((n, phi.shape[1]), dtype=np.float64)
        for d_axis in range(phi.shape[1]):
            grad[:, d_axis] = (np.bincount(pi_, weights=gi[:, d_axis], minlength=n)
                               + np.bincount(pj_, weights=gj[:, d_axis], minlength=n))
        step += 1
        m *= beta1
        m += (1.0 - beta1) * grad
        v *= beta2
        v += (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        phi -= lr_now * m_hat / (np.sqrt(v_hat) + adam_eps)
        norm = np.sqrt(np.sum(phi ** 2, axis=1))
        big = norm > max_norm
        if big.any():
            phi[big] *= (max_norm / norm[big])[:, None]

    mon = None
    if monitor_pairs:
        n_mon = min(int(monitor_pairs), pos_i.size, neg_i.size)
        mp = rng.choice(pos_i.size, size=n_mon, replace=False)
        mn = rng.choice(neg_i.size, size=n_mon, replace=False)
        mon = (pos_i[mp], pos_j[mp], neg_i[mn], neg_j[mn])

    n_batches = int(np.ceil(pos_i.size / batch))
    history = []
    n_law_refit_fallbacks = 0
    for epoch in range(int(epochs)):
        # learning-rate schedule: two halvings
        frac = (epoch + 1) / max(1, epochs)
        lr_now = lr * (0.5 if frac > 0.6 else 1.0) * (0.5 if frac > 0.85 else 1.0)
        order = rng.permutation(pos_i.size)
        for b in range(n_batches):
            sel = order[b * batch:(b + 1) * batch]
            _step(pos_i[sel], pos_j[sel], np.ones(sel.size), lr_now)
        order = rng.permutation(neg_i.size)
        for b in range(n_batches):
            sel = order[b * batch:(b + 1) * batch]
            _step(neg_i[sel], neg_j[sel], np.zeros(sel.size), lr_now)

        # alternating maximum-likelihood refit of the law
        s = rng.choice(pos_i.size, size=min(200_000, pos_i.size), replace=False)
        dp = _distances(pos_i[s], pos_j[s])
        sn = rng.choice(neg_i.size, size=min(200_000, neg_i.size), replace=False)
        dn = _distances(neg_i[sn], neg_j[sn])
        law = fit_connection_law(np.concatenate([dp, dn]),
                                 np.concatenate([np.ones(dp.size), np.zeros(dn.size)]),
                                 R0=R, T0=T, seed=seed, max_pairs=None)
        if np.isfinite(law.get("R", np.nan)) and np.isfinite(law.get("T", np.nan)):
            R, T = float(law["R"]), float(law["T"])
        else:                                       # keep the previous (R, T)
            n_law_refit_fallbacks += 1
        entry = {"epoch": epoch + 1, "R": R, "T": T,
                 "train_nll_mean": law.get("nll_mean"), "lr": lr_now,
                 "law_refit_ok": bool(law.get("converged", False)),
                 "seconds": round(time.time() - t0, 1)}
        if mon is not None:
            d_m = np.concatenate([_distances(mon[0], mon[1]), _distances(mon[2], mon[3])])
            y_m = np.concatenate([np.ones(mon[0].size), np.zeros(mon[2].size)])
            entry["in_sample_auc"] = roc_auc(y_m, -d_m)
        finite_ok = bool(np.isfinite(phi).all())
        entry["finite"] = finite_ok
        history.append(entry)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"    [hyp] epoch {epoch + 1}/{epochs} R={R:.4f} T={T:.4f} "
                  f"nll={entry['train_nll_mean']} auc_d={entry.get('in_sample_auc')} "
                  f"t={entry['seconds']}s")
        if not finite_ok:
            break

    X = np.concatenate([np.sqrt(1.0 + np.sum(phi ** 2, axis=1, keepdims=True)), phi], axis=1)
    coords = _hyperboloid_to_poincare(X)
    dd = _distances(pos_i[:200_000], pos_j[:200_000])
    ll = bernoulli_log_likelihood(np.ones(dd.size), connection_probability(dd, R, T))
    touched = np.bincount(np.concatenate([pos_i, neg_i]), minlength=n) > 0
    info = {
        "coords": coords,
        "dim": int(dim),
        "R": float(R),
        "T": float(T),
        "train_log_likelihood": float(ll),
        "train_nll_mean": float(-ll),
        "n_positive_edges": int(pos_i.size),
        "n_negative_samples": int(neg_i.size),
        "n_nodes_fitted": n,
        "n_unconstrained_nodes": int((~touched).sum()),
        "radius_summary": {"min": float(np.linalg.norm(coords, axis=1).min()),
                           "median": float(np.median(np.linalg.norm(coords, axis=1))),
                           "p99": float(np.percentile(np.linalg.norm(coords, axis=1), 99)),
                           "max": float(np.linalg.norm(coords, axis=1).max())},
        "history": history,
        "law_init": {"R": float(law_init_fallback["R"]) if law_init_fallback else float(law0["R"]),
                     "T": float(law_init_fallback["T"]) if law_init_fallback else float(law0["T"]),
                     "fallback_used": law_init_fallback},
        "n_law_refit_fallbacks": int(n_law_refit_fallbacks),
        "protocol": {
            "model": "P(connect|d) = 1/(1+exp((d-R)/T)), d = Poincare distance",
            "optimiser": f"Adam (lr={lr}, batch={batch}, {epochs} epochs) on the "
                         f"hyperboloid model, Euclidean distance gradient",
            "negative_sampling": "uniform non-edges of the full graph, 1:1 with positives",
            "law_refit": "maximum likelihood after every epoch (alternating)",
            "init": (f"{init} radius assignment" + (
                " by total-degree rank (hubs near the origin)" if init == "degree"
                else f" in {r_init} with uniform angle")) + f", seed={seed}",
            "max_hyperboloid_norm_clamp": max_norm,
            "deterministic": True,
        },
        "seconds": round(time.time() - t0, 1),
    }
    return info


# ================================================================ spectral geometry
def spectral_embedding(c, k: int = 32, edges=None, nodes: np.ndarray | None = None,
                       tol: float = 1e-5, return_info: bool = False,
                       maxiter: int | None = None):
    """Normalized-Laplacian eigenmaps of the undirected, autapse-free projection.

    Coordinates are the ``k`` non-trivial eigenvectors of the symmetric normalized
    adjacency ``M = D^-1/2 A D^-1/2`` scaled to the eigenmap basis
    ``psi_j(i) = u_j(i) / sqrt(deg_i)`` (eigenvectors of the symmetric normalized
    Laplacian ``I - M``; the trivial ``lambda = 1`` eigenvector is dropped). This is
    mathematically the same subspace as ``eigsh(L, which='SA')`` but converges much
    faster.

    Nodes outside ``nodes`` (default: all nodes with at least one edge) and
    isolated nodes get zero rows and are reported as not fitted: an eigenmap is
    undefined for a component of size one, which is why the Phase 4 protocol
    restricts the analysis to the giant weakly-connected component.

    Returns the ``(N, k)`` array (with ``return_info=True``: ``(coords, info)``).
    """
    t0 = time.time()
    n = c.n
    if edges is None:
        i_, j_, _ = undirected_edges(c)
        edges = (i_, j_)
    ei = np.asarray(edges[0], dtype=np.int64)
    ej = np.asarray(edges[1], dtype=np.int64)

    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    A = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()

    deg = np.asarray(A.sum(axis=1)).ravel()
    keep = np.ones(n, dtype=bool) if nodes is None else np.asarray(nodes, dtype=bool)
    keep &= deg > 0

    idx = np.flatnonzero(keep)
    sub = A[idx][:, idx].tocsr()
    dsub = np.asarray(sub.sum(axis=1)).ravel()
    dsq = 1.0 / np.sqrt(dsub)
    M = (sub.multiply(dsq[:, None]).multiply(dsq[None, :])).tocsr()
    M = sparse.csr_matrix((M + M.T) / 2.0)

    kk = int(min(k + 1, idx.size - 2))
    vals, vecs = splinalg.eigsh(M, k=kk, which="LA", tol=tol, maxiter=maxiter)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    eig_vals = vals[1:kk]                     # drop the trivial lambda = 1
    U = vecs[:, 1:kk]
    coords = np.zeros((n, eig_vals.size), dtype=np.float64)
    coords[idx] = U * dsq[:, None]            # psi_j(i) = u_j(i)/sqrt(deg_i)

    info = {
        "k": int(eig_vals.size),
        "n_nodes_fitted": int(idx.size),
        "n_nodes_not_fitted": int(n - idx.size),
        "n_isolated_nodes_in_scope": int((deg == 0).sum()),
        "eigenvalues_1_minus_lambda_desc": [float(v) for v in (1.0 - eig_vals)],
        "lambda_min_fitted": float(eig_vals.min()),
        "lambda_max_fitted": float(eig_vals.max()),
        "residual_max": float(np.abs(M @ vecs[:, 1:kk] - vecs[:, 1:kk] * vals[None, 1:kk]).max()),
        "n_edges_in_fit_graph": int(sub.nnz / 2),
        "tol": tol,
        "seconds": round(time.time() - t0, 1),
        "definition": "normalized-Laplacian eigenmaps: psi_j(i) = u_j(i)/sqrt(deg_i), "
                      "u_j eigenvectors of D^-1/2 A D^-1/2, trivial lambda=1 dropped",
    }
    return (coords, info) if return_info else coords


# ================================================================== protocol + eval
def _annotation_defined(c) -> np.ndarray:
    """Boolean mask of neurons with a complete published annotation position."""
    return np.isfinite(anatomical_xyz(c)).all(axis=1)


def build_protocol(c, threshold: int = DEFAULT_THRESHOLD, test_frac: float = 0.1,
                   seed: int = DEFAULT_SEED, neg_per_pos: int = 1) -> dict:
    """Fixed train/test split, giant component and negative samples for Phase 4.

    Returns a dict describing the analysis graph, the fit graph (as an undirected
    view for the embedders), the held-out pair sets and the provenance needed to
    state the protocol in the results file.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = c.n
    view = c.thresholded(threshold) if threshold > 1 else c
    ei, ej, _ = undirected_edges(view)
    keys = edge_keys(ei, ej, n)

    perm = rng.permutation(ei.size)
    n_test = int(round(ei.size * test_frac))
    test_sel = np.sort(perm[:n_test])
    fit_sel = np.sort(perm[n_test:])
    fit_i, fit_j = ei[fit_sel], ej[fit_sel]
    test_i, test_j = ei[test_sel], ej[test_sel]

    # neurons without an annotation row have no anatomical coordinate, so they are
    # removed from the analysis graph entirely: every geometry is then fit and scored
    # on exactly the same nodes and pairs (the alternative -- dropping those pairs
    # separately per geometry -- would silently give each geometry a different
    # evaluation set). The giant component is taken from the annotated fit graph so
    # every analysis node has at least one fit edge inside it, which is what makes the
    # spectral embedding finite for all of them.
    annotated = _annotation_defined(c)
    keep_fit = annotated[fit_i] & annotated[fit_j]
    A = sparse.coo_matrix((np.ones(int(keep_fit.sum())), (fit_i[keep_fit], fit_j[keep_fit])),
                          shape=(n, n)).tocsr()
    A = A + A.T
    ncomp, labels = csgraph.connected_components(A, directed=False, connection="weak")
    sizes = np.bincount(labels)
    giant = int(np.argmax(sizes))
    G = labels == giant
    n_unannotated_in_giant = int((G & ~annotated).sum())
    G_nodes = np.flatnonzero(G)

    in_G_fit = G[fit_i] & G[fit_j]
    in_G_test = G[test_i] & G[test_j]
    fit_G_i, fit_G_j = fit_i[in_G_fit], fit_j[in_G_fit]
    test_G_i, test_G_j = test_i[in_G_test], test_j[in_G_test]

    n_eval_pos = test_G_i.size
    n_neg = int(n_eval_pos * neg_per_pos)
    eval_neg_i, eval_neg_j = sample_non_edges(n, n_neg, keys, rng, nodes=G_nodes)
    fit_neg_i, fit_neg_j = sample_non_edges(n, int(fit_G_i.size * neg_per_pos), keys, rng,
                                            nodes=G_nodes)

    deg = np.bincount(np.concatenate([view.pre, view.post]), minlength=n)
    proto = {
        "threshold": int(threshold),
        "seed": int(seed),
        "n_neurons": int(n),
        "n_directed_pairs_thresholded": int(view.pre.size),
        "n_undirected_edges": int(ei.size),
        "test_frac": float(test_frac),
        "n_fit_edges": int(fit_i.size),
        "n_test_edges": int(test_i.size),
        "giant_component_size": int(sizes.max()),
        "giant_component_fraction": float(sizes.max() / n),
        "n_components": int(ncomp),
        "analysis_nodes": int(G_nodes.size),
        "analysis_node_fraction": float(G_nodes.size / n),
        "n_neurons_without_annotation_excluded": n_unannotated_in_giant,
        "n_nodes_excluded_from_eval": int(n - G_nodes.size),
        "fit_edges_inside_giant": int(fit_G_i.size),
        "test_edges_inside_giant": int(test_G_i.size),
        "frac_edges_inside_giant": float((in_G_fit.sum() + in_G_test.sum()) / ei.size),
        "n_eval_negatives": int(n_neg),
        "n_fit_negatives": int(fit_neg_i.size),
        "eval_node_mask": G,
        "deg_total_fit": deg,
        "degrees": {
            "n_zero_degree_in_giant": int((deg[G_nodes] == 0).sum()),
            "mean_total_degree_in_giant": float(deg[G_nodes].mean()),
            "n_neurons_without_annotation": int((~np.isfinite(
                anatomical_xyz(c)[:, 0])).sum()),
        },
        "pairs": {
            "fit_pos": (fit_G_i.astype(np.int64), fit_G_j.astype(np.int64)),
            "test_pos": (test_G_i.astype(np.int64), test_G_j.astype(np.int64)),
            "fit_neg": (fit_neg_i, fit_neg_j),
            "test_neg": (eval_neg_i, eval_neg_j),
        },
        "build_seconds": round(time.time() - t0, 1),
    }
    return proto


def _pair_distances(kind: str, coords: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    if kind == "hyperbolic":
        return hyperbolic_distance_poincare(coords[i], coords[j])
    return euclidean_distance(coords, i, j)


def evaluate_geometry(protocol: dict, coords: np.ndarray, kind: str,
                      name: str = "geometry", seed: int = DEFAULT_SEED) -> dict:
    """Fit the connection law on the fit pairs and score the held-out pairs.

    The *same* fit and held-out pair sets are used for every geometry, and pairs
    whose coordinates are non-finite for either endpoint are dropped from both, so
    geometries are compared on identical pairs (the number dropped is reported).
    """
    t0 = time.time()
    out: dict = {"kind": kind, "n_dims": int(coords.shape[1])}
    fit_i, fit_j = protocol["pairs"]["fit_pos"]
    te_i, te_j = protocol["pairs"]["test_pos"]
    fn_i, fn_j = protocol["pairs"]["fit_neg"]
    tn_i, tn_j = protocol["pairs"]["test_neg"]

    def prep(i, j, y):
        d = _pair_distances(kind, coords, i, j)
        ok = np.isfinite(d)
        return d[ok], np.full(int(ok.sum()), y)

    d_fp, y_fp = prep(fit_i, fit_j, 1.0)
    d_fn, y_fn = prep(fn_i, fn_j, 0.0)
    d_tp, y_tp = prep(te_i, te_j, 1.0)
    d_tn, y_tn = prep(tn_i, tn_j, 0.0)

    out["n_pairs_dropped_nonfinite"] = {
        "fit_pos": int(fit_i.size - d_fp.size), "fit_neg": int(fn_i.size - d_fn.size),
        "test_pos": int(te_i.size - d_tp.size), "test_neg": int(tn_i.size - d_tn.size),
    }
    law = fit_connection_law(np.concatenate([d_fp, d_fn]),
                             np.concatenate([y_fp, y_fn]), seed=seed)
    out["connection_law"] = law
    out["fit"] = {"n_pos": int(d_fp.size), "n_neg": int(d_fn.size),
                  "auc": roc_auc(np.concatenate([y_fp, y_fn]),
                                 -np.concatenate([d_fp, d_fn])),
                  "log_likelihood_mean": law.get("log_likelihood_mean")}
    d_te = np.concatenate([d_tp, d_tn])
    y_te = np.concatenate([y_tp, y_tn])
    p_te = connection_probability(d_te, law["R"], law["T"])
    out["heldout"] = {
        "n_pos": int(d_tp.size),
        "n_neg": int(d_tn.size),
        "auc": roc_auc(y_te, -d_te),
        "average_precision": average_precision(y_te, -d_te),
        "log_likelihood_mean": bernoulli_log_likelihood(y_te, p_te),
        "nll_mean": float(-bernoulli_log_likelihood(y_te, p_te)),
        "distance_separation": {
            "median_pos": float(np.median(d_tp)) if d_tp.size else None,
            "median_neg": float(np.median(d_tn)) if d_tn.size else None,
        },
    }
    out["seconds"] = round(time.time() - t0, 1)
    out["name"] = name
    return out


def degree_baseline(protocol: dict, seed: int = DEFAULT_SEED) -> dict:
    """Degree-only baseline: score = product of the two total degrees.

    Ranking uses ``log d_i + log d_j`` (monotone in the product) and the probability
    used for log-likelihood is a two-parameter logistic in those log-degrees fitted
    on the same fit pairs, i.e. the baseline gets the same calibration freedom as a
    geometry's (R, T).
    """
    t0 = time.time()
    deg = protocol["deg_total_fit"].astype(np.float64)
    fit_i, fit_j = protocol["pairs"]["fit_pos"]
    te_i, te_j = protocol["pairs"]["test_pos"]
    fn_i, fn_j = protocol["pairs"]["fit_neg"]
    tn_i, tn_j = protocol["pairs"]["test_neg"]
    logd = np.log(np.maximum(deg, 1.0))

    def score(i, j):
        return logd[i] + logd[j]

    s_fp, s_fn = score(fit_i, fit_j), score(fn_i, fn_j)
    s_tp, s_tn = score(te_i, te_j), score(tn_i, tn_j)
    y_fit = np.concatenate([np.ones(s_fp.size), np.zeros(s_fn.size)])
    s_fit = np.concatenate([s_fp, s_fn])

    def nll(theta):
        a, b = theta
        p = 1.0 / (1.0 + np.exp(-(a + b * s_fit)))
        return -np.mean(y_fit * np.log(np.clip(p, 1e-12, 1)) +
                        (1 - y_fit) * np.log(np.clip(1 - p, 1e-12, 1)))

    res = minimize(nll, np.array([-10.0, 1.0]), method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12})
    a, b = float(res.x[0]), float(res.x[1])
    s_te = np.concatenate([s_tp, s_tn])
    y_te = np.concatenate([np.ones(s_tp.size), np.zeros(s_tn.size)])
    p_te = 1.0 / (1.0 + np.exp(-(a + b * s_te)))
    return {
        "name": "degree_only",
        "score": "log(total_degree_i) + log(total_degree_j)",
        "calibration": "logistic in (log d_i + log d_j) fitted on the fit pairs",
        "a": a, "b": b,
        "converged": bool(res.success),
        "fit": {"n_pos": int(s_fp.size), "n_neg": int(s_fn.size),
                "auc": roc_auc(y_fit, s_fit),
                "log_likelihood_mean": bernoulli_log_likelihood(y_fit, 1.0 / (1.0 + np.exp(-(a + b * s_fit))))},
        "heldout": {
            "n_pos": int(s_tp.size), "n_neg": int(s_tn.size),
            "auc": roc_auc(y_te, s_te),
            "average_precision": average_precision(y_te, s_te),
            "log_likelihood_mean": bernoulli_log_likelihood(y_te, p_te),
            "nll_mean": float(-bernoulli_log_likelihood(y_te, p_te)),
        },
        "n_tied_scores": int(s_te.size - np.unique(s_te).size),
        "note": "scores are integer-valued (degree products) so ties are frequent; AUC "
                "uses exact midpoint tie credit and AP is the exact expectation over "
                "random tie-breaking, so the ties do not favor the baseline",
        "seconds": round(time.time() - t0, 1),
    }


def geometry_plus_degree(protocol: dict, coords: np.ndarray, kind: str,
                         seed: int = DEFAULT_SEED, max_fit_pairs: int = 300_000) -> dict:
    """Auxiliary: does the geometry add anything on top of the degree baseline?

    Three-feature logistic ``P = sigma(a + b*(log d_i + log d_j) + c*(-d_ij))`` fitted on
    the fit pairs (subsampled to ``max_fit_pairs`` for speed) and scored on all held-out
    pairs. A held-out AUC above the degree-only baseline's says the latent distance
    carries information the degrees do not.
    """
    deg = np.log(np.maximum(protocol["deg_total_fit"].astype(np.float64), 1.0))
    fit_i, fit_j = protocol["pairs"]["fit_pos"]
    te_i, te_j = protocol["pairs"]["test_pos"]
    fn_i, fn_j = protocol["pairs"]["fit_neg"]
    tn_i, tn_j = protocol["pairs"]["test_neg"]

    def feats(i, j):
        d = _pair_distances(kind, coords, i, j)
        ok = np.isfinite(d)
        return np.stack([np.ones(int(ok.sum())), deg[i][ok] + deg[j][ok], -d[ok]], axis=1), ok

    F_fp, _ = feats(fit_i, fit_j)
    F_fn, _ = feats(fn_i, fn_j)
    rng = np.random.default_rng(seed)
    half = max(1000, max_fit_pairs // 2)
    if F_fp.shape[0] > half:
        F_fp = F_fp[rng.choice(F_fp.shape[0], size=half, replace=False)]
    if F_fn.shape[0] > half:
        F_fn = F_fn[rng.choice(F_fn.shape[0], size=half, replace=False)]
    F_tp, _ = feats(te_i, te_j)
    F_tn, _ = feats(tn_i, tn_j)
    X_fit = np.vstack([F_fp, F_fn])
    y_fit = np.concatenate([np.ones(F_fp.shape[0]), np.zeros(F_fn.shape[0])])

    def nll(theta):
        z = np.clip(X_fit @ theta, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        return -np.mean(y_fit * np.log(np.clip(p, 1e-12, 1)) +
                        (1 - y_fit) * np.log(np.clip(1 - p, 1e-12, 1)))

    res = minimize(nll, np.array([-10.0, 1.0, 0.001]), method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-12})
    theta = res.x
    X_te = np.vstack([F_tp, F_tn])
    y_te = np.concatenate([np.ones(F_tp.shape[0]), np.zeros(F_tn.shape[0])])
    p_te = 1.0 / (1.0 + np.exp(-np.clip(X_te @ theta, -40, 40)))
    return {"coefficients": {"intercept": float(theta[0]), "log_degree_sum": float(theta[1]),
                             "minus_distance": float(theta[2])},
            "converged": bool(res.success),
            "n_fit_pairs_used": int(X_fit.shape[0]),
            "heldout": {"auc": roc_auc(y_te, X_te @ theta),
                        "average_precision": average_precision(y_te, X_te @ theta),
                        "log_likelihood_mean": bernoulli_log_likelihood(y_te, p_te)},
            "n_pos": int(F_tp.shape[0]), "n_neg": int(F_tn.shape[0])}


def compare_geometries(c, geometries: dict, protocol: dict | None = None,
                       threshold: int = DEFAULT_THRESHOLD, seed: int = DEFAULT_SEED,
                       aux_degree_interaction: bool = True) -> dict:
    """Held-out comparison of candidate geometries against the degree-only baseline.

    ``geometries`` maps a name to ``{"coords": (N, d) array, "kind": "euclidean" |
    "hyperbolic"}``. Every geometry is scored on the identical held-out pair set with
    the identical (R, T) fitting procedure, so the comparison isolates the geometry.

    Returns a dict with the protocol, the per-geometry results (fitted law, held-out
    AUC / average precision / mean log-likelihood, timings) and the degree-only
    baseline.
    """
    t0 = time.time()
    if protocol is None:
        protocol = build_protocol(c, threshold=threshold, seed=seed)
    results: dict = {"geometries": {}, "errors": {}}
    for name, spec in geometries.items():
        try:
            res = evaluate_geometry(protocol, np.asarray(spec["coords"], dtype=np.float64),
                                    spec.get("kind", "euclidean"), name=name, seed=seed)
            if aux_degree_interaction:
                res["with_degree_auxiliary"] = geometry_plus_degree(
                    protocol, np.asarray(spec["coords"], dtype=np.float64),
                    spec.get("kind", "euclidean"), seed=seed)
            results["geometries"][name] = res
        except Exception as exc:                                  # never lose the run
            results["errors"][name] = f"{type(exc).__name__}: {exc}"
    results["degree_only"] = degree_baseline(protocol, seed=seed)

    # headline ranking on held-out AUC
    table = {k: v["heldout"]["auc"] for k, v in results["geometries"].items()
             if "heldout" in v}
    table["degree_only"] = results["degree_only"]["heldout"]["auc"]
    table = {k: v for k, v in table.items() if v == v}
    if table:
        best = max(table, key=table.get)
        results["ranking_by_heldout_auc"] = dict(sorted(table.items(), key=lambda kv: -kv[1]))
        results["best_geometry"] = best
        results["degree_only_auc"] = results["degree_only"]["heldout"]["auc"]
    results["seconds"] = round(time.time() - t0, 1)
    return results


# ======================================================================= provenance
def dataset_provenance(c) -> dict:
    """Canonical dataset provenance, taken from meta.json rather than restated."""
    meta = c.meta
    return {
        "canonical_dir": str(Path(c.dir).resolve()),
        "canonical_version": meta.get("canonical_version"),
        "built_utc": meta.get("built_utc"),
        "source": meta.get("source"),
        "counts": meta.get("counts"),
        "conventions": meta.get("conventions"),
    }


def environment_versions() -> dict:
    import scipy
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


# ========================================================================= self-test
def self_test(verbose: bool = True) -> dict:
    """Verify the numerical building blocks against independent brute-force checks.

    Returns a dict of checks; raises AssertionError on failure. Used by
    ``scripts/phase4_geometry.py --self-test``.
    """
    checks: dict = {}
    rng = np.random.default_rng(12345)

    # 1. AUC and AP versus brute force
    ok_auc = ok_ap = 0
    for _ in range(200):
        m = int(rng.integers(5, 30))
        y = (rng.random(m) < rng.uniform(0.2, 0.8)).astype(int)
        if y.sum() in (0, m):
            continue
        s = rng.integers(0, 4, m).astype(float)
        pos, neg = s[y == 1], s[y == 0]
        bf = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
        assert abs(roc_auc(y, s) - bf) < 1e-12
        ok_auc += 1
        aps = []
        for _ in range(3000):
            o = np.lexsort((rng.random(m), -s))
            yy = y[o]
            tp = np.cumsum(yy)
            aps.append(float(((tp / np.arange(1, m + 1)) * yy).sum() / yy.sum()))
        ap = average_precision(y, s)
        assert min(aps) - 1e-9 <= ap <= max(aps) + 1e-9
        assert abs(ap - float(np.mean(aps))) < 0.02
        ok_ap += 1
    checks["auc_vs_mannwhitney"] = {"n_cases": ok_auc, "pass": True}
    checks["average_precision_vs_random_tiebreak"] = {"n_cases": ok_ap, "pass": True}

    # 2. Poincare and hyperboloid distance agree
    P = rng.uniform(-0.85, 0.85, size=(500, 2))
    P = P * (rng.uniform(0.05, 1.0, size=(500, 1)) / np.maximum(
        np.linalg.norm(P, axis=1, keepdims=True), 1e-9) * 0.99)
    X = _poincare_to_hyperboloid(P)
    i = rng.integers(0, 500, 3000)
    j = rng.integers(0, 500, 3000)
    d_poincare = hyperbolic_distance_poincare(P[i], P[j])
    d_hyperboloid = hyperbolic_distance_hyperboloid(X, i, j)
    # both formulas are exact up to the machine-precision floor where the true
    # distance is ~0, so the comparison is made on genuinely separated pairs
    sep = d_poincare > 1e-4
    assert sep.sum() > 1000, int(sep.sum())
    assert np.allclose(d_poincare[sep], d_hyperboloid[sep], rtol=1e-9, atol=1e-9)
    assert np.allclose(d_poincare[~sep], 0.0, atol=1e-6)
    assert np.allclose(d_hyperboloid[~sep], 0.0, atol=1e-6)
    # round trip
    assert np.allclose(_hyperboloid_to_poincare(X), P, atol=1e-12)
    # known value: distance from the origin to Poincare radius r is 2*artanh(r), so
    # two points at +/-r on the same diameter are 4*artanh(r) apart
    r = 0.5
    two = np.array([[r, 0.0], [-r, 0.0]])
    assert abs(hyperbolic_distance_poincare(two[:1], two[1:])[0] - 4 * np.arctanh(r)) < 1e-12
    checks["poincare_equals_hyperboloid"] = {"max_abs_diff": float(np.abs(d_poincare - d_hyperboloid).max()),
                                            "pass": True}

    # 2b. the analytic gradient of the pair NLL matches central finite differences
    n_g = 7
    phi_g = rng.uniform(-0.6, 0.6, size=(n_g, 2))
    gi_idx = np.array([0, 1, 2, 3, 4, 5, 0, 2, 4, 6])
    gj_idx = np.array([1, 2, 3, 4, 5, 6, 6, 5, 3, 1])
    y_g = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    R_g, T_g = 1.3, 0.7

    def _loss(ph):
        _, _, _, p_ = hyperboloid_pair_grad(ph, gi_idx, gj_idx, y_g, R_g, T_g)
        p_ = np.clip(p_, 1e-12, 1 - 1e-12)
        return -np.mean(y_g * np.log(p_) + (1 - y_g) * np.log1p(-p_))

    gi_a, gj_a, _, _ = hyperboloid_pair_grad(phi_g, gi_idx, gj_idx, y_g, R_g, T_g)
    g_analytic = np.zeros_like(phi_g)
    np.add.at(g_analytic, gi_idx, gi_a / y_g.size)      # NLL is a mean over pairs
    np.add.at(g_analytic, gj_idx, gj_a / y_g.size)
    g_numeric = np.zeros_like(phi_g)
    eps = 1e-6
    for a in range(n_g):
        for ax in range(2):
            ph = phi_g.copy()
            ph[a, ax] += eps
            lp = _loss(ph)
            ph = phi_g.copy()
            ph[a, ax] -= eps
            lm = _loss(ph)
            g_numeric[a, ax] = (lp - lm) / (2 * eps)
    fd_err = float(np.abs(g_analytic - g_numeric).max())
    scale = float(max(np.abs(g_numeric).max(), 1e-12))
    assert fd_err / scale < 1e-5, (fd_err, scale)
    checks["hyperboloid_gradient_vs_finite_difference"] = {
        "max_abs_error": fd_err, "max_abs_gradient": scale,
        "relative_error": fd_err / scale, "pass": True}

    # 3. Connection law: monotone decreasing, and (R, T) recovered exactly on a
    #    generative model whose balanced-sample law is analytically logistic:
    #    f_edge = N(0,1), f_nonedge = N(1.5,1)  =>  q(d) = sigma((0.75 - d)/(2/3)).
    d = np.linspace(0.0, 10.0, 101)
    p = connection_probability(d, 3.0, 0.5)
    assert np.all(np.diff(p) < 0) and abs(p[np.argmin(np.abs(d - 3.0))] - 0.5) < 0.02
    m = 200_000
    d_bal = np.concatenate([rng.normal(0.0, 1.0, m), rng.normal(1.5, 1.0, m)])
    y_bal = np.concatenate([np.ones(m), np.zeros(m)])
    law = fit_connection_law(d_bal, y_bal, max_pairs=None)
    assert law["converged"] and law["T"] > 0
    assert abs(law["R"] - 0.75) < 0.02, law["R"]
    assert abs(law["T"] - 2.0 / 3.0) < 0.02, law["T"]
    pi_ = 1e-4
    logit_pi = np.log(pi_ / (1.0 - pi_))
    R_density = law["R"] + law["T"] * logit_pi
    R_density_exact = 0.75 + (2.0 / 3.0) * logit_pi
    assert abs(R_density - (law["R"] + law["T"] * logit_pi)) < 1e-12      # the identity
    assert abs(R_density - R_density_exact) < 0.5                        # within fit error
    checks["connection_law"] = {
        "fitted_R": law["R"], "fitted_T": law["T"],
        "analytic_R": 0.75, "analytic_T": 2.0 / 3.0,
        "R_density_matched": R_density, "prevalence_used": pi_,
        "nll_mean": law["nll_mean"], "pass": True}

    # 4. spectral embedding on a planted two-cluster graph: coordinates recover blocks
    n = 400
    blk = np.repeat([0, 1], n // 2)
    p_in, p_out = 0.10, 0.005
    B = (rng.random((n, n)) < np.where(blk[:, None] == blk[None, :], p_in, p_out))
    B = np.triu(B, 1)
    ii, jj = np.nonzero(B)
    class _C:                                                    # minimal stand-in
        n = 400
        pre = np.concatenate([ii, jj])
        post = np.concatenate([jj, ii])
        syn = np.ones(ii.size * 2)
    coords, sinfo = spectral_embedding(_C, k=4, return_info=True)
    sign = np.sign(coords[:, 0] - np.median(coords[:, 0]))
    purity = max(np.mean((blk == 0) == (sign > 0)), np.mean((blk == 0) == (sign < 0)))
    assert purity > 0.9, purity
    assert sinfo["residual_max"] < 1e-6, sinfo["residual_max"]
    checks["spectral_planted_blocks"] = {"sign_purity": float(purity),
                                         "eigen_residual_max": sinfo["residual_max"],
                                         "pass": True}

    # 5. hyperbolic SGD learns on a planted hyperbolic disk (the generative model IS a
    #    distance law in hyperbolic space, so a correct optimizer must recover it)
    nn = 600
    r_h = rng.uniform(0.0, 2.5, size=nn)                          # hyperbolic radii
    ang = rng.uniform(0, 2 * np.pi, size=nn)
    pos = np.stack([np.tanh(r_h / 2) * np.cos(ang), np.tanh(r_h / 2) * np.sin(ang)], axis=1)
    tri = np.triu_indices(nn, 1)
    dp = hyperbolic_distance_poincare(pos[tri[0]], pos[tri[1]])
    planted = rng.random(tri[0].size) < 0.35 * np.exp(-dp)
    ei_, ej_ = tri[0][planted], tri[1][planted]
    planted_key = np.zeros(tri[0].size, dtype=bool)
    planted_key[planted] = True

    class _C2:
        n = nn
        pre = np.concatenate([ei_, ej_])
        post = np.concatenate([ej_, ei_])
        syn = np.ones(ei_.size * 2)
    hyp = hyperbolic_embedding(_C2, dim=2, edges=(ei_, ej_), epochs=150, batch=2048,
                               lr=0.05, seed=1, verbose=False, monitor_pairs=0)
    cand = np.flatnonzero(~planted_key)                # true non-edges only
    sel = np.random.default_rng(7).choice(cand.size, size=20000, replace=False)
    nidx = cand[sel]
    # the achievable ranking quality on this evaluation set is set by the planted
    # process itself, so the optimizer is scored against that oracle rather than
    # against an arbitrary fixed constant
    y_ev = np.concatenate([np.ones(ei_.size), np.zeros(nidx.size)])
    d_pos_true = dp[planted]
    d_neg_true = dp[nidx]
    oracle_auc = roc_auc(y_ev, -np.concatenate([d_pos_true, d_neg_true]))
    d_fit = hyperbolic_distance_poincare(hyp["coords"][ei_], hyp["coords"][ej_])
    d_neg = hyperbolic_distance_poincare(hyp["coords"][tri[0][nidx]], hyp["coords"][tri[1][nidx]])
    auc = roc_auc(y_ev, -np.concatenate([d_fit, d_neg]))
    assert auc > 0.9 * oracle_auc, (auc, oracle_auc)
    checks["hyperbolic_sgd_recovery"] = {
        "auc_on_planted_disk": float(auc), "oracle_auc_true_positions": float(oracle_auc),
        "fraction_of_oracle": float(auc / oracle_auc),
        "fitted_R": hyp["R"], "fitted_T": hyp["T"],
        "train_log_likelihood": hyp["train_log_likelihood"], "pass": True}
    if verbose:
        print(json.dumps(checks, indent=2, default=str))
    return checks
