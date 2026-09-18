"""Geometric renormalization and inverse renormalization (PROJECT-VYBFLY.md §11-§12).

Two operations, both driven by the latent geometry rather than by node index or anatomy:

  coarse_grain(g, geometry, factor)  -> 0.5x / 0.25x / 0.1x replicas
      Groups neurons that are close in the latent space (exact hyperbolic / Euclidean
      nearest-neighbour agglomeration over the geometry), collapses each group into one
      supernode, and re-establishes connections between groups. The coarse-edge threshold is
      calibrated so the replica keeps the source graph's mean degree - i.e. the reduction
      preserves biological sparsity instead of drifting dense.

  upscale(g, geometry, factor)       -> 2x / 5x / 10x artificially enlarged connectomes
      A source neuron becomes `factor` children sitting near the parent in the latent
      geometry. Each child inherits the parent's partner *set* (weights preserved) with the
      child-to-child target chosen by the geometric connection law, which keeps mean degree
      and mean connection strength bounded while every count scales linearly with N - the
      property §12 insists on (E proportional to N, never N^2).

Everything is seeded and reversible: `GraphView.parent_index` records the lineage so a
subdivided graph can be coarse-grained back along its own tree, which is exactly the closure
test of §13.

Hyperbolic formulas use the Poincare ball with curvature -1:
    d(x, y) = arcosh(1 + 2|x-y|^2 / ((1-|x|^2)(1-|y|^2)))
    exp_x(tanh(l/2) u) moves x a distance l in direction u
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .synthetic import GraphView, compact_pairs


# --------------------------------------------------------------------------- geometry
def poincare_distance_block(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise hyperbolic distances between rows of x (m,d) and y (k,d) -> (m,k)."""
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    diff = x2 + y2 - 2.0 * (x @ y.T)
    denom = np.maximum((1.0 - x2) * (1.0 - y2), 1e-12)
    arg = 1.0 + 2.0 * diff / denom
    return np.arccosh(np.maximum(arg, 1.0))


def euclidean_distance_block(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * (x @ y.T)
    return np.sqrt(np.maximum(diff, 0.0))


def poincare_exp_map(base: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exponential map at `base` (m,d) for tangent vectors `v` (m,d)."""
    v2 = np.sum(v * v, axis=1, keepdims=True)
    bv = np.sum(base * v, axis=1, keepdims=True)
    b2 = np.sum(base * base, axis=1, keepdims=True)
    num = (1.0 + 2.0 * bv + v2) * base + (1.0 - b2) * v
    den = 1.0 + 2.0 * bv + b2 * v2
    out = num / den
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(norm >= 1.0, out * (0.999 / np.maximum(norm, 1e-12)), out)


@dataclass
class GeometryLaw:
    """Latent coordinates plus the fitted connection-probability law P = 1/(1+e^((d-R)/T))."""

    coords: np.ndarray
    kind: str = "euclidean"           # 'euclidean' | 'hyperbolic'
    R: float = 1.0
    T: float = 0.2
    source: str = "unspecified"       # where the coordinates came from (provenance)

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])

    @property
    def dim(self) -> int:
        return int(self.coords.shape[1])

    def distance(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Distance for index arrays (element-wise broadcasting)."""
        x, y = self.coords[i], self.coords[j]
        if self.kind == "hyperbolic":
            return poincare_distance_block(x, y).diagonal()
        return np.linalg.norm(x - y, axis=1)

    def distance_block(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        x, y = self.coords[i], self.coords[j]
        return poincare_distance_block(x, y) if self.kind == "hyperbolic" \
            else euclidean_distance_block(x, y)

    def connect_prob(self, d: np.ndarray | float) -> np.ndarray:
        z = (np.asarray(d, dtype=np.float64) - self.R) / max(self.T, 1e-9)
        # clamp before exp: distances far beyond R/T overflow exp and numpy warns; the
        # probability is 0 (or 1) there anyway
        return 1.0 / (1.0 + np.exp(np.clip(z, -700.0, 700.0)))

    def note(self) -> dict:
        return {"kind": self.kind, "dim": self.dim, "R": self.R, "T": self.T,
                "source": self.source, "n": self.n}


def typical_neighbour_distance(law: GeometryLaw, sample: int = 4000, k: int = 2,
                               seed: int = 0) -> float:
    """Median distance to the k-th nearest latent neighbour - the natural sub-division scale."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(law.n, size=min(sample, law.n), replace=False)
    d = law.distance_block(idx, idx)                       # (m, m) among the sample
    np.fill_diagonal(d, np.inf)
    kth = np.partition(d, k - 1, axis=1)[:, k - 1]
    return float(np.median(kth))


# --------------------------------------------------------------------------- grouping
def _candidate_neighbours(coords: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours in the ambient Euclidean space (approximate)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    kk = min(k + 1, coords.shape[0])
    _, nbr = tree.query(coords, k=kk)
    return np.atleast_2d(nbr)


def group_by_geometry(law: GeometryLaw, group_size: int, seed: int = 0,
                      candidate_pool: int = 48) -> tuple[np.ndarray, int]:
    """Partition neurons into groups of `group_size` using true latent distances.

    Candidates come from an ambient-Euclidean KD-tree (cheap), then are ranked by the true
    hyperbolic/Euclidean distance, so the grouping is exact among the candidate pool.
    Greedy and seeded: nodes are visited in a seeded random order, each unassigned node seeds
    a group and absorbs its nearest unassigned candidates.
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    rng = np.random.default_rng(seed)
    n = law.n
    nbr = _candidate_neighbours(law.coords, candidate_pool)
    order = rng.permutation(n)
    assignment = np.full(n, -1, dtype=np.int64)
    gid = 0
    for node in order:
        if assignment[node] != -1:
            continue
        cand = nbr[node]
        cand = cand[(cand != node) & (assignment[cand] == -1)]
        if cand.size == 0:
            assignment[node] = gid
            gid += 1
            continue
        d = law.distance(np.full(cand.size, node, dtype=np.int64), cand)
        take = cand[np.argsort(d)[:group_size - 1]]
        assignment[node] = gid
        assignment[take] = gid
        gid += 1
    return assignment, gid


def _aggregate_attributes(g: GraphView, groups: np.ndarray, n_groups: int,
                          law: GeometryLaw) -> dict:
    """Supernode attributes: latent centroid, dominant annotations, representative root id."""
    order = np.argsort(groups, kind="stable")
    counts = np.bincount(groups, minlength=n_groups)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    coords = np.zeros((n_groups, law.coords.shape[1]), dtype=np.float64)
    np.add.at(coords, groups, law.coords)
    coords = coords / np.maximum(counts, 1)[:, None]
    if law.kind == "hyperbolic":
        norm = np.linalg.norm(coords, axis=1, keepdims=True)
        coords = np.where(norm >= 1.0, coords * (0.999 / np.maximum(norm, 1e-12)), coords)

    def dominant(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        vals = np.asarray(values, dtype=object)
        out = np.empty(n_groups, dtype=object)
        for gi in range(n_groups):
            seg = vals[order[starts[gi]:starts[gi] + counts[gi]]]
            uniq, cnt = np.unique(seg.astype(str), return_counts=True)
            out[gi] = uniq[int(np.argmax(cnt))]
        return out

    root_ids = np.full(n_groups, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(root_ids, groups, g.root_ids)
    return {
        "coords": coords,
        "cell_type": dominant(g.cell_type),
        "super_class": dominant(g.super_class),
        "top_nt": dominant(g.top_nt),
        "root_ids": root_ids,
        "sizes": counts,
    }


def fit_connection_law(coords: np.ndarray, pre: np.ndarray, post: np.ndarray,
                       kind: str = "euclidean", n_samples: int = 400_000,
                       seed: int = 0) -> dict:
    """Fit P(connect) = 1/(1+exp((d-R)/T)) to a distance-vs-connection scatter.

    Positive samples are real edges; negatives are uniformly drawn node pairs with no edge in
    either direction. The fit is a one-dimensional logistic regression on distance, solved by
    Newton iterations, which gives (R, T) directly and a log-likelihood for the fit.
    """
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    m = pre.size
    idx = np.arange(m) if m <= n_samples else rng.choice(m, size=n_samples, replace=False)
    pos_i, pos_j = pre[idx], post[idx]
    key = pre.astype(np.int64) * n + post.astype(np.int64)
    keyset = np.unique(np.concatenate([key, post.astype(np.int64) * n + pre.astype(np.int64)]))
    neg_i, neg_j = [], []
    want = len(idx)
    while sum(len(x) for x in neg_i) < want:
        a = rng.integers(0, n, size=want)
        b = rng.integers(0, n, size=want)
        cand = a.astype(np.int64) * n + b.astype(np.int64)
        ok = (a != b) & ~np.isin(cand, keyset)
        neg_i.append(a[ok])
        neg_j.append(b[ok])
    neg_i = np.concatenate(neg_i)[:want]
    neg_j = np.concatenate(neg_j)[:want]

    def dist(i, j):
        """Row-wise (elementwise) distance between paired points.

        Must NOT build a full pairwise block: on the real graph this path pairs 400k sampled
        positives and 400k negatives, and an (n, n) block there is a 1.16 TiB allocation.
        """
        x = coords[i]
        y = coords[j]
        if kind == "hyperbolic":
            x2 = np.einsum("ij,ij->i", x, x)
            y2 = np.einsum("ij,ij->i", y, y)
            diff = x2 + y2 - 2.0 * np.einsum("ij,ij->i", x, y)
            denom = np.maximum((1.0 - x2) * (1.0 - y2), 1e-12)
            return np.arccosh(np.maximum(1.0 + 2.0 * diff / denom, 1.0))
        return np.linalg.norm(x - y, axis=1)

    d = np.concatenate([dist(pos_i, pos_j), dist(neg_i, neg_j)])
    y = np.concatenate([np.ones(len(idx)), np.zeros(len(neg_i))])
    # logistic regression on d: logit(p) = (R - d)/T -> slope -1/T, intercept R/T
    keep = np.isfinite(d)
    d, y = d[keep], y[keep]
    X = np.stack([np.ones_like(d), d], axis=1)
    w = np.zeros(2)
    for _ in range(60):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        grad = X.T @ (y - p)
        hess = (X * (p * (1 - p))[:, None]).T @ X
        try:
            step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    intercept, slope = w
    T = float(-1.0 / slope) if slope < 0 else float("nan")
    R = float(intercept * T) if slope < 0 else float("nan")
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    eps = 1e-12
    ll = float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    # AUC as P(a connected pair is closer than an unconnected pair): rank the distances
    # ascending and score the positives by 1 - rank, i.e. report 1 - (rank statistic), because
    # closeness (not distance) is what predicts a connection.
    order = np.argsort(d)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(d.size) + 1
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    auc = float(1.0 - (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) \
        if n_pos and n_neg else float("nan")
    return {"kind": kind, "R": R, "T": T, "log_likelihood": ll, "auc": auc,
            "auc_definition": "P(distance(connected pair) < distance(random pair))",
            "n_positive": int(n_pos), "n_negative": int(n_neg),
            "distance_median_positive": float(np.median(d[y == 1])),
            "distance_median_negative": float(np.median(d[y == 0]))}


def coarse_grain(g: GraphView, law: GeometryLaw, factor: float, seed: int = 0,
                 threshold: int | None = None, target_mean_degree: float | None = None,
                 keep_autapses: bool = False, max_group_size: int = 400,
                 return_groups: bool = False):
    """Coarse-grain by `factor` (<1) in the latent geometry.

    The coarse-edge threshold defaults to the value that reproduces the source graph's mean
    out-degree (preserving sparsity); passing `threshold` overrides that calibration.
    """
    if not 0.0 < factor < 1.0:
        raise ValueError("coarse_grain factor must be in (0, 1)")
    group_size = int(max(2, round(1.0 / factor)))
    if group_size > max_group_size:
        raise ValueError(f"group_size {group_size} exceeds max_group_size")
    groups, n_groups = group_by_geometry(law, group_size, seed=seed)
    attrs = _aggregate_attributes(g, groups, n_groups, law)

    gpre, gpost = groups[g.pre], groups[g.post]
    same = gpre == gpost
    intra_syn = int(g.syn[same].sum()) if same.any() else 0
    if not keep_autapses:
        keep = ~same
        gpre, gpost, gsyn, gnt = gpre[keep], gpost[keep], g.syn[keep], g.nt_code[keep]
    else:
        gsyn, gnt = g.syn, g.nt_code
    pre, post, syn, nt = compact_pairs(gpre, gpost, gsyn, gnt, threshold=1)

    src_mean_out = float(g.pre.size / g.n)
    if threshold is None:
        target = float(target_mean_degree if target_mean_degree is not None else src_mean_out)
        threshold, calib = _calibrate_threshold(syn, pre, n_groups, target)
    else:
        calib = {"requested": int(threshold), "calibrated": False}

    keep = syn >= threshold
    out = GraphView(
        n=n_groups, pre=pre[keep], post=post[keep], syn=syn[keep], nt_code=nt[keep],
        root_ids=attrs["root_ids"], cell_type=attrs["cell_type"],
        super_class=attrs["super_class"], top_nt=attrs["top_nt"], coords=attrs["coords"],
        provenance={
            "kind": "coarse_grain", "factor": factor, "group_size": group_size,
            "n_groups": int(n_groups), "seed": seed, "geometry": law.note(),
            "edge_threshold": int(threshold), "threshold_calibration": calib,
            "source_n": int(g.n), "source_edges": int(g.pre.size),
            "source_mean_out_degree": round(src_mean_out, 4),
            "intra_group_synapses": intra_syn,
            "group_size_histogram": _hist(attrs["sizes"]),
        },
    )
    if return_groups:
        return out, groups
    return out


def _calibrate_threshold(syn: np.ndarray, pre: np.ndarray, n_groups: int,
                         target_mean_degree: float) -> tuple[int, dict]:
    """Pick the synapse threshold whose coarse graph has mean out-degree closest to target."""
    if syn.size == 0:
        return 1, {"candidates": [], "target_mean_degree": target_mean_degree}
    lo, hi = 1, int(np.percentile(syn, 99.9)) + 1
    cands = sorted(set(int(hi * (i / 24.0)) + 1 for i in range(25)) | {1, 5})
    rows = []
    best, best_err = 1, None
    for t in cands:
        keep = syn >= t
        deg = float(keep.sum() / n_groups)
        err = abs(deg - target_mean_degree)
        rows.append({"threshold": int(t), "mean_out_degree": round(deg, 5)})
        if best_err is None or err < best_err:
            best, best_err = t, err
    achievable = rows[0]["mean_out_degree"] if rows else 0.0   # threshold 1 = densest possible
    return int(best), {"candidates": rows, "target_mean_degree": target_mean_degree,
                       "chosen": int(best), "chosen_error": round(float(best_err), 6),
                       "min_achievable_mean_degree": achievable,
                       "target_reachable": bool(achievable >= target_mean_degree - 1e-9),
                       "calibrated": True}


def _hist(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    u, c = np.unique(values, return_counts=True)
    return {int(a): int(b) for a, b in zip(u, c)}


# --------------------------------------------------------------------------- subdivision
def _assign_children(n_parents: int, factor: float, seed: int) -> np.ndarray:
    """How many children each parent gets (integer counts summing to round(n*factor))."""
    rng = np.random.default_rng(seed)
    total = int(round(n_parents * factor))
    base = int(np.floor(factor))
    counts = np.full(n_parents, base, dtype=np.int64)
    extra = total - base * n_parents
    if extra > 0:
        pick = rng.choice(n_parents, size=extra, replace=False)
        counts[pick] += 1
    elif extra < 0:                                   # factor < 1 handled elsewhere
        raise ValueError("subdivision factor must be >= 1")
    return counts


def upscale(g: GraphView, law: GeometryLaw, factor: float, seed: int = 0,
            perturbation: float | None = None, sibling_edges: bool = True,
            rewire_fraction: float = 0.0, sibling_threshold: int = 5,
            sibling_prob_scale: float = 0.1) -> GraphView:
    """Subdivide each neuron into `factor` children placed near the parent in the geometry.

    Edge policy (§12): a child inherits the parent's partner set with the parent's synapse
    weights (so mean connection strength is preserved), and the concrete target child is chosen
    by the geometric connection law among the partner's children. `rewire_fraction` of each
    child's inherited edges are instead redrawn to a different parent-level partner from the
    same pool, adding diversification without changing degrees. Sibling (intra-group) edges are
    added by the law, which is what §12 asks for as "local microstructure".
    """
    if factor < 1.0:
        raise ValueError("upscale factor must be >= 1")
    rng = np.random.default_rng(seed)
    n = g.n
    counts = _assign_children(n, factor, seed)
    parent_index = np.repeat(np.arange(n, dtype=np.int64), counts)
    new_n = int(parent_index.size)
    # child local index within its family
    child_local = np.concatenate([np.arange(c) for c in counts]) if new_n else np.zeros(0, np.int64)

    eps = float(perturbation if perturbation is not None else 0.5 * typical_neighbour_distance(law))
    dirs = rng.normal(size=(new_n, law.dim))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    if law.kind == "hyperbolic":
        base = law.coords[parent_index]
        v = np.tanh(eps / 2.0) * dirs
        child_coords = poincare_exp_map(base, v)
    else:
        child_coords = law.coords[parent_index] + eps * dirs

    # family offsets so a child can be addressed as offset[parent] + local
    family_start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=family_start[1:])

    pre_par, post_par, syn_par, nt_par = g.pre, g.post, g.syn, g.nt_code
    n_edges = pre_par.size
    # each child inherits every parent edge: children_out[parent] copies
    child_edge_owner = np.repeat(pre_par, counts[pre_par])            # parent of the child
    child_edge_count = counts[pre_par]                                # children per parent
    child_edge_post_parent = np.repeat(post_par, counts[pre_par])
    child_edge_syn = np.repeat(syn_par, counts[pre_par])
    child_edge_nt = np.repeat(nt_par, counts[pre_par])
    # which local child of the presynaptic parent owns this copy (round-robin by edge order)
    owner_local = _repeat_round_robin(counts[pre_par])
    child_edge_pre = family_start[child_edge_owner] + owner_local

    # choose the target child by the geometric law among the partner's children
    tgt_start = family_start[child_edge_post_parent]
    tgt_count = counts[child_edge_post_parent]
    child_edge_post = _sample_child_by_law(law, child_coords, child_edge_pre, tgt_start,
                                           tgt_count, rng)

    if rewire_fraction > 0.0:
        child_edge_post_parent, child_edge_post, child_edge_syn, child_edge_nt = _diversify(
            rng, g, child_edge_pre, child_edge_post_parent, child_edge_post,
            child_edge_syn, child_edge_nt, family_start, counts, law, child_coords,
            rewire_fraction)

    pre = child_edge_pre.astype(np.int64)
    post = child_edge_post.astype(np.int64)
    syn = child_edge_syn.astype(np.int64)
    nt = child_edge_nt.astype(np.int8)

    if sibling_edges:
        sib_pre, sib_post, sib_syn, sib_nt = _sibling_edges(
            law, child_coords, parent_index, family_start, counts, rng, sibling_threshold,
            prob_scale=sibling_prob_scale)
        pre = np.concatenate([pre, sib_pre])
        post = np.concatenate([post, sib_post])
        syn = np.concatenate([syn, sib_syn])
        nt = np.concatenate([nt, sib_nt])
        n_sibling = int(sib_pre.size)
    else:
        n_sibling = 0

    pre, post, syn, nt = compact_pairs(pre, post, syn, nt, threshold=1)

    cell_type = None if g.cell_type is None else np.asarray(g.cell_type, dtype=object)[parent_index]
    super_class = None if g.super_class is None else np.asarray(g.super_class, dtype=object)[parent_index]
    top_nt = None if g.top_nt is None else np.asarray(g.top_nt, dtype=object)[parent_index]
    root_ids = g.root_ids[parent_index]

    return GraphView(
        n=new_n, pre=pre, post=post, syn=syn, nt_code=nt, root_ids=root_ids,
        cell_type=cell_type, super_class=super_class, top_nt=top_nt, coords=child_coords,
        provenance={
            "kind": "upscale", "factor": factor, "seed": seed, "geometry": law.note(),
            "perturbation": eps, "sibling_edges": bool(sibling_edges),
            "sibling_threshold": int(sibling_threshold),
            "sibling_prob_scale": float(sibling_prob_scale),
            "n_sibling_edges_pre_merge": int(n_sibling),
            "rewire_fraction": float(rewire_fraction),
            "source_n": int(g.n), "source_edges": int(g.pre.size),
            "mean_children": round(float(counts.mean()), 4),
            "children_histogram": _hist(counts),
        },
        parent_index=parent_index,
    )


def _repeat_round_robin(counts: np.ndarray) -> np.ndarray:
    """For an array of group sizes, the local child index of each repeated copy (0..c-1)."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    idx = np.arange(total, dtype=np.int64)
    return idx - np.repeat(starts, counts)


def _sample_child_by_law(law: GeometryLaw, coords: np.ndarray, pre: np.ndarray,
                         tgt_start: np.ndarray, tgt_count: np.ndarray,
                         rng: np.random.Generator, chunk: int = 2_000_000) -> np.ndarray:
    """Pick, for every inherited edge, one child of the postsynaptic parent.

    Choice is a softmax over the geometric connection probability (closer children are more
    likely targets), which is what makes the subdivision respect the inferred connectivity law
    instead of replicating the parent's wiring blindly.
    """
    out = np.empty(pre.size, dtype=np.int64)
    max_c = int(tgt_count.max()) if tgt_count.size else 1
    base_prob = law.connect_prob(0.0)
    for lo in range(0, pre.size, chunk):
        hi = min(lo + chunk, pre.size)
        m = hi - lo
        offs = np.arange(max_c, dtype=np.int64)[None, :]
        valid = offs < tgt_count[lo:hi, None]
        idx = tgt_start[lo:hi, None] + offs
        cand = idx.clip(max=coords.shape[0] - 1)
        a = np.repeat(coords[pre[lo:hi]], max_c, axis=0)
        b = coords[cand.reshape(-1)]
        d = _row_distances(law, a, b).reshape(m, max_c)
        p = law.connect_prob(d)
        p = np.where(valid, p, 0.0)
        row_sum = p.sum(axis=1, keepdims=True)
        uniform = valid / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        p = np.where(row_sum > 0, p / np.maximum(row_sum, 1e-12), uniform)
        cum = np.cumsum(p, axis=1)
        r = rng.random((m, 1))
        pick = (cum < r).sum(axis=1).clip(max=max_c - 1)
        out[lo:hi] = tgt_start[lo:hi] + pick
    return out


def _row_distances(law: GeometryLaw, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise distance between two aligned (m,d) coordinate blocks."""
    if law.kind == "hyperbolic":
        a2 = np.sum(a * a, axis=1)
        b2 = np.sum(b * b, axis=1)
        ab = np.sum(a * b, axis=1)
        diff = a2 + b2 - 2 * ab
        denom = np.maximum((1.0 - a2) * (1.0 - b2), 1e-12)
        return np.arccosh(np.maximum(1.0 + 2.0 * diff / denom, 1.0))
    return np.linalg.norm(a - b, axis=1)


def _diversify(rng, g, pre, post_parent, post, syn, nt, family_start, counts, law,
               coords, fraction):
    """Redraw a fraction of inherited edges to a different partner of the same parent."""
    m = pre.size
    if m == 0 or fraction <= 0:
        return post_parent, post, syn, nt
    sel = rng.random(m) < fraction
    if not sel.any():
        return post_parent, post, syn, nt
    # pick an alternative partner: the parent of this child, sampled from its out-partners
    owner_parent = np.repeat(np.arange(g.n, dtype=np.int64), counts)[pre]
    alt = _sample_alt_partner(rng, g, owner_parent[sel])
    post_parent = post_parent.copy()
    post_parent[sel] = alt
    post[sel] = family_start[alt] + (rng.random(int(sel.sum())) * counts[alt]).astype(np.int64)
    post[sel] = np.minimum(post[sel], family_start[alt] + counts[alt] - 1)
    return post_parent, post, syn, nt


def _sample_alt_partner(rng, g, parents: np.ndarray) -> np.ndarray:
    """Uniformly sample one of each parent's existing partners (as a parent index)."""
    start = g.out_indptr[parents]
    cnt = g.out_indptr[parents + 1] - start
    offs = (rng.random(parents.size) * np.maximum(cnt, 1)).astype(np.int64)
    offs = np.minimum(offs, np.maximum(cnt - 1, 0))
    return g.out_indices[start + offs].astype(np.int64)


def _block_from_coords(kind: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance block between two coordinate blocks, honouring the geometry."""
    return poincare_distance_block(x, y) if kind == "hyperbolic" \
        else euclidean_distance_block(x, y)


def _sibling_edges(law, coords, parent_index, family_start, counts, rng, threshold,
                   prob_scale: float = 0.1):
    """Add intra-family connections using the geometric law (close children connect).

    `prob_scale` damps the law's probability: at subdivision distances the raw law is close to
    1, which would add ~c-1 extra out-edges per neuron and visibly inflate mean degree above
    the source graph's. The damped rate keeps the local-synapse contribution a documented,
    inspectable fraction of the budget instead of a hidden degree inflation.
    """
    multi = np.flatnonzero(counts >= 2)
    if multi.size == 0:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.int64), np.zeros(0, np.int8))
    pre_l, post_l, syn_l, nt_l = [], [], [], []
    for p in multi:
        s, c = int(family_start[p]), int(counts[p])
        fam = np.arange(s, s + c, dtype=np.int64)
        d = _block_from_coords(law.kind, coords[fam], coords[fam])
        pmat = np.clip(law.connect_prob(d) * float(prob_scale), 0.0, 1.0)
        np.fill_diagonal(pmat, 0.0)
        draw = rng.random(pmat.shape) < pmat
        ii, jj = np.nonzero(draw)
        if ii.size == 0:
            continue
        pre_l.append(fam[ii])
        post_l.append(fam[jj])
        syn_l.append(np.full(ii.size, threshold, dtype=np.int64))
        nt_l.append(np.zeros(ii.size, dtype=np.int8))
    if not pre_l:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.int64), np.zeros(0, np.int8))
    return (np.concatenate(pre_l), np.concatenate(post_l),
            np.concatenate(syn_l), np.concatenate(nt_l))


# --------------------------------------------------------------------------- closure helper
def coarse_grain_by_lineage(g: GraphView, target_n: int | None = None) -> GraphView:
    """Coarse-grain a subdivided graph back along its own lineage (the inverted upscale).

    This is the strongest possible form of the §13 closure operation: instead of re-learning
    the grouping from geometry, it uses the recorded parent_index so the only error source is
    the scaling operation itself, not the rediscovery of the partition.
    """
    if g.parent_index is None:
        raise ValueError("graph has no parent_index lineage; use coarse_grain instead")
    par = g.parent_index
    uniq, groups = np.unique(par, return_inverse=True)
    n_groups = int(uniq.size)
    cap = GraphView(
        n=n_groups, pre=np.zeros(0, np.int64), post=np.zeros(0, np.int64),
        syn=np.zeros(0, np.int64), nt_code=np.zeros(0, np.int8),
        root_ids=g.root_ids[uniq] if g.root_ids is not None else np.arange(n_groups),
        coords=g.coords[uniq] if g.coords is not None else None,
    )
    gpre, gpost = groups[g.pre], groups[g.post]
    same = gpre == gpost
    keep = ~same
    pre, post, syn, nt = compact_pairs(gpre[keep], gpost[keep], g.syn[keep], g.nt_code[keep],
                                       threshold=1)
    out = GraphView(
        n=n_groups, pre=pre, post=post, syn=syn, nt_code=nt,
        root_ids=cap.root_ids, coords=cap.coords,
        provenance={"kind": "lineage_coarse_grain", "source_kind": g.provenance.get("kind"),
                    "source_n": int(g.n), "n_groups": n_groups,
                    "intra_group_synapses": int(g.syn[same].sum()) if same.any() else 0},
    )
    return out
