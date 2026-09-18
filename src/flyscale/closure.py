"""Renormalization closure metrics (PROJECT-VYBFLY.md §13).

`compare(reference, replica, correspondence)` measures how far a scaled replica is from the
biological source graph. §13 asks for a single composite score *only* for internal
optimization and requires the raw metrics to be preserved, so every metric is returned
individually and the composite is computed separately from them.

Every metric states whether it needed the node correspondence:

  correspondence-free  degree / weighted-degree / spectral / motif / rich-club / connectivity
                       matrix / latent-distance distributions
  correspondence-bound  community agreement (ARI/NMI), per-node rich-club overlap

Distances are all oriented so that 0 = identical. Nothing here is a p-value; these are
descriptive distances between two graphs.
"""
from __future__ import annotations

import numpy as np

from .synthetic import GraphView


# --------------------------------------------------------------------------- helpers
def out_degrees(g: GraphView) -> np.ndarray:
    return g.out_degree().astype(np.float64)


def weighted_out_degrees(g: GraphView) -> np.ndarray:
    return g.weighted_out_degree().astype(np.float64)


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def _hist_frequencies(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.zeros(0), np.zeros(0)
    u, c = np.unique(values, return_counts=True)
    return u.astype(np.float64), (c / c.sum()).astype(np.float64)


def _symmetric_laplacian(g: GraphView, autapses: bool = False):
    from scipy import sparse
    A = g.adjacency(include_autapses=autapses)
    U = sparse.csr_matrix(A + A.T)
    U.data[:] = 1.0
    U.setdiag(0)
    U.eliminate_zeros()
    deg = np.asarray(U.sum(axis=1)).ravel()
    dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg, dtype=float), where=deg > 0)
    D = sparse.diags(dinv)
    return sparse.eye(g.n, format="csr") - D @ U @ D, U, deg


def spectral_signature(g: GraphView, k: int = 50) -> list[float]:
    """Top-k eigenvalues of the symmetric normalized Laplacian (Laplacian eigenvalue proxy)."""
    from scipy.sparse import linalg as splinalg
    L, _, _ = _symmetric_laplacian(g)
    kk = int(min(k, g.n - 2))
    if kk < 1:
        return []
    try:
        vals = splinalg.eigsh(L.tocsc(), k=kk, which="LA", return_eigenvectors=False)
    except Exception:
        return []
    return [float(v) for v in np.sort(vals)[::-1]]


def spectral_distance(sig_a: list[float], sig_b: list[float]) -> float:
    k = min(len(sig_a), len(sig_b))
    if k == 0:
        return float("nan")
    a, b = np.asarray(sig_a[:k]), np.asarray(sig_b[:k])
    return float(np.sqrt(np.mean((a - b) ** 2)))


def two_node_motifs(g: GraphView) -> dict:
    """Two-node motif census: reciprocal, unidirectional and absent pairs among connected nodes."""
    n = g.n
    key = g.pre.astype(np.int64) * n + g.post.astype(np.int64)
    keep = g.pre != g.post
    key_ns = key[keep]
    rev = g.post[keep].astype(np.int64) * n + g.pre[keep].astype(np.int64)
    has_rev = np.isin(rev, key_ns)
    uni = int(key_ns.size - has_rev.sum())
    rec = int(has_rev.sum() // 2)
    total = int(key_ns.size)
    return {
        "reciprocal_pairs": rec,
        "unidirectional_pairs": uni,
        "connections": total,
        "reciprocity": round(float(has_rev.sum() / max(1, total)), 6) if total else float("nan"),
    }


def triad_frequencies(g: GraphView, max_edges: int = 4_000_000) -> dict | None:
    """Normalized 16-class triad census (None when the graph exceeds `max_edges`)."""
    import igraph as ig
    from .metrics import TRIAD_LABELS
    if g.pre.size > max_edges:
        return None
    graph = ig.Graph(n=g.n, edges=list(zip(g.pre.tolist(), g.post.tolist())), directed=True)
    census = np.asarray(graph.triad_census(), dtype=np.float64)
    tot = census.sum()
    return {lab: float(v / tot) for lab, v in zip(TRIAD_LABELS, census)} if tot else None


def motif_divergence(census_a: dict | None, census_b: dict | None) -> float:
    if not census_a or not census_b:
        return float("nan")
    keys = sorted(set(census_a) & set(census_b))
    a = np.asarray([census_a[k] for k in keys])
    b = np.asarray([census_b[k] for k in keys])
    return float(np.abs(a - b).sum())


def rich_club_curve(g: GraphView, ks: np.ndarray, autapses: bool = False) -> dict:
    """Rich-club coefficient Phi(k) = 2 E_k / (N_k (N_k - 1)) on the undirected projection."""
    from .metrics import undirected_binary, union_edges
    A = g.adjacency(include_autapses=autapses)
    U = undirected_binary(A)
    deg = np.asarray(U.sum(axis=1)).ravel()
    ei, ej = union_edges(U)
    out = {}
    for k in ks:
        rich = np.flatnonzero(deg > k)
        if rich.size < 2:
            out[int(k)] = None
            continue
        sel = np.zeros(g.n, dtype=bool)
        sel[rich] = True
        e_in = int((sel[ei] & sel[ej]).sum())
        out[int(k)] = round(float(2 * e_in / (rich.size * (rich.size - 1))), 9)
    return out


def rich_club_distance(ca: dict, cb: dict) -> float:
    keys = [k for k in sorted(set(ca) & set(cb)) if ca[k] is not None and cb[k] is not None]
    if not keys:
        return float("nan")
    diffs = [abs(ca[k] - cb[k]) for k in keys]
    scales = [max(ca[k], cb[k], 1e-12) for k in keys]
    return float(np.mean([d / s for d, s in zip(diffs, scales)]))


def leiden_membership(g: GraphView, seed: int = 0, resolution: float = 1.0) -> np.ndarray | None:
    import random

    import igraph as ig
    from .metrics import undirected_binary, union_edges
    if g.pre.size == 0:
        return None
    ig.set_random_number_generator(random.Random(seed))
    U = undirected_binary(g.adjacency(include_autapses=False))
    i, j = union_edges(U)
    graph = ig.Graph(n=g.n, edges=list(zip(i.tolist(), j.tolist())), directed=False)
    part = graph.community_leiden(objective_function="modularity", resolution=resolution,
                                  n_iterations=2)
    return np.asarray(part.membership)


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI between two label vectors over the same items (no sklearn dependency)."""
    n = a.size
    if n == 0 or a.size != b.size:
        return float("nan")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(cont, (ai, bi), 1)
    comb2 = lambda x: x * (x - 1) / 2.0                                    # noqa: E731
    sum_ij = comb2(cont.astype(np.float64)).sum()
    sum_a = comb2(cont.sum(axis=1).astype(np.float64)).sum()
    sum_b = comb2(cont.sum(axis=0).astype(np.float64)).sum()
    expected = sum_a * sum_b / comb2(float(n))
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0 if sum_ij == expected else 0.0
    return float((sum_ij - expected) / (maximum - expected))


def celltype_matrix(g: GraphView, level: str = "super_class") -> tuple[np.ndarray, list[str]] | None:
    vals = getattr(g, level, None)
    if vals is None:
        return None
    labels = np.asarray([("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
                         for v in vals], dtype=object)
    cats, codes = np.unique(labels, return_inverse=True)
    K = cats.size
    M = np.bincount(codes[g.pre] * K + codes[g.post], weights=g.syn.astype(np.float64),
                    minlength=K * K).reshape(K, K)
    row = M.sum(axis=1, keepdims=True)
    Mn = np.divide(M, row, out=np.zeros_like(M), where=row > 0)
    return Mn, cats.tolist()


def connectivity_matrix_distance(ma: tuple | None, mb: tuple | None) -> float:
    if ma is None or mb is None:
        return float("nan")
    A, ca = ma
    B, cb = mb
    if ca != cb:
        cats = sorted(set(ca) | set(cb))
        ia = {c: i for i, c in enumerate(ca)}
        ib = {c: i for i, c in enumerate(cb)}
        Aa = np.zeros((len(cats), len(cats)))
        Bb = np.zeros((len(cats), len(cats)))
        for i, c in enumerate(cats):
            for j, d in enumerate(cats):
                if c in ia and d in ia:
                    Aa[i, j] = A[ia[c], ia[d]]
                if c in ib and d in ib:
                    Bb[i, j] = B[ib[c], ib[d]]
        A, B = Aa, Bb
    return float(np.abs(A - B).mean())


def latent_distance_distribution(g: GraphView, sample: int = 200_000,
                                 seed: int = 0) -> np.ndarray:
    if g.coords is None or g.pre.size == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    m = g.pre.size
    idx = np.arange(m) if m <= sample else rng.choice(m, size=sample, replace=False)
    a, b = g.coords[g.pre[idx]], g.coords[g.post[idx]]
    return np.linalg.norm(a - b, axis=1)


# --------------------------------------------------------------------------- composite
def compare(reference: GraphView, replica: GraphView,
            correspondence: np.ndarray | None = None,
            reference_groups: np.ndarray | None = None,
            spectral_k: int = 50, rich_club_ks: Iterable[int] = (2, 5, 10, 25, 50, 100, 200),
            triad_max_edges: int = 4_000_000, seed: int = 0,
            reference_signature: dict | None = None) -> dict:
    """Measure replica vs reference.

    Two ways to state the node relationship:
      * `correspondence[i] = j` - replica node i descends from reference node j (subdivision)
      * `reference_groups[j] = i` - reference node j fell into replica node i (partition /
        coarse graining). The replica-side aggregate for a group is the mean over its members.

    Pass `reference_signature` (the output of `signature(reference)`) to avoid recomputing the
    expensive reference-side pieces (triad census, Leiden, spectrum) for every scale.
    """
    from .metrics import undirected_binary, union_edges  # noqa: F401
    if correspondence is not None and reference_groups is not None:
        raise ValueError("pass either correspondence or reference_groups, not both")
    ref_sig = reference_signature or signature(reference, spectral_k=spectral_k,
                                               rich_club_ks=tuple(rich_club_ks),
                                               triad_max_edges=triad_max_edges, seed=seed)

    out: dict = {
        "reference": reference.summary(),
        "replica": replica.summary(),
        "n_reference": int(reference.n),
        "n_replica": int(replica.n),
        "scale_n": round(replica.n / reference.n, 6),
        "scale_edges": round(replica.pre.size / max(1, reference.pre.size), 6),
        "scale_synapses": round(float(replica.syn.sum()) / max(1.0, float(reference.syn.sum())), 6),
    }

    out["degree_wasserstein"] = round(_wasserstein(out_degrees(reference), out_degrees(replica)), 6)
    out["degree_wasserstein_normalised"] = round(
        out["degree_wasserstein"] / max(1.0, float(np.mean(out_degrees(reference)))), 6)
    out["weighted_degree_wasserstein"] = round(
        _wasserstein(weighted_out_degrees(reference), weighted_out_degrees(replica)), 6)

    rep_triad = triad_frequencies(replica, max_edges=triad_max_edges)
    out["triad_census_available"] = rep_triad is not None
    out["motif_divergence_L1"] = round(motif_divergence(ref_sig.get("triad"), rep_triad), 9) \
        if (ref_sig.get("triad") and rep_triad) else None

    rec_ref = two_node_motifs(reference)
    rec_rep = two_node_motifs(replica)
    out["two_node"] = {
        "reference": rec_ref, "replica": rec_rep,
        "reciprocity_delta": round(rec_rep["reciprocity"] - rec_ref["reciprocity"], 6),
    }

    sig_rep = spectral_signature(replica, k=spectral_k)
    out["spectral_distance"] = round(spectral_distance(ref_sig.get("spectral", []), sig_rep), 9)
    out["spectral_reference_head"] = [round(v, 6) for v in ref_sig.get("spectral", [])[:10]]
    out["spectral_replica_head"] = [round(v, 6) for v in sig_rep[:10]]

    rc_ref = ref_sig.get("rich_club", {})
    rc_rep = rich_club_curve(replica, np.asarray(list(rich_club_ks)))
    out["rich_club_distance"] = round(rich_club_distance(rc_ref, rc_rep), 9)
    out["rich_club"] = {"reference": rc_ref, "replica": rc_rep}

    cm_ref = ref_sig.get("celltype_matrix")
    cm_rep = celltype_matrix(replica, "super_class")
    out["connectivity_matrix_divergence"] = round(connectivity_matrix_distance(cm_ref, cm_rep), 9)

    lat_ref = ref_sig.get("latent_distances")
    lat_rep = latent_distance_distribution(replica, seed=seed)
    out["latent_distance_wasserstein"] = round(_wasserstein(np.asarray(lat_ref), lat_rep), 9) \
        if lat_ref is not None and np.asarray(lat_ref).size and lat_rep.size else None

    if correspondence is not None and correspondence.size == replica.n:
        keep = correspondence >= 0
        out["correspondence_coverage"] = round(float(keep.mean()), 6)
        memb_ref = ref_sig.get("leiden")
        memb_rep = leiden_membership(replica, seed=seed)
        if memb_ref is not None and memb_rep is not None:
            mapped = np.where(keep, memb_ref[np.clip(correspondence, 0, reference.n - 1)], -1)
            sel = mapped >= 0
            out["community_ari"] = round(adjusted_rand_index(mapped[sel], memb_rep[sel]), 6)
            out["community_nmi"] = round(_nmi(mapped[sel], memb_rep[sel]), 6)
        out["rich_club_jaccard_matched_quantile"] = round(
            rich_club_jaccard(reference, replica, correspondence, quantile=0.29), 6)
        out["rich_club_jaccard_note"] = (
            "top 29% by total degree in each graph (the published rich-club fraction), "
            "compared node-for-node through the correspondence")
    else:
        out["community_ari"] = None
        out["community_nmi"] = None
        out["rich_club_jaccard_matched_quantile"] = None

    if reference_groups is not None and reference_groups.size == reference.n:
        groups = np.asarray(reference_groups)
        counts = np.bincount(groups, minlength=replica.n).astype(np.float64)
        out["partition_group_size_mean"] = round(float(counts.mean()), 4)
        out["partition_coverage"] = round(float((counts > 0).mean()), 6)
        memb_ref = ref_sig.get("leiden")
        memb_rep = leiden_membership(replica, seed=seed)
        if memb_ref is not None and memb_rep is not None:
            # the reference community each group most of its members belong to
            comb = np.stack([groups, memb_ref], axis=1)
            uniq, cnt = np.unique(comb, axis=0, return_counts=True)
            order = np.lexsort((cnt, uniq[:, 0]))
            u_sorted, c_sorted = uniq[order], cnt[order]
            last = np.append(np.flatnonzero(np.diff(u_sorted[:, 0]) != 0), u_sorted.shape[0] - 1)
            majority = np.full(replica.n, -1, dtype=np.int64)
            majority[u_sorted[last, 0]] = u_sorted[last, 1]
            sel = (majority >= 0) & (memb_rep >= 0)
            if sel.sum() > 1:
                out["community_ari"] = round(adjusted_rand_index(majority[sel], memb_rep[sel]), 6)
                out["community_nmi"] = round(_nmi(majority[sel], memb_rep[sel]), 6)
        # rich-club agreement at matched quantiles, using the aggregated reference degree
        deg_ref = (reference.out_degree() + reference.in_degree()).astype(np.float64)
        agg = np.bincount(groups, weights=deg_ref, minlength=replica.n) / np.maximum(counts, 1)
        deg_rep = (replica.out_degree() + replica.in_degree()).astype(np.float64)
        q = 0.29
        m_ref = agg > np.quantile(agg[agg > 0], 1.0 - q)
        m_rep = deg_rep > np.quantile(deg_rep, 1.0 - q)
        inter, union = int((m_ref & m_rep).sum()), int((m_ref | m_rep).sum())
        out["rich_club_jaccard_matched_quantile"] = round(float(inter / union), 6) if union else None
        out["rich_club_jaccard_note"] = (
            "top 29% by total degree; the reference side is the per-group mean degree, so a "
            "coarse node inherits the richness of the neurons it aggregates")

    out["composite"] = composite_score(out)
    return out


def rich_club_jaccard(reference: GraphView, replica: GraphView, correspondence: np.ndarray,
                      quantile: float = 0.29) -> float:
    """Jaccard overlap of the top-`quantile` degree sets, compared through the correspondence.

    Absolute degree thresholds are not comparable across scales (a coarse node aggregates many
    neurons), so the rich club is defined by the same *quantile* in both graphs.
    """
    def top_mask(g: GraphView) -> np.ndarray:
        deg = (g.out_degree() + g.in_degree()).astype(np.float64)
        if deg.size == 0:
            return np.zeros(g.n, dtype=bool)
        cut = np.quantile(deg, 1.0 - quantile)
        return deg > cut

    m_ref = top_mask(reference)[correspondence]
    m_rep = top_mask(replica)
    inter = int((m_ref & m_rep).sum())
    union = int((m_ref | m_rep).sum())
    return float(inter / union) if union else float("nan")


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.float64)
    np.add.at(cont, (ai, bi), 1)
    n = cont.sum()
    pij = cont / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = pij * np.log(pij / (pi * pj))
    mi = np.nansum(np.where(pij > 0, t, 0.0))
    ha = -np.nansum(pi * np.log(pi))
    hb = -np.nansum(pj * np.log(pj))
    if ha <= 0 or hb <= 0:
        return 1.0 if mi == 0 else 0.0
    return float(2 * mi / (ha + hb))


def signature(g: GraphView, spectral_k: int = 50, rich_club_ks: tuple = (2, 5, 10, 25, 50, 100, 200),
              triad_max_edges: int = 4_000_000, seed: int = 0) -> dict:
    """Expensive reference-side quantities, computed once and reused across scales."""
    return {
        "spectral": spectral_signature(g, k=spectral_k),
        "triad": triad_frequencies(g, max_edges=triad_max_edges),
        "rich_club": rich_club_curve(g, np.asarray(list(rich_club_ks))),
        "leiden": leiden_membership(g, seed=seed),
        "celltype_matrix": celltype_matrix(g, "super_class"),
        "latent_distances": latent_distance_distribution(g, seed=seed),
        "two_node": two_node_motifs(g),
        "summary": g.summary(),
    }


COMPOSITE_WEIGHTS = {
    "degree_wasserstein_normalised": 0.20,
    "motif_divergence_L1": 0.20,
    "spectral_distance": 0.15,
    "rich_club_distance": 0.15,
    "connectivity_matrix_divergence": 0.15,
    "community_ari": 0.15,               # signed the other way (higher = better)
}


def composite_score(metrics: dict) -> float | None:
    """Weighted mean of available normalized distances; ARI enters as (1 - ARI).

    Internal optimisation score only - §13 requires the raw metrics to be kept alongside it,
    and they are (this dict is stored whole in the results JSON).
    """
    total, used = 0.0, 0.0
    for key, w in COMPOSITE_WEIGHTS.items():
        v = metrics.get(key)
        if v is None or (isinstance(v, float) and v != v):
            continue
        val = (1.0 - float(v)) if key == "community_ari" else float(v)
        total += w * val
        used += w
    return round(total / used, 9) if used > 0 else None
