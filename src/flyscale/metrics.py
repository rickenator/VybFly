"""Canonical network metrics for the FlyWire v783 canonical dataset (Phase 0 gate).

Every metric reports the graph variant it was computed on. The variants are named
explicitly because the published network-statistics paper (Lin, Yang et al. 2023) applies
different conventions in different figures:

  full       all published pairs, autapses included
  no_aut                 autapses removed
  central    nodes with out-degree >= 1 (sinks dropped)
  recurrent  iteratively drop sinks until no sink remains
  largest_scc  pairs whose both endpoints live in the largest strongly connected component

All graphs are unweighted (weight 1 per pair) unless a metric says otherwise; synapse
counts are used as *weights* only in the weighted metrics and are never mixed into topology.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as splinalg

from .bfs import bfs_multi

TRIAD_LABELS = ("003", "012", "102", "021D", "021U", "021C", "111D", "111U",
                "030T", "030C", "201", "120D", "120U", "120C", "210", "300")

ARTIFACTS: Path | None = None


def _ad() -> Path:
    if ARTIFACTS is None:
        raise RuntimeError("metrics.ARTIFACTS is unset; call compute_baseline first")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS


# --------------------------------------------------------------------------- helpers
def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


def hill_alpha(values: np.ndarray, tail_frac: float = 0.1) -> float:
    """Hill estimator of a power-law tail exponent on the largest `tail_frac` values."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    x = x[x > 0]
    if x.size < 50:
        return float("nan")
    k = max(10, int(x.size * tail_frac))
    tail = x[-k:]
    xmin = tail[0]
    if xmin <= 0:
        return float("nan")
    return float(1.0 + k / np.log(tail / xmin).sum())


def distribution_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {}
    return {
        "count": int(v.size),
        "mean": round(float(v.mean()), 6),
        "std": round(float(v.std()), 6),
        "median": round(float(np.median(v)), 6),
        "min": round(float(v.min()), 6),
        "max": round(float(v.max()), 6),
        "p90": round(float(np.percentile(v, 90)), 6),
        "p99": round(float(np.percentile(v, 99)), 6),
        "p999": round(float(np.percentile(v, 99.9)), 6),
        "gini": round(gini(v), 6),
        "fraction_zero": round(float((v == 0).mean()), 6),
        "hill_alpha_tail10pct": round(hill_alpha(v), 4),
    }


def to_coo(pre: np.ndarray, post: np.ndarray, n: int, data=None) -> sparse.coo_matrix:
    d = np.ones(pre.size, dtype=np.float64) if data is None else np.asarray(data, dtype=np.float64)
    return sparse.coo_matrix((d, (pre, post)), shape=(n, n))


def undirected_binary(A: sparse.spmatrix) -> sparse.csr_matrix:
    """Symmetric binary projection with no self loops."""
    U = sparse.csr_matrix(A.tocsr() + A.tocsr().T)
    U.data[:] = 1.0
    U.setdiag(0)
    U.eliminate_zeros()
    return U


def union_edges(U: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edge list (i < j) of a symmetric binary matrix."""
    coo = sparse.triu(U, k=1).tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64)


# --------------------------------------------------------------------------- variants
def graph_variants(c) -> dict[str, np.ndarray]:
    """Boolean masks over the canonical pair list, one per analysis convention."""
    pre, post = c.pre, c.post
    n = c.n
    variants: dict[str, np.ndarray] = {"full": np.ones(pre.size, dtype=bool)}
    no_aut = pre != post
    variants["no_aut"] = no_aut

    out_deg = np.bincount(pre[no_aut], minlength=n)
    variants["central"] = no_aut & (out_deg[pre] > 0) & (out_deg[post] > 0)

    # recurrent: iteratively drop sinks
    live = out_deg > 0
    while True:
        deg = np.bincount(pre[no_aut & live[pre]], minlength=n)
        nxt = deg > 0
        if nxt.sum() == live.sum():
            break
        live = nxt
    variants["recurrent"] = no_aut & live[pre] & live[post]

    # largest strongly connected component
    A = to_coo(pre, post, n).tocsr()
    _, labels = csgraph.connected_components(A, directed=True, connection="strong")
    sizes = np.bincount(labels)
    biggest = int(np.argmax(sizes))
    in_scc = labels == biggest
    variants["largest_scc"] = no_aut & in_scc[pre] & in_scc[post]
    return variants


# --------------------------------------------------------------------------- metric blocks
def counts_block(c, variants) -> dict:
    pre, post, syn = c.pre, c.post, c.syn
    out = {}
    for name, mask in variants.items():
        out[name] = {
            "n_pairs": int(mask.sum()),
            "n_synapses": int(syn[mask].sum()),
            "n_neurons_with_edge": int(np.unique(np.concatenate([pre[mask], post[mask]])).size),
        }
    return out


def degree_block(c, variants, sample_sources: np.ndarray | None = None) -> dict:
    out = {}
    for name, mask in variants.items():
        n = c.n
        pre, post, syn = c.pre[mask], c.post[mask], c.syn[mask]
        out_deg = np.bincount(pre, minlength=n)
        in_deg = np.bincount(post, minlength=n)
        w_out = np.bincount(pre, weights=syn, minlength=n)
        w_in = np.bincount(post, weights=syn, minlength=n)
        both = out_deg + in_deg
        out[name] = {
            "out_degree": distribution_stats(out_deg),
            "in_degree": distribution_stats(in_deg),
            "total_degree": distribution_stats(both),
            "weighted_out_degree": distribution_stats(w_out),
            "weighted_in_degree": distribution_stats(w_in),
            "mean_degree": round(float(pre.size * 2 / max(1, (both > 0).sum())), 6),
            "density": round(float(pre.size / (n * (n - 1))), 12),
        }
    return out


def clustering_block(c, variants) -> dict:
    out = {}
    for name, mask in variants.items():
        A = to_coo(c.pre[mask], c.post[mask], c.n).tocsr()
        U = undirected_binary(A)
        deg = np.asarray(U.sum(axis=1)).ravel()
        tri_per_node = None
        # local (undirected, binary) clustering: 2*triangles_i / (k_i (k_i - 1))
        common = (U @ U)
        triangles2 = np.asarray(common.multiply(U).sum(axis=1)).ravel()   # = 2 * triangles_i
        k = deg
        denom = k * (k - 1)
        local = np.divide(triangles2, denom, out=np.zeros_like(triangles2, dtype=float),
                          where=denom > 0)
        n_triangles = float(triangles2.sum() / 6.0)
        # global transitivity = 3 * triangles / connected triples, triples = sum k(k-1)/2
        transitivity = float(3.0 * n_triangles / max(1.0, denom.sum() / 2.0))
        # directed Fagiolo clustering
        A2 = A @ A
        t_diag = np.asarray(A2.multiply(A.T).sum(axis=1)).ravel()          # = (A^3)_ii
        out_deg = np.asarray(A.sum(axis=1)).ravel()
        in_deg = np.asarray(A.sum(axis=0)).ravel()
        rec = np.asarray((A.multiply(A.T)).sum(axis=1)).ravel()
        k_tot = out_deg + in_deg
        denom_d = k_tot * (k_tot - 1) - 2 * rec
        fagiolo = np.divide(t_diag, denom_d, out=np.zeros_like(t_diag, dtype=float),
                            where=denom_d > 0)
        out[name] = {
            "n_triangles_undirected": n_triangles,
            "global_transitivity_undirected": round(transitivity, 6),
            "mean_local_clustering_undirected": round(float(local.mean()), 6),
            "mean_local_clustering_undirected_nonzero": round(float(local[k > 1].mean()) if (k > 1).any() else 0.0, 6),
            "mean_directed_clustering_fagiolo": round(float(fagiolo.mean()), 6),
            "mean_degree_for_clustering": round(float(k[k > 1].mean()) if (k > 1).any() else 0.0, 6),
        }
    return out


def components_block(c) -> dict:
    A = to_coo(c.pre, c.post, c.n).tocsr()
    res = {}
    for kind, conn in (("strong", "strong"), ("weak", "weak")):
        ncomp, labels = csgraph.connected_components(A, directed=True, connection=conn)
        sizes = np.sort(np.bincount(labels))[::-1]
        res[kind] = {
            "n_components": int(ncomp),
            "largest_component_size": int(sizes[0]),
            "largest_component_fraction": round(float(sizes[0] / c.n), 6),
            "components_gt_2_nodes": int((sizes > 2).sum()),
            "components_gt_100_nodes": int((sizes > 100).sum()),
            "largest_10_sizes": [int(s) for s in sizes[:10]],
            "fraction_nodes_in_components_gt_100": round(float(sizes[sizes > 100].sum() / c.n), 6),
        }
    return res


def path_block(c, variants, n_sources: int = 1000, seed: int = 0,
               multigraph_undirected: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, mask in variants.items():
        A = to_coo(c.pre[mask], c.post[mask], c.n).tocsr()
        # sources are drawn from the largest weakly connected component: a BFS started in a
        # disconnected fragment measures nothing about the brain
        ncomp, labels = csgraph.connected_components(A, directed=True, connection="weak")
        sizes = np.bincount(labels)
        inside = np.flatnonzero(labels == int(np.argmax(sizes)))
        sources = rng.choice(inside, size=min(n_sources, inside.size), replace=False)
        out[name] = {"n_nodes_in_largest_wcc": int(inside.size)}
        out[name]["directed"] = bfs_multi(A.indptr.astype(np.int64),
                                          A.indices.astype(np.int32), c.n, sources)
        if multigraph_undirected:
            U = undirected_binary(A)
            out[name]["undirected"] = bfs_multi(U.indptr.astype(np.int64),
                                                U.indices.astype(np.int32), c.n, sources)
    return out


def neurotransmitter_block(c, variants) -> dict:
    from .io import NT_TYPES
    out = {}
    for name, mask in variants.items():
        codes = c.pairs["nt_code"].to_numpy()[mask]
        syn = c.syn[mask]
        tot_c, tot_s = int(codes.size), float(syn.sum())
        out[name] = {
            "by_connections": {t: {"count": int((codes == i).sum()),
                                   "fraction": round(float((codes == i).mean()), 6)}
                               for i, t in enumerate(NT_TYPES)},
            "by_synapses": {t: {"count": int(syn[codes == i].sum()),
                                "fraction": round(float(syn[codes == i].sum() / tot_s), 6)}
                            for i, t in enumerate(NT_TYPES)},
            "n_connections": tot_c,
            "n_synapses": int(tot_s),
        }
    return out


def neuropil_block(c, raw_dir: Path, variants) -> dict:
    """Neuropil-to-neuropil synapse matrix using *dominant-region* neuron assignment."""
    from .io import NT_TYPES  # noqa: F401
    pre_df = pd.read_feather(Path(raw_dir) / "per_neuron_neuropil_count_pre_783.feather")
    post_df = pd.read_feather(Path(raw_dir) / "per_neuron_neuropil_count_post_783.feather")
    names = sorted(set(pre_df["neuropil"].unique()) | set(post_df["neuropil"].unique()))
    code = {nm: i for i, nm in enumerate(names)}
    root = c.root_ids
    pos = {int(r): i for i, r in enumerate(root)}

    def dominant(df: pd.DataFrame, id_col: str) -> np.ndarray:
        df = df.copy()
        df["idx"] = df[id_col].map(pos)
        df = df.dropna(subset=["idx"])
        df["idx"] = df["idx"].astype(np.int64)
        df["code"] = df["neuropil"].map(code)
        # keep the largest count per neuron, ties broken by neuropil code for determinism
        df = df.sort_values(["idx", "Count", "code"], ascending=[True, False, True])
        best = df.drop_duplicates("idx", keep="first")
        dom = np.full(c.n, -1, dtype=np.int64)
        dom[best["idx"].to_numpy()] = best["code"].to_numpy()
        return dom

    pre_col = "pre_pt_root_id" if "pre_pt_root_id" in pre_df.columns else "pre_root_id"
    post_col = "post_pt_root_id" if "post_pt_root_id" in post_df.columns else "post_root_id"
    if "Count" not in pre_df.columns:
        cnt = [c for c in pre_df.columns if c.lower() == "count"][0]
        pre_df = pre_df.rename(columns={cnt: "Count"})
        post_df = post_df.rename(columns={cnt: "Count"})
    dom_pre = dominant(pre_df, pre_col)
    dom_post = dominant(post_df, post_col)

    K = len(names)
    out = {"n_neuropils": K, "assignment": "dominant neuropil by synapse count (pre/post side)",
           "variants": {}}
    for name, mask in variants.items():
        if name not in ("no_aut", "largest_scc"):
            continue
        p, q = dom_pre[c.pre[mask]], dom_post[c.post[mask]]
        ok = (p >= 0) & (q >= 0)
        syn = c.syn[mask][ok]
        M = np.bincount(p[ok] * K + q[ok], weights=syn, minlength=K * K).reshape(K, K)
        row_sums = M.sum(axis=1, keepdims=True)
        Mn = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
        top = np.dstack(np.unravel_index(np.argsort(M, axis=None)[::-1][:15], M.shape))[0]
        out["variants"][name] = {
            "total_synapses_assigned": int(M.sum()),
            "fraction_synapses_assigned": round(float(M.sum() / c.syn[mask].sum()), 6),
            "top_pairs": [{"pre": names[i], "post": names[j], "synapses": int(M[i, j]),
                           "row_fraction": round(float(Mn[i, j]), 5)} for i, j in top],
            "matrix_file": "neuropil_matrix_%s.npy" % name,
        }
        np.save(_ad() / f"neuropil_matrix_{name}.npy", M)
    return out


def celltype_block(c, variants, levels=("super_class", "cell_type")) -> dict:
    out = {}
    for level in levels:
        labels = c.neurons[level].fillna("UNANNOTATED").to_numpy()
        cats, codes = np.unique(labels.astype(str), return_inverse=True)
        K = cats.size
        per_variant = {}
        for name, mask in variants.items():
            if name not in ("no_aut",):
                continue
            a = codes[c.pre[mask]]
            b = codes[c.post[mask]]
            syn = c.syn[mask].astype(np.float64)
            M = np.bincount(a * K + b, weights=syn, minlength=K * K).reshape(K, K)
            np.save(_ad() / f"celltype_{level}_matrix_{name}.npy", M)
            (_ad() / f"celltype_{level}_labels_{name}.txt").write_text(
                "\n".join(cats.tolist()) + "\n")
            rs = M.sum(axis=1, keepdims=True)
            Mn = np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)
            top = np.argsort(M, axis=None)[::-1][:15]
            per_variant[name] = {
                "n_types": int(K),
                "total_synapses": int(M.sum()),
                "diagonal_fraction_asymmetry": None,
                "top_pairs": [{"pre": str(cats[i // K]), "post": str(cats[i % K]),
                               "synapses": int(M[i // K, i % K]),
                               "row_fraction": round(float(Mn[i // K, i % K]), 5)}
                              for i in top],
            }
        out[level] = per_variant
    return out


def spectral_block(c, variants, k: int = 50) -> dict:
    out = {}
    for name, mask in variants.items():
        if name not in ("no_aut", "largest_scc"):
            continue
        A = to_coo(c.pre[mask], c.post[mask], c.n).tocsr()
        U = undirected_binary(A)
        deg = np.asarray(U.sum(axis=1)).ravel()
        dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg, dtype=float), where=deg > 0)
        D = sparse.diags(dinv)
        L = sparse.eye(c.n) - D @ U @ D                     # symmetric normalized Laplacian
        kk = min(k, c.n - 2)
        vals = splinalg.eigsh(L.tocsc(), k=kk, which="LA", return_eigenvectors=False)
        vals = np.sort(vals)[::-1]
        rad = splinalg.eigsh(U.tocsc(), k=1, which="LA", return_eigenvectors=False)
        out[name] = {
            "laplacian_top50": [round(float(v), 6) for v in vals],
            "laplacian_lambda_max": round(float(vals[0]), 6),
            "adjacency_spectral_radius": round(float(rad[0]), 6),
            "n_isolated_nodes": int((deg == 0).sum()),
        }
    return out


def leiden_block(c, variants, resolution: float = 1.0, seed: int = 0) -> dict:
    import random

    import igraph as ig
    from scipy.sparse import csgraph

    ig.set_random_number_generator(random.Random(seed))
    out = {}
    for name, mask in variants.items():
        if name not in ("no_aut",):
            continue
        A = to_coo(c.pre[mask], c.post[mask], c.n).tocsr()
        # communities on the full graph are dominated by the ~5k disconnected fragments,
        # so report the giant weak component separately
        ncomp, labels = csgraph.connected_components(A, directed=True, connection="weak")
        sizes = np.bincount(labels)
        giant = labels == int(np.argmax(sizes))
        for scope, keep in (("full", np.ones(c.n, dtype=bool)), ("giant_wcc", giant)):
            # isolated vertices would each become a singleton community, so the graph is
            # built only over the vertices that are inside `keep` and membership is mapped back
            local = np.flatnonzero(keep)
            remap = np.full(c.n, -1, dtype=np.int64)
            remap[local] = np.arange(local.size)
            m2 = mask & keep[c.pre] & keep[c.post]
            ei, ej = c.pre[m2], c.post[m2]
            sub = to_coo(ei, ej, c.n).tocsr()
            U = undirected_binary(sub)
            i, j = union_edges(U)
            g = ig.Graph(n=int(local.size), edges=list(zip(remap[i].tolist(), remap[j].tolist())),
                         directed=False)
            part = g.community_leiden(objective_function="modularity", resolution=resolution,
                                      n_iterations=2)
            membership = np.full(c.n, -1, dtype=np.int64)
            membership[local] = np.asarray(part.membership)
            sizes2 = np.bincount(np.asarray(part.membership))
            out[f"{name}_{scope}"] = {
                "algorithm": "Leiden (modularity, resolution=%s, seed=%s)" % (resolution, seed),
                "scope": "all neurons" if scope == "full" else "largest weak component",
                "n_communities": int(len(part)),
                "modularity": round(float(part.modularity), 6),
                "largest_community": int(sizes2.max()),
                "largest_community_fraction": round(float(sizes2.max() / c.n), 6),
                "communities_gt_100": int((sizes2 > 100).sum()),
                "size_quantiles": {q: int(np.percentile(sizes2, q)) for q in (50, 90, 99)},
            }
            if scope == "giant_wcc":
                np.save(_ad() / "leiden_communities_no_aut_giant_wcc.npy", membership)
    return out


def rich_club_block(c, variants, n_bins: int = 14, null_reps: int = 3, seed: int = 0,
                    null_swaps_per_edge: int = 10) -> dict:
    import random

    import igraph as ig
    ig.set_random_number_generator(random.Random(seed))
    out = {}
    for name, mask in variants.items():
        if name not in ("no_aut",):
            continue
        A = to_coo(c.pre[mask], c.post[mask], c.n).tocsr()
        U = undirected_binary(A)
        deg = np.asarray(U.sum(axis=1)).ravel()
        ei, ej = union_edges(U)
        kmax = int(deg.max())
        grid = np.unique(np.round(np.geomspace(2, max(3, kmax), n_bins)).astype(int))

        def phi(k: int, deg_arr: np.ndarray, ei_: np.ndarray, ej_: np.ndarray) -> float:
            rich = np.flatnonzero(deg_arr > k)
            if rich.size < 2:
                return float("nan")
            sel = np.zeros(deg_arr.size, dtype=bool)
            sel[rich] = True
            e_in = int((sel[ei_] & sel[ej_]).sum())
            n_r = rich.size
            return float(2 * e_in / (n_r * (n_r - 1)))

        real = {int(k): phi(int(k), deg, ei, ej) for k in grid}
        rng = np.random.default_rng(seed)
        nulls = {int(k): [] for k in grid}
        g = ig.Graph(n=c.n, edges=list(zip(ei.tolist(), ej.tolist())), directed=False)
        for _ in range(null_reps):
            g2 = g.copy()
            g2.rewire(n=int(null_swaps_per_edge * ei.size), allowed_edge_types="simple")
            e2 = np.asarray(g2.get_edgelist(), dtype=np.int64)
            d2 = np.asarray(g2.degree(), dtype=np.int64)
            for k in grid:
                nulls[int(k)].append(phi(int(k), d2, e2[:, 0], e2[:, 1]))
        norm = {}
        for k in grid:
            vals = [v for v in nulls[int(k)] if v == v and v > 0]
            norm[int(k)] = round(real[int(k)] / np.mean(vals), 5) if vals and real[int(k)] == real[int(k)] else None
        rich_norm = {k: v for k, v in norm.items() if v is not None}
        best = max(rich_norm.items(), key=lambda kv: kv[1]) if rich_norm else (None, None)
        out[name] = {
            "null_model": f"degree-preserving rewiring ({null_reps} reps, "
                          f"{null_swaps_per_edge}*E swaps, simple mode)",
            "k_grid": [int(k) for k in grid],
            "phi_k": {int(k): (round(v, 6) if v == v else None) for k, v in real.items()},
            "phi_norm_k": norm,
            "max_phi_norm": best[1],
            "max_phi_norm_k": best[0],
            "degree_threshold_for_rich_club": next(
                (int(k) for k in grid if rich_norm.get(int(k), 0) > 1.0), None),
            "n_nodes_in_rich_club_at_best": int((deg > best[0]).sum()) if best[0] else None,
        }
    return out


def triad_block(c, variants) -> dict:
    import igraph as ig
    out = {}
    for name, mask in variants.items():
        if name not in ("no_aut",):
            continue
        pre, post = c.pre[mask], c.post[mask]
        g = ig.Graph(n=c.n, edges=list(zip(pre.tolist(), post.tolist())), directed=True)
        t0 = time.time()
        census = g.triad_census()
        out[name] = {
            "triad_census": {lab: int(v) for lab, v in zip(TRIAD_LABELS, census)},
            "n_triads": int(sum(census)),
            "n_triads_connected": int(sum(census) - census[0]),
            "seconds": round(time.time() - t0, 1),
        }
    return out


# --------------------------------------------------------------------------- orchestration
def threshold_robustness(c_full, thresholds: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """How counts, strength and reciprocity move with the synapse threshold.

    Mirrors the robustness check of Lin et al. 2024 (Extended Data Fig. 1c): the
    published numbers correspond to a threshold of five synapses per connection.
    """
    out = {}
    n = c_full.n
    for k in thresholds:
        cc = c_full.thresholded(k) if k > 1 else c_full
        pre, post, syn = cc.pre, cc.post, cc.syn
        no_aut = pre != post
        keys = pre.astype(np.int64) * n + post
        rev = post.astype(np.int64) * n + pre
        rec = np.isin(rev, keys) & no_aut
        deg = np.bincount(pre, minlength=n) + np.bincount(post, minlength=n)
        out[str(int(k))] = {
            "n_connections": int(pre.size),
            "n_synapses_in_connections": int(syn.sum()),
            "mean_synapses_per_connection": round(float(syn.mean()), 4),
            "mean_total_degree": round(float(deg.mean()), 3),
            "n_neurons_total_degree_gt_37": int((deg > 37).sum()),
            "reciprocity": round(float(rec.sum() / max(1, int(no_aut.sum()))), 5),
        }
    return out


def compute_baseline(c, raw_dir: Path, out_json: Path, n_bfs_sources: int = 1000,
                     heavy: bool = False, seed: int = 0, synapse_threshold: int = 5,
                     robustness_thresholds: tuple[int, ...] = (1, 2, 3, 4, 5, 10),
                     only_blocks: list[str] | None = None) -> dict:
    """Compute the Phase 0 metric set, checkpointing to disk after each block.

    `synapse_threshold` selects the connectivity convention. The published FlyWire
    network analyses threshold at five synapses per connection, so 5 is the default and
    is the view whose numbers are comparable with the literature.
    """
    global ARTIFACTS
    raw_dir, out_json = Path(raw_dir), Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS = out_json.parent / "artifacts"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if synapse_threshold > 1:
        c_full = c
        c = c.thresholded(synapse_threshold)
    else:
        c_full = c

    result: dict = {
        "canonical_version": c.meta["canonical_version"],
        "canonical_dir": str(c.dir),
        "canonical_counts": c.meta["counts"],
        "synapse_threshold": int(synapse_threshold),
        "threshold_note": (
            "connections are included when they carry >= synapse_threshold synapses; "
            "this matches Lin et al. 2024 / Dorkenwald et al. 2024 (five synapses)"
            if synapse_threshold > 1 else "no synapse threshold: every published pair included"
        ),
        "seed": seed,
        "blocks": {},
        "timings": {},
    }
    if only_blocks and out_json.exists():
        previous = json.loads(out_json.read_text())
        result["blocks"] = previous.get("blocks", {})
        result["timings"] = previous.get("timings", {})
        result["variants"] = previous.get("variants", {})

    def checkpoint(key: str, value) -> None:
        result["blocks"][key] = value
        out_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

    t0 = time.time()
    variants = graph_variants(c)
    result["variants"] = {k: int(v.sum()) for k, v in variants.items()}
    result["timings"]["variants"] = round(time.time() - t0, 1)

    steps = [
        ("threshold_robustness", lambda: threshold_robustness(c_full, robustness_thresholds)),
        ("counts", lambda: counts_block(c, variants)),
        ("degrees", lambda: degree_block(c, variants)),
        ("components", lambda: components_block(c)),
        ("neurotransmitter", lambda: neurotransmitter_block(c, variants)),
        ("clustering", lambda: clustering_block(c, variants)),
        ("paths", lambda: path_block(c, variants, n_sources=n_bfs_sources, seed=seed)),
        ("leiden", lambda: leiden_block(c, variants, seed=seed)),
        ("spectral", lambda: spectral_block(c, variants)),
        ("neuropil", lambda: neuropil_block(c, raw_dir, variants)),
        ("celltype", lambda: celltype_block(c, variants)),
    ]
    if heavy:
        steps += [("triads", lambda: triad_block(c, variants)),
                  ("rich_club", lambda: rich_club_block(c, variants, seed=seed))]
    if only_blocks:
        steps = [s for s in steps if s[0] in set(only_blocks)]

    for name, fn in steps:
        t = time.time()
        try:
            checkpoint(name, fn())
        except Exception as exc:                                     # keep partial progress
            checkpoint(name, {"error": f"{type(exc).__name__}: {exc}"})
        result["timings"][name] = round(time.time() - t, 1)
        out_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        print(f"[{name}] {result['timings'][name]}s")
    return result
