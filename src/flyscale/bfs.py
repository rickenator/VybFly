"""Vectorised breadth-first search over a CSR adjacency.

Used for shortest-path distributions on a graph too large for all-pairs work: we sample
source nodes and expand frontiers with numpy slices, which stays in C loops.
"""
from __future__ import annotations

import numpy as np


def bfs_distances(indptr: np.ndarray, indices: np.ndarray, n: int, source: int,
                  max_depth: int | None = None) -> np.ndarray:
    """Unweighted BFS distances from `source`; -1 means unreachable."""
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0
    frontier = np.array([source], dtype=np.int64)
    depth = 0
    while frontier.size and (max_depth is None or depth < max_depth):
        depth += 1
        starts = indptr[frontier]
        counts = indptr[frontier + 1] - starts
        total = int(counts.sum())
        if total == 0:
            break
        within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        neigh = indices[np.repeat(starts, counts) + within]
        neigh = neigh[dist[neigh] < 0]
        if neigh.size == 0:
            break
        neigh = np.unique(neigh)
        dist[neigh] = depth
        frontier = neigh
    return dist


def bfs_multi(indptr: np.ndarray, indices: np.ndarray, n: int, sources: np.ndarray,
              max_depth: int | None = None) -> dict:
    """BFS from many sources; returns aggregate statistics over reachable pairs."""
    hist = np.zeros((max_depth or n) + 1, dtype=np.int64)
    n_reached, diameters, eccs = [], [], []
    for s in sources:
        d = bfs_distances(indptr, indices, n, int(s), max_depth=max_depth)
        reach = d >= 0
        n_reached.append(int(reach.sum()))
        vals = d[reach]
        if vals.size > 0:
            b = np.bincount(vals)
            hist[:b.size] += b
            eccs.append(int(vals.max()))
    n_sources = int(sources.size)
    total = int(hist.sum())
    dists = np.arange(hist.size)
    mean = float((dists * hist).sum() / total) if total else float("nan")
    csum = np.cumsum(hist)
    def q(p: float) -> int:
        return int(np.searchsorted(csum, p * total)) if total else -1
    return {
        "n_sources": n_sources,
        "n_reachable_pairs": total,
        "mean_reachable_fraction": float(np.mean(n_reached) / n),
        "mean_path_length": mean,
        "median_path_length": q(0.5),
        "p90_path_length": q(0.9),
        "p99_path_length": q(0.99),
        "max_path_length": int(dists[hist > 0].max()) if total else -1,
        "mean_eccentricity": float(np.mean(eccs)) if eccs else float("nan"),
        "distance_histogram": {int(k): int(v) for k, v in zip(dists, hist) if v},
    }
