"""Synthetic connectome containers.

Scaling phases (PROJECT-VYBFLY.md §11-§13) produce graphs that are no longer the canonical
dataset: coarse-grained replicas at 0.5x/0.25x/0.1x and subdivided replicas at 2x-100x. They
must still be usable by the same metric code, so `GraphView` implements the subset of the
`flyscale.connectome.Connectome` interface that `flyscale.metrics` touches, plus the latent
coordinates and per-neuron attributes needed for scaling.

Kept deliberately dependency-light: arrays in, arrays out, no implicit parquet round trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .io import NT_TYPES


def _csr(pre: np.ndarray, post: np.ndarray, syn: np.ndarray, n: int, by_row: bool) -> dict:
    row, col = (pre, post) if by_row else (post, pre)
    order = np.lexsort((col, row))
    row_s, col_s, syn_s = row[order], col[order], syn[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, row_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return {"indptr": indptr, "indices": col_s.astype(np.int32),
            "syn": syn_s.astype(np.int32), "pair_row": order.astype(np.int64)}


@dataclass
class GraphView:
    """A connectome-shaped graph: neuron attributes + directed weighted pairs + CSR."""

    n: int
    pre: np.ndarray
    post: np.ndarray
    syn: np.ndarray
    nt_code: np.ndarray
    root_ids: np.ndarray
    cell_type: np.ndarray | None = None
    super_class: np.ndarray | None = None
    top_nt: np.ndarray | None = None
    coords: np.ndarray | None = None                 # latent geometry (N, d)
    provenance: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    # lineage: which source neuron each neuron descends from (scaling provenance)
    parent_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.pre = np.asarray(self.pre, dtype=np.int64)
        self.post = np.asarray(self.post, dtype=np.int64)
        self.syn = np.asarray(self.syn, dtype=np.int64)
        self.nt_code = np.asarray(self.nt_code, dtype=np.int8)
        self.meta = dict(self.meta or {})
        self.meta.setdefault("canonical_version", self.provenance.get("kind", "synthetic"))
        self.meta.setdefault("counts", {})
        self._rebuild_csr()
        self._build_frames()

    # ------------------------------------------------------------------ plumbing
    def _rebuild_csr(self) -> None:
        oc = _csr(self.pre, self.post, self.syn, self.n, by_row=True)
        ic = _csr(self.pre, self.post, self.syn, self.n, by_row=False)
        self.out_indptr, self.out_indices, self.out_syn = oc["indptr"], oc["indices"], oc["syn"]
        self.in_indptr, self.in_indices, self.in_syn = ic["indptr"], ic["indices"], ic["syn"]

    def _build_frames(self) -> None:
        self.pairs = pd.DataFrame({
            "pre_idx": self.pre.astype(np.int32),
            "post_idx": self.post.astype(np.int32),
            "syn_count": self.syn.astype(np.int32),
            "nt_code": self.nt_code,
        })
        cols = {"idx": np.arange(self.n, dtype=np.int64), "root_id": self.root_ids}
        if self.cell_type is not None:
            cols["cell_type"] = self.cell_type
        if self.super_class is not None:
            cols["super_class"] = self.super_class
        if self.top_nt is not None:
            cols["top_nt"] = self.top_nt
        self.neurons = pd.DataFrame(cols)
        self.neuropils = {"code_to_name": [], "name_to_code": {}}

    def replace_pairs(self, pre: np.ndarray, post: np.ndarray, syn: np.ndarray,
                      nt_code: np.ndarray | None = None, **extra) -> None:
        """Swap the edge set in place (attributes/coords unchanged)."""
        self.pre = np.asarray(pre, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int64)
        self.syn = np.asarray(syn, dtype=np.int64)
        if nt_code is not None:
            self.nt_code = np.asarray(nt_code, dtype=np.int8)
        for k, v in extra.items():
            setattr(self, k, v)
        self._rebuild_csr()
        self._build_frames()

    # ------------------------------------------------------------------ Connectome-compatible API
    def mask_autapses(self) -> np.ndarray:
        return self.pre != self.post

    def out_degree(self) -> np.ndarray:
        return np.diff(self.out_indptr)

    def in_degree(self) -> np.ndarray:
        return np.diff(self.in_indptr)

    def weighted_out_degree(self) -> np.ndarray:
        c = np.concatenate(([0], np.cumsum(self.out_syn, dtype=np.int64)))
        return c[self.out_indptr[1:]] - c[self.out_indptr[:-1]]

    def weighted_in_degree(self) -> np.ndarray:
        c = np.concatenate(([0], np.cumsum(self.in_syn, dtype=np.int64)))
        return c[self.in_indptr[1:]] - c[self.in_indptr[:-1]]

    def adjacency(self, include_autapses: bool = True, weighted: bool = False,
                  mask: np.ndarray | None = None):
        from scipy import sparse
        keep = np.ones(self.pre.size, dtype=bool)
        if not include_autapses:
            keep &= self.mask_autapses()
        if mask is not None:
            keep &= mask
        data = self.syn[keep].astype(np.float64) if weighted else np.ones(int(keep.sum()))
        return sparse.coo_matrix((data, (self.pre[keep], self.post[keep])),
                                 shape=(self.n, self.n)).tocsr()

    def thresholded(self, min_synapses: int) -> "GraphView":
        keep = self.syn >= int(min_synapses)
        return self._sub(keep)

    def restricted_to(self, allowed: np.ndarray) -> "GraphView":
        return self._sub(allowed[self.pre] & allowed[self.post])

    def _sub(self, keep: np.ndarray) -> "GraphView":
        g = GraphView(
            n=self.n, pre=self.pre[keep], post=self.post[keep], syn=self.syn[keep],
            nt_code=self.nt_code[keep], root_ids=self.root_ids, cell_type=self.cell_type,
            super_class=self.super_class, top_nt=self.top_nt, coords=self.coords,
            provenance=dict(self.provenance), meta=dict(self.meta),
            parent_index=self.parent_index,
        )
        return g

    def summary(self) -> dict:
        deg = self.out_degree().astype(np.float64)
        return {
            "n_neurons": int(self.n),
            "n_connections": int(self.pre.size),
            "n_synapses": int(self.syn.sum()),
            "mean_out_degree": round(float(deg.mean()), 4),
            "mean_synapses_per_connection": round(float(self.syn.mean()), 4) if self.syn.size else 0.0,
            "density": round(float(self.pre.size / (self.n * (self.n - 1))), 12) if self.n > 1 else 0.0,
        }

    def nt_counts(self) -> dict:
        return {t: int((self.nt_code == i).sum()) for i, t in enumerate(NT_TYPES)}


def from_connectome(c, coords: np.ndarray | None = None,
                    provenance: dict | None = None) -> GraphView:
    """Wrap a canonical Connectome (or any view of it) as a GraphView."""
    return GraphView(
        n=c.n, pre=c.pre, post=c.post, syn=c.syn,
        nt_code=c.pairs["nt_code"].to_numpy(),
        root_ids=c.neurons["root_id"].to_numpy(),
        cell_type=c.neurons["cell_type"].to_numpy() if "cell_type" in c.neurons else None,
        super_class=c.neurons["super_class"].to_numpy() if "super_class" in c.neurons else None,
        top_nt=c.neurons["top_nt"].to_numpy() if "top_nt" in c.neurons else None,
        coords=coords, provenance=provenance or {"kind": "canonical"},
        meta=dict(c.meta),
    )


def compact_pairs(pre: np.ndarray, post: np.ndarray, syn: np.ndarray,
                  nt_code: np.ndarray, threshold: int = 1) -> tuple[np.ndarray, ...]:
    """Aggregate duplicate (pre, post) pairs produced by a scaling operation.

    Sums synapse counts over duplicates and keeps the transmitter of the largest
    contributing connection as the merged pair's transmitter (vectorised: sort by
    (pair, syn) and take the last row of each segment).
    """
    if pre.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i, empty_i, np.zeros(0, dtype=np.int8)
    n = int(max(pre.max(), post.max())) + 1
    key = pre.astype(np.int64) * n + post.astype(np.int64)
    order = np.lexsort((syn, key))
    key_s, syn_s, nt_s = key[order], syn[order], nt_code[order]
    uniq, start = np.unique(key_s, return_index=True)
    sums = np.add.reduceat(syn_s, start)
    bounds = np.append(start[1:], key_s.size) - 1          # last row of each segment
    nt_out = nt_s[bounds].astype(np.int8)
    keep = sums >= threshold
    u = uniq[keep]
    return ((u // n).astype(np.int64), (u % n).astype(np.int64),
            sums[keep].astype(np.int64), nt_out[keep])


def save_graph(g: GraphView, path: str | Path) -> None:
    """Persist a synthetic graph (arrays + attributes; coords as a separate .npy)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "graph.npz",
        n=np.array([g.n]), pre=g.pre, post=g.post, syn=g.syn, nt_code=g.nt_code,
        root_ids=g.root_ids,
        parent_index=g.parent_index if g.parent_index is not None else np.zeros(0, dtype=np.int64),
    )
    if g.coords is not None:
        np.save(path / "coords.npy", g.coords)
    import json
    (path / "graph.json").write_text(json.dumps({
        "n": int(g.n), "summary": g.summary(), "provenance": g.provenance,
    }, indent=2, sort_keys=True, default=str) + "\n")


def load_graph(path: str | Path) -> GraphView:
    path = Path(path)
    z = np.load(path / "graph.npz")
    coords = np.load(path / "coords.npy") if (path / "coords.npy").exists() else None
    import json
    prov = json.loads((path / "graph.json").read_text()).get("provenance", {})
    pi = z["parent_index"]
    return GraphView(n=int(z["n"][0]), pre=z["pre"], post=z["post"], syn=z["syn"],
                     nt_code=z["nt_code"], root_ids=z["root_ids"], coords=coords,
                     provenance=prov, parent_index=pi if pi.size else None)
