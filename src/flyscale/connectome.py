"""Canonical, reproducible FlyWire v783 connectome representation (FlyScale Phase 0).

A canonical dataset directory contains:

  meta.json        provenance: source files + checksums, thresholds, counts, conventions
  neurons.parquet  one row per proofread neuron, idx 0..N-1 ordered by ascending root_id,
                   with the Schlegel et al. 2024 annotation columns joined on root_id
  pairs.parquet    one row per unique directed (pre, post) neuron pair:
                   pre_idx, post_idx, syn_count, nt_code, nt_prob_{gaba,ach,glut,oct,ser,da}
  edges.parquet    one row per (pre, post, neuropil) combination exactly as published:
                   pre_idx, post_idx, neuropil_code, syn_count, nt_code
  neuropils.json   neuropil code <-> name
  csr.npz          outgoing and incoming CSR adjacency over the pair graph

Conventions (also written into meta.json so downstream phases cannot drift):

  * neurons are sorted by root_id; all arrays are indexed by that position
  * the pair graph aggregates the published rows over neuropil; it is the canonical graph
    for network statistics
  * autapses (pre_idx == post_idx) are KEPT in the arrays and exposed via mask_autapses so
    every analysis states explicitly whether it drops them
  * nt_code is the argmax of the synapse-count-weighted mean of the six Eckstein, Bates
    et al. 2024 neurotransmitter probabilities; index order is NT_TYPES
  * no synapse or connection threshold beyond the release's own (cleft_score >= 50)
  * all connections are weight 1 in the pair graph; syn_count is carried separately as a
    weight, never silently mixed into topology
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from . import io as fio

CANONICAL_VERSION = "flywire-v783-canon-1"

ANNOTATION_COLUMNS = (
    "super_class", "cell_class", "cell_sub_class", "supertype", "cell_type",
    "hemibrain_type", "ito_lee_hemilineage", "hartenstein_hemilineage",
    "top_nt", "top_nt_conf", "known_nt", "known_nt_source", "side", "nerve",
    "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "status",
    "supervoxel_id", "nucleus_id", "flow", "vfb_id", "fbbt_id",
)


def _sha256_head(path: Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()[:16]


NP_SCALE = 1 << 17   # neuropil code slot in a combined (pair, neuropil) integer key


def _codes_for(names: np.ndarray, registry: dict[str, int]) -> np.ndarray:
    """Map neuropil names to stable integer codes, extending `registry` as needed."""
    cat = pd.Categorical(names)
    lookup = np.empty(len(cat.categories), dtype=np.int64)
    for i, name in enumerate(cat.categories):
        code = registry.get(name)
        if code is None:
            code = len(registry)
            registry[name] = code
        lookup[i] = code
    if len(registry) > NP_SCALE:
        raise ValueError("neuropil registry exceeded the key slot width")
    return lookup[cat.codes].astype(np.int64)


def _reduce_pairs(keys: np.ndarray, weights: np.ndarray, prob_matrix: np.ndarray | None,
                  n_nt: int):
    """Group rows by pair key; return (unique_key, syn_sum, weighted_prob_sum)."""
    uniq, inv = np.unique(keys, return_inverse=True)
    syn = np.bincount(inv, weights=weights, minlength=uniq.size)
    if prob_matrix is None:
        return uniq, syn, None
    ws = np.empty((uniq.size, n_nt), dtype=np.float64)
    for j in range(n_nt):
        ws[:, j] = np.bincount(inv, weights=weights * prob_matrix[:, j], minlength=uniq.size)
    return uniq, syn, ws


def build_canonical(raw_dir: str | Path, out_dir: str | Path,
                    chunk_rows: int = 2_000_000, force: bool = False) -> dict:
    """Build (or reuse) the canonical dataset from the raw FlyWire release files."""
    raw, out = Path(raw_dir), Path(out_dir)
    meta_path = out / "meta.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text())

    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------- neurons
    root_ids = fio.load_root_ids(raw / "proofread_root_ids_783.npy")
    n = int(root_ids.size)

    ann = fio.load_annotations(raw / "Supplemental_file1_neuron_annotations.tsv")
    ann_dupes = int(ann["root_id"].duplicated().sum())
    ann = ann.drop_duplicates(subset="root_id", keep="first")
    ann_ids = set(ann["root_id"].tolist())
    annotated = np.isin(root_ids, list(ann_ids))
    ann = ann.set_index("root_id").reindex(root_ids)
    ann.index.name = "root_id"
    for col in ANNOTATION_COLUMNS:
        if col not in ann.columns:
            ann[col] = np.nan
    neurons = ann[list(ANNOTATION_COLUMNS)].copy()
    neurons.insert(0, "idx", np.arange(n, dtype=np.int32))
    neurons = neurons.reset_index()
    neurons.to_parquet(out / "neurons.parquet", index=False)

    # ---------------------------------------------------------------- connections
    neuropil_codes: dict[str, int] = {}
    part_keys, part_syn, part_ws = [], [], []
    epart_keys, epart_syn, epart_ws = [], [], []
    rows_total = 0
    rows_unmatched = 0
    pair_index: dict[int, int] = {}

    conn_path = raw / "proofread_connections_783.feather"
    for df in fio.iter_feather_chunks(conn_path, chunk_rows):
        fio.check_connections_schema(df)
        rows_total += len(df)
        pre = df["pre_pt_root_id"].to_numpy(np.int64)
        post = df["post_pt_root_id"].to_numpy(np.int64)
        syn = df["syn_count"].to_numpy(np.float64)

        pre_idx = np.searchsorted(root_ids, pre)
        post_idx = np.searchsorted(root_ids, post)
        pre_idx = np.clip(pre_idx, 0, n - 1)
        post_idx = np.clip(post_idx, 0, n - 1)
        ok = (root_ids[np.clip(np.searchsorted(root_ids, pre), 0, n - 1)] == pre) & \
             (root_ids[np.clip(np.searchsorted(root_ids, post), 0, n - 1)] == post)
        rows_unmatched += int((~ok).sum())
        if not ok.all():
            pre_idx, post_idx, syn = pre_idx[ok], post_idx[ok], syn[ok]
            df = df.loc[ok]
        pre_idx = pre_idx.astype(np.int64)
        post_idx = post_idx.astype(np.int64)

        npi = _codes_for(df["neuropil"].astype(str).to_numpy(), neuropil_codes)

        probs = df[list(fio.NT_PROB_COLUMNS)].to_numpy(np.float64)

        key = pre_idx * n + post_idx
        u, s, w = _reduce_pairs(key, syn, probs, len(fio.NT_PROB_COLUMNS))
        part_keys.append(u)
        part_syn.append(s)
        part_ws.append(w)

        key2 = key * NP_SCALE + npi                       # (pair, neuropil) combined key
        u2, inv2 = np.unique(key2, return_inverse=True)
        m = u2.size
        epart_keys.append(u2)
        epart_syn.append(np.bincount(inv2, weights=syn, minlength=m))
        epart_ws.append(np.stack(
            [np.bincount(inv2, weights=syn * probs[:, j], minlength=m)
             for j in range(probs.shape[1])], axis=1))

    # final reduction across chunks
    all_keys = np.concatenate(part_keys)
    all_syn = np.concatenate(part_syn)
    all_ws = np.concatenate(part_ws, axis=0)
    u, inv = np.unique(all_keys, return_inverse=True)
    pair_syn = np.bincount(inv, weights=all_syn, minlength=u.size)
    pair_ws = np.stack([np.bincount(inv, weights=all_ws[:, j], minlength=u.size)
                        for j in range(all_ws.shape[1])], axis=1)

    pair_pre = (u // n).astype(np.int32)
    pair_post = (u % n).astype(np.int32)
    pair_nt_prob = pair_ws / pair_syn[:, None]
    pair_nt = pair_nt_prob.argmax(axis=1).astype(np.int8)

    order = np.lexsort((pair_post, pair_pre))          # sort by pre, then post
    pair_pre, pair_post = pair_pre[order], pair_post[order]
    pair_syn = np.rint(pair_syn[order]).astype(np.int64)
    pair_nt_prob = pair_nt_prob[order]
    pair_nt = pair_nt[order]

    pairs = pd.DataFrame({
        "pre_idx": pair_pre,
        "post_idx": pair_post,
        "syn_count": pair_syn.astype(np.int32),
        "nt_code": pair_nt,
    })
    for j, col in enumerate(fio.NT_PROB_COLUMNS):
        pairs["nt_prob_" + fio.NT_TYPES[j]] = pair_nt_prob[:, j].astype(np.float32)
    pairs.to_parquet(out / "pairs.parquet", index=False)

    # edges table: one row per (pre, post, neuropil)
    ekeys = np.concatenate(epart_keys)
    esyn = np.concatenate(epart_syn)
    ews = np.concatenate(epart_ws, axis=0)
    eu, einv = np.unique(ekeys, return_inverse=True)
    m = eu.size
    e_syn = np.bincount(einv, weights=esyn, minlength=m)
    e_ws = np.stack([np.bincount(einv, weights=ews[:, j], minlength=m)
                     for j in range(ews.shape[1])], axis=1)
    edges = pd.DataFrame({
        "pre_idx": (eu // NP_SCALE // n).astype(np.int32),
        "post_idx": ((eu // NP_SCALE) % n).astype(np.int32),
        "neuropil_code": (eu % NP_SCALE).astype(np.int16),
        "syn_count": np.rint(e_syn).astype(np.int64).astype(np.int32),
        "nt_code": (e_ws / e_syn[:, None]).argmax(axis=1).astype(np.int8),
    })
    edges.to_parquet(out / "edges.parquet", index=False)

    (out / "neuropils.json").write_text(json.dumps({
        "code_to_name": [k for k, _ in sorted(neuropil_codes.items(), key=lambda kv: kv[1])],
        "name_to_code": neuropil_codes,
    }, indent=2) + "\n")

    # ---------------------------------------------------------------- CSR (pair graph)
    out_csr = _csr(pair_pre, pair_post, pair_syn, n, by_row=True)
    in_csr = _csr(pair_pre, pair_post, pair_syn, n, by_row=False)
    np.savez_compressed(
        out / "csr.npz",
        out_indptr=out_csr["indptr"], out_indices=out_csr["indices"],
        out_syn=out_csr["syn"], out_pair_row=out_csr["pair_row"],
        in_indptr=in_csr["indptr"], in_indices=in_csr["indices"],
        in_syn=in_csr["syn"], in_pair_row=in_csr["pair_row"],
    )

    syn_total = int(pair_syn.sum())
    meta = {
        "canonical_version": CANONICAL_VERSION,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t_start, 1),
        "source": {
            "dataset": "FlyWire FAFB v783 (FAFB v783 (CB))",
            "zenodo_doi": "10.5281/zenodo.10676866",
            "annotations": "flyconnectome/flywire_annotations Supplemental_file1 (Schlegel et al. 2024)",
            "connections_file": conn_path.name,
            "connections_sha256_head16": _sha256_head(conn_path),
        },
        "counts": {
            "n_neurons": n,
            "n_annotated_neurons": int(annotated.sum()),
            "n_pairs": int(pair_pre.size),
            "n_edges_rows": int(edges.shape[0]),
            "n_raw_rows_streamed": int(rows_total),
            "n_raw_rows_with_unmatched_root_id": int(rows_unmatched),
            "n_synapses_pairs": syn_total,
            "n_autapse_pairs": int((pair_pre == pair_post).sum()),
            "n_synapses_autapse": int(pair_syn[pair_pre == pair_post].sum()),
            "n_neuropils": len(neuropil_codes),
            "n_reciprocal_pairs": int(_reciprocal_mask(pair_pre, pair_post, n).sum() // 2),
            "n_edges_in_reciprocal_pairs": int(_reciprocal_mask(pair_pre, pair_post, n).sum()),
            "reciprocity": float(_reciprocal_mask(pair_pre, pair_post, n).sum()
                                 / max(1, int((pair_pre != pair_post).sum()))),
        },
        "annotation_duplicate_root_ids": ann_dupes,
        "conventions": {
            "indexing": "neurons sorted by root_id; idx = position",
            "pair_graph": "published rows aggregated over neuropil; all pair weights 1",
            "autapses": "kept; use mask_autapses()",
            "nt_order": list(fio.NT_TYPES),
            "clustering_input": "see flyscale.metrics.graph_variants",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def _reciprocal_mask(pre: np.ndarray, post: np.ndarray, n: int) -> np.ndarray:
    """True for each directed edge whose reverse also exists (autapses excluded)."""
    keys = pre.astype(np.int64) * n + post
    rev = post.astype(np.int64) * n + pre
    out = np.isin(rev, keys) & (pre != post)
    return out


def _csr(pre: np.ndarray, post: np.ndarray, syn: np.ndarray, n: int, by_row: bool) -> dict:
    """CSR adjacency. by_row=True: row=pre (outgoing). by_row=False: row=post (incoming)."""
    if by_row:
        row, col = pre, post
    else:
        row, col = post, pre
    order = np.lexsort((col, row))
    row_s, col_s, syn_s = row[order], col[order], syn[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, row_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return {"indptr": indptr, "indices": col_s.astype(np.int32),
            "syn": syn_s.astype(np.int32), "pair_row": order.astype(np.int64)}


class Connectome:
    """Immutable view over a canonical dataset directory."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.neurons = pd.read_parquet(self.dir / "neurons.parquet")
        self.pairs = pd.read_parquet(self.dir / "pairs.parquet")
        self.edges = pd.read_parquet(self.dir / "edges.parquet")
        self.neuropils = json.loads((self.dir / "neuropils.json").read_text())
        z = np.load(self.dir / "csr.npz")
        self.out_indptr = z["out_indptr"]
        self.out_indices = z["out_indices"]
        self.out_syn = z["out_syn"]
        self.in_indptr = z["in_indptr"]
        self.in_indices = z["in_indices"]
        self.in_syn = z["in_syn"]

    # ---------------------------------------------------------------- basics
    @property
    def n(self) -> int:
        return int(self.meta["counts"]["n_neurons"])

    @property
    def root_ids(self) -> np.ndarray:
        return self.neurons["root_id"].to_numpy(np.int64)

    @property
    def pre(self) -> np.ndarray:
        return self.pairs["pre_idx"].to_numpy()

    @property
    def post(self) -> np.ndarray:
        return self.pairs["post_idx"].to_numpy()

    @property
    def syn(self) -> np.ndarray:
        return self.pairs["syn_count"].to_numpy()

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
                  mask: np.ndarray | None = None) -> sparse.csr_matrix:
        """Directed adjacency over the pair graph, shaped (n, n)."""
        keep = np.ones(self.pre.size, dtype=bool)
        if not include_autapses:
            keep &= self.mask_autapses()
        if mask is not None:
            keep &= mask
        data = self.syn[keep].astype(np.float64) if weighted else np.ones(int(keep.sum()))
        return sparse.csr_matrix((data, (self.pre[keep], self.post[keep])), shape=(self.n, self.n))

    def restricted_to(self, allowed: np.ndarray) -> "Connectome":
        """A shallow sub-connectome view keeping only pairs inside `allowed` (bool array)."""
        return self._view(np.flatnonzero(self.mask_view(allowed)))

    def mask_view(self, allowed: np.ndarray) -> np.ndarray:
        return allowed[self.pre] & allowed[self.post]

    def thresholded(self, min_synapses: int) -> "Connectome":
        """A shallow view keeping only connections with >= `min_synapses` synapses.

        The published FlyWire network analyses apply a threshold of five synapses per
        connection (Lin et al. 2024; Dorkenwald et al. 2024), so this is the view that
        published numbers must be compared against.
        """
        keep = (self.pairs["syn_count"].to_numpy() >= int(min_synapses))
        return self._view(np.flatnonzero(keep))

    def _view(self, row_positions: np.ndarray) -> "Connectome":
        sub = object.__new__(Connectome)
        sub.dir = self.dir
        sub.meta = dict(self.meta)
        sub.neurons, sub.edges, sub.neuropils = self.neurons, self.edges, self.neuropils
        sub.pairs = self.pairs.iloc[row_positions].reset_index(drop=True)
        pre = sub.pairs["pre_idx"].to_numpy()
        post = sub.pairs["post_idx"].to_numpy()
        syn = sub.pairs["syn_count"].to_numpy()
        oc = _csr(pre, post, syn, self.n, by_row=True)
        ic = _csr(pre, post, syn, self.n, by_row=False)
        sub.out_indptr, sub.out_indices, sub.out_syn = oc["indptr"], oc["indices"], oc["syn"]
        sub.in_indptr, sub.in_indices, sub.in_syn = ic["indptr"], ic["indices"], ic["syn"]
        return sub

    def summary(self) -> dict:
        return {
            "n_neurons": self.n,
            "n_connections": int(self.pre.size),
            "n_synapses": int(self.syn.sum()),
            "mean_out_degree": float(self.out_degree().mean()),
            "mean_in_degree": float(self.in_degree().mean()),
            "mean_synapses_per_pair": float(self.syn.mean()),
            "autapse_fraction": float((~self.mask_autapses()).mean()),
        }
