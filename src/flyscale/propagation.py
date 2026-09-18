"""Signal-propagation dynamics for cross-scale validation (PROJECT-VYBFLY.md §11).

§11 requires that identical, normalized stimuli be run through G1, G0.5 and G0.25 and that
propagation patterns, firing statistics, latency and output distributions be compared. The LIF
engine (Phase 1) is the fuller dynamical model; this module provides the cheap deterministic
cascade that can be run at every scale in seconds, so a scale can be rejected before a
long simulation is spent on it.

Model: synchronous threshold cascade on the directed, synapse-weighted graph. At step t a
neuron integrates the synapse counts arriving from neurons that fired at t-1 (optionally
weighted by the connection's own strength) and fires when the input reaches `threshold`. The
threshold is expressed in *relative* units - a multiple of the graph's own mean synapses per
connection - so the same stimulus is comparable across scales, where a coarse edge aggregates
the synapses of many neuron pairs.
"""
from __future__ import annotations

import numpy as np

from .synthetic import GraphView


def synapse_scale(g: GraphView) -> float:
    """Mean synapses per connection in this graph (the natural threshold unit)."""
    return float(g.syn.mean()) if g.syn.size else 1.0


def normalised_threshold(g: GraphView, relative: float) -> float:
    return float(relative * synapse_scale(g))


def cascade(g: GraphView, seeds: np.ndarray, steps: int = 12, relative_threshold: float = 1.0,
            use_weights: bool = True, max_fraction: float = 1.0) -> dict:
    """Deterministic threshold cascade; returns per-step activation and summary statistics."""
    n = g.n
    active = np.zeros(n, dtype=bool)
    seeds = np.asarray(seeds, dtype=np.int64)
    active[seeds] = True
    curve = [int(active.sum())]
    newly = [int(active.sum())]
    frontier = seeds
    thr = normalised_threshold(g, relative_threshold)
    syn = g.out_syn.astype(np.float64) if use_weights else np.ones(g.out_syn.size)
    limit = int(max_fraction * n)
    for _ in range(steps):
        if frontier.size == 0:
            break
        starts = g.out_indptr[frontier]
        counts = g.out_indptr[frontier + 1] - starts
        total = int(counts.sum())
        if total == 0:
            break
        within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        tgt = g.out_indices[np.repeat(starts, counts) + within]
        w = syn[np.repeat(starts, counts) + within]
        inp = np.bincount(tgt, weights=w, minlength=n)
        fired = (inp >= thr) & (~active)
        if active.sum() + fired.sum() > limit:
            order = np.argsort(-inp[fired], kind="stable")
            allow = max(0, limit - int(active.sum()))
            idx = np.flatnonzero(fired)[order[:allow]]
            fired = np.zeros(n, dtype=bool)
            fired[idx] = True
        active |= fired
        frontier = np.flatnonzero(fired)
        curve.append(int(active.sum()))
        newly.append(int(fired.sum()))
    total_active = int(active.sum())
    half = 0.5 * total_active
    latency = next((i for i, c in enumerate(curve) if c >= half), len(curve) - 1) if half else 0
    return {
        "n_neurons": int(n),
        "steps": len(curve) - 1,
        "relative_threshold": relative_threshold,
        "absolute_threshold": thr,
        "n_seeds": int(seeds.size),
        "seed_fraction": round(float(seeds.size / n), 8),
        "active_curve": curve,
        "active_fraction_curve": [round(c / n, 8) for c in curve],
        "newly_active": newly,
        "final_active": total_active,
        "final_active_fraction": round(total_active / n, 8),
        "latency_steps_to_half": int(latency),
        "peak_newly_active": int(max(newly)),
        "active_set_fraction": round(float(active.sum() / n), 8),
        "active_mask_hash": int(active.view(np.uint8).sum()),
        "_active": active,
    }


def compare_dynamics(a: dict, b: dict, mapping: np.ndarray | None = None) -> dict:
    """Compare two cascades over the same normalized stimulus.

    `mapping[j] = i` means reference neuron j corresponds to replica node i; with it, the
    output distributions are compared as set overlap (Jaccard), otherwise only the curve
    summaries are compared.
    """
    ca = np.asarray(a["active_fraction_curve"], dtype=np.float64)
    cb = np.asarray(b["active_fraction_curve"], dtype=np.float64)
    k = min(ca.size, cb.size)
    out = {
        "curve_L1": round(float(np.abs(ca[:k] - cb[:k]).sum()), 8),
        "curve_max_abs_delta": round(float(np.abs(ca[:k] - cb[:k]).max()), 8),
        "reference_final_fraction": a["final_active_fraction"],
        "replica_final_fraction": b["final_active_fraction"],
        "final_fraction_ratio": round(b["final_active_fraction"] /
                                      max(a["final_active_fraction"], 1e-12), 6),
        "latency_delta_steps": int(b["latency_steps_to_half"] - a["latency_steps_to_half"]),
        "peak_delta": int(b["peak_newly_active"] - a["peak_newly_active"]),
    }
    if mapping is not None and np.asarray(mapping).size:
        # mapping[j] = i maps REFERENCE node j onto REPLICA node i (partition membership or
        # lineage). A replica node counts as active if any of its members is active, so the
        # reference activity is pushed forward through the mapping rather than indexed by an
        # array of different length.
        m = np.asarray(mapping)
        ref_idx = np.flatnonzero(np.asarray(a["_active"], dtype=bool))
        ref_idx = ref_idx[ref_idx < m.size]
        rep_active = np.asarray(b["_active"], dtype=bool)
        mapped_active = np.zeros(rep_active.size, dtype=bool)
        if ref_idx.size:
            mapped_active[np.unique(m[ref_idx])] = True
        k2 = min(mapped_active.size, rep_active.size)
        inter = int((mapped_active[:k2] & rep_active[:k2]).sum())
        union = int((mapped_active[:k2] | rep_active[:k2]).sum())
        out["active_set_jaccard"] = round(float(inter / union), 6) if union else None
        out["active_set_precision"] = round(float(inter / max(int(rep_active[:k2].sum()), 1)), 6)
        out["active_set_recall"] = round(float(inter / max(int(mapped_active[:k2].sum()), 1)), 6)
        out["mapped_nodes"] = int(m.size)
    else:
        out["active_set_jaccard"] = None
    return out


def sensory_seeds(g: GraphView, fraction: float = 0.01, seed: int = 0,
                  prefer_classes: tuple[str, ...] = ("sensory", "visual", "olfactory")) -> np.ndarray:
    """Seeds drawn from annotated sensory classes when available, else uniformly at random.

    Selecting by class rather than by index keeps the stimulus *biologically* matched across
    scales: a coarse node or a parent neuron carries the class annotation forward.
    """
    rng = np.random.default_rng(seed)
    n = g.n
    want = max(1, int(round(fraction * n)))
    cls = getattr(g, "super_class", None)
    if cls is not None:
        vals = np.asarray(["" if v is None else str(v) for v in cls], dtype=object)
        m = np.isin(vals, np.asarray(prefer_classes, dtype=object))
        if m.sum() >= want:
            return rng.choice(np.flatnonzero(m), size=want, replace=False)
    return rng.choice(n, size=want, replace=False)
