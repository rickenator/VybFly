"""Capability scaling benchmarks for renormalized connectomes (PROJECT-VYBFLY.md §9, §19; M12-M14).

This module answers one question with measurements rather than demonstrations: *does the
larger connectome do more?* It runs the same normalized stimulus battery on every scale and
reduces each battery to a small set of capability numbers that can be fitted against N.

WHAT IS MEASURED (exact protocol - identical at every scale, expressed as fractions of N)
---------------------------------------------------------------------------------------------
Dynamics
    The cheap deterministic threshold cascade of ``flyscale.propagation`` (not the LIF engine):
    a neuron fires at step t iff the synapse count arriving from neurons that fired at t-1
    reaches ``relative_threshold`` x the graph's own mean synapses-per-connection
    (``propagation.normalised_threshold``). 8 steps. The relative threshold is calibrated once,
    on the 1x graph, by the pre-declared rule in ``calibrate_relative_threshold`` and then used
    unchanged at every scale.

Stimuli
    A stimulus is a set of ``sample_fraction * N`` seed neurons, drawn without replacement from
    a *stimulus class pool* of ``pool_fraction * N`` neurons. Both are fractions of N, so the
    stimulus occupies the same share of the network at 1x and 10x. Pools are drawn only from
    neurons annotated ``sensory``, ``visual_projection`` or ``optic`` (STIMULUS_CLASSES; 73.6%
    of v783), so the stimulus enters through the same annotated population at every scale.
    Trial-to-trial variability comes from re-drawing the seeds inside the pool; the cascade
    itself is deterministic. Every draw is seeded, so a run is reproducible.

Readout (the SAME readout at every scale)
    ``readout_fraction * N`` neurons chosen uniformly at random with a fixed seed. The response
    feature of one trial is the **binary final activation mask over those readout neurons**
    (1 = active at the last step). Capability is read out by a **nearest-centroid classifier**
    (template matching: the class template is the mean feature vector; the prediction is the
    closest template in Euclidean distance). It is analytic - no iterative training, no
    hyper-parameters, no per-scale tuning - and its cost is O(trials x readout_dim), i.e. linear
    in N. Readout dimension is therefore proportional to N by design; the readout fraction and
    the classifier are never changed between scales.

The seven benchmarks of §9
    sensory discrimination    2-pool 2AFC; the pools overlap by a controlled fraction J, so
                              the stimulus distance is Delta = 1 - J. Reported: accuracy(Delta)
                              and the minimum separable distance Delta* = smallest Delta with
                              test accuracy >= discrimination_threshold (0.75), plus the exact
                              mutual information (bits) between pool identity and decoded
                              response from the confusion matrix.
    memory capacity           M stimulus classes, nearest-centroid trained on
                              ``train_trials_per_class`` trials per class and tested on held-out
                              trials of the same classes; M is grown until accuracy drops below
                              ``capacity_threshold`` (0.9). Reported: accuracy(M), M*, and whether
                              M* is censored by the tested grid.
    temporal depth            delayed match-to-sample: a cue stimulus is injected, then d steps
                              with no input, then a probe stimulus that either matches the cue's
                              class or not. Reported: accuracy(d) and the longest d with accuracy
                              >= temporal_threshold (0.75).
    sequence learning         k stimuli are injected in successive epochs (``sequence_dwell``
                              steps each); the task is to tell the presented order from its
                              reversal. Reported: accuracy(k) and the longest k >= threshold.
    generalization            the class pools are split in half; the readout is trained on one
                              half and tested on the other (novel variants of the same class).
                              Reported: novel-variant accuracy next to the matched
                              same-distribution accuracy, and their gap.
    robustness                neurons or synapses are removed at a fixed random rate, and a fixed
                              reference discrimination task (overlap J = robustness_overlap) is
                              re-presented with identical seeds. Reported: accuracy(rate), the
                              surviving readout fraction, and the interpolated rate at which
                              accuracy falls halfway to chance.
    dynamical complexity      participation ratio of the response covariance (effective
                              dimensionality), response entropy, exact mutual information
                              between stimulus class and decoded response, avalanche/recruitment
                              statistics (branching ratio, log-log size exponent with R^2),
                              and state-space coverage (fraction of neurons recruited by the
                              battery, fraction of readout units ever active, distinct-pattern
                              ratio).

Grounding rule (PROJECT-VYBFLY.md §30)
    This is a measurement module. Nothing here tunes a protocol to make a curve appear: if a
    capability is flat, saturated, or unmeasurable at some scale, the numbers are returned as
    they are and the caller is expected to say so. Every benchmark returns its own
    ``censored``/``note`` fields when its resolution is exhausted.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from .propagation import cascade, normalised_threshold, synapse_scale
from .synthetic import GraphView

#: Annotation classes treated as the stimulable population. The v783 ``super_class`` vocabulary is
#: optic / central / sensory / visual_projection / sensory_ascending / ... ; "sensory" alone is only
#: 12% of N and cannot host many disjoint class pools, so the sensory-facing classes are used
#: together (73.6% of N) and the choice is recorded in every result.
STIMULUS_CLASSES: tuple[str, ...] = ("sensory", "visual_projection", "optic")

#: Revision marker of the *default* protocol below. A protocol change is recorded, never silently
#: applied: v1 was the first full pass (64 classes at 4/2/2 trials, 6 temporal trials per delay,
#: 8 sequence instances, lesion rates 0.1/0.25/0.5); v1 was kept on disk under
#: results/phase9/scales_archive/protocol_v1/, and v2 below raises the resolution of every
#: measurement whose grid or trial count was exhausted at 1x (capacity censored at 64, temporal and
#: sequence accuracies quantised at 1/6 and 1/8, robustness still undegraded at 50% lesion).
PROTOCOL_REVISION = "v3"
PROTOCOL_REVISIONS: dict[str, dict] = {
    "v1": {"n_classes": 64, "train_trials_per_class": 4, "test_trials_per_class": 2,
           "novel_trials_per_class": 2, "capacity_grid": (2, 4, 8, 16, 24, 32, 48, 64),
           "temporal_trials": 6, "sequence_instances": 8,
           "reference_train": 3, "reference_test": 3, "discrimination_train": 3,
           "discrimination_test": 3,
           "lesion_neuron_rates": (0.1, 0.25, 0.5), "lesion_synapse_rates": (0.1, 0.25, 0.5)},
    "v2": {"change": ("more classes with fewer trials each so that capacity is resolvable below the "
                      "grid maximum; 10 trials per temporal delay and 10 sequence instances for "
                      "1/10 accuracy resolution; lesion rates extended to 0.9 because 50% neuron "
                      "loss left the reference task at full accuracy")},
    "v3": {"change": ("test-sample counts raised so that a significance test is possible at the "
                      "same 0.05 level, paid for by trimming measurements whose range was already "
                      "resolved: discrimination 4+4 per side (16 test samples per Delta), temporal "
                      "12 trials per condition per delay (24 test samples), sequences 12 instances "
                      "(12 test samples per k), robustness 5+5 per side (10 test samples per "
                      "lesion rate); class battery 96x3+1+1 trials, temporal delays (0,1,2,4,8), "
                      "neuron-lesion rates (0.5,0.9) because 1x-5x tolerated everything up to 0.7. "
                      "v2 results are kept under results/phase9/scales_archive/protocol_v2/ and "
                      "artifacts_archive/protocol_v2/ so the five v1/v2 headline numbers can be "
                      "compared against v3"),
           "v2_headline_numbers": {"memory_capacity_Mstar": {"1x": 16, "2x": 96, "5x": 96},
                                   "min_separable_delta": {"1x": 0.2, "2x": 0.1, "5x": 0.4},
                                   "temporal_depth_steps": {"1x": 4, "2x": 2, "5x": 4},
                                   "sequence_depth": {"1x": 0, "2x": 0, "5x": 0},
                                   "participation_ratio": {"1x": 194.1, "2x": 295.8, "5x": 382.7},
                                   "stimulus_response_mi_bits": {"1x": 6.143, "2x": 6.517,
                                                                 "5x": 6.585}}},
}


class CapabilityError(RuntimeError):
    """A benchmark could not be run at this scale. Recorded as an explicit error field."""


# --------------------------------------------------------------------------- protocol
@dataclass(frozen=True)
class Protocol:
    """All knobs of the benchmark battery. Fractions are of N, never absolute counts."""

    # stimulus geometry
    pool_fraction: float = 0.004          # stimulus-class pool size, as a fraction of N
    sample_fraction: float = 0.002        # seeds per trial, as a fraction of N
    # readout
    readout_fraction: float = 0.05        # readout population, as a fraction of N
    # dynamics
    relative_threshold: float = 3.0       # calibrated on 1x by calibrate_relative_threshold()
    cascade_steps: int = 8
    max_active_fraction: float = 1.0      # no hard cap on recruited fraction
    # class battery (memory capacity + generalization)
    n_classes: int = 96
    train_trials_per_class: int = 3
    test_trials_per_class: int = 1        # same pool half as training
    novel_trials_per_class: int = 1       # other pool half: novel variants
    capacity_grid: tuple[int, ...] = (2, 4, 8, 16, 32, 48, 64, 80, 96)
    capacity_threshold: float = 0.9
    generalization_classes: int = 16
    # sensory discrimination
    discrimination_overlaps: tuple[float, ...] = (1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0)
    discrimination_pairs: int = 2
    discrimination_train: int = 4
    discrimination_test: int = 4          # 2 pairs x 2 sides x 4 = 16 test samples per Delta
    discrimination_threshold: float = 0.75
    # temporal depth (delayed match-to-sample)
    temporal_delays: tuple[int, ...] = (0, 1, 2, 4, 8)
    temporal_trials: int = 12             # per delay, per condition (match / mismatch)
    temporal_threshold: float = 0.75
    # sequence learning
    sequence_lengths: tuple[int, ...] = (2, 3, 4)
    sequence_dwell: int = 2
    sequence_instances: int = 12          # forward/reversed pair per instance
    sequence_threshold: float = 0.75
    # robustness
    reference_overlap: float = 0.9
    reference_train: int = 5
    reference_test: int = 5
    lesion_neuron_rates: tuple[float, ...] = (0.5, 0.9)
    lesion_synapse_rates: tuple[float, ...] = (0.25, 0.5, 0.7, 0.9)
    # seeds
    seed: int = 0

    def note(self) -> dict:
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        d["stimulus_classes"] = list(STIMULUS_CLASSES)
        d["protocol_revision"] = PROTOCOL_REVISION
        d["protocol_revisions"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                       for kk, vv in v.items()}
                                   for k, v in PROTOCOL_REVISIONS.items()}
        d["dynamics"] = ("synchronous threshold cascade (flyscale.propagation.cascade): fire iff "
                         "synapses arriving from the previous step's active set >= "
                         "relative_threshold x mean synapses per connection in that graph")
        d["readout"] = ("binary final activation mask over readout_fraction x N neurons drawn with "
                        "a fixed seed; nearest-centroid classifier (analytic, no training loop)")
        return d

    def class_pool_size(self, n: int) -> int:
        return max(2, int(round(self.pool_fraction * n)))

    def sample_size(self, n: int) -> int:
        return max(1, int(round(self.sample_fraction * n)))

    def readout_dim(self, n: int) -> int:
        return max(1, int(round(self.readout_fraction * n)))


def fingerprint(proto: Protocol) -> str:
    """Short stable hash of the protocol, for the manifest."""
    payload = repr(sorted((k, str(v)) for k, v in proto.note().items())).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# --------------------------------------------------------------------------- helpers
def _rng(*parts: int) -> np.random.Generator:
    """Seeded generator from an integer tuple (stable across processes and Python versions)."""
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def stimulus_candidates(g: GraphView) -> np.ndarray:
    """Indices of the neurons the stimulus is allowed to enter (STIMULUS_CLASSES)."""
    if getattr(g, "super_class", None) is None:
        return np.arange(g.n, dtype=np.int64)
    vals = np.asarray(["" if v is None else str(v) for v in g.super_class], dtype=object)
    m = np.isin(vals, np.asarray(STIMULUS_CLASSES, dtype=object))
    if not m.any():
        return np.arange(g.n, dtype=np.int64)
    return np.flatnonzero(m).astype(np.int64)


def readout_indices(g: GraphView, proto: Protocol) -> np.ndarray:
    """The fixed readout population: readout_fraction x N neurons, seeded, sorted."""
    rng = _rng(proto.seed, 11)
    size = min(proto.readout_dim(g.n), g.n)
    return np.sort(rng.choice(g.n, size=size, replace=False).astype(np.int64))


def _weights_f64(g: GraphView) -> np.ndarray:
    """Synapse weights as float64, cached on the graph object.

    ``np.bincount(weights=...)`` needs float64, and converting the 28M-edge 10x replica on every
    cascade would dominate the per-cascade cost. The cache is keyed by array size and only valid
    while the graph's edge set is not mutated, which is true for every battery here (lesioning is
    done with masks, never by rebuilding the graph). ``self_check`` asserts that the update rule is
    unchanged by this.
    """
    w = getattr(g, "_capability_syn_f64", None)
    if w is None or w.size != g.out_syn.size:
        w = g.out_syn.astype(np.float64)
        try:
            g._capability_syn_f64 = w
        except Exception:                                      # pragma: no cover - read-only obj
            pass
    return w


def cascade_masked(g: GraphView, seeds: np.ndarray, proto: Protocol,
                   keep_node: np.ndarray | None = None,
                   keep_edge: np.ndarray | None = None) -> dict:
    """The propagation.cascade update rule with optional node/edge keep-masks (lesioning).

    ``keep_node`` (bool over neurons) and ``keep_edge`` (bool over ``g.pre``) delete the masked
    neurons/connections *without* rebuilding the graph: the update is identical to
    ``propagation.cascade`` on the lesioned graph, which ``self_check`` asserts on a small graph.
    With no masks this is exactly ``propagation.cascade`` (also asserted), which is why every
    battery in this module uses it.
    """
    n = g.n
    active = np.zeros(n, dtype=bool)
    seeds = np.asarray(seeds, dtype=np.int64)
    if keep_node is not None:
        seeds = seeds[keep_node[seeds]]
    active[seeds] = True
    thr = normalised_threshold(g, proto.relative_threshold)
    syn = _weights_f64(g)
    frontier = seeds
    curve = [int(active.sum())]
    newly = [int(active.sum())]
    for _ in range(proto.cascade_steps):
        if frontier.size == 0:
            break
        starts = g.out_indptr[frontier]
        counts = g.out_indptr[frontier + 1] - starts
        total = int(counts.sum())
        if total == 0:
            break
        within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        eid = np.repeat(starts, counts) + within
        tgt = g.out_indices[eid]
        w = syn[eid]
        if keep_edge is not None:
            ok = keep_edge[eid]
            tgt, w = tgt[ok], w[ok]
        inp = np.bincount(tgt, weights=w, minlength=n)
        fired = (inp >= thr) & (~active)
        if keep_node is not None:
            fired &= keep_node
        if proto.max_active_fraction < 1.0:
            limit = int(proto.max_active_fraction * n)
            if int(active.sum()) + int(fired.sum()) > limit:
                order = np.argsort(-inp[fired], kind="stable")
                allow = max(0, limit - int(active.sum()))
                idx = np.flatnonzero(fired)[order[:allow]]
                fired = np.zeros(n, dtype=bool)
                fired[idx] = True
        active |= fired
        frontier = np.flatnonzero(fired)
        curve.append(int(active.sum()))
        newly.append(int(fired.sum()))
    return {"_active": active, "active_curve": curve, "newly_active": newly,
            "final_active_fraction": float(active.sum() / n),
            "threshold": thr}


def cascade_epochs(g: GraphView, seed_groups: Sequence[np.ndarray], proto: Protocol) -> dict:
    """Cascade with stimulation injected at successive epochs; empty groups are pure delay steps.

    Same update rule as ``propagation.cascade`` with one step per call: the injected neurons are
    added to the active set *and* to the propagation frontier before the step, then the next epoch
    propagates from the neurons that just fired. An empty group therefore behaves exactly like a
    step with no new input, and ``cascade_epochs([seeds] * steps)`` reproduces ``cascade`` (asserted
    in :func:`self_check`). Returns the activation mask at the end of every epoch.
    """
    n = g.n
    active = np.zeros(n, dtype=bool)
    thr = normalised_threshold(g, proto.relative_threshold)
    syn = _weights_f64(g)
    masks: list[np.ndarray] = []
    newly: list[int] = []
    frontier = np.zeros(0, dtype=np.int64)
    for grp in seed_groups:
        grp = np.asarray(grp, dtype=np.int64)
        if grp.size:
            add = grp[~active[grp]]
            active[add] = True
            frontier = np.concatenate([frontier, add])
        if frontier.size:
            starts = g.out_indptr[frontier]
            counts = g.out_indptr[frontier + 1] - starts
            total = int(counts.sum())
            if total:
                within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
                eid = np.repeat(starts, counts) + within
                inp = np.bincount(g.out_indices[eid], weights=syn[eid], minlength=n)
                fired = (inp >= thr) & (~active)
                active |= fired
                frontier = np.flatnonzero(fired)
                newly.append(int(fired.sum()))
            else:
                frontier = np.zeros(0, dtype=np.int64)
                newly.append(0)
        else:
            newly.append(0)
        masks.append(active.copy())
    return {"epoch_masks": masks, "newly_active": newly, "threshold": thr,
            "final_active_fraction": float(active.sum() / n)}


def _delay_groups(epoch_groups: Sequence[np.ndarray], dwell: int) -> list[np.ndarray]:
    """Expand each epoch into `dwell` stimulation steps (empty groups stay empty = delay)."""
    out: list[np.ndarray] = []
    for grp in epoch_groups:
        for _ in range(max(1, dwell)):
            out.append(np.asarray(grp, dtype=np.int64))
    return out


# --------------------------------------------------------------------------- batteries
def class_pools(g: GraphView, proto: Protocol) -> np.ndarray:
    """(n_classes, pool_size) disjoint stimulus-class pools drawn from the stimulable neurons."""
    cand = stimulus_candidates(g)
    pool = proto.class_pool_size(g.n)
    need = proto.n_classes * pool
    if need > cand.size:
        raise CapabilityError(
            f"need {need} stimulable neurons for {proto.n_classes} pools of {pool}, only "
            f"{cand.size} available at N={g.n}")
    rng = _rng(proto.seed, 101)
    perm = rng.permutation(cand)
    return perm[:need].reshape(proto.n_classes, pool).astype(np.int64)


def run_class_battery(g: GraphView, proto: Protocol, readout: np.ndarray,
                      on_trial=None) -> dict:
    """Stimulus classes with train / same-distribution test / novel-variant test trials."""
    pools = class_pools(g, proto)
    half = pools.shape[1] // 2
    sample = proto.sample_size(g.n)
    per_class = proto.train_trials_per_class + proto.test_trials_per_class + proto.novel_trials_per_class
    total = proto.n_classes * per_class
    R = readout.size
    X = np.zeros((total, R), dtype=np.uint8)
    y = np.zeros(total, dtype=np.int32)
    split = np.zeros(total, dtype=np.int8)
    tid = np.zeros(total, dtype=np.int32)
    newly = np.zeros((total, proto.cascade_steps + 1), dtype=np.int32)
    final_frac = np.zeros(total, dtype=np.float32)
    union = np.zeros(g.n, dtype=bool)
    row = 0
    for c in range(proto.n_classes):
        for t in range(per_class):
            if t < proto.train_trials_per_class:
                half_sel, sp = 0, 0
            elif t < proto.train_trials_per_class + proto.test_trials_per_class:
                half_sel, sp = 0, 1
            else:
                half_sel, sp = 1, 2
            pool = pools[c] if half_sel == 0 else pools[c][half:]
            rng = _rng(proto.seed, 201, c, t)
            seeds = rng.choice(pool, size=min(sample, pool.size), replace=False)
            r = cascade_masked(g, seeds, proto)
            act = r["_active"]
            X[row] = act[readout]
            y[row], split[row], tid[row] = c, sp, t
            nw = np.zeros(proto.cascade_steps + 1, dtype=np.int32)
            nw[:len(r["newly_active"])] = np.asarray(r["newly_active"], dtype=np.int32)
            newly[row] = nw
            final_frac[row] = r["final_active_fraction"]
            union |= act
            row += 1
            if on_trial is not None:
                on_trial(row, total)
    return {
        "name": "classes", "features": X, "class_id": y, "split": split, "trial_index": tid,
        "newly_active": newly, "final_active_fraction": final_frac,
        "union_fraction": float(union.sum() / g.n),
        "n_trials": int(total), "readout_dim": int(R), "n_classes": proto.n_classes,
        "pool_size": int(pools.shape[1]), "sample_size": int(sample),
        "class_pool_fraction_of_stimulable": float(
            proto.n_classes * pools.shape[1] / max(stimulus_candidates(g).size, 1)),
    }


def discrimination_pools(g: GraphView, proto: Protocol) -> list[dict]:
    """Pool pairs whose overlap sets the stimulus distance Delta = 1 - J."""
    cand = stimulus_candidates(g)
    pool = proto.class_pool_size(g.n)
    rng = _rng(proto.seed, 301)
    perm = rng.permutation(cand)
    cursor = 0
    out: list[dict] = []
    for j_index, J in enumerate(proto.discrimination_overlaps):
        shared = int(round(J * pool))
        unique = pool - shared
        for pair in range(proto.discrimination_pairs):
            need = shared + 2 * unique
            if cursor + need > perm.size:
                raise CapabilityError("not enough stimulable neurons for discrimination pools")
            seg = perm[cursor:cursor + need]
            cursor += need
            A = np.concatenate([seg[:shared], seg[shared:shared + unique]])
            B = np.concatenate([seg[:shared], seg[shared + unique:shared + 2 * unique]])
            out.append({"overlap": float(J), "delta": float(1.0 - J), "pair": int(pair),
                        "A": A.astype(np.int64), "B": B.astype(np.int64)})
    return out


def run_discrimination_battery(g: GraphView, proto: Protocol, readout: np.ndarray,
                               pools: list[dict] | None = None, on_trial=None) -> dict:
    pairs = discrimination_pools(g, proto) if pools is None else pools
    sample = proto.sample_size(g.n)
    per_pool = proto.discrimination_train + proto.discrimination_test
    total = len(pairs) * 2 * per_pool
    R = readout.size
    X = np.zeros((total, R), dtype=np.uint8)
    delta = np.zeros(total, dtype=np.float32)
    overlap = np.zeros(total, dtype=np.float32)
    side = np.zeros(total, dtype=np.int8)
    pair_id = np.zeros(total, dtype=np.int32)
    split = np.zeros(total, dtype=np.int8)
    newly = np.zeros((total, proto.cascade_steps + 1), dtype=np.int32)
    final_frac = np.zeros(total, dtype=np.float32)
    union = np.zeros(g.n, dtype=bool)
    row = 0
    for pi, p in enumerate(pairs):
        for s, pool in ((0, p["A"]), (1, p["B"])):
            for t in range(per_pool):
                sp = 0 if t < proto.discrimination_train else 1
                rng = _rng(proto.seed, 401, pi, s, t)
                seeds = rng.choice(pool, size=min(sample, pool.size), replace=False)
                r = cascade_masked(g, seeds, proto)
                act = r["_active"]
                X[row] = act[readout]
                delta[row], overlap[row], side[row] = p["delta"], p["overlap"], s
                pair_id[row], split[row] = pi, sp
                nw = np.zeros(proto.cascade_steps + 1, dtype=np.int32)
                nw[:len(r["newly_active"])] = np.asarray(r["newly_active"], dtype=np.int32)
                newly[row] = nw
                final_frac[row] = r["final_active_fraction"]
                union |= act
                row += 1
                if on_trial is not None:
                    on_trial(row, total)
    return {"name": "discrimination", "features": X, "delta": delta, "overlap": overlap,
            "side": side, "pair_id": pair_id, "split": split, "newly_active": newly,
            "final_active_fraction": final_frac, "union_fraction": float(union.sum() / g.n),
            "n_trials": int(total), "readout_dim": int(R),
            "n_pairs": len(pairs), "pools": pairs}


def run_temporal_battery(g: GraphView, proto: Protocol, readout: np.ndarray,
                         pools: np.ndarray | None = None, on_trial=None) -> dict:
    """Delayed match-to-sample: cue -> d delay steps -> probe (matching or not)."""
    pools = class_pools(g, proto) if pools is None else pools
    sample = proto.sample_size(g.n)
    delays = list(proto.temporal_delays)
    total = len(delays) * proto.temporal_trials * 2
    R = readout.size
    X = np.zeros((total, R), dtype=np.uint8)
    delay_col = np.zeros(total, dtype=np.int32)
    label = np.zeros(total, dtype=np.int8)          # 1 = probe matches the cue class
    trial = np.zeros(total, dtype=np.int32)
    steps_used = np.zeros(total, dtype=np.int32)
    final_frac = np.zeros(total, dtype=np.float32)
    union = np.zeros(g.n, dtype=bool)
    row = 0
    for d in delays:
        for t in range(proto.temporal_trials):
            for lab in (1, 0):
                rng = _rng(proto.seed, 501, d, t, lab)
                cue_class = int(rng.integers(0, proto.n_classes))
                if lab == 1:
                    probe_class = cue_class
                else:
                    probe_class = int((cue_class + 1 + int(rng.integers(0, proto.n_classes - 1)))
                                      % proto.n_classes)
                cue = rng.choice(pools[cue_class], size=min(sample, pools.shape[1]), replace=False)
                probe = rng.choice(pools[probe_class], size=min(sample, pools.shape[1]), replace=False)
                groups = [cue] + [np.zeros(0, dtype=np.int64)] * d + [probe]
                r = cascade_epochs(g, _delay_groups(groups, 1), proto)
                act = r["epoch_masks"][-1]
                X[row] = act[readout]
                delay_col[row], label[row], trial[row] = d, lab, t
                steps_used[row] = len(r["epoch_masks"])
                final_frac[row] = r["final_active_fraction"]
                union |= act
                row += 1
                if on_trial is not None:
                    on_trial(row, total)
    return {"name": "temporal", "features": X, "delay": delay_col, "label": label,
            "trial_index": trial, "steps_used": steps_used,
            "final_active_fraction": final_frac, "union_fraction": float(union.sum() / g.n),
            "n_trials": int(total), "readout_dim": int(R)}


def run_sequence_battery(g: GraphView, proto: Protocol, readout: np.ndarray,
                         pools: np.ndarray | None = None, on_trial=None) -> dict:
    """Order learning: k stimuli in successive epochs; decode the order against its reversal."""
    pools = class_pools(g, proto) if pools is None else pools
    sample = proto.sample_size(g.n)
    lengths = list(proto.sequence_lengths)
    total = len(lengths) * proto.sequence_instances * 2
    R = readout.size
    dwell = max(1, proto.sequence_dwell)
    dim = max(lengths) * dwell * R
    X = np.zeros((total, dim), dtype=np.uint8)
    k_col = np.zeros(total, dtype=np.int32)
    cond = np.zeros(total, dtype=np.int8)          # 1 = as drawn, 0 = reversed
    inst = np.zeros(total, dtype=np.int32)
    dwell = max(1, proto.sequence_dwell)
    for li, k in enumerate(lengths):
        for i in range(proto.sequence_instances):
            rng = _rng(proto.seed, 601, k, i)
            classes = rng.choice(proto.n_classes, size=k, replace=False)
            base = [rng.choice(pools[int(c)], size=min(sample, pools.shape[1]), replace=False)
                    for c in classes]
            for ci, order in enumerate((base, list(reversed(base)))):
                cue_groups = _delay_groups(order, dwell)
                r = cascade_epochs(g, cue_groups, proto)
                row = li * proto.sequence_instances * 2 + i * 2 + ci
                # response = readout mask at the end of every epoch, concatenated (dim = k*R)
                for e, m in enumerate(r["epoch_masks"]):
                    X[row, e * R:(e + 1) * R] = m[readout]
                k_col[row] = k
                cond[row] = 1 if ci == 0 else 0
                inst[row] = i
                if on_trial is not None:
                    on_trial(row + 1, total)
    return {"name": "sequence", "features": X, "length": k_col, "condition": cond,
            "instance": inst, "n_trials": int(total), "readout_dim": int(R),
            "feature_dim": int(dim), "dwell": dwell}


def lesion_masks(g: GraphView, proto: Protocol, rate: float, kind: str) -> np.ndarray:
    """Deterministic keep-mask for removing `rate` of the neurons or of the connections."""
    if kind == "neurons":
        rng = _rng(proto.seed, 701, int(round(rate * 1000)))
        return rng.random(g.n) >= rate
    rng = _rng(proto.seed, 702, int(round(rate * 1000)))
    return rng.random(g.pre.size) >= rate


def run_robustness_battery(g: GraphView, proto: Protocol, readout: np.ndarray,
                           pools: list[dict] | None = None, on_trial=None) -> dict:
    """A fixed reference discrimination task re-presented under neuron/synapse deletion."""
    all_pairs = discrimination_pools(g, proto) if pools is None else pools
    ref = [p for p in all_pairs if abs(p["overlap"] - proto.reference_overlap) < 1e-9]
    if not ref:
        raise CapabilityError(f"no discrimination pool pair with overlap {proto.reference_overlap}")
    ref = ref[:1]                                  # one fixed pool pair: identical stimulus set
    p = ref[0]
    sample = proto.sample_size(g.n)
    per_pool = proto.reference_train + proto.reference_test
    conditions: list[dict] = [{"kind": "none", "rate": 0.0}]
    conditions += [{"kind": "neurons", "rate": r} for r in proto.lesion_neuron_rates]
    conditions += [{"kind": "synapses", "rate": r} for r in proto.lesion_synapse_rates]
    total = len(conditions) * 2 * per_pool
    R = readout.size
    X = np.zeros((total, R), dtype=np.uint8)
    side = np.zeros(total, dtype=np.int8)
    split = np.zeros(total, dtype=np.int8)
    cond_id = np.zeros(total, dtype=np.int32)
    rate_col = np.zeros(total, dtype=np.float32)
    kind_col = np.zeros(total, dtype=np.int8)      # 0 none, 1 neurons, 2 synapses
    surviving_readout = np.zeros(total, dtype=np.float32)
    final_frac = np.zeros(total, dtype=np.float32)
    row = 0
    for ci, cond in enumerate(conditions):
        keep_node = keep_edge = None
        if cond["kind"] == "neurons":
            keep_node = lesion_masks(g, proto, cond["rate"], "neurons")
        elif cond["kind"] == "synapses":
            keep_edge = lesion_masks(g, proto, cond["rate"], "synapses")
        surv = 1.0 if keep_node is None else float(keep_node[readout].mean())
        for s, pool in ((0, p["A"]), (1, p["B"])):
            # the stimulus keeps its size (fraction of N): seeds are re-drawn from the surviving
            # part of the pool instead of shrinking with the lesion
            pool_eff = pool if keep_node is None else pool[keep_node[pool]]
            for t in range(per_pool):
                sp = 0 if t < proto.reference_train else 1
                rng = _rng(proto.seed, 401, 0, s, t)       # same stimuli as the baseline pair 0
                seeds = rng.choice(pool_eff, size=min(sample, pool_eff.size), replace=False)
                r = cascade_masked(g, seeds, proto, keep_node=keep_node, keep_edge=keep_edge)
                X[row] = r["_active"][readout]
                side[row], split[row], cond_id[row] = s, sp, ci
                rate_col[row] = cond["rate"]
                kind_col[row] = {"none": 0, "neurons": 1, "synapses": 2}[cond["kind"]]
                surviving_readout[row] = surv
                final_frac[row] = r["final_active_fraction"]
                row += 1
                if on_trial is not None:
                    on_trial(row, total)
    return {"name": "robustness", "features": X, "side": side, "split": split,
            "condition": cond_id, "lesion_rate": rate_col, "lesion_kind": kind_col,
            "surviving_readout_fraction": surviving_readout, "final_active_fraction": final_frac,
            "n_trials": int(total), "readout_dim": int(R),
            "conditions": [dict(c, index=i) for i, c in enumerate(conditions)],
            "reference_overlap": float(p["overlap"]), "reference_pair": int(p["pair"])}


# --------------------------------------------------------------------------- readout
def nearest_centroid_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Template matching: prediction = label of the closest class-mean feature vector.

    Equivalent to a linear discriminant with a Euclidean metric and a shared-identity
    covariance; analytic and hyper-parameter free, so nothing here can be tuned per scale.
    """
    classes = np.unique(ytr)
    Xf = Xtr.astype(np.float32)
    protos = np.stack([Xf[ytr == c].mean(axis=0) for c in classes]).astype(np.float32)
    scores = Xte.astype(np.float32) @ protos.T
    scores -= 0.5 * np.sum(protos * protos, axis=1)[None, :]
    return classes[np.argmax(scores, axis=1)]


def confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray | None = None) -> dict:
    """Confusion matrix plus exact entropies and mutual information (bits)."""
    classes = np.unique(y_true) if classes is None else np.asarray(classes)
    idx = {int(c): i for i, c in enumerate(classes)}
    k = classes.size
    C = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(np.asarray(y_true), np.asarray(y_pred)):
        if int(a) in idx and int(b) in idx:
            C[idx[int(a)], idx[int(b)]] += 1
    n = C.sum()
    if n == 0:
        return {"n": 0, "accuracy": None, "mutual_information_bits": None}
    P = C / n
    py = P.sum(axis=1)
    pp = P.sum(axis=0)

    def _H(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(P > 0, P / (py[:, None] * pp[None, :]), 1.0)
        mi = float(np.nansum(np.where(P > 0, P * np.log2(ratio), 0.0)))
    H_true, H_pred = _H(py), _H(pp)
    return {
        "n": int(n), "classes": int(k), "n_correct": int(np.trace(C)),
        "accuracy": float(np.trace(C) / n),
        "chance_accuracy": float(1.0 / k),
        "mutual_information_bits": mi,
        "normalised_mutual_information": float(mi / H_true) if H_true > 0 else None,
        "entropy_true_bits": H_true, "entropy_predicted_bits": H_pred,
        "confusion": C.astype(int).tolist(),
    }


def binomial_p_value(k: int, n: int, p0: float = 0.5) -> dict:
    """One-sided exact binomial test that k/n successes exceed chance p0.

    Trial counts here are small (6-16 test samples per condition), so an accuracy above a
    threshold can easily be a noise excursion: every capability decision that depends on a
    threshold is gated on this p-value as well, and both numbers are reported. Caveat recorded in
    the results: trials inside one stimulus class are not independent draws of the same
    population (they share a pool), so this p-value is optimistic - it is a lower bound on the
    evidence, not a calibrated one.
    """
    if n <= 0:
        return {"n_test": 0, "n_correct": 0, "p_value": None, "significant_at_0.05": False}
    try:
        from scipy import stats
        p = float(stats.binomtest(int(k), int(n), p0, alternative="greater").pvalue)
    except Exception:                                          # pragma: no cover - scipy present
        from math import comb
        p = float(sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(int(k), n + 1)))
    return {"n_test": int(n), "n_correct": int(k), "chance": float(p0),
            "p_value": p, "significant_at_0.05": bool(p < 0.05)}


def fisher_improvement(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Fisher exact test of whether condition 2 is better than condition 1 (accuracy counts)."""
    if min(n1, n2) <= 0:
        return {"p_value": None, "significant_at_0.05": False}
    try:
        from scipy import stats
        table = [[int(k1), int(n1 - k1)], [int(k2), int(n2 - k2)]]
        _, p = stats.fisher_exact(table, alternative="less")
    except Exception:                                          # pragma: no cover
        return {"p_value": None, "significant_at_0.05": False, "note": "scipy unavailable"}
    return {"n_test_lower_scale": int(n1), "n_test_higher_scale": int(n2), "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05)}


def fisher_difference(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Two-sided Fisher exact test of a difference between two accuracy counts."""
    if min(n1, n2) <= 0:
        return {"p_value": None, "significant_at_0.05": False}
    try:
        from scipy import stats
        _, p = stats.fisher_exact([[int(k1), int(n1 - k1)], [int(k2), int(n2 - k2)]],
                                  alternative="two-sided")
    except Exception:                                          # pragma: no cover
        return {"p_value": None, "significant_at_0.05": False, "note": "scipy unavailable"}
    return {"p_value": float(p), "significant_at_0.05": bool(p < 0.05),
            "counts": [[int(k1), int(n1 - k1)], [int(k2), int(n2 - k2)]]}


# --------------------------------------------------------------------------- benchmarks
def memory_capacity(battery: dict, proto: Protocol) -> dict:
    """Accuracy as a function of the number of trained stimulus->response associations."""
    y = battery["class_id"]
    curve: dict[str, dict] = {}
    order = list(proto.capacity_grid)
    for M in order:
        if M > proto.n_classes:
            continue
        m_tr = (battery["split"] == 0) & (y < M)
        m_te = (battery["split"] == 1) & (y < M)
        if m_tr.sum() < M or m_te.sum() < 1:
            curve[str(M)] = {"status": "insufficient_trials"}
            continue
        pred = nearest_centroid_predict(battery["features"][m_tr], y[m_tr],
                                        battery["features"][m_te])
        conf = confusion(y[m_te], pred, classes=np.arange(M))
        sig = binomial_p_value(conf["n_correct"], conf["n"], 1.0 / M)
        curve[str(M)] = {"accuracy": conf["accuracy"], "chance_accuracy": conf["chance_accuracy"],
                         "mutual_information_bits": conf["mutual_information_bits"],
                         "n_test": conf["n"], "significance_vs_chance": sig}
    ok = [(int(M), v["accuracy"], v["significance_vs_chance"]["significant_at_0.05"])
          for M, v in curve.items()
          if isinstance(v, dict) and v.get("accuracy") is not None]
    thresholds = (proto.capacity_threshold, 0.95, 0.99)
    at_threshold = {}
    for th in thresholds:
        passing_th = [M for M, a, s in ok if a >= th and s]
        max_grid = max(order)
        at_threshold[str(th)] = {
            "capacity": int(max(passing_th)) if passing_th else 0,
            "censored_at_grid_max": bool(passing_th and max(passing_th) == max_grid)}
    passing = [M for M, a, s in ok if a >= proto.capacity_threshold and s]
    saturated = bool(passing and max(passing) == max(order))
    return {
        "protocol": (f"nearest-centroid on {proto.train_trials_per_class} train trials per class, "
                     f"tested on {proto.test_trials_per_class} held-out trials per class; "
                     f"capacity = largest M with accuracy >= {proto.capacity_threshold} AND "
                     f"p<0.05 above chance (1/M); capacities are also reported at stricter "
                     f"thresholds (0.95, 0.99) so a grid-max censored value still yields a number"),
        "curve": curve,
        "capacity": int(max(passing)) if passing else 0,
        "capacity_bits": float(np.log2(max(passing))) if passing else 0.0,
        "capacity_at_thresholds": at_threshold,
        "capacity_above_chance_only": int(max([M for M, a, _ in ok if a > 1.0 / M])) if ok else 0,
        "censored_at_grid_max": saturated,
        "accuracy_falls_below_threshold_at": next(
            (M for M, a, _ in ok if a < proto.capacity_threshold), None),
        "n_classes_tested": int(max(proto.capacity_grid)),
        "accuracy_at_grid_max": next((v["accuracy"] for k, v in curve.items()
                                      if int(k) == max(order) and isinstance(v, dict)
                                      and v.get("accuracy") is not None), None),
    }


def discrimination(battery: dict, proto: Protocol) -> dict:
    """Minimum separable stimulus distance, from the accuracy-vs-Delta curve."""
    curve: dict[str, dict] = {}
    deltas = sorted({float(d) for d in battery["delta"]})
    for d in deltas:
        accs, infos, ns, ncorr = [], [], 0, 0
        for pair in np.unique(battery["pair_id"]):
            m_tr = (battery["delta"] == d) & (battery["pair_id"] == pair) & (battery["split"] == 0)
            m_te = (battery["delta"] == d) & (battery["pair_id"] == pair) & (battery["split"] == 1)
            if m_tr.sum() < 2 or m_te.sum() < 1:
                continue
            pred = nearest_centroid_predict(battery["features"][m_tr], battery["side"][m_tr],
                                            battery["features"][m_te])
            conf = confusion(battery["side"][m_te], pred, classes=np.array([0, 1]))
            accs.append(conf["accuracy"])
            infos.append(conf["mutual_information_bits"])
            ns += conf["n"]
            ncorr += conf["n_correct"]
        if accs:
            curve[f"{d:.4f}"] = {"delta": round(d, 4), "overlap": round(1.0 - d, 4),
                                 "accuracy": float(ncorr / ns) if ns else None,
                                 "accuracy_per_pair": [float(a) for a in accs],
                                 "mutual_information_bits": float(np.mean(infos)),
                                 "n_test": int(ns),
                                 "significance_vs_chance": binomial_p_value(ncorr, ns, 0.5)}
    sep = [v["delta"] for v in curve.values()
           if v["accuracy"] is not None and v["accuracy"] >= proto.discrimination_threshold
           and v["significance_vs_chance"]["significant_at_0.05"]]
    noisy = [v["delta"] for v in curve.values()
             if v["accuracy"] is not None and v["accuracy"] >= proto.discrimination_threshold
             and not v["significance_vs_chance"]["significant_at_0.05"]]
    return {
        "protocol": (f"2-pool 2AFC on stimulus pools whose overlap J sets Delta = 1 - J; "
                     f"nearest-centroid on {proto.discrimination_train} train trials per side, "
                     f"{proto.discrimination_test} held-out; separable = accuracy >= "
                     f"{proto.discrimination_threshold} AND p<0.05 vs chance"),
        "curve": curve,
        "min_separable_delta": float(min(sep)) if sep else None,
        "separable_within_tested_range": bool(sep),
        "accuracies_above_threshold_but_not_significant": sorted(noisy),
        "pools_identical_control_accuracy": (curve.get("0.0000", {}) or {}).get("accuracy"),
        "accuracy_at_full_difference": (curve.get(f"{max(deltas):.4f}", {}) or {}).get("accuracy"),
    }


def temporal_depth(battery: dict, proto: Protocol) -> dict:
    """Longest delay over which a cue can still be matched to a probe."""
    curve: dict[str, dict] = {}
    for d in sorted({int(x) for x in battery["delay"]}):
        m_tr = (battery["delay"] == d) & (battery["trial_index"] % 2 == 0)
        m_te = (battery["delay"] == d) & (battery["trial_index"] % 2 == 1)
        if m_tr.sum() < 2 or m_te.sum() < 1:
            continue
        pred = nearest_centroid_predict(battery["features"][m_tr], battery["label"][m_tr],
                                        battery["features"][m_te])
        conf = confusion(battery["label"][m_te], pred, classes=np.array([0, 1]))
        sig = binomial_p_value(conf["n_correct"], conf["n"], 0.5)
        curve[str(d)] = {"delay_steps": int(d), "accuracy": conf["accuracy"],
                         "mutual_information_bits": conf["mutual_information_bits"],
                         "n_test": conf["n"], "significance_vs_chance": sig}
    passing = [v["delay_steps"] for v in curve.values()
               if v["accuracy"] >= proto.temporal_threshold
               and v["significance_vs_chance"]["significant_at_0.05"]]
    above = [v["delay_steps"] for v in curve.values()
             if v["accuracy"] >= proto.temporal_threshold]
    return {
        "protocol": ("delayed match-to-sample: cue seeds, d steps with no input, probe seeds; "
                     "decode matching vs mismatching probe from the final activation mask; depth = "
                     "longest delay with accuracy >= threshold AND p<0.05 vs chance"),
        "curve": curve,
        "temporal_depth_steps": int(max(passing)) if passing else 0,
        "delays_above_threshold_but_not_significant": sorted(above),
        "delay_zero_accuracy": (curve.get("0", {}) or {}).get("accuracy"),
        "all_delays_at_chance": bool(not passing and not above),
    }


def sequence_learning(battery: dict, proto: Protocol) -> dict:
    """Longest sequence whose order can be decoded against its reversal."""
    curve: dict[str, dict] = {}
    for k in sorted({int(x) for x in battery["length"]}):
        m = battery["length"] == k
        feats = battery["features"][m]
        label = battery["condition"][m]
        inst = battery["instance"][m]
        tr = np.isin(inst, np.unique(inst)[:: 2])
        te = ~tr
        if tr.sum() < 2 or te.sum() < 1:
            continue
        pred = nearest_centroid_predict(feats[tr], label[tr], feats[te])
        conf = confusion(label[te], pred, classes=np.array([0, 1]))
        curve[str(k)] = {"length": int(k), "accuracy": conf["accuracy"],
                         "mutual_information_bits": conf["mutual_information_bits"],
                         "n_train": int(tr.sum()), "n_test": conf["n"],
                         "significance_vs_chance": binomial_p_value(conf["n_correct"], conf["n"], 0.5)}
    passing = [v["length"] for v in curve.values()
               if v["accuracy"] >= proto.sequence_threshold
               and v["significance_vs_chance"]["significant_at_0.05"]]
    above = [v["length"] for v in curve.values() if v["accuracy"] >= proto.sequence_threshold]
    return {
        "protocol": (f"k stimuli injected in successive epochs ({proto.sequence_dwell} steps each); "
                     "decode presented order vs its reversal from the concatenated per-epoch "
                     "readout masks; depth = longest k with accuracy >= threshold AND p<0.05"),
        "curve": curve,
        "sequence_depth": int(max(passing)) if passing else 0,
        "lengths_above_threshold_but_not_significant": sorted(above),
    }


def generalization(battery: dict, proto: Protocol) -> dict:
    """Accuracy on novel variants of the trained classes vs matched same-distribution trials."""
    M = min(proto.generalization_classes, proto.n_classes)
    y = battery["class_id"]
    out: dict = {"protocol": (f"{M} classes; readout trained on one half of each class pool, "
                              "tested on the other half (novel variants) and on held-out draws "
                              "from the training half (matched control)"),
                 "n_classes": int(M)}
    m_tr = (battery["split"] == 0) & (y < M)
    for name, split_value in (("matched", 1), ("novel", 2)):
        m_te = (battery["split"] == split_value) & (y < M)
        if m_tr.sum() < 2 or m_te.sum() < 1:
            out[name] = {"status": "insufficient_trials"}
            continue
        pred = nearest_centroid_predict(battery["features"][m_tr], y[m_tr],
                                        battery["features"][m_te])
        conf = confusion(y[m_te], pred, classes=np.arange(M))
        out[name] = {"accuracy": conf["accuracy"], "chance_accuracy": conf["chance_accuracy"],
                     "mutual_information_bits": conf["mutual_information_bits"],
                     "n_test": conf["n"], "n_correct": conf["n_correct"],
                     "significance_vs_chance": binomial_p_value(conf["n_correct"], conf["n"],
                                                                1.0 / M)}
    if "accuracy" in out.get("matched", {}) and "accuracy" in out.get("novel", {}):
        out["generalization_gap"] = float(out["matched"]["accuracy"] - out["novel"]["accuracy"])
        out["novel_to_matched_ratio"] = float(out["novel"]["accuracy"] /
                                              max(out["matched"]["accuracy"], 1e-9))
        out["novel_vs_matched_fisher"] = fisher_difference(
            out["novel"]["n_correct"], out["novel"]["n_test"],
            out["matched"]["n_correct"], out["matched"]["n_test"])
    return out


def robustness(battery: dict, proto: Protocol) -> dict:
    """Graceful degradation of the reference task under neuron / synapse removal."""
    out: dict = {"protocol": (f"reference 2-pool discrimination at overlap "
                              f"{proto.reference_overlap}, identical stimulus seeds at every "
                              "lesion rate; per-rate accuracy tested against chance and against "
                              "the unlesioned baseline (Fisher exact)"),
                 "reference_overlap": float(proto.reference_overlap),
                 "curves": {}}
    # baseline (no lesion) accuracy, from the same stimuli
    baseline_conf: dict = {}
    mn = battery["lesion_kind"] == 0
    m_tr, m_te = mn & (battery["split"] == 0), mn & (battery["split"] == 1)
    if m_tr.sum() >= 2 and m_te.sum() >= 1:
        pred = nearest_centroid_predict(battery["features"][m_tr], battery["side"][m_tr],
                                        battery["features"][m_te])
        baseline_conf = confusion(battery["side"][m_te], pred, classes=np.array([0, 1]))
        out["baseline_accuracy"] = baseline_conf["accuracy"]
        out["baseline_mutual_information_bits"] = baseline_conf["mutual_information_bits"]
        out["baseline_n_test"] = baseline_conf["n"]
        out["baseline_significance_vs_chance"] = binomial_p_value(
            baseline_conf["n_correct"], baseline_conf["n"], 0.5)

    for kind, code in (("neurons", 1), ("synapses", 2)):
        rows: list[dict] = []
        for rate in sorted({float(r) for r, k in zip(battery["lesion_rate"], battery["lesion_kind"])
                            if int(k) == code}):
            m = battery["lesion_kind"] == code
            m &= np.isclose(battery["lesion_rate"], rate)
            m_tr = m & (battery["split"] == 0)
            m_te = m & (battery["split"] == 1)
            if m_tr.sum() < 2 or m_te.sum() < 1:
                continue
            pred = nearest_centroid_predict(battery["features"][m_tr], battery["side"][m_tr],
                                            battery["features"][m_te])
            conf = confusion(battery["side"][m_te], pred, classes=np.array([0, 1]))
            row = {"rate": rate, "accuracy": conf["accuracy"],
                   "mutual_information_bits": conf["mutual_information_bits"],
                   "surviving_readout_fraction": float(
                       battery["surviving_readout_fraction"][m].mean()),
                   "mean_final_active_fraction": float(
                       battery["final_active_fraction"][m].mean()),
                   "n_test": conf["n"],
                   "significance_vs_chance": binomial_p_value(conf["n_correct"], conf["n"], 0.5)}
            if baseline_conf:
                row["significance_vs_baseline"] = fisher_improvement(
                    conf["n_correct"], conf["n"],
                    baseline_conf["n_correct"], baseline_conf["n"])
            rows.append(row)
        out["curves"][kind] = rows
        if rows:
            out[f"half_degradation_rate_{kind}"] = _half_degradation(
                [r["rate"] for r in rows],
                [out.get("baseline_accuracy", rows[0]["accuracy"])] + [r["accuracy"] for r in rows])
            base_mi = out.get("baseline_mutual_information_bits")
            if base_mi:
                out[f"mi_half_degradation_rate_{kind}"] = half_degradation_rate(
                    [r["rate"] for r in rows],
                    [base_mi] + [r["mutual_information_bits"] for r in rows], floor=0.0)
                out[f"mi_retention_at_largest_rate_{kind}"] = float(
                    rows[-1]["mutual_information_bits"] / base_mi)
            drops = [r["rate"] for r in rows
                     if (r.get("significance_vs_baseline") or {}).get("significant_at_0.05")]
            out[f"lowest_rate_with_significant_drop_{kind}"] = min(drops) if drops else None
        if rows:
            out[f"maximum_tolerated_rate_{kind}"] = max(
                [r["rate"] for r in rows
                 if r["accuracy"] >= out.get("baseline_accuracy", 1.0) - 1e-9] or [0.0])
    return out


def half_degradation_rate(rates: Sequence[float], values_with_baseline: Sequence[float],
                          floor: float = 0.5) -> float | None:
    """Rate at which a metric falls halfway from its baseline to `floor`, interpolated.

    Used for both robustness curves: accuracy (floor 0.5 = chance, so the target is halfway to
    chance) and mutual information (floor 0.0, so the target is half the baseline bits). Accuracy
    at these test-set sizes picks whole samples and saturates; the MI curve is continuous and is
    reported next to it for that reason.
    """
    ys = np.asarray(list(values_with_baseline), dtype=float)
    xs = np.concatenate([[0.0], np.asarray(rates, dtype=float)])
    if ys.size != xs.size or ys.size < 2:
        return None
    base = float(ys[0])
    target = floor + 0.5 * (base - floor)
    for i in range(1, xs.size):
        if ys[i] <= target <= ys[i - 1] and ys[i - 1] > ys[i]:
            f = (ys[i - 1] - target) / (ys[i - 1] - ys[i])
            return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))
    return None


def _half_degradation(rates: Sequence[float], accs_with_baseline: Sequence[float]) -> float | None:
    """Backwards-compatible wrapper: accuracy halfway to chance."""
    return half_degradation_rate(rates, accs_with_baseline, floor=0.5)


# --------------------------------------------------------------------------- complexity
def participation_ratio(X: np.ndarray) -> dict:
    """Effective dimensionality: participation ratio of the response covariance.

    Computed from the trial-space Gram matrix (identical non-zero eigenvalues to the covariance),
    so the cost is O(T^2 * dim) instead of O(dim^2). PR <= min(T, dim): when it approaches T the
    measurement is censored by the number of trials, which is reported.
    """
    Xf = X.astype(np.float64)
    T = Xf.shape[0]
    if T < 3:
        return {"participation_ratio": None, "n_trials": int(T)}
    Xc = Xf - Xf.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T / (T - 1)
    ev = np.linalg.eigvalsh(G)[::-1]
    ev = np.clip(ev, 0.0, None)
    s1, s2 = float(ev.sum()), float((ev ** 2).sum())
    pr = (s1 ** 2) / s2 if s2 > 0 else None
    ev_n = ev / s1 if s1 > 0 else ev
    cum = np.cumsum(ev_n)
    return {
        "participation_ratio": pr,
        "n_trials": int(T), "readout_dim": int(Xf.shape[1]),
        "top_eigenvalue_fraction": float(ev_n[0]) if s1 > 0 else None,
        "components_for_90pct_variance": int(np.searchsorted(cum, 0.9) + 1) if s1 > 0 else None,
        "censored_at_n_trials": bool(pr is not None and pr >= 0.9 * T),
    }


def response_entropy(X: np.ndarray) -> dict:
    """Mean per-unit binary entropy plus the entropy of the distinct-pattern distribution."""
    Xf = X.astype(np.float64)
    p = Xf.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(np.where(p > 0, p, 1.0)) + (1 - p) * np.log2(np.where(p < 1, 1 - p, 1.0)))
    h = np.nan_to_num(h)
    packed = np.packbits(Xf.astype(np.uint8), axis=1)
    uniq, counts = np.unique(packed, axis=0, return_counts=True)
    q = counts / counts.sum()
    H_pat = float(-(q * np.log2(q)).sum())
    return {
        "mean_unit_binary_entropy_bits": float(h.mean()),
        "max_unit_entropy_bits": 1.0,
        "distinct_patterns": int(uniq.shape[0]),
        "n_trials": int(Xf.shape[0]),
        "distinct_pattern_ratio": float(uniq.shape[0] / Xf.shape[0]),
        "pattern_entropy_bits": H_pat,
        "max_pattern_entropy_bits": float(np.log2(Xf.shape[0])) if Xf.shape[0] > 1 else 0.0,
        "readout_unit_coverage": float((Xf.max(axis=0) > 0).mean()),
        "readout_mean_activity": float(Xf.mean()),
    }


def avalanche_statistics(newly: np.ndarray) -> dict:
    """Recruitment/avalanche statistics over the per-step newly-active counts of a battery.

    "Avalanche" here is the exact object the cascade produces: the number of neurons recruited at
    one step. The size exponent is an empirical descriptor of the empirical distribution, fitted
    on log-binned counts with its R^2 - it is not offered as evidence of criticality.
    """
    A = np.asarray(newly, dtype=np.float64)
    sizes = A.reshape(-1)
    sizes = sizes[sizes > 0]
    per_trial_ratio = []
    ratio_of_sums = []
    for row in A:
        r = row[row > 0]
        if r.size >= 2:
            per_trial_ratio.append(float(np.mean(r[1:] / r[:-1])))
            ratio_of_sums.append(float(r[1:].sum() / max(r[:-1].sum(), 1e-12)))
    out: dict = {
        "n_steps_with_recruitment": int(sizes.size),
        "mean_recruits_per_step": float(sizes.mean()) if sizes.size else None,
        "median_recruits_per_step": float(np.median(sizes)) if sizes.size else None,
        "max_recruits_per_step": float(sizes.max()) if sizes.size else None,
        "branching_ratio_mean_of_ratios": float(np.mean(per_trial_ratio)) if per_trial_ratio else None,
        "branching_ratio_ratio_of_sums": float(np.mean(ratio_of_sums)) if ratio_of_sums else None,
    }
    if sizes.size >= 50:
        lo, hi = np.log10(max(sizes.min(), 1)), np.log10(sizes.max())
        if hi - lo > 0.5:
            bins = np.linspace(lo, hi, 13)
            hist, edges = np.histogram(np.log10(sizes), bins=bins)
            centres = 0.5 * (edges[1:] + edges[:-1])
            keep = hist > 0
            if keep.sum() >= 3:
                x, yl = centres[keep], np.log10(hist[keep])
                slope, intercept = np.polyfit(x, yl, 1)
                pred = slope * x + intercept
                ss_res = float(((yl - pred) ** 2).sum())
                ss_tot = float(((yl - yl.mean()) ** 2).sum())
                out["size_distribution_exponent"] = float(slope)
                out["size_distribution_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
                out["size_distribution_bins"] = int(keep.sum())
    return out


def dynamical_complexity(battery: dict, proto: Protocol, recruited_fraction: float) -> dict:
    """The §9 dynamical-complexity block, all from the class battery."""
    X = battery["features"]
    y = battery["class_id"]
    m_tr = battery["split"] == 0
    m_te = battery["split"] == 1
    ent = response_entropy(X)
    dec = {}
    if m_tr.sum() >= 2 and m_te.sum() >= 1:
        pred = nearest_centroid_predict(X[m_tr], y[m_tr], X[m_te])
        dec = confusion(y[m_te], pred, classes=np.unique(y[m_tr]))
    mi_ceiling = float(np.log2(proto.n_classes))
    mi = dec.get("mutual_information_bits")
    return {
        "effective_dimensionality": participation_ratio(X),
        "response_entropy": ent,
        "stimulus_response_information": {
            "decoder": "nearest-centroid, train trials -> held-out trials",
            "mutual_information_bits": mi,
            "normalised_mutual_information": dec.get("normalised_mutual_information"),
            "entropy_decoded_bits": dec.get("entropy_predicted_bits"),
            "accuracy": dec.get("accuracy"), "chance_accuracy": dec.get("chance_accuracy"),
            "mutual_information_ceiling_bits": mi_ceiling,
            "at_or_above_ceiling": bool(mi is not None and mi >= mi_ceiling - 1e-9),
            "ceiling_note": ("the decoder can never exceed log2(n_classes) = "
                             f"{mi_ceiling:.4f} bits; a value at that ceiling means the "
                             "measurement is bounded by the number of stimulus classes, not by "
                             "the network"),
        },
        "recruitment_statistics": avalanche_statistics(battery["newly_active"]),
        "state_space_coverage": {
            "neurons_recruited_fraction": float(recruited_fraction),
            "mean_final_active_fraction": float(battery["final_active_fraction"].mean()),
            "max_final_active_fraction": float(battery["final_active_fraction"].max()),
            "distinct_patterns": int(ent["distinct_patterns"]),
            "n_trials": int(X.shape[0]),
        },
    }


# --------------------------------------------------------------------------- fits
def fit_power_law(sizes: Sequence[float], values: Sequence[float], label: str = "") -> dict:
    """Least-squares fit of value ~ size^alpha on log-log axes, with R^2."""
    s = np.asarray(sizes, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(s) & np.isfinite(v) & (s > 0) & (v > 0)
    out: dict = {"label": label, "n_points": int(ok.sum()), "sizes": s[ok].tolist(),
                 "values": v[ok].tolist()}
    if ok.sum() < 3:
        out.update({"alpha": None, "r2": None,
                    "note": "fewer than 3 usable positive points - not fitted"})
        return out
    if np.unique(v[ok]).size == 1:
        out.update({"alpha": 0.0, "r2": None,
                    "note": "constant across the fitted scales (no growth in this range)"})
        return out
    ln_s, ln_v = np.log(s[ok]), np.log(v[ok])
    slope, intercept = np.polyfit(ln_s, ln_v, 1)
    pred = slope * ln_s + intercept
    ss_res = float(((ln_v - pred) ** 2).sum())
    ss_tot = float(((ln_v - ln_v.mean()) ** 2).sum())
    out.update({"alpha": float(slope), "intercept": float(intercept),
                "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
                "log_log_residual_rms": float(np.sqrt(ss_res / max(ok.sum() - 2, 1)))})
    return out


def fit_linear(sizes: Sequence[float], values: Sequence[float], label: str = "") -> dict:
    """Least-squares linear fit (for bounded metrics, where a power law is not meaningful)."""
    s = np.asarray(sizes, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(s) & np.isfinite(v)
    if ok.sum() < 3:
        return {"label": label, "n_points": int(ok.sum()), "slope": None, "r2": None}
    slope, intercept = np.polyfit(s[ok], v[ok], 1)
    pred = slope * s[ok] + intercept
    ss_res = float(((v[ok] - pred) ** 2).sum())
    ss_tot = float(((v[ok] - v[ok].mean()) ** 2).sum())
    return {"label": label, "n_points": int(ok.sum()), "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None}


# --------------------------------------------------------------------------- calibration
def choose_reference_overlap(disc: dict, threshold: float = 0.75) -> dict:
    """Pick the robustness reference point from the 1x discrimination curve (protocol setting).

    Rule, applied once on the 1x graph and then frozen for every scale: the smallest stimulus
    distance Delta whose 1x accuracy reaches ``threshold``. If nothing reaches it, the most
    separable Delta actually measured is used and the fallback is recorded - a saturation warning
    for the robustness measurement, not a silent substitution.
    """
    curve = disc.get("curve") or {}
    rows = sorted((v for v in curve.values() if v.get("accuracy") is not None),
                  key=lambda v: v["delta"])
    passing = [v for v in rows if v["accuracy"] >= threshold]
    if passing:
        pick = passing[0]
        rule = f"smallest Delta with 1x accuracy >= {threshold}"
        fallback = False
    elif rows:
        pick = max(rows, key=lambda v: v["accuracy"])
        rule = (f"no Delta reached accuracy {threshold} at 1x; using the most separable Delta "
                f"measured")
        fallback = True
    else:
        return {"overlap": None, "delta": None, "rule": "discrimination curve empty at 1x",
                "fallback": True}
    return {"overlap": float(pick["overlap"]), "delta": float(pick["delta"]),
            "accuracy_at_1x": float(pick["accuracy"]), "rule": rule, "fallback": bool(fallback),
            "curve_at_1x": [{"delta": v["delta"], "accuracy": v["accuracy"]} for v in rows]}


def calibrate_relative_threshold(g: GraphView, proto: Protocol,
                                 grid: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 6.0),
                                 activity_target: float = 0.4,
                                 trials: int = 6) -> dict:
    """Pick the cascade threshold on the 1x graph by a rule fixed before seeing any capability.

    The rule: the smallest relative threshold in the grid whose mean final active fraction stays
    at or below ``activity_target`` (0.4), i.e. the cascade must be sub-saturating so the readout
    carries a distinguishable pattern rather than "everything active". If no grid point qualifies,
    the largest is used and that is recorded. The same value is then used at every scale; the
    per-scale activity fraction that results is reported next to each capability number.
    """
    readout = readout_indices(g, proto)
    cand = stimulus_candidates(g)
    pool_size = proto.class_pool_size(g.n)
    sample = proto.sample_size(g.n)
    rng = _rng(proto.seed, 901)
    pools = rng.permutation(cand)[:2 * pool_size].reshape(2, pool_size)
    table = []
    chosen = grid[-1]
    for rt in grid:
        acts, contrasts = [], []
        feats = {}
        for pi in range(2):
            for t in range(trials):
                r2 = _rng(proto.seed, 911, pi, t)
                seeds = r2.choice(pools[pi], size=sample, replace=False)
                r = cascade(g, seeds, steps=proto.cascade_steps, relative_threshold=float(rt))
                feats[(pi, t)] = r["_active"][readout]
                acts.append(r["final_active_fraction"])
        keys = sorted(feats)
        within, between = [], []
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                dist = float(np.mean(feats[k1] != feats[k2]))
                (within if k1[0] == k2[0] else between).append(dist)
        table.append({"relative_threshold": float(rt),
                      "absolute_threshold": float(normalised_threshold(g, float(rt))),
                      "mean_final_active_fraction": float(np.mean(acts)),
                      "mean_readout_activity": float(np.mean([np.mean(v) for v in feats.values()])),
                      "within_pool_hamming": float(np.mean(within)) if within else None,
                      "between_pool_hamming": float(np.mean(between)) if between else None,
                      "hamming_contrast": (float(np.mean(between) - np.mean(within))
                                           if within and between else None)})
    for row in table:
        if row["mean_final_active_fraction"] <= activity_target:
            chosen = row["relative_threshold"]
            break
    return {"rule": (f"smallest relative threshold in {list(grid)} whose mean final active "
                     f"fraction on the 1x graph is <= {activity_target}; else the largest tested"),
            "activity_target": activity_target, "grid": table, "chosen": float(chosen),
            "chosen_row": next(r for r in table if r["relative_threshold"] == chosen),
            "note": ("calibration is done once on the 1x graph and the chosen value is applied "
                     "unchanged at every scale; the calibration table is kept so the choice can "
                     "be audited")}


# --------------------------------------------------------------------------- self-checks
def self_check() -> dict:
    """Assert the equivalence the robustness battery and the sequence battery rely on.

    On a small synthetic graph: (1) cascade_masked with no masks == propagation.cascade;
    (2) cascade_masked with masks == propagation.cascade on the explicitly lesioned graph;
    (3) cascade_epochs with a single epoch and dwell == cascade steps reproduces cascade exactly.
    """
    import numpy as _np

    rng = _np.random.default_rng(0)
    n = 300
    coords = rng.random((n, 3))
    d = _np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    A = (d < 0.30) & (d > 0)
    pre, post = _np.nonzero(A)
    g = GraphView(n=n, pre=pre, post=post, syn=rng.integers(1, 9, size=pre.size),
                  nt_code=_np.zeros(pre.size, dtype=_np.int8),
                  root_ids=_np.arange(1000, 1000 + n, dtype=_np.int64),
                  super_class=_np.array(["sensory"] * n, dtype=object),
                  coords=coords, provenance={"kind": "self_check"})
    proto = Protocol()
    seeds = _np.arange(0, 5, dtype=_np.int64)
    base = cascade(g, seeds, steps=proto.cascade_steps, relative_threshold=proto.relative_threshold)
    m0 = cascade_masked(g, seeds, proto)
    assert _np.array_equal(base["_active"], m0["_active"]), "masked cascade != cascade (no masks)"

    keep_node = rng.random(n) > 0.3
    keep_edge = rng.random(pre.size) > 0.3
    masked = cascade_masked(g, seeds, proto, keep_node=keep_node, keep_edge=keep_edge)
    keep = keep_edge & keep_node[pre] & keep_node[post]
    les = GraphView(n=n, pre=pre[keep], post=post[keep], syn=g.syn[keep],
                    nt_code=g.nt_code[keep], root_ids=g.root_ids, super_class=g.super_class,
                    coords=coords, provenance={"kind": "self_check_lesion"})
    ref = cascade(les, seeds[keep_node[seeds]], steps=proto.cascade_steps,
                  relative_threshold=proto.relative_threshold)
    assert _np.array_equal(masked["_active"] & keep_node, ref["_active"]), \
        "masked cascade != cascade on the lesioned graph"
    assert _np.array_equal(masked["_active"] & ~keep_node, _np.zeros(n, dtype=bool)), \
        "deleted neurons must never activate"

    ep = cascade_epochs(g, [seeds] * proto.cascade_steps, proto)
    assert _np.array_equal(ep["epoch_masks"][-1], base["_active"]), \
        "cascade_epochs (1 epoch, dwell=steps) != cascade"
    return {"masked_equals_cascade": True, "masks_equal_lesioned_graph": True,
            "epochs_equal_cascade": True,
            "self_check_n": int(n), "self_check_edges": int(pre.size)}
