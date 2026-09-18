"""Leaky integrate-and-fire whole-connectome model (FlyScale Phase 1, PROJECT-VYBFLY.md §7).

Phase 1 asks for "the smallest defensible whole-connectome dynamical model": the point is
reproducibility and comparability, not maximal biological realism.  This module therefore
implements exactly one neuron model (LIF with refractory state), one weight rule
(anatomical synapse counts, configurable exponent), one delay rule (configurable), and
**two independent execution engines over the same state update**:

  * :meth:`LIFNetwork.simulate_dense`  - every neuron is integrated on every timestep and
    the synaptic current is a full gather/scatter over the whole edge list (scipy CSR
    matvec per delay group, or a pure-numpy bincount variant).  Work is O(steps * (E + N))
    regardless of how many neurons are active.  This is the **validity reference**.
  * :meth:`LIFNetwork.simulate_events` - sparse, bucket-driven execution: only neurons that
    receive input expand, and only they are state-updated.  A firing neuron scatters its
    outgoing edges into future time buckets (one bucket per future timestep); a silent
    neuron keeps a stale ``Vm`` plus the timestep it was last updated and is advanced
    lazily with the analytic decay factor ``lambda**k``.  Work is
    O(sum over active neurons * out-degree).  This is the CPU reference for the Phase 3
    discrete-event runtime ("CPU DES ~ GPU DES ~ timestep reference", §9).

Both engines share the identical discrete-time recurrence, so they are comparable
bit-for-bit up to floating-point rounding (the dense engine multiplies by ``lambda`` k
times, the event engine multiplies once by a precomputed ``lambda**k``):

    if still refractory:      V = v_reset                                  (input discarded)
    else:                     V = v_rest + (V - v_rest) * lambda + I(t)
    if V >= v_thresh:         emit spike at t; V = v_reset; refractory for ref_steps

with ``lambda = 1 - dt/tau_m`` and ``I(t)`` the total synaptic input delivered at t
(in threshold units: a neuron fires when its input-integrated V reaches ``v_thresh``).

Physics/units
-------------
Time in milliseconds.  Voltages are dimensionless "threshold units" with
``v_rest = 0``, ``v_reset = 0``, ``v_thresh = 1``: one unit of input current integrated
over ``tau_m`` is exactly the threshold.  Synaptic weights carry their own scale
(``g_syn``), so firing rates are set by ``g_syn`` against the measured in-degree.

Connectivity and weight conventions (all configurable, all recorded in the output)
----------------------------------------------------------------------------------
* graph: the canonical FlyWire v783 pair graph, by default thresholded at >= 5 synapses
  per connection, which is the published convention (Lin et al. 2024; Dorkenwald et al.
  2024).  Autapse count is verified to be zero in v783 and asserted.
* weight from anatomy: ``w_anat = syn_count**alpha`` normalised so that the **mean edge
  weight is 1** (``alpha`` defaults to 1.0, i.e. weight proportional to synapse count).
* sign: ``ach`` excitatory, ``gaba`` inhibitory, ``glut`` inhibitory (documented choice:
  most glutamatergic neurons in the adult fly brain are inhibitory), and
  ``oct``/``ser``/``da`` treated as *weak* excitatory modulatory drive scaled by
  ``modulatory_scale`` (they are ~1.7% of connections).  Recorded in the output JSON.
* delay: integer timesteps per edge.  ``kind='distance'`` derives the delay from the
  anatomical distance between the two neurons' soma positions --
  ``delay = clip(round((base_ms + dist_um / speed_um_per_ms) / dt), 1, max_steps)`` --
  which is a *specified* model (the connectome contains no measured delays), not a
  measurement.  ``kind='uniform'`` puts every edge at ``uniform_steps``.

The edge arrays are in CSR order for rows = presynaptic neuron: the canonical dataset
sorts the pair list by ``(pre, post)``, which is verified in :meth:`LIFNetwork._check_csr`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .connectome import Connectome
from .io import NT_TYPES

#: Transmitter -> sign of the synaptic current.  Documented model choice (PROJECT-VYBFLY.md
#: §7 "configurable excitatory/inhibitory behaviour"): acetylcholine is the fast excitatory
#: transmitter of the fly brain, GABA the fast inhibitory one, and glutamate is treated as
#: inhibitory because most glutamatergic neurons in the adult fly brain are inhibitory.
NT_SIGN = {"gaba": -1.0, "ach": +1.0, "glut": -1.0, "oct": +1.0, "ser": +1.0, "da": +1.0}

#: Transmitters that are modulatory rather than fast synaptic; scaled by modulatory_scale.
MODULATORY_NTS = ("oct", "ser", "da")

#: canonical positions are in nanometres (FlyWire FAFB space)
POS_UNITS_PER_UM = 1000.0


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class LIFParams:
    """Leaky integrate-and-fire parameters (one shared set for the whole network)."""

    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    v_rest: float = 0.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    t_ref_ms: float = 2.0

    @property
    def lam(self) -> float:
        """Per-timestep leak factor (V -> v_rest + (V - v_rest) * lam)."""
        return 1.0 - self.dt_ms / self.tau_m_ms

    @property
    def ref_steps(self) -> int:
        """Refractory steps after a spike (minimum inter-spike interval)."""
        return max(1, int(round(self.t_ref_ms / self.dt_ms)))

    def validate(self) -> None:
        if self.dt_ms <= 0 or self.tau_m_ms <= 0:
            raise ValueError("dt_ms and tau_m_ms must be positive")
        if self.v_thresh <= self.v_rest:
            raise ValueError("v_thresh must be above v_rest")
        if not self.v_rest <= self.v_reset <= self.v_thresh:
            raise ValueError("need v_rest <= v_reset <= v_thresh so that a silent neuron "
                             "can never cross the threshold by leaking alone")
        if not 0.0 < self.lam < 1.0:
            raise ValueError(f"dt/tau out of range for a stable leak: lam={self.lam}")

    def to_dict(self) -> dict:
        return {"dt_ms": self.dt_ms, "tau_m_ms": self.tau_m_ms, "v_rest": self.v_rest,
                "v_thresh": self.v_thresh, "v_reset": self.v_reset,
                "t_ref_ms": self.t_ref_ms, "lam_per_step": self.lam,
                "refractory_steps": self.ref_steps}


@dataclass(frozen=True)
class SynapseParams:
    """How anatomical synapse counts become signed synaptic input."""

    alpha: float = 1.0
    normalize_mean_weight: bool = True
    g_syn: float = 0.05
    modulatory_scale: float = 0.25
    inhibitory_gain: float = 1.0
    sign_map: dict = field(default_factory=lambda: dict(NT_SIGN))

    def to_dict(self) -> dict:
        return {"weight_from_synapses": f"w_anat = syn_count**{self.alpha}"
                + (" / mean(syn_count**alpha)  (mean edge weight = 1)"
                   if self.normalize_mean_weight else ""),
                "alpha": self.alpha,
                "normalize_mean_weight": self.normalize_mean_weight,
                "g_syn": self.g_syn,
                "inhibitory_gain": self.inhibitory_gain,
                "input_offset_to_first_spike_note": (
                    "a neuron spikes when the sum of delivered weights reaches 1.0; a "
                    "constant input I per step holds V at I*tau/dt in steady state"),
                "sign_map": dict(self.sign_map),
                "modulatory_transmitters": list(MODULATORY_NTS),
                "modulatory_scale": self.modulatory_scale,
                "sign_convention_note": (
                    "ach excitatory; gaba inhibitory; glut inhibitory (most adult-fly "
                    "glutamatergic neurons are inhibitory); oct/ser/da modulatory, treated "
                    "as weak excitatory and scaled by modulatory_scale; inhibitory edges "
                    "are additionally multiplied by inhibitory_gain, which is a model "
                    "parameter, not an anatomical measurement")}


@dataclass(frozen=True)
class DelayParams:
    """Per-edge synaptic delay, in integer timesteps (the connectome has no measured delays)."""

    kind: str = "distance"          # 'uniform' | 'distance'
    base_ms: float = 1.0
    speed_um_per_ms: float = 100.0
    max_steps: int = 5
    uniform_steps: int = 1

    def to_dict(self) -> dict:
        if self.kind == "uniform":
            model = f"all edges delay {self.uniform_steps} step(s)"
        else:
            model = ("delay_steps = clip(round((base_ms + dist_um/speed_um_per_ms)/dt_ms), "
                     "1, max_steps), dist = Euclidean distance between soma positions "
                     "(nm -> um); a specified model, not a measurement")
        return {"kind": self.kind, "base_ms": self.base_ms,
                "speed_um_per_ms": self.speed_um_per_ms, "max_steps": self.max_steps,
                "uniform_steps": self.uniform_steps, "model": model}


# --------------------------------------------------------------------------- drive
class Drive:
    """External input: for each timestep, a set of target neurons and their input amounts.

    The same object is handed to both engines, so the two engines see byte-identical
    stimulus event streams and any difference is engine numerics, not stimulus sampling.
    """

    n_events: int = 0

    def at(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(target_neuron_indices, input_amounts) injected at timestep ``t``."""
        raise NotImplementedError

    def events_by_step(self) -> np.ndarray:
        """Number of events per timestep (used for reporting/testing)."""
        raise NotImplementedError


class PoissonDrive(Drive):
    """Independent Bernoulli-per-timestep Poisson drive on a fixed target set.

    ``t_start``/``t_end`` restrict the drive to a timestep window (a stimulus pulse);
    outside the window no events exist, so both engines see exactly nothing there.
    """

    def __init__(self, targets: np.ndarray, rate_hz, steps: int, dt_ms: float,
                 amplitude: float, rng: np.random.Generator,
                 t_start: int = 0, t_end: int | None = None):
        targets = np.asarray(targets, dtype=np.int64)
        rate = np.asarray(rate_hz, dtype=np.float64)
        if rate.ndim == 0:
            rate = np.full(targets.size, float(rate))
        p = rate * dt_ms / 1000.0
        if np.any(p > 1.0):
            raise ValueError("drive rate * dt exceeds 1 Hz-ms: not Poisson-approximable")
        w0 = max(0, int(t_start))
        w1 = steps if t_end is None else min(int(steps), int(t_end))
        if w1 < w0:
            raise ValueError("drive window is empty")
        mask = rng.random((targets.size, max(0, w1 - w0))) < p[:, None]
        tgt_i, step_i = np.nonzero(mask)
        step_i = step_i + w0
        order = np.argsort(step_i, kind="stable")
        step_i = step_i[order]
        self.indices = targets[tgt_i[order]].astype(np.int64)
        self.indptr = np.concatenate(
            ([0], np.cumsum(np.bincount(step_i, minlength=steps)))).astype(np.int64)
        self.data = np.full(self.indices.size, float(amplitude))
        self.n_events = int(self.indices.size)
        self.targets = targets
        self.rate_hz = rate
        self.amplitude = float(amplitude)
        self.steps = int(steps)
        self.window = (w0, w1)
        self.expected_events = float((rate * dt_ms / 1000.0).sum() * max(0, w1 - w0))

    def at(self, t: int):
        if t < 0 or t >= self.steps:
            return self.indices[:0], self.data[:0]
        lo, hi = self.indptr[t], self.indptr[t + 1]
        return self.indices[lo:hi], self.data[lo:hi]

    def events_by_step(self) -> np.ndarray:
        return np.diff(self.indptr)

    def summary(self) -> dict:
        rates = np.unique(np.round(self.rate_hz, 6))
        return {"kind": "poisson", "n_target_neurons": int(self.targets.size),
                "rate_hz": (float(rates[0]) if rates.size == 1 else rates.tolist()),
                "amplitude_per_event": self.amplitude, "steps": self.steps,
                "window_steps": list(self.window),
                "n_events": self.n_events,
                "expected_events": round(self.expected_events, 3),
                "mean_events_per_step": round(self.n_events / max(1, self.steps), 3)}


class ConstantDrive(Drive):
    """Constant current injected into a fixed target set over a timestep window."""

    def __init__(self, targets: np.ndarray, amplitude: float, t_start: int, t_end: int):
        self.targets = np.asarray(targets, dtype=np.int64)
        self.amplitude = float(amplitude)
        self.t_start, self.t_end = int(t_start), int(t_end)
        self._amp = np.full(self.targets.size, self.amplitude)
        self.n_events = int(self.targets.size * max(0, self.t_end - self.t_start))

    def at(self, t: int):
        if self.t_start <= t < self.t_end:
            return self.targets, self._amp
        return self.targets[:0], self._amp[:0]

    def events_by_step(self) -> np.ndarray:
        return np.full(self.t_end, self.targets.size)

    def summary(self) -> dict:
        return {"kind": "constant", "n_target_neurons": int(self.targets.size),
                "amplitude": self.amplitude,
                "window_steps": [self.t_start, self.t_end], "n_events": self.n_events}


class CombinedDrive(Drive):
    """Sum of several drives (e.g. a light pulse plus an odour pulse)."""

    def __init__(self, drives: list[Drive]):
        self.drives = list(drives)
        self.n_events = int(sum(d.n_events for d in self.drives))

    def at(self, t: int):
        parts = [d.at(t) for d in self.drives]
        parts = [(i, a) for i, a in parts if i.size]
        if not parts:
            return np.empty(0, dtype=np.int64), np.empty(0)
        return (np.concatenate([p[0] for p in parts]),
                np.concatenate([p[1] for p in parts]))

    def events_by_step(self) -> np.ndarray:
        return np.sum([d.events_by_step() for d in self.drives], axis=0)

    def summary(self) -> dict:
        return {"kind": "combined", "n_events": self.n_events,
                "components": [d.summary() for d in self.drives]}


# --------------------------------------------------------------------------- result
@dataclass
class SimResult:
    """Outcome of one engine run."""

    engine: str
    steps: int
    dt_ms: float
    n_neurons: int
    spike_steps: np.ndarray
    spike_neurons: np.ndarray
    wall_seconds: float
    n_synaptic_events: int
    n_state_updates: int
    n_active_neurons: int
    rate_hz: np.ndarray
    probe: np.ndarray | None = None
    probe_steps: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_spikes(self) -> int:
        return int(self.spike_steps.size)

    @property
    def duration_s(self) -> float:
        return self.steps * self.dt_ms / 1000.0

    def spike_count_per_neuron(self) -> np.ndarray:
        return np.bincount(self.spike_neurons, minlength=self.n_neurons)

    def summary(self) -> dict:
        counts = self.spike_count_per_neuron()
        dur = self.duration_s
        active = int((counts > 0).sum())
        return {
            "engine": self.engine,
            "steps": self.steps,
            "dt_ms": self.dt_ms,
            "simulated_seconds": dur,
            "n_neurons": self.n_neurons,
            "n_spikes": self.n_spikes,
            "spikes_per_biological_second": round(self.n_spikes / dur, 3),
            "mean_rate_hz_over_all_neurons": round(float(counts.sum() / dur / self.n_neurons), 6),
            "mean_rate_hz_over_active_neurons": round(
                float(counts[counts > 0].mean() / dur), 6) if active else 0.0,
            "fraction_neurons_active": round(active / self.n_neurons, 6),
            "n_synaptic_events_delivered": int(self.n_synaptic_events),
            "synaptic_events_per_biological_second": round(self.n_synaptic_events / dur, 1),
            "n_neuron_state_updates": int(self.n_state_updates),
            "neuron_state_updates_per_biological_second": round(self.n_state_updates / dur, 1),
            "n_spiking_neurons": active,
            "spiking_fraction_of_population": round(active / self.n_neurons, 6),
            "wall_seconds": round(self.wall_seconds, 3),
            "wall_seconds_per_simulated_second": round(self.wall_seconds / dur, 3),
            **self.extra,
        }


# --------------------------------------------------------------------------- network
class LIFNetwork:
    """The whole-connectome LIF network: static structure + both execution engines."""

    def __init__(self, c: Connectome, lif: LIFParams | None = None,
                 syn: SynapseParams | None = None, delay: DelayParams | None = None,
                 threshold: int = 5):
        self.connectome = c
        self.threshold = int(threshold)
        self.lif = lif or LIFParams()
        self.syn = syn or SynapseParams()
        self.delay = delay or DelayParams()
        self.lif.validate()

        self.n = int(c.n)
        self.pre = np.asarray(c.pre, dtype=np.int64)
        self.post = np.asarray(c.post, dtype=np.int64)
        self.syn_count = np.asarray(c.syn, dtype=np.int64)
        self.nt_code = np.asarray(c.pairs["nt_code"].to_numpy(), dtype=np.int64)
        self.e = int(self.pre.size)

        if int((self.pre == self.post).sum()) != 0:
            raise ValueError("autapses present; the v783 canonical graph has none, so the "
                             "engines below are not validated for them")

        self.indptr = self._build_csr_indptr()
        self.w_anat = self._edge_weights()
        self.sign = self._edge_signs()
        self.w_edge = self.sign * self.w_anat * self.syn.g_syn
        self.delay_steps = self._edge_delays()

        self.max_delay = int(self.delay_steps.max())
        self.delay_histogram = np.bincount(self.delay_steps,
                                          minlength=self.max_delay + 1).astype(np.int64)
        self.out_degree = np.diff(self.indptr)
        self.nt_histogram = {t: int((self.nt_code == i).sum())
                             for i, t in enumerate(NT_TYPES)}
        self.sign_histogram = {
            "excitatory_edges": int((self.sign > 0).sum()),
            "inhibitory_edges": int((self.sign < 0).sum()),
            "excitatory_synapses": int(self.syn_count[self.sign > 0].sum()),
            "inhibitory_synapses": int(self.syn_count[self.sign < 0].sum())}

        self._csr_by_delay: dict[int, sparse.csr_matrix] | None = None
        self._gather_groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None

    # ---------------------------------------------------------------- structure
    def _build_csr_indptr(self) -> np.ndarray:
        indptr = np.concatenate(
            ([0], np.cumsum(np.bincount(self.pre, minlength=self.n)))).astype(np.int64)
        # the canonical pair list is sorted by (pre, post) => it *is* the CSR edge order;
        # verify rather than assume, because every engine below relies on it
        if not np.array_equal(self.post, np.asarray(self.connectome.out_indices, dtype=np.int64)):
            raise ValueError("pair order is not CSR (row=pre) order; edge arrays cannot be "
                             "used as CSR arrays")
        if not np.array_equal(indptr, np.asarray(self.connectome.out_indptr, dtype=np.int64)):
            raise ValueError("reconstructed indptr disagrees with the canonical CSR")
        return indptr

    def _edge_weights(self) -> np.ndarray:
        """``syn_count**alpha``, normalised so the mean edge weight is 1."""
        raw = np.power(self.syn_count.astype(np.float64), self.syn.alpha)
        if self.syn.normalize_mean_weight:
            raw = raw / raw.mean()
        return raw

    def _edge_signs(self) -> np.ndarray:
        base = np.asarray([self.syn.sign_map[t] for t in NT_TYPES], dtype=np.float64)
        sign = base[self.nt_code]
        scale = np.ones(len(NT_TYPES), dtype=np.float64)
        for t in MODULATORY_NTS:
            scale[NT_TYPES.index(t)] = self.syn.modulatory_scale
        sign = sign * scale[self.nt_code]
        sign[sign < 0] *= self.syn.inhibitory_gain
        return sign

    def _edge_delays(self) -> np.ndarray:
        if self.delay.kind == "uniform":
            return np.full(self.e, int(self.delay.uniform_steps), dtype=np.int64)
        if self.delay.kind != "distance":
            raise ValueError(f"unknown delay kind {self.delay.kind!r}")

        neu = self.connectome.neurons
        pos = neu[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
        ok = np.isfinite(pos).all(axis=1)
        if ok.mean() < 0.99:
            raise ValueError("more than 1% of neurons lack positions; use delay kind 'uniform'")
        p_pre, p_post = pos[self.pre], pos[self.post]
        bad = ~(ok[self.pre] & ok[self.post])
        if bad.any():
            # fall back to the median distance for the few edges with a missing position,
            # so no edge is silently given a degenerate delay
            p_pre = p_pre.copy()
            p_post = p_post.copy()
            med = np.median(np.linalg.norm(p_post[~bad] - p_pre[~bad], axis=1))
            p_pre[bad] = 0.0
            p_post[bad] = med
        dist_um = np.linalg.norm(p_post - p_pre, axis=1) / POS_UNITS_PER_UM
        delay_ms = self.delay.base_ms + dist_um / self.delay.speed_um_per_ms
        steps = np.rint(delay_ms / self.lif.dt_ms).astype(np.int64)
        self.distance_um = dist_um
        return np.clip(steps, 1, int(self.delay.max_steps))

    # ---------------------------------------------------------------- operators
    def csr_in_by_delay(self) -> dict[int, sparse.csr_matrix]:
        """One CSR matrix of signed input weights per distinct delay value.

        The matrices are indexed ``[post, pre]`` (row = target neuron) so that the dense
        timestep gather is the plain matrix-vector product
        ``cur = W_in @ spikes``: ``cur[v] = sum over edges u->v of w(u->v) * spikes[u]``.
        Building them with row = pre instead would silently compute the *transposed*
        network (it turns every edge around), which is what
        :meth:`check_operators` exists to catch.
        """
        if self._csr_by_delay is None:
            groups = {}
            for k in np.unique(self.delay_steps):
                m = self.delay_steps == k
                # (row, col) = (post, pre): the matvec is then the synaptic gather
                groups[int(k)] = sparse.csr_matrix(
                    (self.w_edge[m], (self.post[m], self.pre[m])), shape=(self.n, self.n))
            self._csr_by_delay = groups
        return self._csr_by_delay

    def gather_groups(self) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Edge lists (pre, post, weight) per delay value, for the pure-numpy dense step."""
        if self._gather_groups is None:
            self._gather_groups = {
                int(k): (self.pre[m], self.post[m], self.w_edge[m])
                for k in np.unique(self.delay_steps)
                for m in [self.delay_steps == k]}
        return self._gather_groups

    def check_operators(self, seed: int = 0, n_sources: int = 5000,
                        n_checks: int = 3, tolerance: float = 1e-12) -> dict:
        """Self-check that the synaptic operators inject current in the correct direction.

        For random spike vectors this compares three independent reductions of the same
        edge list: the CSR matvec, the numpy ``bincount`` scatter, and a reference scatter
        over the raw edge arrays.  It also checks the direction explicitly (a single
        active source may only put current on its *targets*).  This matters because the
        transposed operator - row = pre instead of row = post - reverses every edge, which
        is nearly invisible on a random toy graph and completely changes the dynamics of a
        real connectome.
        """
        rng = np.random.default_rng(seed)
        per_delay = {int(k): {"n_edges": int((self.delay_steps == k).sum()),
                              "max_abs_csr_minus_scatter": 0.0,
                              "max_abs_signal": 0.0}
                     for k in np.unique(self.delay_steps)}
        full = 0.0
        for c in range(n_checks):
            s = np.zeros(self.n)
            s[rng.choice(self.n, min(n_sources, self.n), replace=False)] = 1.0
            ref_full = np.bincount(self.post, weights=self.w_edge * s[self.pre],
                                   minlength=self.n)
            got_full = np.zeros(self.n)
            for k, W in self.csr_in_by_delay().items():
                gp, gn, gw = self.gather_groups()[k]
                ref = np.bincount(gn, weights=gw * s[gp], minlength=self.n)
                a = W @ s
                per_delay[k]["max_abs_csr_minus_scatter"] = max(
                    per_delay[k]["max_abs_csr_minus_scatter"], float(np.abs(a - ref).max()))
                per_delay[k]["max_abs_signal"] = max(per_delay[k]["max_abs_signal"],
                                                     float(np.abs(ref).max()))
                got_full += a
            full = max(full, float(np.abs(got_full - ref_full).max()))

        src = int(rng.integers(0, self.n))
        direction = {"checked": False}
        if self.indptr[src + 1] > self.indptr[src]:
            one = np.zeros(self.n)
            one[src] = 1.0
            cur = np.zeros(self.n)
            for k, W in self.csr_in_by_delay().items():
                cur += W @ one
            st, en = self.indptr[src], self.indptr[src + 1]
            targets = np.unique(self.post[st:en])
            got = np.flatnonzero(cur)
            direction = {"checked": True, "source": src, "out_degree": int(en - st),
                         "n_unique_targets": int(targets.size),
                         "n_neurons_with_current": int(got.size),
                         "only_targets_received_current": bool(
                             np.isin(got, targets).all()),
                         "source_received_current": bool(abs(cur[src]) > 0)}

        ok = (full < tolerance
              and all(v["max_abs_csr_minus_scatter"] < tolerance for v in per_delay.values())
              and (not direction["checked"] or direction["only_targets_received_current"]))
        return {"ok": bool(ok), "tolerance": tolerance,
                "n_random_spike_vectors": n_checks, "n_sources_per_vector": n_sources,
                "full_edge_list_max_abs_difference": full,
                "per_delay": {str(k): v for k, v in per_delay.items()},
                "direction_check": direction}

    # ---------------------------------------------------------------- laps
    def _lam_pow(self, steps: int) -> np.ndarray:
        """lambda**k for k = 0..steps (``lam`` is a Python float, so the table reproduces
        the dense engine's repeated multiplication)."""
        k = np.arange(steps + 1, dtype=np.int64)
        return np.power(self.lif.lam, k)

    # ---------------------------------------------------------------- dense engine
    def simulate_dense(self, drive: Drive, steps: int, method: str = "csr",
                       seed: int | None = None, probe_steps: np.ndarray | None = None,
                       v_thresh: float | None = None, t_offset: int = 0) -> SimResult:
        """Timestep integration of every neuron: the validity reference.

        ``method='csr'``    synaptic current by one CSR matvec per delay group (the CSR is
                            indexed [post, pre], so this is the synaptic gather)
        ``method='gather'`` synaptic current by a pure-numpy gather over the edge list and
                            ``np.bincount`` scatter into the postsynaptic neurons

        Both touch every edge on every step, independent of activity.  ``v_thresh``
        overrides the threshold for subthreshold probe runs (no spikes at all then).
        ``t_offset`` shifts the drive/stimulus clock (so a second engine can be restarted
        at a later stimulus time while keeping identical inputs).
        """
        t0 = time.perf_counter()
        lif = self.lif
        lam = lif.lam
        vth = lif.v_thresh if v_thresh is None else float(v_thresh)
        size = self.max_delay + 1
        hist = np.zeros((size, self.n), dtype=np.float64)
        v = np.full(self.n, lif.v_rest, dtype=np.float64)
        refrac_until = np.full(self.n, -1, dtype=np.int64)

        groups_csr = self.csr_in_by_delay() if method == "csr" else None
        groups_np = self.gather_groups() if method == "gather" else None
        if method not in ("csr", "gather"):
            raise ValueError(f"unknown dense method {method!r}")

        spikes_step: list[np.ndarray] = []
        spikes_neuron: list[np.ndarray] = []
        n_events = 0

        probe_idx = None if probe_steps is None else np.asarray(probe_steps, dtype=np.int64)
        probe = np.empty((0 if probe_idx is None else probe_idx.size, self.n))
        probe_at = {} if probe_idx is None else {int(s): i for i, s in enumerate(probe_idx)}

        for t in range(steps):
            cur = np.zeros(self.n, dtype=np.float64)
            if method == "csr":
                for k, W in groups_csr.items():
                    src = hist[(t - k) % size]
                    if src.any():
                        cur += W @ src
            else:
                for k, (gp, gn, gw) in groups_np.items():
                    src = hist[(t - k) % size]
                    if src.any():
                        cur += np.bincount(gn, weights=gw * src[gp], minlength=self.n)

            tg, amp = drive.at(t + t_offset)
            if tg.size:
                cur += self._scatter(tg, amp)

            slot = t % size
            sv = hist[slot]
            sv[:] = 0.0

            refr = refrac_until > t
            if refr.any():
                v[refr] = lif.v_reset
            free = ~refr
            vn = lif.v_rest + (v - lif.v_rest) * lam + cur
            fired = free & (vn >= vth)
            v[free] = vn[free]
            if fired.any():
                idx = np.flatnonzero(fired)
                v[idx] = lif.v_reset
                refrac_until[idx] = t + lif.ref_steps
                n_events += int(self.out_degree[idx].sum())
                spikes_step.append(np.full(idx.size, t, dtype=np.int32))
                spikes_neuron.append(idx.astype(np.int32))
                sv[idx] = 1.0
            if probe_at and t in probe_at:
                probe[probe_at[t]] = v

        counts = np.bincount(np.concatenate(spikes_neuron) if spikes_neuron else
                             np.empty(0, dtype=np.int64), minlength=self.n)
        wall = time.perf_counter() - t0
        return SimResult(
            engine=f"dense_timestep_{method}", steps=steps, dt_ms=lif.dt_ms, n_neurons=self.n,
            spike_steps=(np.concatenate(spikes_step) if spikes_step
                         else np.empty(0, np.int32)),
            spike_neurons=(np.concatenate(spikes_neuron) if spikes_neuron
                           else np.empty(0, np.int32)),
            wall_seconds=wall, n_synaptic_events=n_events,
            n_state_updates=int(self.n * steps),
            n_active_neurons=int((counts > 0).sum()),
            rate_hz=counts / (steps * lif.dt_ms / 1000.0),
            probe=(probe if probe_idx is not None else None),
            probe_steps=probe_idx,
            extra={"method": method, "seed": seed, "v_thresh_used": vth,
                   "work_model": "O(steps * (edges + neurons)): every neuron and every edge "
                                 "is touched on every timestep, independent of activity"})

    # ---------------------------------------------------------------- event engine
    def simulate_events(self, drive: Drive, steps: int, seed: int | None = None,
                        probe_steps: np.ndarray | None = None, v_thresh: float | None = None,
                        t_offset: int = 0) -> SimResult:
        """Sparse bucket-driven engine: only active neurons are expanded and updated.

        Structure: one input bucket per future timestep inside the delay window
        (``acc[k]`` = current to be delivered ``k`` steps from now, plus a list of touched
        targets), a lazily-decoded ``Vm`` per neuron, and per-bucket scatter of the firing
        neurons' outgoing edges.  A neuron that receives nothing is never touched: its
        state is advanced analytically with ``lambda**k`` when it next wakes.
        """
        t0 = time.perf_counter()
        lif = self.lif
        vth = lif.v_thresh if v_thresh is None else float(v_thresh)
        size = self.max_delay + 1
        lam_pow = self._lam_pow(steps + lif.ref_steps + 2)

        indptr, post_e, w_e, d_e = self.indptr, self.post, self.w_edge, self.delay_steps
        bucket = np.zeros((size, self.n), dtype=np.float64)
        touched: list[list[np.ndarray]] = [[] for _ in range(size)]

        v = np.full(self.n, lif.v_rest, dtype=np.float64)
        last = np.zeros(self.n, dtype=np.int64)          # state valid as of this step
        next_ok = np.zeros(self.n, dtype=np.int64)       # first step not refractory

        spikes_step: list[np.ndarray] = []
        spikes_neuron: list[np.ndarray] = []
        n_events = 0
        n_updates = 0
        woken_any = np.zeros(self.n, dtype=bool)

        probe_idx = None if probe_steps is None else np.asarray(probe_steps, dtype=np.int64)
        probe = np.empty((0 if probe_idx is None else probe_idx.size, self.n))
        probe_at = {} if probe_idx is None else {int(s): i for i, s in enumerate(probe_idx)}

        for t in range(steps):
            slot = t % size

            # --- inputs delivered at this step: internal bucket + external drive
            if touched[slot]:
                # a target can be appended more than once (several sources, or the same
                # source reached from different steps in this bucket), and the bucket
                # already holds the *summed* value, so the index list must be deduplicated
                # before it is read - otherwise the total is delivered twice
                idx_in = np.unique(np.concatenate(touched[slot]))
                touched[slot] = []
            else:
                idx_in = np.empty(0, dtype=np.int64)
            in_val = bucket[slot, idx_in]
            if idx_in.size:
                bucket[slot, idx_in] = 0.0
            tg, amp = drive.at(t + t_offset)
            if idx_in.size or tg.size:
                wake = np.concatenate([idx_in, tg.astype(np.int64)])
                vals = np.concatenate([in_val, amp])
                wake, inv = np.unique(wake, return_inverse=True)
                cur = np.bincount(inv, weights=vals, minlength=wake.size)
            else:
                wake = np.empty(0, dtype=np.int64)
                cur = np.empty(0, dtype=np.float64)

            if wake.size:
                k = t - last[wake]
                na = next_ok[wake]
                still = na > t                            # still refractory: input discarded
                resumed = (na > last[wake]) & ~still      # left refractory during the gap
                vv = lif.v_rest + (v[wake] - lif.v_rest) * lam_pow[k] + cur
                if resumed.any():
                    # state at step (na - 1) was v_reset, so na - 1 leak steps precede t
                    vv[resumed] = (lif.v_rest + (lif.v_reset - lif.v_rest)
                                   * lam_pow[t - na[resumed] + 1] + cur[resumed])
                if still.any():
                    vv[still] = lif.v_reset
                fired_local = vv >= vth
                v[wake] = vv
                last[wake] = t
                n_updates += int(wake.size)
                woken_any[wake] = True
                if fired_local.any():
                    fired = wake[fired_local]
                    v[fired] = lif.v_reset
                    next_ok[fired] = t + lif.ref_steps
                    last[fired] = t
                    # expand only the firing neurons' outgoing edges into future buckets
                    st = indptr[fired]
                    cnt = indptr[fired + 1] - st
                    tot = int(cnt.sum())
                    if tot:
                        rep = np.repeat(np.arange(fired.size), cnt)
                        flat = np.repeat(st, cnt) + (np.arange(tot) - np.repeat(
                            np.cumsum(cnt) - cnt, cnt))
                        tgts = post_e[flat]
                        ws = w_e[flat]
                        ds = d_e[flat]
                        for k_d in np.unique(ds):
                            sel = ds == k_d
                            b = (t + int(k_d)) % size
                            u, inv = np.unique(tgts[sel], return_inverse=True)
                            bucket[b, u] += np.bincount(inv, weights=ws[sel], minlength=u.size)
                            touched[b].append(u)
                        n_events += tot
                    spikes_step.append(np.full(fired.size, t, dtype=np.int32))
                    spikes_neuron.append(fired.astype(np.int32))
            if probe_at and t in probe_at:
                probe[probe_at[t]] = self._materialize(v, last, next_ok, t, lam_pow)

        counts = np.bincount(np.concatenate(spikes_neuron) if spikes_neuron else
                             np.empty(0, dtype=np.int64), minlength=self.n)
        wall = time.perf_counter() - t0
        return SimResult(
            engine="sparse_events", steps=steps, dt_ms=lif.dt_ms, n_neurons=self.n,
            spike_steps=(np.concatenate(spikes_step) if spikes_step
                         else np.empty(0, np.int32)),
            spike_neurons=(np.concatenate(spikes_neuron) if spikes_neuron
                           else np.empty(0, np.int32)),
            wall_seconds=wall, n_synaptic_events=n_events, n_state_updates=n_updates,
            n_active_neurons=int((counts > 0).sum()),
            rate_hz=counts / (steps * lif.dt_ms / 1000.0),
            probe=(probe if probe_idx is not None else None),
            probe_steps=probe_idx,
            extra={"seed": seed, "v_thresh_used": vth,
                   "work_model": "O(sum over active neurons * out-degree): silent neurons are "
                                 "never touched, their state is recovered analytically",
                   "n_neurons_ever_woken": int(woken_any.sum()),
                   "n_neurons_never_touched": int((~woken_any).sum())})

    def _scatter(self, targets: np.ndarray, amounts: np.ndarray) -> np.ndarray:
        """Accumulate duplicate targets (pure add.at for a handful, bincount above)."""
        if targets.size < 64:
            out = np.zeros(self.n, dtype=np.float64)
            np.add.at(out, targets, amounts)
            return out
        return np.bincount(targets, weights=amounts, minlength=self.n)

    def _materialize(self, v, last, next_ok, t, lam_pow) -> np.ndarray:
        """Full ``Vm`` vector at step ``t`` from the lazily-updated event state.

        A neuron that never woke since step 0 is recovered from ``v_rest``; one that is
        still refractory sits at ``v_reset``; one that left refractory during the gap
        decays from ``v_reset`` starting at its first free step.
        """
        k = t - last
        vm = self.lif.v_rest + (v - self.lif.v_rest) * lam_pow[k]
        resumed = (next_ok > last) & (next_ok <= t)      # left refractory during the gap
        if resumed.any():
            # state at step (next_ok - 1) is v_reset, so the number of leak steps applied
            # is t - (next_ok - 1)
            vm[resumed] = (self.lif.v_rest
                           + (self.lif.v_reset - self.lif.v_rest)
                           * lam_pow[t - next_ok[resumed] + 1])
        vm[next_ok > t] = self.lif.v_reset               # still refractory
        return vm

    # ---------------------------------------------------------------- reporting
    def summary(self) -> dict:
        deg = self.out_degree
        return {
            "n_neurons": self.n,
            "n_connections": self.e,
            "synapse_threshold_used": self.threshold,
            "n_synapses_in_connections": int(self.syn_count.sum()),
            "mean_edges_per_neuron": round(float(self.e / self.n), 4),
            "mean_out_degree": round(float(deg.mean()), 4),
            "mean_synapses_per_connection": round(float(self.syn_count.mean()), 4),
            "n_neurons_with_no_outgoing_edges": int((deg == 0).sum()),
            "weight_rule": self.syn.to_dict(),
            "mean_abs_edge_weight_anatomical": round(float(np.abs(self.w_anat).mean()), 6),
            "mean_abs_edge_input_after_scale": round(float(np.abs(self.w_edge).mean()), 8),
            "delay_rule": {**self.delay.to_dict(),
                           "measured_delay_histogram_steps": {
                               int(k): int(n) for k, n in enumerate(self.delay_histogram) if n},
                           "measured_mean_delay_steps": round(float(self.delay_steps.mean()), 4)},
            "transmitter_edges": self.nt_histogram,
            "sign_balance": self.sign_histogram,
            "neuron_model": self.lif.to_dict(),
        }


# --------------------------------------------------------------------------- comparison
def compare_runs(a: SimResult, b: SimResult, count_tolerance: float = 0.01,
                 rate_correlation_tolerance: float = 0.99,
                 probe_tolerance: float = 1e-9) -> dict:
    """Agreement between two engine runs on the same stimulus.

    Reports the quantities the Phase 3 gate needs ("CPU DES ~ GPU DES ~ timestep
    reference within stated numerical tolerances"): spike-count delta, per-neuron rate
    correlation, the fraction of individual spikes that coincide in (step, neuron), and -
    when probe traces were recorded - the elementwise ``Vm`` agreement.
    """
    if a.steps != b.steps or a.n_neurons != b.n_neurons:
        raise ValueError("runs are not comparable: different steps or neuron counts")

    ca, cb = a.spike_count_per_neuron(), b.spike_count_per_neuron()
    na, nb = int(ca.sum()), int(cb.sum())
    set_a = set(zip(a.spike_steps.tolist(), a.spike_neurons.tolist()))
    set_b = set(zip(b.spike_steps.tolist(), b.spike_neurons.tolist()))
    common = len(set_a & set_b)
    both_active = (ca > 0) | (cb > 0)

    def corr(x, y):
        if x.size < 3 or x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    active_corr = corr(ca[both_active].astype(np.float64), cb[both_active].astype(np.float64))
    all_corr = corr(ca.astype(np.float64), cb.astype(np.float64))
    rate_norm_a = ca / (a.steps * a.dt_ms / 1000.0)
    rate_norm_b = cb / (b.steps * b.dt_ms / 1000.0)

    out = {
        "engine_a": a.engine, "engine_b": b.engine,
        "steps": a.steps, "simulated_seconds": a.duration_s,
        "n_spikes_a": na, "n_spikes_b": nb,
        "spike_count_delta": nb - na,
        "spike_count_delta_fraction": round((nb - na) / max(1, na), 8),
        "identical_spikes": common,
        "identical_spike_fraction_of_a": round(common / max(1, na), 8),
        "spikes_a_not_in_b": na - common, "spikes_b_not_in_a": nb - common,
        "n_active_neurons_a": int((ca > 0).sum()), "n_active_neurons_b": int((cb > 0).sum()),
        "n_active_neurons_in_either": int(both_active.sum()),
        "n_neurons_active_in_one_only": int((ca > 0).sum() + (cb > 0).sum() - 2 * int(((ca > 0) & (cb > 0)).sum())),
        "per_neuron_rate_pearson_r_all": None if all_corr != all_corr else round(all_corr, 8),
        "per_neuron_rate_pearson_r_neurons_active_in_either": (
            None if active_corr != active_corr else round(active_corr, 8)),
        "mean_abs_rate_difference_hz": round(float(np.abs(rate_norm_a - rate_norm_b)[both_active].mean()), 8)
        if both_active.any() else 0.0,
        "max_abs_rate_difference_hz": round(float(np.abs(rate_norm_a - rate_norm_b).max()), 8),
        "per_step_spike_count_pearson_r": None,
        "wall_seconds_a": round(a.wall_seconds, 3), "wall_seconds_b": round(b.wall_seconds, 3),
        "speedup_b_over_a": round(a.wall_seconds / b.wall_seconds, 3) if b.wall_seconds else None,
    }
    pa = np.bincount(a.spike_steps, minlength=a.steps).astype(np.float64)
    pb = np.bincount(b.spike_steps, minlength=b.steps).astype(np.float64)
    ps = corr(pa, pb)
    out["per_step_spike_count_pearson_r"] = None if ps != ps else round(ps, 8)

    if a.probe is not None and b.probe is not None:
        d = np.abs(a.probe - b.probe)
        out["probe"] = {
            "n_probe_steps": int(a.probe.shape[0]),
            "probe_steps": [int(s) for s in a.probe_steps],
            "max_abs_vm_difference": float(d.max()),
            "rms_vm_difference": float(np.sqrt((d ** 2).mean())),
            "max_abs_vm_a": float(np.abs(a.probe).max()),
            "max_abs_vm_b": float(np.abs(b.probe).max()),
        }

    checks = {
        "spike_count_within_tolerance": abs(nb - na) / max(1, na) <= count_tolerance,
    }
    if na + nb == 0:
        # nothing spiked anywhere: the spike-based comparisons are undefined, not failed;
        # this happens in the deliberately subthreshold probe runs, where the Vm traces are
        # the meaningful comparison
        checks["rate_correlation_above_tolerance"] = None
        out["spike_based_checks_applicable"] = False
    else:
        checks["rate_correlation_above_tolerance"] = (
            active_corr == active_corr and active_corr >= rate_correlation_tolerance)
        out["spike_based_checks_applicable"] = True
    if "probe" in out:
        checks["vm_traces_within_probe_tolerance"] = (
            out["probe"]["max_abs_vm_difference"] <= probe_tolerance)
    out["tolerances"] = {"spike_count_relative_delta": count_tolerance,
                         "per_neuron_rate_pearson_r": rate_correlation_tolerance,
                         "vm_trace_max_abs": probe_tolerance}
    out["checks"] = {k: (None if v is None else bool(v)) for k, v in checks.items()}
    out["agreement"] = bool(all(v for v in checks.values() if v is not None))
    return out


# --------------------------------------------------------------------------- standalone checks
def check_single_neuron(lif: LIFParams | None = None, steps: int = 40,
                        const_input: float = 0.08) -> dict:
    """Analytic sanity check of the update rule, independent of the graph.

    With no input the state decays geometrically to rest (compared against the closed form
    ``v_rest + (v0 - v_rest) * lam**steps``).  With a constant input ``I`` per timestep the
    no-firing steady state is ``I * tau/dt`` and, once that exceeds the threshold, the
    neuron fires periodically with an interval bounded below by the refractory period.
    """
    lif = lif or LIFParams()
    lam = lif.lam

    v = 1.0                                        # start above thresh to test pure decay
    for _ in range(steps):
        v = lif.v_rest + (v - lif.v_rest) * lam
    expected = lif.v_rest + (1.0 - lif.v_rest) * lam ** steps
    decays_to_rest = abs(v - expected) < 1e-12

    v = lif.v_rest
    fired: list[int] = []
    refrac_until = -1
    for t in range(int(200)):
        if refrac_until > t:
            v = lif.v_reset
        else:
            v = lif.v_rest + (v - lif.v_rest) * lam + const_input
            if v >= lif.v_thresh:
                fired.append(t)
                v = lif.v_reset
                refrac_until = t + lif.ref_steps
    isis = np.diff(fired)
    return {"decays_to_rest_matches_closed_form": bool(decays_to_rest),
            "v_after_decay": round(float(v), 15),
            "closed_form_v_after_decay": round(float(expected), 15),
            "constant_input_per_step": const_input,
            "steady_state_V_without_firing": const_input * lif.tau_m_ms / lif.dt_ms,
            "firing_because_steady_state_exceeds_threshold": bool(
                const_input * lif.tau_m_ms / lif.dt_ms >= lif.v_thresh),
            "n_spikes_200_steps": len(fired),
            "inter_spike_intervals": isis.tolist(),
            "min_inter_spike_interval": int(isis.min()) if isis.size else None,
            "refractory_steps": lif.ref_steps,
            "refractory_respected": bool(isis.size == 0 or isis.min() >= lif.ref_steps)}
