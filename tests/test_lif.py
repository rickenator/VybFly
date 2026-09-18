"""Self-test for the Phase 1 whole-connectome LIF model (src/flyscale/lif.py).

Runs without pytest:   python tests/test_lif.py
(also collected by pytest as tests/test_lif.py::test_*)

The tests use the *real* canonical builder on a synthetic release file with the published
schema, so the network under test is built through exactly the production code path
(Connectome -> thresholded view -> LIFNetwork -> both engines).  What is checked:

  * the weight rule (syn_count**alpha normalized to mean edge weight 1) and its
    configurability,
  * the transmitter sign rule (ach +, gaba -, glut -, modulatory scaled) and that it moves
    the postsynaptic membrane potential in the documented direction,
  * the delay rule (uniform and distance-derived, bounds, histogram completeness),
  * the neuron model against its closed form (geometric decay; constant-input steady state
    and periodic firing; refractory period respected),
  * that the Poisson drive delivers the number of events its rate implies,
  * dense-timestep (CSR and pure-numpy gather) vs sparse event-driven agreement, both with
    spiking enabled (spike-set agreement) and in a subthreshold probe run where no spikes
    exist and the Vm traces must match to floating-point rounding,
  * determinism (same seed -> identical spike trains).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flyscale import io as fio                                              # noqa: E402
from flyscale.connectome import Connectome, build_canonical                 # noqa: E402
from flyscale.lif import (CombinedDrive, ConstantDrive, DelayParams, LIFNetwork,  # noqa: E402
                          LIFParams, NT_SIGN, PoissonDrive, SynapseParams,
                          check_single_neuron, compare_runs)

ROOT_IDS = np.arange(1_000_000, 1_000_400, dtype=np.int64) * 1  # 400 synthetic neurons
N = ROOT_IDS.size
SEED = 20260917


# --------------------------------------------------------------------------- fixtures
def make_small_canonical(raw: Path, out: Path) -> dict:
    """A deterministic synthetic release file in the published schema (400 neurons)."""
    rng = np.random.default_rng(SEED)
    np.save(raw / "proofread_root_ids_783.npy", ROOT_IDS)

    n_edge = 2600
    pre = rng.integers(0, N, n_edge)
    post = rng.integers(0, N, n_edge)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    # unique pairs, with synapse counts spanning both sides of the published threshold of 5
    key = pre.astype(np.int64) * N + post
    _, first = np.unique(key, return_index=True)
    pre, post = pre[first], post[first]
    syn = np.rint(1.0 + rng.pareto(1.3, pre.size) * 4.0).astype(np.int64)
    syn = np.clip(syn, 1, 400)

    nt_choice = rng.integers(0, 6, pre.size)
    probs = np.full((pre.size, 6), 0.03)
    probs[np.arange(pre.size), nt_choice] = 0.85
    neuropil = np.where(rng.random(pre.size) < 0.5, "ME_L", "AL_R")

    cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count",
            *fio.NT_PROB_COLUMNS]
    rows = np.column_stack([ROOT_IDS[pre], ROOT_IDS[post],
                            np.zeros(pre.size), syn, probs]).astype(object)
    df = pd.DataFrame(rows, columns=cols)
    df["pre_pt_root_id"] = ROOT_IDS[pre]
    df["post_pt_root_id"] = ROOT_IDS[post]
    df["neuropil"] = neuropil
    df["syn_count"] = syn
    for j, col in enumerate(fio.NT_PROB_COLUMNS):
        df[col] = probs[:, j]
    df.to_feather(raw / "proofread_connections_783.feather")

    ann = pd.DataFrame({
        "root_id": ROOT_IDS,
        "super_class": np.where(np.arange(N) < 40, "sensory", "central"),
        "cell_type": [f"CT{i % 7}" for i in range(N)],
        "top_nt": [fio.NT_TYPES[int(c)] for c in rng.integers(0, 6, N)],
        "side": np.where(np.arange(N) % 2 == 0, "left", "right"),
        # synthetic soma positions in the canonical nanometre space, so the distance-derived
        # delay rule is exercised on real code paths
        "pos_x": rng.uniform(0.0, 300_000.0, N),
        "pos_y": rng.uniform(0.0, 300_000.0, N),
        "pos_z": rng.uniform(0.0, 300_000.0, N),
    })
    ann.to_csv(raw / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)
    return build_canonical(raw, out, chunk_rows=700, force=True)   # forces chunk merges


_CACHE: dict = {}


def small_net(lif: LIFParams | None = None, syn: SynapseParams | None = None,
              delay: DelayParams | None = None, threshold: int = 5) -> LIFNetwork:
    """One canonical-dataset build, reused for every test in this module."""
    if "dir" not in _CACHE:
        td = tempfile.mkdtemp(prefix="flyscale_lif_test_")
        raw, out = Path(td) / "raw", Path(td) / "canon"
        raw.mkdir(parents=True)
        _CACHE["meta"] = make_small_canonical(raw, out)
        _CACHE["dir"] = out
    c = Connectome(_CACHE["dir"])
    cview = c.thresholded(threshold) if threshold > 1 else c
    return LIFNetwork(cview, lif=lif or LIFParams(), syn=syn or SynapseParams(),
                      delay=delay or DelayParams(kind="uniform"), threshold=threshold)


# --------------------------------------------------------------------------- structure
def test_weight_rule() -> None:
    net = small_net(syn=SynapseParams(alpha=1.0, g_syn=0.05))
    syn = net.syn_count.astype(float)
    assert abs(net.w_anat.mean() - 1.0) < 1e-12, net.w_anat.mean()
    assert np.allclose(net.w_anat, syn / syn.mean())
    flat = small_net(syn=SynapseParams(alpha=0.0))
    assert np.allclose(flat.w_anat, 1.0)
    steep = small_net(syn=SynapseParams(alpha=2.0))
    assert abs(steep.w_anat.mean() - 1.0) < 1e-12
    # stronger connections must get strictly more weight under alpha > 0
    assert steep.w_anat.max() > net.w_anat.max()
    # the graph must span the published threshold so the test is not vacuous
    assert (net.syn_count >= 5).all() and net.syn_count.min() == 5
    print(f"[weight rule] E={net.e} mean syn={net.syn_count.mean():.2f} "
          f"max syn={net.syn_count.max()} mean |w|={abs(net.w_anat).mean():.6f}")


def test_sign_rule() -> None:
    net = small_net()
    for t, code in (("ach", +1.0), ("gaba", -1.0), ("glut", -1.0)):
        m = net.nt_code == fio.NT_TYPES.index(t)
        if m.any():
            assert np.all(np.sign(net.sign[m]) == code), t
    for t in ("oct", "ser", "da"):
        m = net.nt_code == fio.NT_TYPES.index(t)
        if m.any():
            assert np.allclose(net.sign[m], NT_SIGN[t] * net.syn.modulatory_scale), t
    assert set(NT_SIGN) == set(fio.NT_TYPES)
    print("[sign rule] edges per transmitter:", net.nt_histogram,
          "| sign balance:", net.sign_histogram)


def test_delay_rule() -> None:
    uni = small_net(delay=DelayParams(kind="uniform", uniform_steps=1))
    assert np.all(uni.delay_steps == 1)
    dist = small_net(delay=DelayParams(kind="distance", base_ms=1.0,
                                       speed_um_per_ms=100.0, max_steps=5))
    assert dist.delay_steps.min() >= 1 and dist.delay_steps.max() <= 5
    assert dist.delay_histogram.sum() == dist.e
    assert dist.max_delay == int(dist.delay_steps.max())
    print("[delay rule] uniform=1 step; distance histogram (steps -> edges):",
          {k: int(v) for k, v in enumerate(dist.delay_histogram) if v},
          f"mean={dist.delay_steps.mean():.3f}")


def test_structure_invariants() -> None:
    net = small_net()
    assert net.e == int(net.out_degree.sum())
    assert np.array_equal(np.diff(net.indptr), net.out_degree)
    assert int((net.pre == net.post).sum()) == 0
    assert net.delay_steps.size == net.e == net.w_edge.size == net.nt_code.size
    print(f"[structure] N={net.n} E={net.e} mean out-degree={net.out_degree.mean():.2f}")


# --------------------------------------------------------------------------- neuron model
def test_single_neuron_analytic() -> None:
    r = check_single_neuron(LIFParams())
    assert r["decays_to_rest_matches_closed_form"]
    assert r["refractory_respected"], r["inter_spike_intervals"]
    assert r["n_spikes_200_steps"] > 5
    # the analytic no-firing fixed point
    assert abs(r["steady_state_V_without_firing"] - 0.08 * 20.0) < 1e-12
    print("[single neuron]", {k: r[k] for k in
                             ("steady_state_V_without_firing", "n_spikes_200_steps",
                              "inter_spike_intervals", "min_inter_spike_interval")})


def test_drive_statistics() -> None:
    rng = np.random.default_rng(1)
    targets = np.arange(0, 100, dtype=np.int64)
    d = PoissonDrive(targets, rate_hz=20.0, steps=500, dt_ms=1.0, amplitude=0.1, rng=rng)
    expected = 20.0 * 0.001 * 500 * targets.size
    assert abs(d.n_events - expected) / expected < 0.1, (d.n_events, expected)
    assert d.events_by_step().sum() == d.n_events
    assert np.all(np.diff(d.indptr) >= 0)
    c = ConstantDrive(np.array([3, 7]), 0.5, t_start=10, t_end=20)
    assert c.at(9)[0].size == 0 and c.at(10)[0].size == 2 and c.at(20)[0].size == 0
    print(f"[drive] poisson events={d.n_events} expected={expected:.1f}")


# --------------------------------------------------------------------------- engines
def _drive(steps: int, seed: int, rate: float = 8.0, amplitude: float = 0.05) -> PoissonDrive:
    rng = np.random.default_rng(seed)
    targets = np.arange(0, int(0.6 * max(1, small_net().n)), dtype=np.int64)
    return PoissonDrive(targets, rate_hz=rate, steps=steps, dt_ms=1.0,
                        amplitude=amplitude, rng=rng)


def test_operator_orientation() -> None:
    """The synaptic operator must fire along the edges, not against them.

    A tiny asymmetric feed-forward chain (0->1, 0->2, 1->3) is enough to catch the
    transposed-operator bug: with the edges reversed nothing ever reaches neurons 1 and 2
    (nobody targets 0 in the toy graph), so the sign test below would see no input at all.
    """
    steps = 40
    with tempfile.TemporaryDirectory() as td:
        raw, out = Path(td) / "raw", Path(td) / "canon"
        raw.mkdir(parents=True)
        ids = np.arange(100, 105, dtype=np.int64)
        np.save(raw / "proofread_root_ids_783.npy", ids)
        cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count",
                *fio.NT_PROB_COLUMNS]
        rows = []
        for a, b in ((0, 1), (0, 2), (1, 3)):
            probs = [0.02, 0.9, 0.02, 0.02, 0.02, 0.02]      # acetylcholine
            rows.append((ids[a], ids[b], "ME_L", 10, *probs))
        pd.DataFrame(rows, columns=cols).to_feather(raw / "proofread_connections_783.feather")
        pd.DataFrame({"root_id": ids, "super_class": ["central"] * 5}).to_csv(
            raw / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)
        build_canonical(raw, out, force=True)
        net = LIFNetwork(Connectome(out), delay=DelayParams(kind="uniform", uniform_steps=1))
        chk = net.check_operators(seed=5, n_sources=3)
        assert chk["ok"], chk
        assert chk["full_edge_list_max_abs_difference"] < 1e-12
        assert chk["direction_check"]["only_targets_received_current"]

        # one step after neuron 0 fires, current must appear on its targets (1 and 2) only
        drv = ConstantDrive(np.array([0]), amplitude=0.9, t_start=0, t_end=5)
        r = net.simulate_dense(drv, steps, method="csr", probe_steps=np.arange(steps))
        fired = r.spike_steps[r.spike_neurons == 0]
        assert fired.size >= 1, "the driven neuron never fired"
        t_star = int(fired[0])
        v_at = r.probe[t_star]
        v_next = r.probe[t_star + 1]
        assert v_at[1] == 0.0 and v_at[2] == 0.0, v_at      # nothing delivered yet
        assert abs(v_next[1] - net.w_edge[0]) < 1e-12, v_next
        assert abs(v_next[2] - net.w_edge[1]) < 1e-12, v_next
        assert v_next[3] == 0.0 and v_next[4] == 0.0, v_next  # 1->3 needs neuron 1 to fire
        e = net.simulate_events(drv, steps, probe_steps=np.arange(steps))
        assert e.n_spikes == r.n_spikes
        assert np.abs(e.probe - r.probe).max() < 1e-12
    print(f"[operator orientation] source 0 fires at t={t_star}; targets 1,2 get "
          f"w={float(net.w_edge[0]):.6f}; neurons 3,4 stay at rest; "
          f"dense csr == sparse events (max |dV| = "
          f"{float(np.abs(e.probe - r.probe).max()):.2e})")


def test_bucket_deduplication() -> None:
    """Two edges arriving at the same step must be summed once, not delivered twice.

    Constructed so that both deliveries land on the same event bucket: A -> C has delay 2
    and B -> C delay 1 (set through the soma distance), A fires at t=1 and B at t=2, so
    both hit C at t=3. The bucket already holds the summed input, so a duplicated target
    index in the touched list would double the delivered current.
    """
    steps = 12
    with tempfile.TemporaryDirectory() as td:
        raw, out = Path(td) / "raw", Path(td) / "canon"
        raw.mkdir(parents=True)
        ids = np.array([10, 20, 30], dtype=np.int64)          # A, B, C
        np.save(raw / "proofread_root_ids_783.npy", ids)
        cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count",
                *fio.NT_PROB_COLUMNS]
        ach = [0.02, 0.9, 0.02, 0.02, 0.02, 0.02]
        rows = [(ids[0], ids[2], "ME_L", 10, *ach),
                (ids[1], ids[2], "ME_L", 10, *ach)]
        pd.DataFrame(rows, columns=cols).to_feather(raw / "proofread_connections_783.feather")
        # A is 50 um from C (delay 2); B sits on C (delay 1); positions are in nm
        pd.DataFrame({
            "root_id": ids, "super_class": ["central"] * 3, "cell_type": ["A", "B", "C"],
            "pos_x": [0.0, 50_000.0, 50_000.0], "pos_y": [0.0, 0.0, 0.0],
            "pos_z": [0.0, 0.0, 0.0],
        }).to_csv(raw / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)
        build_canonical(raw, out, force=True)
        net = LIFNetwork(Connectome(out), delay=DelayParams(
            kind="distance", base_ms=1.0, speed_um_per_ms=50.0, max_steps=5))
        assert net.delay_steps.tolist() == [2, 1], net.delay_steps.tolist()

        drv = CombinedDrive([
            ConstantDrive(np.array([0]), 0.9, t_start=0, t_end=3),
            ConstantDrive(np.array([1]), 0.9, t_start=1, t_end=4)])
        probe = np.arange(steps)
        a = net.simulate_dense(drv, steps, method="csr", probe_steps=probe)
        e = net.simulate_events(drv, steps, probe_steps=probe)
        common = net.w_edge[0] + net.w_edge[1]
        assert a.spike_neurons.tolist() == [0, 1], a.spike_neurons.tolist()
        assert abs(a.probe[3, 2] - common) < 1e-12, a.probe[3]
        assert abs(e.probe[3, 2] - common) < 1e-12, ("event engine delivered the summed "
                                                     "bucket twice", e.probe[3])
        assert np.abs(e.probe - a.probe).max() < 1e-12
    print(f"[bucket dedup] A(delay 2) + B(delay 1) both arrive at t=3 on C: "
          f"{float(a.probe[3, 2]):.6f} == {float(common):.6f}; "
          f"max |dV| dense-events = {float(np.abs(e.probe - a.probe).max()):.2e}")


def test_engines_agree_with_spiking() -> None:
    steps = 120
    net = small_net()
    drv = _drive(steps, seed=SEED + 1, rate=40.0, amplitude=0.5)
    a = net.simulate_dense(drv, steps, method="csr", seed=SEED)
    b = net.simulate_dense(drv, steps, method="gather", seed=SEED)
    e = net.simulate_events(drv, steps, seed=SEED)
    cmp_ab = compare_runs(a, b)
    cmp_ae = compare_runs(a, e)
    assert a.n_spikes > 100, a.n_spikes            # the test network must actually fire
    assert cmp_ab["checks"]["spike_count_within_tolerance"], cmp_ab
    assert cmp_ae["checks"]["spike_count_within_tolerance"], cmp_ae
    assert cmp_ab["per_neuron_rate_pearson_r_neurons_active_in_either"] > 0.99, cmp_ab
    assert cmp_ae["per_neuron_rate_pearson_r_neurons_active_in_either"] > 0.99, cmp_ae
    # event engine must touch far fewer neurons than the dense engine update count
    assert e.n_state_updates < a.n_state_updates
    print(f"[engines spiking] spikes csr={a.n_spikes} gather={b.n_spikes} events={e.n_spikes} "
          f"| identical={cmp_ae['identical_spikes']} | "
          f"r={cmp_ae['per_neuron_rate_pearson_r_neurons_active_in_either']} "
          f"| updates dense={a.n_state_updates} events={e.n_state_updates}")


def test_engine_agreement_subthreshold() -> None:
    """No spikes: both engines must produce the *same* Vm trajectory up to rounding."""
    steps = 120
    net = small_net()
    drv = _drive(steps, seed=SEED + 2, amplitude=0.5)
    probe = np.arange(5, steps, 15)
    kw = dict(probe_steps=probe, v_thresh=1e9, seed=SEED)
    a = net.simulate_dense(drv, steps, method="csr", **kw)
    b = net.simulate_dense(drv, steps, method="gather", **kw)
    e = net.simulate_events(drv, steps, **kw)
    assert a.n_spikes == 0 and b.n_spikes == 0 and e.n_spikes == 0
    assert a.probe.max() > 0.1, "probe saw no input at all; the check would be vacuous"
    for other, label in ((b, "dense_gather"), (e, "sparse_events")):
        d = np.abs(a.probe - other.probe)
        assert d.max() < 1e-9, (label, d.max())
        print(f"[subthreshold] csr vs {label}: max |dVm| = {d.max():.3e} "
              f"rms = {np.sqrt((d ** 2).mean()):.3e} over {probe.size} probes")


def test_determinism() -> None:
    steps = 60
    net = small_net()
    drv = _drive(steps, seed=SEED + 3, rate=40.0, amplitude=0.5)
    r1 = net.simulate_events(drv, steps, seed=SEED)
    r2 = net.simulate_events(drv, steps, seed=SEED)
    assert np.array_equal(r1.spike_steps, r2.spike_steps)
    assert np.array_equal(r1.spike_neurons, r2.spike_neurons)
    r3 = net.simulate_dense(drv, steps, method="csr", seed=SEED)
    r4 = net.simulate_dense(drv, steps, method="csr", seed=SEED)
    assert np.array_equal(r3.spike_steps, r4.spike_steps)
    assert np.array_equal(r3.spike_neurons, r4.spike_neurons)
    assert r1.n_spikes > 0 and r3.n_spikes > 0
    print(f"[determinism] identical spike trains over {steps} steps "
          f"({r1.n_spikes} spikes events, {r3.n_spikes} spikes dense)")


def test_refractory_in_network() -> None:
    steps = 200
    net = small_net()
    # strong wide drive: neurons will saturate and must still respect the refractory period
    rng = np.random.default_rng(SEED + 4)
    drv = PoissonDrive(np.arange(net.n, dtype=np.int64), rate_hz=200.0, steps=steps,
                       dt_ms=1.0, amplitude=0.6, rng=rng)
    r = net.simulate_dense(drv, steps, method="csr", seed=SEED)
    order = np.lexsort((r.spike_steps, r.spike_neurons))
    sn, ss = r.spike_neurons[order], r.spike_steps[order]
    same = np.diff(sn) == 0
    isi = np.diff(ss)[same]
    assert r.n_spikes > 500
    assert isi.min() >= net.lif.ref_steps, isi.min()
    print(f"[refractory] {r.n_spikes} spikes, min ISI = {isi.min()} steps "
          f"(refractory {net.lif.ref_steps})")


def test_sign_dynamics() -> None:
    """An ach edge must push Vm up and a gaba edge down, with equal magnitude."""
    steps = 60
    rng = np.random.default_rng(SEED + 5)
    # a three-neuron canonical dataset is not needed: reuse the small net and override the
    # edge set by building a LIFNetwork over a hand-made Connectome view
    with tempfile.TemporaryDirectory() as td:
        raw, out = Path(td) / "raw", Path(td) / "canon"
        raw.mkdir(parents=True)
        ids = np.array([10, 20, 30], dtype=np.int64)
        np.save(raw / "proofread_root_ids_783.npy", ids)
        # 0 -> 1 inhibitory (gaba), 0 -> 2 excitatory (ach), identical synapse counts
        rows = [(10, 20, "ME_L", 10, 0.9, 0.02, 0.02, 0.02, 0.02, 0.02),
                (10, 30, "ME_L", 10, 0.02, 0.9, 0.02, 0.02, 0.02, 0.02)]
        cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count",
                *fio.NT_PROB_COLUMNS]
        pd.DataFrame(rows, columns=cols).to_feather(raw / "proofread_connections_783.feather")
        ann = pd.DataFrame({"root_id": ids, "super_class": ["central"] * 3,
                            "cell_type": ["A", "B", "C"]})
        ann.to_csv(raw / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)
        build_canonical(raw, out, force=True)
        c = Connectome(out)
        net = LIFNetwork(c, delay=DelayParams(kind="uniform", uniform_steps=1))
        assert np.allclose(np.abs(net.w_edge), abs(net.w_edge[0]))   # same magnitude
        assert net.w_edge[0] < 0 < net.w_edge[1]
        drv = ConstantDrive(np.array([0]), amplitude=0.08, t_start=0, t_end=30)
        steps = 60
        probe = np.arange(steps)
        r = net.simulate_events(drv, steps, probe_steps=probe)
        fired = r.spike_steps[r.spike_neurons == 0]
        assert fired.size >= 1, "the driven neuron never fired"
        t_spike = int(fired[0])
        at_spike = r.probe[t_spike]
        at_delivery = r.probe[t_spike + 1]        # delay = 1 step
        # one step after delivery the postsynaptic potentials decay but keep their sign
        assert at_delivery[1] < 0.0 < at_delivery[2], at_delivery
        assert abs(abs(at_delivery[1]) - abs(at_delivery[2])) < 1e-12
        d_dense = net.simulate_dense(drv, steps, method="csr", probe_steps=probe)
        assert np.abs(d_dense.probe - r.probe).max() < 1e-9
        assert at_spike[1] == 0.0 and at_spike[2] == 0.0   # nothing delivered yet
    print("[sign dynamics] gaba target Vm =", at_delivery[1],
          "ach target Vm =", at_delivery[2], f"(spike at t={t_spike})")


TESTS = (test_weight_rule, test_sign_rule, test_delay_rule, test_structure_invariants,
         test_single_neuron_analytic, test_drive_statistics, test_operator_orientation,
         test_bucket_deduplication, test_engines_agree_with_spiking,
         test_engine_agreement_subthreshold, test_determinism,
         test_refractory_in_network, test_sign_dynamics)


def run() -> None:
    for fn in TESTS:
        fn()
    print(f"phase 1 LIF self-test PASSED ({len(TESTS)} checks)")


def test_lif() -> None:
    run()


if __name__ == "__main__":
    run()
