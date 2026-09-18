"""Phase 1 (PROJECT-VYBFLY.md §7): the smallest defensible whole-connectome LIF baseline.

    python scripts/phase1_lif.py [--seconds 1.0] [--g-syn 0.15] ...

Runs the FlyWire v783 connectome (139,255 neurons, connections thresholded at >= 5
synapses, the published convention) as a leaky integrate-and-fire network with
synapse-count-weighted connectivity, per-edge synaptic delays, transmitter sign and a
refractory state, for 1.0 s of biological time, and validates the two execution engines
against each other on identical stimulus event streams:

  dense_timestep_csr / dense_timestep_gather   every neuron and every edge, every step
                                               (the validity reference, §9)
  sparse_events                                bucket-driven: only active neurons are
                                               expanded and state-updated (the CPU
                                               reference for the Phase 3 DES, §8/§9)

Stimuli: the §20 Level-1 environment (Poisson light pulse on photoreceptors plus a Poisson
odor pulse on olfactory receptor neurons) and the engine-agreement protocol suggested for
the Phase 3 gate (20,000 uniformly random neurons driven with a known Poisson input).

Everything written to results/phase1/lif_baseline.json is measured by this run: the
parameters, seeds, wall-clock timings, spike statistics, engine deltas and the canonical
dataset provenance all come from the run itself.  A block that fails is recorded under
"failures" instead of being silently dropped.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flyscale.connectome import Connectome                                    # noqa: E402
from flyscale.lif import (CombinedDrive, DelayParams, LIFNetwork, LIFParams,  # noqa: E402
                          PoissonDrive, SynapseParams, compare_runs)


# --------------------------------------------------------------------------- helpers
def git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                 timeout=15)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "commit_note": None if commit else "repository has no commits yet (files staged only)",
        "working_tree_dirty": bool(status) if status is not None else None,
        "n_modified_or_untracked_paths": (len(status.splitlines()) if status else 0),
    }


def stimulus_sets(c: Connectome) -> dict:
    """Annotation-derived neuron groups for the Level-1 stimulus."""
    ct = c.neurons["cell_type"].fillna("NA").astype(str).to_numpy(dtype=str)
    sc = c.neurons["super_class"].fillna("NA").astype(str).to_numpy(dtype=str)
    photoreceptors = np.flatnonzero(np.isin(ct, ["R1-6", "R7", "R8"]))
    orns = np.flatnonzero(np.char.startswith(ct, "ORN_"))
    other = np.flatnonzero((sc == "sensory")
                           & ~np.isin(np.arange(c.n), np.concatenate([photoreceptors, orns])))
    return {"photoreceptors": photoreceptors, "olfactory_receptor_neurons": orns,
            "other_sensory": other}


def run_engines(net: LIFNetwork, drive, steps: int, engines: tuple[str, ...],
                probe_steps=None, v_thresh: float | None = None, tag: str = "") -> dict:
    """Run the requested engines on the same drive; keep going if one fails."""
    results, failures = {}, {}
    for engine in engines:
        t0 = time.time()
        try:
            if engine == "sparse_events":
                r = net.simulate_events(drive, steps, probe_steps=probe_steps,
                                        v_thresh=v_thresh)
            else:
                method = engine.replace("dense_timestep_", "")
                r = net.simulate_dense(drive, steps, method=method, probe_steps=probe_steps,
                                       v_thresh=v_thresh)
            results[engine] = r
            print(f"  [{tag}] {engine}: {r.n_spikes} spikes, "
                  f"{r.n_state_updates} state updates, {time.time() - t0:.1f}s wall")
        except Exception as exc:                                     # noqa: BLE001
            failures[engine] = f"{type(exc).__name__}: {exc}"
            print(f"  [{tag}] {engine}: FAILED ({failures[engine]})")
    return {"results": results, "failures": failures}


def compare_engine_set(results: dict, reference: str, tolerances: dict) -> dict:
    out = {}
    for name, r in results.items():
        if name == reference:
            continue
        out[f"{reference}_vs_{name}"] = compare_runs(
            results[reference], r,
            count_tolerance=tolerances["spike_count_relative_delta"],
            rate_correlation_tolerance=tolerances["per_neuron_rate_pearson_r"],
            probe_tolerance=tolerances["vm_trace_max_abs"])
    return out


def response_profile(r, driven_idx: np.ndarray, steps: int, n_bins: int = 10) -> dict:
    """How the response is distributed in time and between driven and recruited neurons."""
    width = max(1, steps // n_bins)
    counts = np.bincount(r.spike_steps // width, minlength=n_bins)
    from_driven = np.isin(r.spike_neurons, driven_idx)
    recruited = ~from_driven
    return {
        "engine": r.engine,
        "spikes_per_bin": [int(c) for c in counts],
        "bin_width_steps": int(width),
        "n_spikes_from_driven_neurons": int(from_driven.sum()),
        "n_spikes_from_recruited_neurons": int(recruited.sum()),
        "fraction_of_spikes_from_recruited_neurons": round(float(recruited.mean()), 6),
        "n_recruited_neurons": int(np.unique(r.spike_neurons[recruited]).size)
        if recruited.any() else 0,
    }


def save_run_artifacts(artifacts: Path, tag: str, name: str, r) -> list[str]:
    written = []
    counts = r.spike_count_per_neuron().astype(np.int32)
    p = artifacts / f"spike_counts_{tag}_{name}.npy"
    np.save(p, counts)
    written.append(str(p.relative_to(ROOT)))
    p = artifacts / f"spikes_{tag}_{name}.npz"
    np.savez_compressed(p, steps=r.spike_steps, neurons=r.spike_neurons)
    written.append(str(p.relative_to(ROOT)))
    if r.probe is not None:
        p = artifacts / f"probe_{tag}_{name}.npy"
        np.save(p, r.probe)
        written.append(str(p.relative_to(ROOT)))
    return written


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", default=str(ROOT / "data" / "processed" / "canonical_v783"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase1" / "lif_baseline.json"))
    ap.add_argument("--artifacts", default=str(ROOT / "results" / "phase1" / "artifacts"))
    ap.add_argument("--seconds", type=float, default=1.0,
                    help="simulated biological seconds (dt = 1 ms -> 1000 steps per second)")
    ap.add_argument("--threshold", type=int, default=5,
                    help="minimum synapses per connection (5 = published convention)")
    ap.add_argument("--g-syn", type=float, default=0.15,
                    help="synaptic gain: input units delivered per unit anatomical weight")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="weight rule exponent: w = syn_count**alpha, mean-normalized")
    ap.add_argument("--inhibitory-gain", type=float, default=1.0)
    ap.add_argument("--modulatory-scale", type=float, default=0.25)
    ap.add_argument("--tau-ms", type=float, default=20.0)
    ap.add_argument("--t-ref-ms", type=float, default=2.0)
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--delay-kind", default="distance", choices=("distance", "uniform"))
    ap.add_argument("--delay-speed-um-per-ms", type=float, default=50.0)
    ap.add_argument("--delay-max-steps", type=int, default=5)
    ap.add_argument("--drive-rate-hz", type=float, default=20.0,
                    help="Poisson rate of the Level-1 light and odor pulses")
    ap.add_argument("--drive-amplitude", type=float, default=0.9,
                    help="external input per drive event, in threshold units")
    ap.add_argument("--random-n", type=int, default=20000,
                    help="neurons driven in the engine-agreement protocol")
    ap.add_argument("--random-rate-hz", type=float, default=5.0)
    ap.add_argument("--random-amplitude", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--scan-g-syn", default="0.05,0.15,0.25,0.5",
                    help="comma-separated g_syn ladder for the operating-point scan")
    ap.add_argument("--probe-seconds", type=float, default=0.3,
                    help="subthreshold equivalence probe duration (seconds)")
    ap.add_argument("--skip-gather", action="store_true",
                    help="skip the pure-numpy dense variant (it is ~2x slower)")
    ap.add_argument("--skip-scan", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    steps = int(round(args.seconds * 1000.0 / args.dt_ms))
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "phase": 1,
        "title": "FlyScale Phase 1: smallest defensible whole-connectome LIF model "
                 "(dense timestep reference vs sparse event engine)",
        "scope_doc": "PROJECT-VYBFLY.md §7 (with the §9 agreement gate and §20 Level 1 "
                     "stimuli); execution model per §18",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
            "scipy": __import__("scipy").__version__,
        },
        "git": git_state(),
        "arguments": vars(args),
        "seed": args.seed,
        "failures": [],
        "notes": [],
        "artifacts": [],
    }

    def fail(block: str, exc: Exception) -> None:
        report["failures"].append({"block": block, "error": f"{type(exc).__name__}: {exc}"})
        print(f"!! {block} failed: {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- dataset + network
    c = Connectome(args.canonical)
    c5 = c.thresholded(args.threshold) if args.threshold > 1 else c
    report["canonical_dataset"] = {
        "directory": str(Path(args.canonical).resolve()),
        "canonical_version": c.meta["canonical_version"],
        "built_utc": c.meta.get("built_utc"),
        "source": c.meta.get("source"),
        "counts": c.meta.get("counts"),
        "conventions": c.meta.get("conventions"),
    }

    lif = LIFParams(dt_ms=args.dt_ms, tau_m_ms=args.tau_ms, t_ref_ms=args.t_ref_ms)
    syn = SynapseParams(alpha=args.alpha, g_syn=args.g_syn,
                        modulatory_scale=args.modulatory_scale,
                        inhibitory_gain=args.inhibitory_gain)
    delay = DelayParams(kind=args.delay_kind, speed_um_per_ms=args.delay_speed_um_per_ms,
                        max_steps=args.delay_max_steps)
    net = LIFNetwork(c5, lif=lif, syn=syn, delay=delay, threshold=args.threshold)
    report["graph"] = net.summary()
    report["units_note"] = (
        "voltages are dimensionless threshold units: v_rest = 0, v_reset = 0, threshold "
        "1.0, tau_m = %s ms, dt = %s ms. An edge of anatomical weight 1 delivers "
        "g_syn = %s units of input current on each presynaptic spike, so a neuron spikes "
        "once its integrated input exceeds 1.0." % (lif.tau_m_ms, lif.dt_ms, syn.g_syn))
    report["driven_vs_silent_sensory_note"] = (
        "the §20 Level-1 stimulus drives photoreceptors and olfactory receptor neurons "
        "only; the remaining sensory neurons (mechanosensory, gustatory, hygrosensory) are "
        "left silent so that the recorded response has a defined input set")
    np.save(artifacts / "delay_histogram_steps.npy", net.delay_histogram)
    report["artifacts"].append(str((artifacts / "delay_histogram_steps.npy").relative_to(ROOT)))
    print(f"network: {net.n} neurons, {net.e} connections, max delay {net.max_delay} steps")

    # ---------------------------------------------------------------- operator self-check
    try:
        t0 = time.time()
        chk = net.check_operators(seed=args.seed)
        chk["wall_seconds"] = round(time.time() - t0, 3)
        report["operator_self_check"] = chk
        print(f"operator self-check: ok={chk['ok']} "
              f"(max |csr - scatter| = {chk['full_edge_list_max_abs_difference']:.3e})")
    except Exception as exc:                                          # noqa: BLE001
        fail("operator_self_check", exc)

    if net.max_delay + 1 > steps:
        report["failures"].append({"block": "configuration",
                                   "error": "delay window exceeds the simulated duration"})

    tolerances = {"spike_count_relative_delta": 0.01,
                  "per_neuron_rate_pearson_r": 0.99,
                  "vm_trace_max_abs": 1e-9}
    report["agreement_tolerances"] = tolerances
    engines = ["dense_timestep_csr", "sparse_events"] if args.skip_gather else \
              ["dense_timestep_csr", "dense_timestep_gather", "sparse_events"]

    # ---------------------------------------------------------------- protocol A: Level 1
    sets = stimulus_sets(c5)
    cell_type = c5.neurons["cell_type"].fillna("NA").astype(str).to_numpy(dtype=str)
    set_info = {}
    for name, idx in sets.items():
        types, counts = np.unique(cell_type[idx], return_counts=True)
        order = np.argsort(-counts)
        set_info[name] = {
            "n_neurons": int(idx.size),
            "n_cell_types": int(types.size),
            "top_cell_types": [{"cell_type": str(types[i]), "n": int(counts[i])}
                               for i in order[:15]],
        }
    report["level1_stimulus_sets"] = set_info
    report["protocols"] = {}

    rng = np.random.default_rng(args.seed)
    light = PoissonDrive(sets["photoreceptors"], rate_hz=args.drive_rate_hz, steps=steps,
                         dt_ms=args.dt_ms, amplitude=args.drive_amplitude, rng=rng,
                         t_start=int(round(0.1 * steps)), t_end=int(round(0.6 * steps)))
    odor = PoissonDrive(sets["olfactory_receptor_neurons"], rate_hz=args.drive_rate_hz,
                         steps=steps, dt_ms=args.dt_ms, amplitude=args.drive_amplitude,
                         rng=rng, t_start=int(round(0.4 * steps)), t_end=int(round(0.9 * steps)))
    level1 = CombinedDrive([light, odor])
    print(f"protocol level1_light_odour: {level1.n_events} external events, {steps} steps")
    t0 = time.time()
    out1 = run_engines(net, level1, steps, engines, tag="level1")
    report["protocols"]["level1_light_odour"] = {
        "stimulus": {
            "description": "§20 Level 1: Poisson light pulse on photoreceptors "
                           "(R1-6/R7/R8) during 100-600 ms, Poisson odor pulse on "
                           "olfactory receptor neurons (ORN_*) during 400-900 ms",
            "components": level1.summary(),
            "drive_seed": args.seed,
            "external_events": int(level1.n_events),
        },
        "wall_seconds_all_engines": round(time.time() - t0, 3),
        "runs": {name: r.summary() for name, r in out1["results"].items()},
        "failures": out1["failures"],
        "comparisons": compare_engine_set(out1["results"], "dense_timestep_csr", tolerances),
    }
    if out1["results"]:
        driven = np.concatenate([sets["photoreceptors"], sets["olfactory_receptor_neurons"]])
        report["protocols"]["level1_light_odour"]["response_profile"] = {
            name: response_profile(r, driven, steps)
            for name, r in out1["results"].items()}
    for name, r in out1["results"].items():
        report["artifacts"] += save_run_artifacts(artifacts, "level1", name, r)

    # reproducibility: same seed twice through the sparse engine
    try:
        t0 = time.time()
        rep_a = net.simulate_events(level1, steps)
        identical = (np.array_equal(rep_a.spike_steps, out1["results"]["sparse_events"].spike_steps)
                     and np.array_equal(rep_a.spike_neurons,
                                        out1["results"]["sparse_events"].spike_neurons))
        report["reproducibility"] = {
            "check": "sparse_events re-run on the same drive and seed",
            "identical_spike_train": bool(identical),
            "n_spikes_first": int(out1["results"]["sparse_events"].n_spikes),
            "n_spikes_second": int(rep_a.n_spikes),
            "wall_seconds": round(time.time() - t0, 3),
        }
        print(f"reproducibility: identical={identical}")
    except Exception as exc:                                          # noqa: BLE001
        fail("reproducibility", exc)

    # ---------------------------------------------------------------- protocol B: rand 20k
    rng = np.random.default_rng(args.seed + 1)
    rand_targets = np.sort(rng.choice(net.n, min(args.random_n, net.n), replace=False))
    rng = np.random.default_rng(args.seed + 2)
    rand_drive = PoissonDrive(rand_targets, rate_hz=args.random_rate_hz, steps=steps,
                              dt_ms=args.dt_ms, amplitude=args.random_amplitude, rng=rng)
    print(f"protocol random20k: {rand_drive.n_events} external events on "
          f"{rand_targets.size} random neurons")
    t0 = time.time()
    out2 = run_engines(net, rand_drive, steps, engines, tag="rand20k")
    report["protocols"]["random20k_poisson"] = {
        "stimulus": {
            "description": "engine-agreement protocol suggested for the §9 gate: "
                           f"{rand_targets.size} uniformly random neurons driven with the "
                           "same known Poisson input in every engine",
            "components": rand_drive.summary(),
            "drive_seed": args.seed + 2,
            "target_neurons_seed": args.seed + 1,
            "external_events": int(rand_drive.n_events),
        },
        "wall_seconds_all_engines": round(time.time() - t0, 3),
        "runs": {name: r.summary() for name, r in out2["results"].items()},
        "failures": out2["failures"],
        "comparisons": compare_engine_set(out2["results"], "dense_timestep_csr", tolerances),
    }
    if out2["results"]:
        report["protocols"]["random20k_poisson"]["response_profile"] = {
            name: response_profile(r, rand_targets, steps)
            for name, r in out2["results"].items()}
    for name, r in out2["results"].items():
        report["artifacts"] += save_run_artifacts(artifacts, "rand20k", name, r)

    # ---------------------------------------------------------------- subthreshold equivalence
    try:
        steps_p = int(round(args.probe_seconds * 1000.0 / args.dt_ms))
        probe_steps = np.arange(0, steps_p, max(1, steps_p // 12))
        rng = np.random.default_rng(args.seed + 2)
        probe_drive = PoissonDrive(rand_targets, rate_hz=args.random_rate_hz, steps=steps_p,
                                   dt_ms=args.dt_ms, amplitude=args.random_amplitude, rng=rng)
        t0 = time.time()
        outp = run_engines(net, probe_drive, steps_p, engines, probe_steps=probe_steps,
                           v_thresh=1e9, tag="subthreshold")
        cmp = compare_engine_set(outp["results"], "dense_timestep_csr", tolerances)
        report["subthreshold_probe_equivalence"] = {
            "purpose": "no spike can be emitted (threshold set to 1e9), so the two engines "
                       "must reproduce the same subthreshold Vm trajectory up to "
                       "floating-point rounding; this isolates engine numerics from "
                       "spike-timing amplification",
            "steps": steps_p, "v_thresh_used": 1e9,
            "probe_steps": [int(s) for s in probe_steps],
            "probe_drive": probe_drive.summary(),
            "max_abs_vm": float(np.abs(outp["results"]["dense_timestep_csr"].probe).max()),
            "runs": {name: r.summary() for name, r in outp["results"].items()},
            "comparisons": cmp,
            "pass": all(c["checks"].get("vm_traces_within_probe_tolerance", False)
                        for c in cmp.values()),
            "wall_seconds": round(time.time() - t0, 3),
            "failures": outp["failures"],
        }
        for name, r in outp["results"].items():
            report["artifacts"] += save_run_artifacts(artifacts, "subthreshold", name, r)
        print("subthreshold equivalence:", {k: c["probe"]["max_abs_vm_difference"]
                                            for k, c in cmp.items()})
    except Exception as exc:                                          # noqa: BLE001
        fail("subthreshold_probe_equivalence", exc)

    # ---------------------------------------------------------------- operating-point scan
    if not args.skip_scan:
        ladder = [float(x) for x in str(args.scan_g_syn).split(",") if x.strip()]
        scan_steps = min(steps, 500)
        scan: dict = {"purpose": (
            "documents the g_syn operating point: the same Level-1 stimulus at several "
            "synaptic gains, 0.5 s each, dense engine only. The default g_syn = 0.15 is an "
            "intermediate point of the ladder - sparse enough to keep the event-driven "
            "engine cheap, but with clear recurrent recruitment beyond the driven sensory "
            "neurons. The ladder is recorded so the choice can be re-derived rather than "
            "trusted; it is a recorded operating point, not an optimized one."),
            "ladder": ladder, "steps": scan_steps, "default_g_syn": args.g_syn,
            "measured_trend": None, "runs": {}}
        try:
            for g in ladder:
                rng = np.random.default_rng(args.seed)
                li = PoissonDrive(sets["photoreceptors"], rate_hz=args.drive_rate_hz,
                                  steps=scan_steps, dt_ms=args.dt_ms,
                                  amplitude=args.drive_amplitude, rng=rng,
                                  t_start=int(round(0.1 * scan_steps)),
                                  t_end=int(round(0.6 * scan_steps)))
                od = PoissonDrive(sets["olfactory_receptor_neurons"], rate_hz=args.drive_rate_hz,
                                  steps=scan_steps, dt_ms=args.dt_ms,
                                  amplitude=args.drive_amplitude, rng=rng,
                                  t_start=int(round(0.4 * scan_steps)),
                                  t_end=int(round(0.9 * scan_steps)))
                net_g = LIFNetwork(c5, lif=lif, delay=delay, threshold=args.threshold,
                                   syn=SynapseParams(alpha=args.alpha, g_syn=g,
                                                     modulatory_scale=args.modulatory_scale,
                                                     inhibitory_gain=args.inhibitory_gain))
                t0 = time.time()
                r = net_g.simulate_dense(CombinedDrive([li, od]), scan_steps, method="csr")
                s = r.summary()
                bins = np.bincount(r.spike_steps // max(1, scan_steps // 10),
                                   minlength=10).tolist()
                scan["runs"][str(g)] = {
                    "g_syn": g, "n_spikes": int(s["n_spikes"]),
                    "mean_rate_hz_over_all_neurons": s["mean_rate_hz_over_all_neurons"],
                    "fraction_neurons_active": s["fraction_neurons_active"],
                    "mean_rate_hz_over_active_neurons": s["mean_rate_hz_over_active_neurons"],
                    "spikes_per_100_steps_deciles": bins,
                    "synaptic_events_per_biological_second":
                        s["synaptic_events_per_biological_second"],
                    "wall_seconds": round(time.time() - t0, 3)}
                print(f"  scan g_syn={g}: spikes={s['n_spikes']} "
                      f"rate={s['mean_rate_hz_over_all_neurons']:.3f} Hz "
                      f"active={s['fraction_neurons_active']:.3f}")
            scan["measured_trend"] = {
                "n_spikes_by_g_syn": {k: v["n_spikes"] for k, v in scan["runs"].items()},
                "mean_rate_hz_by_g_syn": {k: v["mean_rate_hz_over_all_neurons"]
                                          for k, v in scan["runs"].items()},
                "fraction_active_by_g_syn": {k: v["fraction_neurons_active"]
                                             for k, v in scan["runs"].items()},
                "synaptic_events_per_second_by_g_syn":
                    {k: v["synaptic_events_per_biological_second"]
                     for k, v in scan["runs"].items()},
            }
            report["operating_point_scan"] = scan
        except Exception as exc:                                      # noqa: BLE001
            report["operating_point_scan"] = {**scan, "error": f"{type(exc).__name__}: {exc}"}
            fail("operating_point_scan", exc)

    # ---------------------------------------------------------------- verdicts
    def gate_of(block: str) -> dict:
        p = report["protocols"].get(block, {})
        comps = p.get("comparisons", {})
        ok = bool(comps) and all(c["agreement"] for c in comps.values())
        return {"protocol": block, "agreement": ok,
                "checks": {k: c["checks"] for k, c in comps.items()},
                "n_spikes_per_engine": {k: c["n_spikes_b"] for k, c in comps.items()}}

    gates = [gate_of(name) for name in report["protocols"]]
    probe_block = report.get("subthreshold_probe_equivalence", {})
    report["engine_agreement_gate"] = {
        "statement": "CPU DES (sparse events) == dense timestep reference within the stated "
                     "tolerances, on identical stimulus event streams",
        "tolerances": tolerances,
        "protocols": gates,
        "subthreshold_vm_traces_pass": bool(probe_block.get("pass", False)),
        "all_protocols_agree": bool(all(g["agreement"] for g in gates)) if gates else False,
        "passed": bool(gates and all(g["agreement"] for g in gates)
                       and probe_block.get("pass", False)),
    }

    # ---------------------------------------------------------------- notes + write
    report["notes"] = [
        "Weights: w_anat = syn_count**alpha normalized so the mean edge weight is 1; the "
        "input per spike is g_syn * sign(transmitter) * modulatory_scale. 'alpha' and "
        "'g_syn' are model parameters, not measured quantities.",
        "Signs: ach excitatory, gaba and glut inhibitory (most adult-fly glutamatergic "
        "neurons are inhibitory), and oct/ser/da modulatory (weak excitatory, scaled). "
        "The mapping is recorded in graph.weight_rule.sign_map.",
        "Delays: per-edge integer timesteps from the soma-to-soma distance with a specified "
        "conduction speed. The connectome contains no measured delays, so the delay rule is "
        "a model, and its measured histogram is included under graph.delay_rule.",
        "The dense engines iterate all 139,255 neurons and all 2.7M edges on every step; "
        "the sparse engine only expands and updates active neurons, so its state-update "
        "count is the quantity that should fall as activity sparsens (§9).",
        "A synapse-count-weighted model is not a model of learned synaptic strength "
        "(an explicit non-goal of the scope document).",
        "Engine-equivalence traps that are now covered by tests/test_lif.py, and that any "
        "Phase 2/3 runtime must also avoid: (a) the synaptic operator has to be indexed "
        "[post, pre] - building the CSR with row = pre silently reverses every edge, which "
        "is almost invisible on a random toy graph and completely changes real connectome "
        "dynamics; (b) the per-step touched-target list must be deduplicated before the "
        "accumulated bucket is read back, otherwise a neuron that receives from two "
        "sources in the same bucket is charged twice.",
        "n_synaptic_events_delivered counts presynaptic spikes times out-degree, so it is "
        "identical in every engine by construction and is not itself evidence of "
        "agreement; the quantity that differs between engines is the neuron state-update "
        "count (steps x population for the dense engines versus active neurons only for "
        "the event engine).",
    ]
    report["wall_seconds_total"] = round(time.time() - t_start, 3)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    # ---------------------------------------------------------------- console summary
    print()
    hdr = f"{'protocol':22s} {'engine':22s} {'spikes':>9s} {'rate Hz':>8s} {'active':>7s} " \
          f"{'updates':>12s} {'wall s':>7s} {'wall s/sim s':>13s}"
    print(hdr)
    print("-" * len(hdr))
    for pname, pdata in report["protocols"].items():
        for ename, s in pdata["runs"].items():
            print(f"{pname:22s} {ename:22s} {s['n_spikes']:9d} "
                  f"{s['mean_rate_hz_over_all_neurons']:8.3f} "
                  f"{s['fraction_neurons_active']:7.4f} "
                  f"{s['n_neuron_state_updates']:12d} {s['wall_seconds']:7.2f} "
                  f"{s['wall_seconds_per_simulated_second']:13.2f}")
        for cname, c in pdata["comparisons"].items():
            print(f"  {cname}: delta={c['spike_count_delta']} "
                  f"({c['spike_count_delta_fraction'] * 100:+.4f}%), "
                  f"identical={c['identical_spike_fraction_of_a'] * 100:.4f}%, "
                  f"r={c['per_neuron_rate_pearson_r_neurons_active_in_either']}, "
                  f"agreement={c['agreement']}")
        prof = next(iter(pdata.get("response_profile", {}).values()), None)
        if prof:
            print(f"  response: recruited {prof['n_recruited_neurons']} neurons, "
                  f"{prof['fraction_of_spikes_from_recruited_neurons'] * 100:.2f}% of spikes "
                  f"not from driven neurons; spikes per {prof['bin_width_steps']} steps "
                  f"= {prof['spikes_per_bin']}")
    print()
    print("engine agreement gate passed:", report["engine_agreement_gate"]["passed"])
    if report["failures"]:
        print("failures:", json.dumps(report["failures"], indent=2))
    print("wrote", out_path, f"({report['wall_seconds_total']}s total)")
    return 0 if report["engine_agreement_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
