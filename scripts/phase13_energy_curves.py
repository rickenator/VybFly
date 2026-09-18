"""M13/M14: energy curves across connectome scales, and capability per watt.

    python scripts/phase13_energy_curves.py --scales 1 2 5 10

For each scale this runs the instrumented sparse-propagation payload
(scripts/cuda/sparse_prop.cu, built to results/energy/build/sparse_prop) on the real CSR of that
scale's graph while power is sampled, and records both energy tracks side by side:

  hardware    measured joules (GPU via NVML, CPU via RAPL), joules per spike, joules per synaptic
              event, wall seconds per simulated biological second
  biological  P_bio(N) = P0 * N / N0 from the cited whole-brain calorimetry anchor - a property
              of the *scale*, not of the simulated workload, so it is reported separately and
              never added to the hardware numbers

§19 capability per watt is emitted only where a capability number exists for the same scale
(read from results/phase9/capability.json); otherwise it is left null with the reason recorded.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_target, write_json, strip_arrays  # noqa: E402

from flyscale.energy import BiologicalEnergyModel, PowerSampler  # noqa: E402
from flyscale.synthetic import load_graph  # noqa: E402

CURVES = ROOT / "results" / "energy" / "curves"
EXE = ROOT / "results" / "energy" / "build" / "sparse_prop"
DT_MS = 0.5
STEPS = 1000
EPOCHS = 8
SYN_GAIN = 6.0
REFR_STEPS = 20


def export_csr(g, dest: Path) -> dict:
    """Write the harness's expected CSR files (indptr i64, indices i32, syn i32)."""
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest / "bin"
    dest.mkdir(parents=True, exist_ok=True)
    indptr = np.zeros(g.n + 1, dtype=np.int64)
    np.add.at(indptr, g.pre + 1, 1)
    np.cumsum(indptr, out=indptr)
    order = np.argsort(g.pre, kind="stable")
    indices = g.post[order].astype(np.int32)
    syn = np.rint(g.syn[order]).astype(np.int32)
    indptr.tofile(dest / "csr.out.indptr.i64")
    indices.tofile(dest / "csr.out.indices.i32")
    syn.tofile(dest / "csr.out.syn.i32")
    return {"n_neurons": int(g.n), "n_edges": int(g.pre.size),
            "n_synapses": int(syn.astype(np.int64).sum()),
            "indptr_bytes": indptr.nbytes, "indices_bytes": indices.nbytes,
            "syn_bytes": syn.nbytes}


def graph_for_scale(scale: int):
    if scale == 1:
        return load_target(threshold=5)
    for base in (ROOT / "results" / "phase6" / "replicas",
                 ROOT / "results" / "phase9" / "replicas"):
        for name in (f"g{scale}", f"g{float(scale)}", f"g{float(scale):.1f}", f"g{scale}.npz"):
            candidate = base / name
            if (candidate / "graph.npz").exists() or candidate.suffix == ".npz" and candidate.exists():
                return load_graph(candidate)
    raise SystemExit(f"no saved replica for {scale}x; run scripts/phase6_upscale.py --save first")


def parse_result(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        if line.startswith("RESULT"):
            for tok in line.split()[1:]:
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    try:
                        out[k] = float(v) if "." in v or "e" in v else int(v)
                    except ValueError:
                        out[k] = v
    return out


def capability_for_scale(scale: int) -> dict:
    path = ROOT / "results" / "phase9" / "capability.json"
    if not path.exists():
        return {"available": False, "reason": "no capability.json"}
    data = json.loads(path.read_text())
    block = (data.get("metrics_by_scale") or {}).get(str(scale))
    if not block:
        return {"available": False, "reason": f"no capability battery row for {scale}x"}
    keys = ("memory_capacity", "stimulus_response_mi_bits", "discrimination_bits_at_zero_overlap",
            "participation_ratio")
    m = block.get("metrics", block)
    return {"available": True, "source": "results/phase9/capability.json",
            "metrics": {k: m.get(k) for k in keys if isinstance(m, dict) and k in m}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--stim-neurons", type=int, default=500)
    ap.add_argument("--stim-hz", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "energy" / "curves.json"))
    args = ap.parse_args()

    if not EXE.exists():
        raise SystemExit(f"missing {EXE}; build with: nvcc -O3 -arch=sm_86 -o <exe> "
                         f"scripts/cuda/sparse_prop.cu")
    bio = BiologicalEnergyModel()
    per_scale: dict = {}
    for scale in args.scales:
        try:
            g = graph_for_scale(scale)
        except SystemExit as exc:
            per_scale[str(scale)] = {"skipped": str(exc)}
            continue
        dest = CURVES / f"s{scale}"
        export = export_csr(g, dest)
        bio_seconds = args.steps * DT_MS / 1000.0
        cmd = [str(EXE), "--dir", str(dest), "--epochs", str(args.epochs),
               "--steps", str(args.steps), "--dt-ms", str(DT_MS),
               "--syn-gain", str(SYN_GAIN), "--refr-steps", str(REFR_STEPS),
               "--stim-neurons", str(args.stim_neurons), "--stim-hz", str(args.stim_hz)]
        sampler = PowerSampler(interval_s=args.interval)
        sampler.start()
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        wall = time.perf_counter() - t0
        sampler.stop()
        rep = sampler.report()
        if proc.returncode != 0:
            per_scale[str(scale)] = {"error": f"harness exit {proc.returncode}",
                                     "stderr_tail": proc.stderr[-400:], **export}
            continue
        res = parse_result(proc.stdout)
        gpu_j = None
        cpu_j = None
        for dev_name, dev in (rep.get("devices") or {}).items():
            if dev.get("joules_total") is None:
                continue
            if "gpu" in dev_name:
                gpu_j = dev["joules_total"]
            else:
                cpu_j = dev.get("joules_total")
        spikes = res.get("spikes")
        events = res.get("synaptic_events")
        per_scale[str(scale)] = {
            **export,
            "steps": args.steps, "epochs": args.epochs, "dt_ms": DT_MS,
            "simulated_biological_seconds": bio_seconds,
            "wall_seconds": round(wall, 6),
            "wall_seconds_per_bio_second": round(wall / bio_seconds, 6),
            "harness": {k: res.get(k) for k in
                        ("spikes", "stimulus_spikes", "synaptic_events", "epochs_identical",
                         "kernel_s", "max_active") if k in res},
            "hardware": {
                "gpu_joules": gpu_j, "cpu_joules": cpu_j,
                "gpu_mean_watts": ((rep.get("devices") or {}).get("gpu:0") or {}).get("mean_w"),
                "gpu_median_watts": ((rep.get("devices") or {}).get("gpu:0") or {}).get("median_w"),
                "cpu_mean_watts": next((d.get("mean_w") for k, d in
                                        (rep.get("devices") or {}).items() if "gpu" not in k and
                                        isinstance(d, dict)), None),
                "joules_per_spike": (round(gpu_j / spikes, 12) if gpu_j and spikes else None),
                "joules_per_synaptic_event": (round(gpu_j / events, 15) if gpu_j and events else None),
                "track_note": "measured watts on this machine; GPU joules only (the payload is "
                              "a CUDA kernel), CPU joules reported separately and never summed in",
            },
            "biological": {
                "power_watts": bio.power_w(g.n),
                "joules_over_simulated_time": bio.joules(g.n, bio_seconds),
                "per_neuron_watts": bio.per_neuron_watts,
                "anchor": bio.as_dict().get("anchor"),
                "track_note": "P_bio depends on N only - a property of the scale, not of this run",
            },
            "capability": capability_for_scale(scale),
        }
        cap = per_scale[str(scale)]["capability"]
        if gpu_j and cap.get("available") and cap["metrics"].get("memory_capacity"):
            per_scale[str(scale)]["capability_per_watt"] = {
                "memory_capacity_per_watt": round(cap["metrics"]["memory_capacity"] /
                                                  ((rep.get("devices") or {})
                                                   .get("gpu:0", {}).get("mean_w") or float("nan")), 8),
                "note": "capability unit per measured GPU watt at this scale",
            }
        print(f"scale {scale}: N={export['n_neurons']} edges={export['n_edges']} "
              f"gpu_j={gpu_j} spikes={spikes} events={events} wall={wall:.1f}s")

    payload = {
        "phase": 13, "milestone": "M13/M14",
        "title": "energy curves across connectome scales and capability per watt",
        "track_separation": "the hardware and biological blocks are never summed or conflated; "
                           "the hardware block measures this machine on a CUDA payload, the "
                           "biological block is a property of the neuron count alone",
        "caveat": "the CUDA payload (scripts/cuda/sparse_prop.cu) is a bounded, counter-rich "
                  "sparse workload, not the project's biological baseline model; the numbers "
                  "characterise the *machine* at each scale",
        "protocol": {"steps": args.steps, "epochs": args.epochs, "dt_ms": DT_MS,
                     "sampling_interval_s": args.interval, "exe": str(EXE.relative_to(ROOT))},
        "per_scale": per_scale,
    }
    write_json(Path(args.out), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
