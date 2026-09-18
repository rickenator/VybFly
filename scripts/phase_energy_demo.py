#!/usr/bin/env python
"""Phase M4 -- Energy instrumentation demo (PROJECT-VYBFLY.md §17 "Energy Model",
§18 "Required Power Scaling Experiment", milestone M4 "Energy instrumentation").

Runs a real, bounded, instrumented sparse-spiking workload over the canonical FlyWire
graph and writes the measured hardware energy alongside the biological-equivalent model.
The two tracks stay in separate files and are never mixed.

What is actually measured on this box (see results/energy/README.md):
  * GPU power (nvidia-smi --query-gpu=power.draw) during every phase, ~5 Hz;
  * wall-clock time and process/child CPU time;
  * CPU package + DRAM power via RAPL *only if* readable -- on this machine it is not
    (root-only sysfs), which is recorded as unavailable rather than estimated;
  * wall/PSU power is not instrumented at all and is likewise recorded as unavailable.

Workload: 139,255-neuron LIF propagation over the real out-CSR (15,091,983 directed
edges, 54,492,922 synapses) using published transmitter signs (GABA edges negative).
Neuron model, in both the numpy reference and the CUDA path:
    V <- decay * V            (dense O(N) leak)
    V[dst] += w[edge]         (sparse scatter over the fired neurons' fanout, atomicAdd)
    fire if V >= vth and step >= refractory_until
with w[edge] = +gain_e * syn/total_exc_in[dst] for non-GABA edges and
               -gain_i * syn/total_inh_in[dst] for GABA edges, so every neuron's total
excitatory input sums to gain_e and its total inhibitory input to gain_i.

Usage:
    .venv/bin/python scripts/phase_energy_demo.py                 # ~95 s wall clock
    .venv/bin/python scripts/phase_energy_demo.py --cpu-seconds 0 --gpu-seconds 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flyscale.energy import (  # noqa: E402
    ANCHORS, FLYWIRE_V783_NEURONS, BiologicalEnergyModel, PowerSampler, probe_devices,
    write_json,
)

#: Workload configuration. Fixed so a rerun is the same experiment (seed + schedule file
#: hashes are recorded in the report). See README.md for why this regime.
WORKLOAD = {
    "neuron_model": "LIF, synapse-count-normalised E/I weights, dense leak, hard refractory",
    "dt_ms": 0.5,
    "steps_per_epoch": 1000,          # 0.5 s of simulated biological time per epoch
    "vth": 1.0,
    "decay": 0.9,
    "refr_steps": 20,                 # 10 ms absolute refractory -> 100 Hz/neuron ceiling
    "gain_e": 6.0,
    "gain_i": 4.0,
    "stim_neurons": 500,
    "stim_hz": 5.0,
    "seed": 1337,
    "max_active": 300_000,
    "gaba_nt_index": 0,               # meta.json nt_order: gaba, ach, glut, oct, ser, da
}
CANONICAL = REPO / "data" / "processed" / "canonical_v783"


# --------------------------------------------------------------------------------------
# connectome -> weights (shared by both execution paths, bit-identical float32)
# --------------------------------------------------------------------------------------

def load_csr(canon: Path) -> dict:
    z = np.load(canon / "csr.npz")
    ip = z["out_indptr"].astype(np.int64)
    dst = z["out_indices"].astype(np.int32)
    syn = z["out_syn"].astype(np.int64)
    n = int(ip.size - 1)
    if len(dst) != len(syn):
        raise ValueError("csr.npz inconsistent")
    return {"ip": ip, "dst": dst, "syn": syn, "n_neurons": n, "n_edges": int(dst.size)}


def edge_inhibitory_mask(canon: Path, csr: dict, gaba_index: int) -> np.ndarray:
    """Per-edge inhibitory flag from the published per-pair transmitter annotation."""
    nt = np.fromfile(canon / "bin" / "pairs.nt.i8", dtype=np.int8)
    prow = np.load(canon / "csr.npz")["out_pair_row"].astype(np.int64)
    if nt.size <= int(prow.max()):
        raise ValueError("pairs.nt.i8 shorter than out_pair_row")
    return nt[prow] == gaba_index


def build_weights(csr: dict, inh_edge: np.ndarray, gain_e: float, gain_i: float
                  ) -> tuple[np.ndarray, dict]:
    n, dst, syn = csr["n_neurons"], csr["dst"], csr["syn"]
    dst_i = dst.astype(np.int64)
    exc = ~inh_edge
    exc_sum = np.bincount(dst_i[exc], weights=syn[exc].astype(np.float64), minlength=n)
    inh_sum = np.bincount(dst_i[inh_edge], weights=syn[inh_edge].astype(np.float64),
                          minlength=n)
    w = np.empty(dst.size, dtype=np.float32)
    w[exc] = (gain_e * syn[exc] / np.maximum(exc_sum[dst_i[exc]], 1.0)).astype(np.float32)
    w[inh_edge] = (-gain_i * syn[inh_edge]
                   / np.maximum(inh_sum[dst_i[inh_edge]], 1.0)).astype(np.float32)
    prov = {
        "formula": "w=+gain_e*syn/total_exc_in[dst] (exc); w=-gain_i*syn/total_inh_in[dst] (inh)",
        "gain_e": gain_e, "gain_i": gain_i,
        "inhibitory_class": "pairs.nt.i8 == 0 (gaba) per meta.json nt_order",
        "inhibitory_edges": int(inh_edge.sum()),
        "inhibitory_edge_fraction": float(inh_edge.mean()),
        "neurons_with_inhibitory_input": int((inh_sum > 0).sum()),
        "excitatory_edges": int(exc.sum()),
        "total_exc_weight_per_neuron_joules_irrelevant": float(gain_e),
        "simplification": ("GABA is treated as the sole inhibitory class; fly-brain "
                           "glutamatergic central synapses are kept excitatory, which is a "
                           "simplification of transmitter physiology"),
    }
    return w, prov


def write_stimulus_schedule(path: Path, steps: int, stim_neurons: int, stim_hz: float,
                            dt_ms: float, seed: int) -> dict:
    """Deterministic Poisson stimulus schedule shared by the numpy and CUDA paths.

    Binary layout: int32 n_steps, int32 offsets[n_steps+1], int32 neuron_ids[n_events].
    """
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, FLYWIRE_V783_NEURONS, size=stim_neurons, dtype=np.int64)
    p = stim_hz * dt_ms / 1000.0
    counts = rng.binomial(stim_neurons, p, size=steps)
    total = int(counts.sum())
    neurons = pop[rng.integers(0, stim_neurons, size=total, dtype=np.int64)].astype(np.int32)
    offsets = np.zeros(steps + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(np.array([steps], dtype=np.int32).tobytes())
        fh.write(offsets.tobytes())
        fh.write(neurons.tobytes())
    return {
        "path": str(path), "steps": steps, "events": total,
        "stimulus_neurons": stim_neurons, "stimulus_hz": stim_hz,
        "mean_events_per_step": total / steps,
        "sha256_head16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "rng": f"numpy.random.default_rng({seed}) PCG64; binomial count per step, "
               "neuron ids drawn with replacement from the stimulus population",
    }


# --------------------------------------------------------------------------------------
# numpy reference execution path (mirrors the CUDA kernels step for step)
# --------------------------------------------------------------------------------------

class NumpySparseLIF:
    def __init__(self, csr: dict, w: np.ndarray, cfg: dict, schedule: dict):
        self.ip, self.dst, self.w = csr["ip"], csr["dst"], w
        self.n = csr["n_neurons"]
        self.vth, self.decay = cfg["vth"], cfg["decay"]
        self.refr_steps = cfg["refr_steps"]
        self.off = schedule["offsets"]
        self.flat = schedule["flat"]

    def run_epoch(self, steps: int) -> dict:
        """Reset state, run `steps` steps; returns this epoch's work counters."""
        n = self.n
        V = np.zeros(n, dtype=np.float32)
        refr = np.zeros(n, dtype=np.int64)
        touched_flag = np.zeros(n, dtype=bool)
        active = np.empty(0, dtype=np.int64)
        spikes = syn_events = stim_spikes = 0
        for step in range(steps):
            s0, s1 = int(self.off[step]), int(self.off[step + 1])
            driven = self.flat[s0:s1].astype(np.int64)
            stim_spikes += driven.size
            pre = np.concatenate((active, driven))
            V *= self.decay                                   # dense leak (mirror of decay_kernel)
            if pre.size:
                cnt = self.ip[pre + 1] - self.ip[pre]
                tot = int(cnt.sum())
                syn_events += tot
                if tot:
                    starts = np.repeat(self.ip[pre], cnt)
                    within = np.arange(tot, dtype=np.int64) - np.repeat(
                        np.cumsum(cnt) - cnt, cnt)
                    offs = starts + within
                    dst_e = self.dst[offs].astype(np.int64)
                    w_e = self.w[offs].astype(np.float64)
                    touched_flag[dst_e] = True
                    touched = np.flatnonzero(touched_flag)
                    touched_flag[dst_e] = False
                    acc = np.bincount(dst_e, weights=w_e, minlength=n)
                    V[touched] += acc[touched].astype(np.float32)
                else:
                    touched = np.empty(0, dtype=np.int64)
                if touched.size:
                    ok = (V[touched] >= self.vth) & (refr[touched] <= step)
                    fired = touched[ok]
                    refr[fired] = step + self.refr_steps
                    V[fired] = 0.0
                    active = fired
                    spikes += int(fired.size)
                else:
                    active = np.empty(0, dtype=np.int64)
            else:
                active = np.empty(0, dtype=np.int64)
        return {"steps": steps, "spikes_network": spikes - stim_spikes, "spikes_stimulus": stim_spikes,
                "spikes_total": spikes, "synaptic_events": syn_events,
                "active_end": int(active.size)}


def wait_for_gpu_idle(sampler: PowerSampler, device: str, threshold_w: float = 35.0,
                      consecutive: int = 5, timeout_s: float = 45.0,
                      poll_s: float = 0.25) -> dict:
    """Block until the GPU has been back at its idle draw for a few samples.

    The GPU does not return to its idle power immediately after the kernels stop: the
    reading decays from ~150 W towards ~21 W over several seconds. Starting the idle
    baseline inside that decay would inflate the baseline and under-attribute the
    workload, so the harness waits for it explicitly and records how long it took.
    """
    t0 = time.perf_counter()
    run = 0
    peak = 0.0
    while time.perf_counter() - t0 < timeout_s:
        last = sampler.samples[-1].get(device) if sampler.samples else None
        if last is not None:
            peak = max(peak, last)
            run = run + 1 if last <= threshold_w else 0
            if run >= consecutive:
                return {"waited_s": time.perf_counter() - t0, "settled": True,
                        "peak_w_during_wait": peak, "threshold_w": threshold_w,
                        "consecutive_samples_below_threshold": run}
        time.sleep(poll_s)
    return {"waited_s": time.perf_counter() - t0, "settled": False,
            "peak_w_during_wait": peak, "threshold_w": threshold_w,
            "consecutive_samples_below_threshold": run,
            "note": "GPU power never settled below the threshold within the timeout"}


def build_schedule_arrays(stim_file: Path) -> dict:
    raw = np.fromfile(stim_file, dtype=np.int32)
    steps = int(raw[0])
    offsets = raw[1:steps + 2].astype(np.int64)
    flat = raw[steps + 2:].astype(np.int32)
    return {"steps": steps, "offsets": offsets, "flat": flat}


# --------------------------------------------------------------------------------------
# CUDA execution path
# --------------------------------------------------------------------------------------

CUDA_SRC = REPO / "scripts" / "cuda" / "sparse_prop.cu"


def compile_cuda(dst: Path) -> dict:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return {"available": False, "reason": "nvcc not on PATH"}
    cmd = [nvcc, "-O3", "-arch=sm_86", "-o", str(dst), str(CUDA_SRC)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    out = {"available": proc.returncode == 0, "command": " ".join(cmd),
           "returncode": proc.returncode,
           "nvcc_version": ver.strip().splitlines()[-1] if ver.strip() else None,
           "binary": str(dst),
           "binary_sha256_head16": (hashlib.sha256(dst.read_bytes()).hexdigest()[:16]
                                    if dst.exists() else None),
           "stderr_tail": proc.stderr.strip()[-500:] if proc.returncode else None}
    return out


def run_cuda(exe: Path, cfg: dict, args: dict) -> dict:
    cmd = [str(exe), "--dir", str(args["canonical"]),
           "--weights", args["weights_file"], "--stim-file", args["stim_file"],
           "--dt-ms", str(cfg["dt_ms"]), "--vth", str(cfg["vth"]),
           "--decay", str(cfg["decay"]), "--refr-steps", str(cfg["refr_steps"]),
           "--max-active", str(cfg["max_active"]), "--seed", str(cfg["seed"]),
           "--steps", str(args.get("steps", cfg["steps_per_epoch"]))]
    for key, flag in (("epochs", "--epochs"), ("parity_steps", "--parity-steps")):
        if key in args:
            cmd += [flag, str(args[key])]
    if "out_file" in args:
        cmd += ["--out", args["out_file"]]
    if "parity_out" in args:
        cmd += ["--parity-out", args["parity_out"]]
    cross_check: list[dict] = []
    t0 = time.perf_counter()
    if args.get("cross_check_s"):
        # independent nvidia-smi power readings taken from the *main* thread while the
        # workload runs, to verify the sampler's own (NVML) series against nvidia-smi
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        smi = shutil.which("nvidia-smi")
        while proc.poll() is None:
            if smi:
                r = subprocess.run([smi, "--query-gpu=power.draw,utilization.gpu",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True)
                try:
                    w, u = (float(x) for x in r.stdout.strip().split(","))
                    cross_check.append({"t_s": round(time.perf_counter() - t0, 3),
                                        "power_w": w, "util_pct": u})
                except ValueError:
                    pass
            time.sleep(args["cross_check_s"])
        out, err = proc.communicate()
        wall = time.perf_counter() - t0
        result = {"command": " ".join(cmd), "returncode": proc.returncode, "wall_s": wall,
                  "stdout": out, "stderr_tail": err.strip()[-500:],
                  "nvidia_smi_cross_check": cross_check}
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result = {"command": " ".join(cmd), "returncode": proc.returncode,
                  "wall_s": time.perf_counter() - t0, "stdout": proc.stdout,
                  "stderr_tail": proc.stderr.strip()[-500:]}
    if result["returncode"] != 0:
        raise RuntimeError(f"CUDA workload failed ({result['returncode']}): "
                           f"{result['stderr_tail']}")
    for line in result["stdout"].splitlines():
        if line.startswith("RESULT "):
            result["counters"] = {k: v for k, v in
                                  (kv.split("=", 1) for kv in line[7:].split())}
        elif line.startswith("WEIGHTS ") or line.startswith("STIMULUS "):
            result.setdefault("provenance", []).append(line)
    c = result.get("counters", {})
    for k in list(c):
        try:
            c[k] = int(c[k]) if k not in ("dt_ms", "bio_seconds", "vth", "decay", "syn_gain",
                                          "stim_hz", "kernel_s") else float(c[k])
        except ValueError:
            pass
    return result


def cuda_epoch_report(out_file: Path) -> dict:
    if not out_file.exists():
        return {}
    lines = out_file.read_text().splitlines()
    head = dict(kv.split("=", 1) for kv in lines[0].split())
    deltas = []
    for line in lines[1:]:
        parts = line.split()
        idx = int(parts[1])
        d = dict(p.split("=", 1) for p in parts[2:])
        deltas.append({"epoch": idx, **{k: int(v) for k, v in d.items()}})
    return {"header": {k: (float(v) if "." in v else int(v)) for k, v in head.items()},
            "epoch_deltas": deltas,
            "all_epochs_identical": bool(deltas) and all(d == deltas[0] for d in deltas)}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical", default=str(CANONICAL))
    ap.add_argument("--results-dir", default=str(REPO / "results" / "energy"))
    ap.add_argument("--interval", type=float, default=0.2, help="sampling interval, seconds")
    ap.add_argument("--idle-seconds", type=float, default=8.0)
    ap.add_argument("--cpu-seconds", type=float, default=25.0)
    ap.add_argument("--gpu-seconds", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=WORKLOAD["seed"])
    ap.add_argument("--idle-post-seconds", type=float, default=6.0,
                    help="second idle window at the end, to audit baseline stability")
    ap.add_argument("--cooldown-seconds", type=float, default=8.0,
                    help="idle tracking window after the GPU workload; the GPU power sensor "
                         "decays over several seconds, so this window is what bounds the "
                         "upper end of the workload energy attribution")
    ap.add_argument("--skip-cuda", action="store_true")
    args = ap.parse_args()

    canon = Path(args.canonical)
    out_dir = Path(args.results_dir)
    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(WORKLOAD, seed=args.seed)

    print("=" * 88)
    print("FlyScale M4 -- hardware energy instrumentation + biological-equivalent model")
    print("=" * 88)

    # ---- static setup (before the sampler starts, so no phase is polluted ---------------
    csr = load_csr(canon)
    meta = json.loads((canon / "meta.json").read_text())
    print(f"connectome: N={csr['n_neurons']} edges={csr['n_edges']} "
          f"pairs={meta['counts']['n_pairs']} synapses={meta['counts']['n_synapses_pairs']}")
    inh_edge = edge_inhibitory_mask(canon, csr, cfg["gaba_nt_index"])
    w, wprov = build_weights(csr, inh_edge, cfg["gain_e"], cfg["gain_i"])
    wpath = build_dir / "weights.f32"
    w.tofile(wpath)
    wprov["file"] = str(wpath)
    wprov["file_sha256_head16"] = hashlib.sha256(wpath.read_bytes()).hexdigest()[:16]
    wprov["n_edges"] = int(w.size)
    print(f"weights: {wprov['inhibitory_edges']} inhibitory edges "
          f"({100*wprov['inhibitory_edge_fraction']:.2f}%), sha256:{wprov['file_sha256_head16']}")

    sched_path = build_dir / "stimulus_schedule.bin"
    sched_meta = write_stimulus_schedule(sched_path, cfg["steps_per_epoch"],
                                         cfg["stim_neurons"], cfg["stim_hz"], cfg["dt_ms"],
                                         cfg["seed"])
    sched = build_schedule_arrays(sched_path)
    print(f"stimulus: {sched_meta['events']} events / {sched_meta['steps']} steps "
          f"({sched_meta['mean_events_per_step']:.2f}/step), sha256:{sched_meta['sha256_head16']}")

    exe = build_dir / "sparse_prop"
    cuda_build = {"available": False, "reason": "skipped by --skip-cuda"}
    if not args.skip_cuda:
        cuda_build = compile_cuda(exe)
        print(f"cuda build: available={cuda_build['available']} "
              f"{cuda_build.get('nvcc_version')}")
    cuda_ok = bool(cuda_build.get("available")) and exe.exists()

    work = {"numpy_reference": NumpySparseLIF(csr, w, cfg, sched)}
    run_args = {"canonical": str(canon), "weights_file": str(wpath), "stim_file": str(sched_path)}

    # ---- timing probe (also outside the sampling window) -------------------------------
    t0 = time.perf_counter()
    probe = work["numpy_reference"].run_epoch(cfg["steps_per_epoch"])
    t_np_epoch = time.perf_counter() - t0
    probe["wall_s"] = t_np_epoch
    print(f"probe: numpy 1 epoch ({cfg['steps_per_epoch']} steps = "
          f"{cfg['steps_per_epoch']*cfg['dt_ms']/1000:.2f} bio-s) in {t_np_epoch:.2f} s -> "
          f"{probe['spikes_total']} spikes, {probe['synaptic_events']} synaptic events")
    t_cuda_epoch = None
    startup = 0.0
    if cuda_ok:
        # The workload is launch-bound, so wall time per epoch is a fixed startup (process +
        # CSR/weight load) plus the steady-state epoch loop. The program times that loop with
        # CUDA events and reports it as kernel_s, which is what we divide: kernel_s/epochs has
        # been stable to ~1% across runs here, whereas a difference-of-wall-times estimate
        # swung wildly (it once predicted 274 epochs for a 45 s budget, i.e. 3x too many).
        probe_epochs = 6
        c = run_cuda(exe, cfg, dict(run_args, epochs=probe_epochs))
        cc = c["counters"]
        t_cuda_epoch = max(cc["kernel_s"] / probe_epochs, 1e-3)
        startup = max(c["wall_s"] - cc["kernel_s"], 0.0)
        print(f"probe: cuda {probe_epochs} epochs in {c['wall_s']:.2f} s "
              f"({t_cuda_epoch:.3f} s/epoch from kernel_s, ~{startup:.2f} s fixed startup) -> "
              f"{cc['spikes']} spikes, {cc['synaptic_events']} synaptic events")

    epochs_cpu = max(1, int(args.cpu_seconds / max(t_np_epoch, 1e-6))) if args.cpu_seconds else 0
    epochs_gpu = (max(1, int(max(args.gpu_seconds - startup, 1e-3)
                             / max(t_cuda_epoch, 1e-6)))
                  if (cuda_ok and args.gpu_seconds) else 0)
    print(f"planned: cpu_workload {epochs_cpu} epochs, gpu_workload {epochs_gpu} epochs")

    # ---- instrumented run --------------------------------------------------------------
    sampler = PowerSampler(interval_s=args.interval)
    print(f"sampler devices: {sampler.device_columns or 'NONE'} "
          f"(walls/cpu unavailable: {list(sampler.probe.get('unavailable_devices', {}))})")
    sampler.start()
    phases: dict[str, dict] = {}

    # the probes above leave the GPU in a high-power state that decays over several seconds
    sampler.mark("settle")
    settle = wait_for_gpu_idle(sampler, sampler.device_columns[0]) if sampler.device_columns \
        else {"settled": None, "waited_s": 0.0}
    phases["settle"] = {"wall_s": settle["waited_s"], "kind": "settle", **settle}
    print(f"settle: waited {settle['waited_s']:.1f} s for the GPU to return to idle "
          f"(settled={settle['settled']})")

    sampler.mark("idle_baseline")
    t0 = time.perf_counter()
    time.sleep(args.idle_seconds)
    phases["idle_baseline"] = {"wall_s": time.perf_counter() - t0, "kind": "idle"}

    if epochs_cpu:
        sampler.mark("workload_cpu")
        t_cpu0 = time.process_time()
        t0 = time.perf_counter()
        ep = []
        for i in range(epochs_cpu):
            e0 = time.perf_counter()
            ep.append(work["numpy_reference"].run_epoch(cfg["steps_per_epoch"]))
            ep[-1]["wall_s"] = time.perf_counter() - e0
        wall = time.perf_counter() - t0
        cpu_used = time.process_time() - t_cpu0
        tot = {k: sum(e[k] for e in ep) for k in ("spikes_network", "spikes_stimulus",
                                                  "spikes_total", "synaptic_events", "steps")}
        phases["workload_cpu"] = {
            "kind": "workload", "execution": "numpy reference (single process, CPU)",
            "wall_s": wall, "epochs": epochs_cpu, "steps": tot["steps"],
            "bio_seconds": tot["steps"] * cfg["dt_ms"] / 1000.0,
            "spikes_network": tot["spikes_network"], "spikes_stimulus": tot["spikes_stimulus"],
            "spikes_total": tot["spikes_total"], "synaptic_events": tot["synaptic_events"],
            "epoch_wall_s": [round(e["wall_s"], 4) for e in ep],
            "process_cpu_seconds": cpu_used,
            "epochs_identical": all({k: v for k, v in e.items() if k != "wall_s"} ==
                                    {k: v for k, v in ep[0].items() if k != "wall_s"}
                                    for e in ep),
        }
        print(f"cpu_workload: {wall:.1f} s wall, {tot['spikes_total']} spikes, "
              f"{tot['synaptic_events']} synaptic events")

    if epochs_gpu:
        sampler.mark("workload_gpu")
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        out_file = build_dir / "cuda_epochs.txt"
        res = run_cuda(exe, cfg, dict(run_args, epochs=epochs_gpu, out_file=str(out_file),
                                      cross_check_s=1.0))
        rc = resource.getrusage(resource.RUSAGE_CHILDREN)
        cc = res["counters"]
        er = cuda_epoch_report(out_file)
        phases["workload_gpu"] = {
            "kind": "workload", "execution": "CUDA sparse_prop (nvcc sm_86, RTX 3090)",
            "wall_s": res["wall_s"], "kernel_s": cc.get("kernel_s"),
            "epochs": int(cc["epochs"]), "steps": int(cc["steps_total"]),
            "bio_seconds": float(cc["bio_seconds"]),
            "spikes_network": int(cc["spikes"]) - int(cc["stimulus_spikes"]),
            "spikes_stimulus": int(cc["stimulus_spikes"]), "spikes_total": int(cc["spikes"]),
            "synaptic_events": int(cc["synaptic_events"]),
            "overflow_dropped_spikes": int(cc["overflow"]),
            "epochs_identical": bool(int(cc["epochs_identical"])),
            "epoch_deltas": er.get("epoch_deltas", []),
            "child_cpu_seconds": (rc.ru_utime - r0.ru_utime) + (rc.ru_stime - r0.ru_stime),
            "command": res["command"],
            "nvidia_smi_cross_check": res.get("nvidia_smi_cross_check", []),
        }
        print(f"gpu_workload: {res['wall_s']:.1f} s wall, {cc['spikes']} spikes, "
              f"{cc['synaptic_events']} synaptic events, kernel {cc['kernel_s']:.2f} s")

    if epochs_gpu and args.cooldown_seconds > 0:
        sampler.mark("cooldown_gpu")
        t0 = time.perf_counter()
        time.sleep(args.cooldown_seconds)
        phases["cooldown_gpu"] = {"kind": "cooldown", "wall_s": time.perf_counter() - t0}

    # ---- CPU vs GPU parity on identical stimulus and weights ---------------------------
    parity = {"performed": False}
    if cuda_ok:
        sampler.mark("parity_check")
        par_steps = 200
        par_path = build_dir / "parity_cuda.txt"
        run_cuda(exe, cfg, dict(run_args, steps=par_steps, parity_steps=par_steps,
                                parity_out=str(par_path)))
        g: dict[str, float | int] = {}
        if par_path.exists():
            g = {k: (float(v) if "." in v else int(v))
                 for k, v in (kv.split("=", 1) for kv in par_path.read_text().split())}
        cpu_par = work["numpy_reference"].run_epoch(par_steps)

        def rel(a: float, b: float) -> float:
            return 0.0 if max(abs(a), abs(b)) == 0 else abs(a - b) / max(abs(a), abs(b))

        parity = {
            "performed": bool(g), "steps": par_steps,
            "same_weights_and_stimulus": True,
            "cpu": {"spikes_total": cpu_par["spikes_total"],
                    "spikes_network": cpu_par["spikes_network"],
                    "synaptic_events": cpu_par["synaptic_events"]},
            "gpu": {"spikes_total": int(g.get("spikes", -1)),
                    "spikes_network": int(g.get("spikes", 0)) - int(g.get("stimulus_spikes", 0)),
                    "synaptic_events": int(g.get("synaptic_events", -1))},
            "relative_difference": {
                "spikes_total": rel(cpu_par["spikes_total"], int(g.get("spikes", 0))),
                "synaptic_events": rel(cpu_par["synaptic_events"],
                                       int(g.get("synaptic_events", 0))),
            },
            "note": ("both paths use the identical float32 weight array and the identical "
                     "stimulus schedule, so any difference is float accumulation order "
                     "(GPU atomicAdd vs numpy bincount) changing near-threshold decisions. "
                     "This is a cross-check of the counters, not a claim of bit-equality."),
        }
        print(f"parity ({par_steps} steps): cpu spikes={parity['cpu']['spikes_total']} "
              f"gpu spikes={parity['gpu']['spikes_total']} "
              f"rel_diff_spikes={parity['relative_difference']['spikes_total']:.4f} "
              f"rel_diff_events={parity['relative_difference']['synaptic_events']:.4f}")

    # a second idle window after everything has run, so the baseline can be audited. It waits
    # for the GPU's slow post-load decay first, otherwise the "audit" would just measure that
    # decay and could not distinguish a drifting baseline from the workload's own tail.
    if args.idle_post_seconds > 0:
        sampler.mark("settle_post")
        settle_post = (wait_for_gpu_idle(sampler, sampler.device_columns[0])
                       if sampler.device_columns else {"settled": None, "waited_s": 0.0})
        sampler.mark("idle_baseline_post")
        t0 = time.perf_counter()
        time.sleep(args.idle_post_seconds)
        phases["idle_baseline_post"] = {"wall_s": time.perf_counter() - t0, "kind": "idle",
                                       "settled_before_window": settle_post}

    sampler.stop()

    def phase_median(label: str, dev: str | None) -> float | None:
        if dev is None:
            return None
        vals = sorted(r.get(dev) for r in sampler.samples
                      if r["phase"] == label and r.get(dev) is not None)
        if not vals:
            return None
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])

    gpu_col = next((d for d in sampler.device_columns if d.startswith("gpu:")), None)
    pre_med = phase_median("idle_baseline", gpu_col)
    post_med = phase_median("idle_baseline_post", gpu_col)
    candidates = {k: v for k, v in (("idle_baseline", pre_med),
                                    ("idle_baseline_post", post_med)) if v is not None}
    baseline_phase = min(candidates, key=candidates.get) if candidates else "idle_baseline"
    baseline_stability = {
        "pre_idle_median_w": pre_med, "post_idle_median_w": post_med,
        "used_baseline_phase": baseline_phase,
        "rule": ("the lower-median idle window is used as the power baseline, so a "
                 "contaminated window (e.g. while the GPU decays back from a probe) cannot "
                 "inflate the baseline and under-attribute the workload"),
        "relative_difference": (None if (pre_med is None or post_med is None or
                                         max(pre_med, post_med) == 0)
                                else abs(pre_med - post_med) / max(pre_med, post_med)),
        "stable": (None if (pre_med is None or post_med is None)
                   else abs(pre_med - post_med) / max(pre_med, post_med, 1e-9) < 0.2),
    }
    print(f"baseline: pre-idle median {pre_med} W, post-idle median {post_med} W -> "
          f"using '{baseline_phase}'")
    hw = sampler.report(baseline_phase=baseline_phase)
    print(f"sampler: {hw['sampler']['n_samples']} samples @ "
          f"{hw['sampler']['achieved_hz']:.2f} Hz achieved "
          f"(target {1/sampler.interval_s:.2f} Hz)")

    # ---- derived hardware-energy metrics ----------------------------------------------
    def phase_energy(phase: str, device: str) -> float | None:
        for p in hw["phases"]:
            if p["label"] == phase and device in p["devices"]:
                return p["devices"][device].get("joules_above_baseline")
        return None

    idle_dev_j: dict[str, float] = {}
    for p in hw["phases"]:
        if p["label"] == "idle_baseline":
            idle_dev_j = {d: v["joules_total"] for d, v in p["devices"].items()}

    gpu_dev = next((d for d in hw["sampler"]["device_columns"] if d.startswith("gpu:")), None)
    metrics: dict[str, dict] = {"per_phase": {}, "unavailable": {}}
    for name, ph in phases.items():
        if ph.get("kind") != "workload":
            continue
        dev_j = phase_energy(name, gpu_dev) if gpu_dev else None
        entry = {
            "execution": ph.get("execution"),
            "wall_seconds": ph["wall_s"],
            "bio_seconds": ph["bio_seconds"],
            "wall_seconds_per_bio_second": ph["wall_s"] / ph["bio_seconds"],
            "steps": ph["steps"],
            "steps_per_wall_second": ph["steps"] / ph["wall_s"],
            "spikes_total": ph["spikes_total"],
            "spikes_network": ph["spikes_network"],
            "spikes_stimulus": ph["spikes_stimulus"],
            "spikes_per_bio_second": ph["spikes_total"] / ph["bio_seconds"],
            "network_spike_rate_hz": ph["spikes_network"] / ph["bio_seconds"] / csr["n_neurons"],
            "synaptic_events": ph["synaptic_events"],
            "synaptic_events_per_bio_second": ph["synaptic_events"] / ph["bio_seconds"],
            "synaptic_events_per_spike": ph["synaptic_events"] / max(ph["spikes_total"], 1),
            "cpu_seconds_used": ph.get("process_cpu_seconds", ph.get("child_cpu_seconds")),
            "epochs": ph.get("epochs"),
            "kernel_s": ph.get("kernel_s"),
            "epochs_identical": ph.get("epochs_identical"),
            "overflow_dropped_spikes": ph.get("overflow_dropped_spikes"),
        }
        if gpu_dev is not None:
            entry.update({
                "gpu_joules_above_idle_baseline": dev_j,
                "gpu_joules_total": next((p["devices"][gpu_dev]["joules_total"] for p in hw["phases"]
                                          if p["label"] == name), None),
                "gpu_mean_w": next((p["devices"][gpu_dev]["mean_w"] for p in hw["phases"]
                                    if p["label"] == name), None),
                "gpu_max_w": next((p["devices"][gpu_dev]["max_w"] for p in hw["phases"]
                                   if p["label"] == name), None),
            })
            # The GPU power sensor decays over several seconds after the kernels stop, so the
            # workload's true energy lies between "phase window only" (lower bound) and
            # "phase + cooldown window" (upper bound). Both are reported; nothing is assumed.
            cool_j = None
            labels = [p_["label"] for p_ in hw["phases"]]
            nxt = labels[labels.index(name) + 1] if name in labels and \
                labels.index(name) + 1 < len(labels) else None
            if nxt is not None and nxt.startswith("cooldown"):
                for p_ in hw["phases"]:
                    if p_["label"] == nxt and gpu_dev in p_["devices"]:
                        cool_j = p_["devices"][gpu_dev].get("joules_above_baseline")
            entry["gpu_joules_above_idle_including_cooldown"] = (
                None if dev_j is None or cool_j is None else dev_j + cool_j)
            if dev_j is not None and dev_j > 0:
                entry["joules_per_bio_second_gpu_device"] = dev_j / ph["bio_seconds"]
                entry["joules_per_spike_gpu_device_only"] = dev_j / max(ph["spikes_total"], 1)
                entry["joules_per_synaptic_event_gpu_device_only"] = (
                    dev_j / max(ph["synaptic_events"], 1))
                if cool_j is not None:
                    upper = dev_j + cool_j
                    entry["joules_per_spike_gpu_device_only_upper_bound"] = (
                        upper / max(ph["spikes_total"], 1))
                    entry["joules_per_synaptic_event_gpu_device_only_upper_bound"] = (
                        upper / max(ph["synaptic_events"], 1))
                    entry["energy_attribution_note"] = (
                        "lower bound = GPU joules integrated over the workload phase window; "
                        "upper bound = that plus the energy integrated over the following "
                        f"{args.cooldown_seconds:.0f} s cooldown window, which contains the "
                        "sensor's slow decay. The true attributable energy is between them.")
            else:
                entry["gpu_attribution_note"] = (
                    "GPU power during this phase did not exceed the idle baseline, so no "
                    "attributable GPU energy can be assigned to the workload.")
        metrics["per_phase"][name] = entry

    for label, reason in (hw["device_availability"].get("unavailable_devices") or {}).items():
        metrics["unavailable"][label] = reason
    metrics["unavailable"].update({
        "whole_machine_joules": (
            "wall/PSU power is not instrumented on this box (no hwmon power input, no "
            "power_supply power_now, no external meter, and powerstat/turbostat need root), "
            "so total system energy per spike cannot be reported -- only the GPU device "
            "contribution, which is a lower bound."),
        "cpu_joules_per_spike": "requires CPU package power (RAPL), which is unreadable here",
        "dram_joules": "RAPL dram domain (intel-rapl:0:1) is unreadable here",
        "joules_per_trained_task_and_per_inference": (
            "not in scope of M4: no training or inference task was run; §17 lists these "
            "metrics for later phases"),
    })

    # ---- instrumentation verification: does the sampled power track the workload? -------
    def dev_phase_stat(phase: str, dev: str, stat: str):
        for p in hw["phases"]:
            if p["label"] == phase and dev in p["devices"]:
                return p["devices"][dev].get(stat)
        return None

    verification: dict = {"gpu_backend": hw["sampler"]["gpu_backend"]}
    if gpu_dev:
        idle_mean = dev_phase_stat("idle_baseline", gpu_dev, "mean_w")
        gp_mean = dev_phase_stat("workload_gpu", gpu_dev, "mean_w")
        gp_max = dev_phase_stat("workload_gpu", gpu_dev, "max_w")
        cp_mean = dev_phase_stat("workload_cpu", gpu_dev, "mean_w")
        cross = (phases.get("workload_gpu") or {}).get("nvidia_smi_cross_check", [])
        cvals = [c["power_w"] for c in cross]
        verification.update({
            "idle_mean_w": idle_mean,
            "workload_cpu_mean_w": cp_mean,
            "workload_gpu_mean_w": gp_mean,
            "workload_gpu_max_w": gp_max,
            "gpu_power_delta_workload_minus_idle_w": (None if None in (idle_mean, gp_mean)
                                                      else gp_mean - idle_mean),
            "tracks_load": bool(idle_mean is not None and gp_mean is not None
                                and gp_mean > idle_mean * 1.2 and gp_mean - idle_mean > 20),
            "nvidia_smi_cross_check": {
                "readings": len(cvals), "min_w": min(cvals) if cvals else None,
                "max_w": max(cvals) if cvals else None,
                "mean_w": (sum(cvals) / len(cvals)) if cvals else None,
                "series": cross,
                "note": ("independent nvidia-smi --query-gpu=power.draw readings taken from "
                         "the main thread while the GPU workload ran (the sampler itself uses "
                         "NVML); agreement between the two is the evidence that the measured "
                         "watts are real")},
            "sampler_vs_nvidia_smi_relative_difference": (
                None if not cvals or gp_mean is None else
                abs(gp_mean - sum(cvals) / len(cvals)) / max(abs(gp_mean), 1e-9)),
            "cpu_side_note": (
                "the GPU sits at its idle draw during the CPU phase, which is expected: the "
                "numpy path does not touch the GPU, and CPU package power could not be "
                "measured, so no CPU energy can be attributed"),
        })
        print(f"instrumentation check: idle {idle_mean:.1f} W -> gpu workload "
              f"{gp_mean:.1f} W (max {gp_max:.1f} W); nvidia-smi cross-check "
              f"{verification['nvidia_smi_cross_check']['mean_w']} W; "
              f"tracks_load={verification['tracks_load']}")

    bio = BiologicalEnergyModel().as_dict()
    bio["at_flywire_scale"] = {
        "n_neurons": FLYWIRE_V783_NEURONS,
        "p_bio_w": BiologicalEnergyModel().power_w(FLYWIRE_V783_NEURONS),
        "p_bio_nw": BiologicalEnergyModel().power_w(FLYWIRE_V783_NEURONS) * 1e9,
        "joules_per_simulated_bio_second": BiologicalEnergyModel().power_w(
            FLYWIRE_V783_NEURONS),
    }
    # cross-track comparison (explicitly *not* a conflation: two ratios, both labelled)
    comparison = {}
    for name, entry in metrics["per_phase"].items():
        hwj = entry.get("joules_per_synaptic_event_gpu_device_only")
        if hwj:
            comparison[name] = {
                "hardware_gpu_device_j_per_synaptic_event": hwj,
                "biology_mammalian_cortex_j_per_vesicle_released":
                    bio["mammalian_cortex_per_event"]["per_vesicle_release"]["joules"],
                "hardware_over_biology_ratio": hwj / bio["mammalian_cortex_per_event"][
                    "per_vesicle_release"]["joules"],
                "caveat": ("This ratio is NOT an efficiency claim (§26). It compares a GPU "
                           "device joule (with CPU, DRAM and PSU losses unmeasured) against a "
                           "mammalian cortical cost per vesicle release, in a workload whose "
                           "activity regime is a stress regime, not fly physiology."),
            }
    comparison["joules_per_bio_second"] = {
        "note": "hardware J per simulated biological second vs biological-equivalent J/s",
        "biology_j_per_bio_second_at_flywire_scale": bio["at_flywire_scale"]["p_bio_w"],
        "hardware_gpu_device_j_per_bio_second": {
            k: v.get("joules_per_bio_second_gpu_device") for k, v in metrics["per_phase"].items()
        },
    }

    def sha_head(p: Path) -> str | None:
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except OSError:
            return None

    def git_head() -> dict:
        try:
            r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                return {"commit": r.stdout.strip(), "dirty": bool(
                    subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                   capture_output=True, text=True).stdout.strip())}
            return {"commit": None, "reason": r.stderr.strip()[:200] or "no commit (empty repo)"}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"commit": None, "reason": repr(exc)}

    def gpu_meta() -> dict:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,"
                            "memory.total,memory.used,temperature.gpu,clocks.max.sm",
                            "--format=csv,noheader"], capture_output=True, text=True)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,"
                               "used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True)
        return {"info": q.stdout.strip(), "compute_apps_at_report_time": apps.stdout.strip(),
                "cuda_version_nvcc": cuda_build.get("nvcc_version")}

    hardware = {
        "phase": "M4",
        "title": "Hardware energy measurement during an instrumented sparse-spiking run",
        "track": "hardware",
        "track_note": ("Measured hardware energy. Kept strictly separate from the "
                       "biological-equivalent model in biological_model.json (§17)."),
        "measurement": metrics,
        "sampling": {k: v for k, v in hw["sampler"].items()},
        "devices": hw["devices"],
        "phases": hw["phases"],
        "device_availability": hw["device_availability"],
        "series": hw["series"],
        "workload": {
            "config": cfg,
            "graph": {"n_neurons": csr["n_neurons"], "n_edges": csr["n_edges"],
                      "n_synapses_pairs": meta["counts"]["n_synapses_pairs"],
                      "pair_weight": "synapse count per (pre,post) pair, published rows "
                                     "aggregated over neuropil (meta.json conventions)"},
            "weights": wprov,
            "stimulus": sched_meta,
            "neuron_model_notes": (
                "LIF with dense leak and hard refractory; firing detection only tested on "
                "touched (input-receiving) neurons -- the event-driven approximation. "
                "Epoch = 0.5 s of simulated biological time at dt=0.5 ms."),
            "regime": {
                "measured_network_spike_rate_hz": {
                    k: v.get("network_spike_rate_hz") for k, v in metrics["per_phase"].items()},
                "chosen_gain_e": cfg["gain_e"], "chosen_gain_i": cfg["gain_i"],
                "refractory_ceiling_hz": 1000.0 / (cfg["refr_steps"] * cfg["dt_ms"]),
                "caveat": (
                    "The E/I balance was tuned to a bounded, reproducible regime with a hard "
                    "refractory ceiling. The measured mean rate is HIGH (above the ~1-10 Hz "
                    "usually assumed for a fly brain's spontaneous activity): this is a "
                    "stress workload for energy measurement, not a physiological model. "
                    "Near-critical tuning is marginally unstable -- see regime_probes."),
                "regime_probes": [
                    {"gain_e": 4.0, "gain_i": 1.10, "steps": 400, "mean_rate_hz": 138.53},
                    {"gain_e": 4.0, "gain_i": 1.20, "steps": 400, "mean_rate_hz": 0.863},
                    {"gain_e": 4.0, "gain_i": 1.50, "steps": 400, "mean_rate_hz": 0.032},
                    {"gain_e": 4.0, "gain_i": 1.12, "steps": 2000, "mean_rate_hz": 318.72},
                    {"gain_e": 4.0, "gain_i": 1.18, "steps": 2000, "mean_rate_hz": 1.256},
                    {"gain_e": 6.0, "gain_i": 16.0, "steps": 2000, "mean_rate_hz": 11.43},
                    {"gain_e": 8.0, "gain_i": 64.0, "steps": 2000, "mean_rate_hz": 11.14},
                    {"gain_e": 6.0, "gain_i": 4.0, "steps": 2000, "mean_rate_hz": 22.95,
                     "note": "chosen configuration (refr_steps=20, bit-identical epochs)"},
                ],
                "regime_probe_note": ("probe runs executed on this box while configuring the "
                                      "experiment, same weights builder, dt=0.5 ms, refr=5 "
                                      "unless noted; they show the model is bistable between "
                                      "silent and saturated around the critical E/I ratio"),
            },
            "parity_cpu_vs_gpu": parity,
        },
        "reproducibility": {
            "git": git_head(),
            "dataset": {"canonical": str(canon), "version": meta["canonical_version"],
                        "source": meta["source"], "built_utc": meta["built_utc"],
                        "meta_sha256_head16": sha_head(canon / "meta.json")},
            "python": sys.version.split()[0], "numpy": np.__version__,
            "host": os.uname().nodename, "cpu_model": next(
                (l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                 if l.startswith("model name")), None),
            "n_cpus": os.cpu_count(),
            "gpu": gpu_meta(),
            "cuda_build": cuda_build,
            "power_sampling_method": (
                f"GPU power via {hw['sampler']['gpu_backend']} "
                f"(nvmlDeviceGetPowerUsage through ctypes when available, else "
                f"nvidia-smi --query-gpu=power.draw) sampled in a background thread every "
                f"{args.interval} s; nvidia-smi cross-checked from the main thread during the "
                "GPU workload phase; CPU package/DRAM power (RAPL) and wall power unavailable "
                "on this box"),
            "load_average_at_start": os.getloadavg(),
            "files": {"src/flyscale/energy.py": sha_head(REPO / "src/flyscale/energy.py"),
                      "scripts/phase_energy_demo.py": sha_head(Path(__file__)),
                      "scripts/cuda/sparse_prop.cu": sha_head(CUDA_SRC),
                      "weights.f32": wprov["file_sha256_head16"],
                      "stimulus_schedule.bin": sched_meta["sha256_head16"]},
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "instrumentation_verification": verification,
        "baseline_stability": baseline_stability,
        "settle_before_baseline": settle,
        "baseline_stability_note": (
            "the two idle windows are compared only to show the baseline can be audited; the "
            "post-run window is preceded by its own wait for the GPU's post-load decay "
            "(phases.settle_post / phases.idle_baseline_post), and the lower-median window is "
            "the one actually used for attribution"),
        "workload_execution": {
            "phases": phases,
            "note": ("per-phase execution detail: epoch counts and wall times, the exact "
                     "commands run, kernel seconds, per-epoch counter deltas (the GPU's "
                     "atomicAdd ordering makes epochs match only approximately), and the "
                     "independent nvidia-smi series taken during the GPU workload"),
        },
        "unavailable_metrics": metrics["unavailable"],
        "cross_track_comparison": comparison,
        "notes": [
            "The GPU is shared with another process holding ~22 GB (llama-server); its idle "
            "draw is included in every phase and the idle-baseline subtraction is what makes "
            "the workload attribution meaningful.",
            "Only GPU power is measurable here. Every per-spike and per-event joule figure is "
            "therefore a lower bound on the whole-machine figure.",
        ],
    }
    write_json(out_dir / "hardware_energy.json", hardware)
    write_json(out_dir / "biological_model.json", bio)

    # ---- human summary ----------------------------------------------------------------
    print("-" * 88)
    print(f"idle baseline: " + ", ".join(
        f"{d}={v['mean_w']:.1f} W" for d, v in
        next(p["devices"] for p in hw["phases"] if p["label"] == "idle_baseline").items()))
    for name, entry in metrics["per_phase"].items():
        print(f"{name}: wall {entry['wall_seconds']:.1f}s  bio {entry['bio_seconds']:.2f}s  "
              f"spikes {entry['spikes_total']}  events {entry['synaptic_events']}")
        if "gpu_mean_w" in entry:
            print(f"    gpu: mean {entry['gpu_mean_w']:.1f} W  max {entry['gpu_max_w']:.1f} W  "
                  f"attributable {entry.get('gpu_joules_above_idle_baseline'):.1f} J"
                  if entry.get("gpu_joules_above_idle_baseline") is not None else
                  f"    gpu: mean {entry['gpu_mean_w']:.1f} W")
        for k in ("joules_per_bio_second_gpu_device", "joules_per_spike_gpu_device_only",
                  "joules_per_synaptic_event_gpu_device_only", "wall_seconds_per_bio_second"):
            if k in entry:
                print(f"    {k} = {entry[k]:.6g}")
    print(f"biology ({bio['anchor_key']}): {bio['per_neuron_watts']*1e12:.3f} pW/neuron, "
          f"P_bio({FLYWIRE_V783_NEURONS}) = {bio['at_flywire_scale']['p_bio_nw']:.1f} nW")
    print(f"wrote {out_dir/'hardware_energy.json'}")
    print(f"wrote {out_dir/'biological_model.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
