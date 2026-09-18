"""Phase 3 gate: CPU DES vs GPU DES (PROJECT-VYBFLY.md §9 "CPU DES ~ GPU DES ~ timestep reference").

    python scripts/phase3_gpu_check.py

Rebuilds the same network the Vyb GPU runner builds (src/vyb_kernels/phase3_gpu.vyb) and
simulates the same event-bucket semantics in numpy, then compares spike trains, per-neuron
counts and membrane state against the runner's printed output.

Semantics being reproduced (must match the kernels exactly):
  accumulate: for each fired neuron in the bucket, for each outgoing edge, accum[target] += g*w
  update:     v_new = v_rest + leak*(v - v_rest) + accum
              fire when t >= refractory_until and v_new >= v_thresh -> reset, set refractory,
              append the neuron id to the next bucket
              then v = v_new and accum = 0
The GPU fills the next bucket by atomic slot reservation, so the ORDER inside a bucket is not
deterministic; the bucket's SET is. This checker therefore compares per-neuron spike counts and
the total spike count, and separately reports whether any neuron's membrane potential sits within
a hair of the threshold (where float addition order could matter).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build_network(n: int = 512):
    offs = (1, 3, 7)
    pre, post, w = [], [], []
    for i in range(n):
        for d in offs:
            pre.append(i)
            post.append((i + d) % n)
            w.append(-(3 + d) if i % 5 == 0 else (3 + d))
    return np.array(pre, np.int64), np.array(post, np.int64), np.array(w, np.float64)


def simulate(n: int = 512, ticks: int = 30, n_seed: int = 10, g: float = 0.02,
             v_thresh: float = 0.3, leak: float = 0.9, refractory: int = 2,
             v_rest: float = 0.0, v_reset: float = 0.0):
    pre, post, w = build_network(n)
    order = np.argsort(pre, kind="stable")
    pre_s, post_s, w_s = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, pre_s + 1, 1)
    np.cumsum(indptr, out=indptr)

    v = np.full(n, v_rest, dtype=np.float64)
    refr = np.full(n, 0, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    per_step = []
    bucket = np.arange(n_seed, dtype=np.int64)
    for t in range(ticks):
        accum = np.zeros(n, dtype=np.float64)
        if bucket.size:
            src = np.concatenate([np.arange(indptr[s], indptr[s + 1]) for s in bucket]) \
                if bucket.size < 50000 else None
            # vectorised equivalent of "for each fired neuron, walk its edges"
            rep = np.repeat(bucket, np.diff(indptr)[bucket])
            counts_per = np.diff(indptr)[bucket]
            starts = indptr[bucket]
            within = np.arange(rep.size, dtype=np.int64) - np.repeat(
                np.cumsum(counts_per) - counts_per, counts_per)
            edge_ids = np.repeat(starts, counts_per) + within
            np.add.at(accum, post_s[edge_ids], g * w_s[edge_ids])
        dv = v - v_rest
        ldv = leak * dv
        v_new = v_rest + ldv + accum
        eligible = t >= refr
        fires = eligible & (v_new >= v_thresh)
        fired_ids = np.flatnonzero(fires)
        counts[fired_ids] += 1
        v_new[fired_ids] = v_reset
        refr[fired_ids] = t + refractory
        v = v_new
        per_step.append(int(fired_ids.size))
        bucket = fired_ids
    return {
        "n": n, "edges": int(pre.size), "ticks": ticks,
        "spikes_total": int(counts.sum()),
        "spikes_per_step": per_step,
        "spike_counts": counts.tolist(),
        "v_milli": [int(round(float(x) * 1000.0)) for x in v],
        "v": v.tolist(),
        "v_thresh": v_thresh,
        "min_distance_to_threshold": float(np.min(np.abs(v - v_thresh))),
    }


def parse_vyb_output(path: Path) -> dict:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith(" "):
            k, _, val = line.partition("=")
            out[k.strip()] = val.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-output", default=str(ROOT / "results" / "phase3" / "gpu_run.txt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase3" / "gpu_equivalence.json"))
    args = ap.parse_args()

    gpu_path = Path(args.gpu_output)
    if not gpu_path.exists():
        raise SystemExit(f"missing {gpu_path}; run the Vyb GPU runner first")
    gpu = parse_vyb_output(gpu_path)
    ref = simulate()

    result: dict = {
        "gate": "CPU DES vs GPU DES (event-bucket LIF pipeline)",
        "gpu_artifacts": {
            "runner": "src/vyb_kernels/phase3_gpu.vyb",
            "kernel": "src/vyb_kernels/spike_bucket.vyb",
            "ptx": "results/phase3/spike_bucket.ptx",
            "output": str(gpu_path.relative_to(ROOT)),
        },
        "semantics": "identical network, stimulus, parameters and bucket update order rules; "
                     "bucket membership is a set (GPU uses atomic slot reservation)",
        "cpu_reference": {k: v for k, v in ref.items() if k != "v"},
        "checks": {},
    }

    def cmp_int(key: str, cpu_val) -> dict:
        got = gpu.get(key)
        ok = got is not None and str(got).strip() == str(cpu_val)
        result["checks"][key] = {"gpu": got, "cpu": cpu_val, "match": bool(ok)}
        return result["checks"][key]

    cmp_int("gpu_spikes_total", ref["spikes_total"])
    cmp_int("gpu_edges", ref["edges"])
    cmp_int("gpu_n", ref["n"])

    # per-step and per-neuron comparison
    gpu_steps = [int(x) for x in gpu.get("gpu_spikes_per_step", "").split(",") if x.strip()]
    gpu_counts = [int(x) for x in gpu.get("gpu_spike_counts", "").split(",") if x.strip()]
    result["checks"]["spikes_per_step"] = {
        "gpu": gpu_steps, "cpu": ref["spikes_per_step"],
        "match": gpu_steps == ref["spikes_per_step"],
    }
    if len(gpu_counts) == ref["n"]:
        mism = [i for i, (a, b) in enumerate(zip(gpu_counts, ref["spike_counts"])) if a != b]
        result["checks"]["per_neuron_spike_counts"] = {
            "n_compared": ref["n"], "n_mismatched": len(mism), "first_mismatches": mism[:20],
            "match": not mism,
        }
    else:
        result["checks"]["per_neuron_spike_counts"] = {
            "error": f"gpu reported {len(gpu_counts)} counts, expected {ref['n']}"}

    gpu_v = gpu.get("gpu_v_probe_milli_first8")
    if gpu_v:
        vals = [int(x) for x in gpu_v.split(",")]
        cpu_first8 = ref["v_milli"][:8]
        result["checks"]["membrane_probe_first8_milli"] = {
            "gpu": vals, "cpu": cpu_first8, "match": vals == cpu_first8}

    required = ["gpu_spikes_total", "gpu_edges", "gpu_n", "spikes_per_step",
                "per_neuron_spike_counts"]
    mismatch = [k for k in required if not result["checks"].get(k, {}).get("match")]
    result["passed"] = not mismatch
    result["mismatched_checks"] = mismatch
    result["float_order_note"] = (
        f"CPU reference min |v - threshold| = {ref['min_distance_to_threshold']:.6g}; a value "
        "much larger than the accumulated rounding error means atomic ordering cannot change "
        "the fired set")
    result["launch_errors_reported_by_gpu"] = gpu.get("gpu_launch_errors")

    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v.get("match", v.get("error", "?"))
                      for k, v in result["checks"].items()}, indent=2))
    print(f"passed: {result['passed']}  mismatches: {mismatch}")
    print("wrote", args.out)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
