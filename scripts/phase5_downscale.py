"""Phase 5 + M6/M7: geometric renormalization DOWN (0.5x / 0.25x / 0.1x) and dynamic validation.

    python scripts/phase5_downscale.py [--factors 0.5 0.25 0.1] [--geometry auto]
                                      [--no-dynamics] [--force-signature]

§11: "Before scaling upward, demonstrate that scaling downward works." Two validations:
  * structural - graph statistics of each replica against the biological source graph
    (closure.compare, with the partition correspondence so community and rich-club metrics work)
  * dynamical   - identical normalized stimuli through G1 and each replica (a deterministic
    threshold cascade, plus the LIF engine when it exists), compared on propagation curves,
    latency, peak activity and output-set overlap.
Outputs: results/phase5/downscale.json, results/phase5/dynamics.json, replicas under
results/phase5/replicas/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import (PHASE5, fit_summary, get_signature, load_geometry, load_target,
                     strip_arrays, write_json)
from flyscale import closure, propagation, renorm, synthetic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="*", type=float, default=[0.5, 0.25, 0.1])
    ap.add_argument("--geometry", default="auto",
                    choices=["auto", "anatomical", "hyperbolic", "spectral"])
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dynamics", action="store_true")
    ap.add_argument("--force-signature", action="store_true")
    ap.add_argument("--save", action="store_true", default=True)
    ap.add_argument("--out", default=str(PHASE5 / "downscale.json"),
                    help="write the structural results here (lets geometries be compared "
                         "side by side instead of overwriting each other)")
    ap.add_argument("--dynamics-out", default=str(PHASE5 / "dynamics.json"))
    args = ap.parse_args()

    t_start = time.time()
    g1 = load_target(args.threshold)
    print("reference:", json.dumps(g1.summary()))
    law = load_geometry(g1, args.geometry, threshold=args.threshold, seed=args.seed)
    print("geometry:", json.dumps(fit_summary(law)))

    tag = f"v783thr{args.threshold}_{law.source.replace(':', '_').replace('/', '_')}"
    sig = get_signature(g1, tag, seed=args.seed, force=args.force_signature)
    print("reference signature ready (cached:", sig.get("_from_cache", False), ")")

    results: dict = {
        "phase": "5 + M6/M7 (downscaling and dynamic validation)",
        "reference": g1.summary(),
        "threshold": args.threshold,
        "geometry": fit_summary(law),
        "seed": args.seed,
        "factors": {},
        "timings": {},
    }
    write_json(Path(args.out), results)

    replicas = {}
    for factor in args.factors:
        t0 = time.time()
        rep, groups = renorm.coarse_grain(g1, law, factor, seed=args.seed, return_groups=True)
        t_cg = time.time() - t0
        print(f"coarse {factor}: {json.dumps(rep.summary())}  ({t_cg:.1f}s)")
        cmp = closure.compare(g1, rep, reference_groups=groups, reference_signature=sig,
                             seed=args.seed)
        results["factors"][str(factor)] = {
            "replica": rep.summary(),
            "replica_provenance": strip_arrays(rep.provenance),
            "closure": cmp,
            "seconds_coarse_grain": round(t_cg, 1),
        }
        replicas[str(factor)] = (rep, groups)
        results["timings"][f"coarse_{factor}"] = round(time.time() - t0, 1)
        write_json(Path(args.out), results)
        if args.save:
            synthetic.save_graph(rep, PHASE5 / "replicas" / f"g{factor}")
            np.save(PHASE5 / "replicas" / f"g{factor}" / "groups.npy", groups)

    if not args.no_dynamics:
        dyn: dict = {"model": "deterministic threshold cascade (flyscale.propagation)",
                     "stimulus": "1% of neurons, sensory super_classes preferred, "
                                 "identical relative threshold across scales",
                     "reference": None, "replicas": {}}
        seeds1 = propagation.sensory_seeds(g1, fraction=0.01, seed=args.seed)
        base = propagation.cascade(g1, seeds1, steps=12, relative_threshold=1.0)
        dyn["reference"] = strip_arrays(base)
        print("dynamics reference:", base["final_active_fraction"], base["latency_steps_to_half"])
        for factor, (rep, groups) in replicas.items():
            rep_seeds = np.unique(groups[seeds1])       # the same biological stimulus
            r = propagation.cascade(rep, rep_seeds, steps=12, relative_threshold=1.0)
            cmp = propagation.compare_dynamics(base, r, mapping=groups)
            dyn["replicas"][factor] = {"cascade": strip_arrays(r), "comparison": cmp}
            print(f"dynamics {factor}: final {r['final_active_fraction']} "
                  f"latency {r['latency_steps_to_half']} jaccard {cmp['active_set_jaccard']}")
        write_json(Path(args.dynamics_out), dyn)

    results["total_seconds"] = round(time.time() - t_start, 1)
    write_json(Path(args.out), results)
    print(f"\nphase 5 done in {results['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
