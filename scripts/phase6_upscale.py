"""Phase 6 + M8/M10: inverse geometric renormalization UP (2x / 5x / 10x, optional 25x-100x).

    python scripts/phase6_upscale.py [--factors 2 5 10] [--geometry auto] [--save]

§12: node subdivision in the latent geometry, never neuron/edge duplication. What is measured
and reported for every scale:
  * count scaling - N, connections and synapses against the source, plus the fitted exponent
    alpha in X ~ N^alpha, the doc's requirement that sparsity is preserved (E proportional to N,
    not N^2)
  * invariant preservation - mean degree and mean connection strength against the source
  * closure distance - closure.compare with the lineage correspondence (replica node -> parent)
Outputs: results/phase6/upscale.json and, with --save, replicas under results/phase6/replicas/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import (PHASE6, fit_summary, get_signature, load_geometry, load_target,
                     strip_arrays, write_json)
from flyscale import closure, renorm, synthetic


def fit_exponent(ns: list[float], xs: list[float]) -> dict:
    """Fit X ~ N^alpha on log-log axes; returns alpha and R^2 for the fit."""
    n = np.asarray(ns, dtype=float)
    x = np.asarray(xs, dtype=float)
    ok = (n > 0) & (x > 0)
    if ok.sum() < 2:
        return {"alpha": None, "r2": None, "n_points": int(ok.sum())}
    ln_n, ln_x = np.log(n[ok]), np.log(x[ok])
    slope, intercept = np.polyfit(ln_n, ln_x, 1)
    pred = slope * ln_n + intercept
    ss_res = float(((ln_x - pred) ** 2).sum())
    ss_tot = float(((ln_x - ln_x.mean()) ** 2).sum())
    return {"alpha": round(float(slope), 6), "intercept": round(float(intercept), 6),
            "r2": round(1 - ss_res / ss_tot, 6) if ss_tot > 0 else None,
            "n_points": int(ok.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="*", type=float, default=[2.0, 5.0, 10.0])
    ap.add_argument("--geometry", default="auto",
                    choices=["auto", "anatomical", "hyperbolic", "spectral"])
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rewire-fraction", type=float, default=0.1)
    ap.add_argument("--sibling-prob-scale", type=float, default=0.1)
    ap.add_argument("--no-sibling-edges", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--force-signature", action="store_true")
    ap.add_argument("--out", default=str(PHASE6 / "upscale.json"),
                    help="write results here, so runs with different geometries can be "
                         "compared side by side instead of overwriting each other")
    args = ap.parse_args()

    t_start = time.time()
    g1 = load_target(args.threshold)
    law = load_geometry(g1, args.geometry, threshold=args.threshold, seed=args.seed)
    print("reference:", json.dumps(g1.summary()))
    print("geometry:", json.dumps(fit_summary(law)))
    tag = f"v783thr{args.threshold}_{law.source.replace(':', '_').replace('/', '_')}"
    sig = get_signature(g1, tag, seed=args.seed, force=args.force_signature)

    results: dict = {
        "phase": "6 + M8/M10 (inverse renormalization / upscaling)",
        "reference": g1.summary(),
        "threshold": args.threshold,
        "geometry": fit_summary(law),
        "seed": args.seed,
        "node_subdivision": {
            "policy": "each neuron becomes `factor` children placed near the parent in the "
                      "latent geometry; a child inherits the parent's partner set with the "
                      "parent's synapse weights and the concrete target child is drawn by the "
                      "geometric connection law; sibling edges added by the law at a damped rate",
            "rewire_fraction": args.rewire_fraction,
            "sibling_edge_prob_scale": args.sibling_prob_scale,
            "sibling_edges": not args.no_sibling_edges,
        },
        "scales": {},
        "timings": {},
    }
    write_json(Path(args.out), results)

    ns, conns, syns = [g1.n], [int(g1.pre.size)], [int(g1.syn.sum())]
    for factor in args.factors:
        t0 = time.time()
        rep = renorm.upscale(g1, law, factor, seed=args.seed,
                             rewire_fraction=args.rewire_fraction,
                             sibling_edges=not args.no_sibling_edges,
                             sibling_prob_scale=args.sibling_prob_scale)
        t_up = time.time() - t0
        cmp = closure.compare(g1, rep, correspondence=rep.parent_index, reference_signature=sig,
                              seed=args.seed)
        entry = {
            "replica": rep.summary(),
            "replica_provenance": strip_arrays(rep.provenance),
            "n_ratio": round(rep.n / g1.n, 6),
            "connection_ratio": round(rep.pre.size / g1.pre.size, 6),
            "synapse_ratio": round(float(rep.syn.sum()) / float(g1.syn.sum()), 6),
            "mean_degree_reference": round(float(g1.pre.size / g1.n), 6),
            "mean_degree_replica": round(float(rep.pre.size / rep.n), 6),
            "mean_strength_reference": round(float(g1.syn.mean()), 6),
            "mean_strength_replica": round(float(rep.syn.mean()), 6),
            "closure": cmp,
            "seconds_upscale": round(t_up, 1),
        }
        results["scales"][str(factor)] = entry
        results["timings"][f"upscale_{factor}"] = round(t_up, 1)
        ns.append(rep.n)
        conns.append(int(rep.pre.size))
        syns.append(int(rep.syn.sum()))
        print(f"upscale x{factor}: n={rep.n} E={rep.pre.size} syn={int(rep.syn.sum())} "
              f"mean_deg={entry['mean_degree_replica']} ({t_up:.1f}s) "
              f"composite={cmp['composite']}")
        if args.save:
            synthetic.save_graph(rep, PHASE6 / "replicas" / f"g{factor}")
        write_json(Path(args.out), results)

    results["scaling_exponents"] = {
        "connections_vs_neurons": fit_exponent(ns, conns),
        "synapses_vs_neurons": fit_exponent(ns, syns),
        "note": "alpha ~ 1 means sparsity is preserved (E proportional to N); alpha ~ 2 would "
                "mean the enlargement drifted into a dense all-to-all regime",
    }
    results["total_seconds"] = round(time.time() - t_start, 1)
    write_json(Path(args.out), results)
    print("\nexponents:", json.dumps(results["scaling_exponents"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
