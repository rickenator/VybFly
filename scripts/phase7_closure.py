"""Phase 7 + M9: the renormalization closure test  R(G_s) ~ G1.

    python scripts/phase7_closure.py [--factors 2 5 10] [--geometry auto] [--from-saved]

§13 is the primary structural acceptance criterion: coarse-graining a generated scale back
down must recover the biological source graph. This script:

  1. takes each enlarged graph from Phase 6 (rebuilt, or reloaded with --from-saved),
  2. applies the renormalization operator: lineage coarse-graining back to the parent neurons,
  3. measures R(G_s) against G1 with the full metric suite of closure.compare - degree and
     weighted-degree Wasserstein distance, triad-census divergence, spectral distance,
     rich-club distance, connectivity-matrix divergence, community agreement through the
     correspondence, latent-distance distribution,
  4. runs the same normalized stimulus through both and compares the dynamics,
  5. writes the composite score per scale, with the raw metrics preserved alongside it.
Outputs: results/phase7/closure.json, results/phase7/SUMMARY.txt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import (PHASE6, PHASE7, fit_summary, get_signature, load_geometry, load_target,
                     strip_arrays, write_json)
from flyscale import closure, propagation, renorm, synthetic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="*", type=float, default=[2.0, 5.0, 10.0])
    ap.add_argument("--geometry", default="auto",
                    choices=["auto", "anatomical", "hyperbolic", "spectral"])
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-saved", action="store_true",
                    help="reload replicas from results/phase6/replicas instead of rebuilding")
    ap.add_argument("--rewire-fraction", type=float, default=0.1)
    ap.add_argument("--no-dynamics", action="store_true")
    ap.add_argument("--force-signature", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    g1 = load_target(args.threshold)
    law = load_geometry(g1, args.geometry, threshold=args.threshold, seed=args.seed)
    tag = f"v783thr{args.threshold}_{law.source.replace(':', '_').replace('/', '_')}"
    sig = get_signature(g1, tag, seed=args.seed, force=args.force_signature)

    # Merge with any scales already measured instead of resetting the file: a later run for one
    # factor must not erase the factors an earlier run established.
    prior_scales, prior_timings = {}, {}
    if (PHASE7 / "closure.json").exists():
        try:
            prior = json.loads((PHASE7 / "closure.json").read_text())
            prior_scales = prior.get("scales") or {}
            prior_timings = prior.get("timings") or {}
        except Exception:
            pass
    out: dict = {
        "phase": "7 + M9 (renormalization closure test)",
        "criterion": "R(G_s) approaches G1: coarse-graining a generated scale must recover the "
                     "biological source graph",
        "reference": g1.summary(),
        "geometry": fit_summary(law),
        "seed": args.seed,
        "scales": dict(prior_scales),
        "timings": dict(prior_timings),
        "prior_scales_carried_over": sorted(prior_scales),
    }
    write_json(PHASE7 / "closure.json", out)

    seeds1 = None if args.no_dynamics else propagation.sensory_seeds(g1, fraction=0.01,
                                                                    seed=args.seed)
    base_dyn = None if args.no_dynamics else propagation.cascade(g1, seeds1, steps=12,
                                                                 relative_threshold=1.0)

    for factor in args.factors:
        t0 = time.time()
        p = PHASE6 / "replicas" / f"g{factor}"
        if args.from_saved and not (p / "graph.npz").exists():
            # Phase 6 writes the replica directory with the float form of the factor (g2.0)
            for alt in (f"g{float(factor)}", f"g{float(factor):.1f}"):
                q = PHASE6 / "replicas" / alt
                if (q / "graph.npz").exists():
                    p = q
                    break
        if args.from_saved and (p / "graph.npz").exists():
            gs = synthetic.load_graph(p)
            source = f"loaded:{p}"
        else:
            gs = renorm.upscale(g1, law, factor, seed=args.seed,
                                rewire_fraction=args.rewire_fraction)
            source = "rebuilt"
        t_up = time.time() - t0

        t1 = time.time()
        rg = renorm.coarse_grain_by_lineage(gs)
        t_rg = time.time() - t1
        correspondence = np.arange(min(g1.n, rg.n))          # lineage nodes ARE the parents
        cmp = closure.compare(g1, rg, correspondence=correspondence, reference_signature=sig,
                              seed=args.seed)

        entry = {
            "source_graph": {"n": gs.n, "edges": int(gs.pre.size), "synapses": int(gs.syn.sum()),
                             "origin": source},
            "renormalised": {"n": rg.n, "edges": int(rg.pre.size),
                             "synapses": int(rg.syn.sum()),
                             "provenance": strip_arrays(rg.provenance)},
            "closure": cmp,
            "seconds_rebuild": round(t_up, 1),
            "seconds_renormalise": round(t_rg, 1),
        }
        if not args.no_dynamics:
            rep_seeds = np.unique(gs.parent_index[seeds1])
            dyn = propagation.cascade(gs, rep_seeds, steps=12, relative_threshold=1.0)
            entry["dynamics"] = {
                "cascade": strip_arrays(dyn),
                "comparison_vs_reference": propagation.compare_dynamics(
                    base_dyn, dyn, mapping=gs.parent_index),
            }
        out["scales"][str(factor)] = entry
        out["timings"][f"scale_{factor}"] = round(time.time() - t0, 1)
        print(f"scale x{factor}: R(G) n={rg.n} E={rg.pre.size} composite={cmp['composite']} "
              f"deg_wass_norm={cmp['degree_wasserstein_normalised']} ARI={cmp['community_ari']}")
        write_json(PHASE7 / "closure.json", out)

    # plain-text summary that can be quoted directly
    lines = ["Renormalization closure: R(G_s) vs the biological source graph (v783, thr 5)",
             f"geometry: {law.source} kind={law.kind} R={law.R:.4g} T={law.T:.4g}",
             "",
             f"{'scale':>6} {'N(G_s)':>12} {'E(G_s)':>13} {'N(R)':>8} {'E(R)':>10} "
             f"{'composite':>10} {'deg_wass_n':>11} {'motif_L1':>10} {'ARI':>7}"]
    for factor, e in out["scales"].items():
        c = e["closure"]
        lines.append(f"{factor:>6} {e['source_graph']['n']:>12,} {e['source_graph']['edges']:>13,} "
                     f"{e['renormalised']['n']:>8,} {e['renormalised']['edges']:>10,} "
                     f"{str(c['composite']):>10} {str(c['degree_wasserstein_normalised']):>11} "
                     f"{str(c['motif_divergence_L1']):>10} {str(c['community_ari']):>7}")
    lines += ["",
              "composite = weighted mean of the normalized distances (lower is closer to G1);",
              "raw metrics for every scale are kept in closure.json - the composite exists only",
              "for internal comparison between scales."]
    (PHASE7 / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    out["total_seconds"] = round(time.time() - t_start, 1)
    write_json(PHASE7 / "closure.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
