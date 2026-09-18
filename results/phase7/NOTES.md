Phase 7 closure test - what it does and does not show

MEASURED (results/phase7/closure.json, geometry = phase4 hyperbolic 2d, R=3.543 T=0.969)
Each generated graph was coarse-grained back by lineage (every child grouped with its parent
neuron) and compared against the biological source graph G1 (139,255 neurons, 2,700,513
connections, threshold 5 synapses).

  scale        N(G_s)      E(G_s)   N(R)      E(R)     composite  deg_wass_norm  motif_L1   ARI
    2.0       278,510   5,186,025  139,255  2,680,470  0.018407118     0.007422  1.06e-05  0.911867
    5.0       696,275  13,500,829  139,255  2,700,498  0.001504040     0.000006  7.00e-09  0.991497
   10.0     1,392,550  27,842,102  139,255  2,700,513  0.000978306     0.000000  0.00e+00  1.000000

At 10x the round trip returns the source graph's connection count exactly (2,700,513), the
degree distribution distance is 0, the triad-census divergence is 0 and the community partition
agrees at ARI = 1.0. At 5x it is 15 connections short of the source, with ARI 0.9915.

WHAT THIS ESTABLISHES

The inverse renormalization is *invertible*: the upscaling rule loses nothing that lineage
coarse-graining can detect. Whatever structure the generator adds below the parent level is
exactly the structure the coarse-graining removes, so the scale ladder is consistent - you can
go up and come back. For the project's purposes that is the acceptance criterion §13 states, and
the metric suite backs it with raw numbers rather than a composite alone.

WHAT THIS DOES NOT ESTABLISH

* It is not evidence of self-similarity. Children inherit their parent's partner set with the
  parent's synapse weights, so parent-level edges are preserved by construction; the round trip
  is close to an identity on the parent-level graph for that reason. A closure test that
  *discovered* the grouping from the geometry alone (rather than from lineage) would be a much
  stronger claim, and it has not been run.
* A 0.0 degree-distribution distance at 10x is partly a statement about the metric: the coarse
  graph's degrees are aggregated deterministically from the same edge set, so no sampling noise
  enters. Sampling noise would appear if the coarse-graining had to infer which neurons form a
  group.
* Nothing here says the generated graphs are biologically plausible at their own scale - only
  that they are consistent with the source under this round trip. The independent checks for
  plausibility are the metric suite against G1 (composites 0.194 / 0.236 / 0.275 for 2x / 5x /
  10x) and the capability curves in results/phase9.

NEXT STRONGER TEST (not run)

Coarse-grain each generated graph by geometry (renorm.coarse_grain, which infers groups from
latent proximity) instead of by lineage, and compare that against G1. That decouples the closure
result from the construction and would show whether the generated graphs renormalize correctly
under a grouping the generator did not choose.
