Phase 6 provenance note - unresolved, recorded so it is not silently trusted

TWO OBSERVATIONS

1. The geometry recorded in results/phase6/upscale.json is
   `phase4:spectral_16_coords.npy+law_fit` (kind euclidean, R=3.964e-04, T=2.115e-04,
   AUC 0.8945). An earlier read of the same file, taken while the run was in progress,
   reported `annotation_xyz+law_fit` (kind euclidean, R=1.748, T=0.756, AUC 0.8025). The
   scaling scripts resolve their geometry once at startup, so a single run cannot produce
   both; the file on disk now corresponds to the spectral_16 geometry.

2. The 2x entry is bit-identical between the two readings:
   N=278,510  E=5,186,025  synapses=68,428,042  mean_out_degree=18.620606
   composite=0.194123045  degree_wasserstein_normalised=0.04061
   Identical to every printed digit, even though the child placement, the sibling-edge law
   and the target-child sampling are all geometry-dependent, and the two geometries put
   children in different places. An exact match across a geometry change is not what the
   implementation predicts.

WHAT THIS MEANS

The 2x/5x/10x numbers in the file are internally consistent (they were written by one
completed run whose stdout is reproduced in the process log: upscales of 10.9 s / 25.6 s /
73.8 s, mean degrees 18.62 / 19.39 / 19.99), and the scaling exponents computed from them
(connections ~ N^1.0175, synapses ~ N^1.0067, both with R^2 > 0.999) stand on their own.
What is NOT established is which embedding produced them, and observation 2 suggests either
a stale entry carried into the file or a cache keyed more coarsely than the geometry.

HOW TO SETTLE IT (do not guess)

  cd ~/Projects/VybFly
  rm -rf results/phase6/replicas && rm -f results/phase6/upscale.json
  .venv/bin/python scripts/phase6_upscale.py --geometry hyperbolic --factors 2 5 10 \
      --save --out results/phase6/upscale_hyperbolic.json
  .venv/bin/python scripts/phase6_upscale.py --geometry anatomical --factors 2 5 10 \
      --save --out results/phase6/upscale_anatomical.json

Comparing the two outputs settles both questions at once: whether the geometry changes the
replica at all, and which geometry the delivered numbers belong to. The `--out` flag keeps
the two runs from overwriting each other.

Related, and already fixed: scripts/phase7_closure.py used to reset its output file at
startup, so starting a run for one factor erased the factors a previous run established. It
now merges with any scales already on disk. The 2x closure result (composite 0.018407118,
normalized degree Wasserstein 0.007422, motif L1 1.0638e-05, community ARI 0.911867) was
measured before that overwrite and is reproduced in the report and in docs/ROADMAP.md.
