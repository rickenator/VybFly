# FlyScale status against PROJECT-VYBFLY.md

Living tracker: each phase and milestone of the scope document mapped to what exists, how it is
verified, and what is still open. "Verified" always means a real run whose artifact path is
listed; anything unverified is stated as such.

## Environment

* Canonical dataset: FlyWire FAFB v783, 139,255 neurons, 15,091,983 directed pairs (threshold 1),
  54,492,922 synapses, sources md5-verified against Zenodo (see `data/raw/MANIFEST.json`).
* Connectivity convention for every published comparison: **>= 5 synapses per connection**
  (2,700,513 connections), see README.md.
* Reference implementation: Python package `flyscale` (the oracle). Production target: Vyb.

## Milestones

| # | Milestone | State |
|---|---|---|
| M0 | Reproduce FlyWire graph statistics | **done** - `results/phase0/`, gate 11/11 required checks |
| M1 | CPU neural baseline (LIF whole connectome) | **done** - 13/13 tests, dense vs sparse delta 0 spikes (rho=1.0), 1.0 s sim in 120 s wall |
| M2 | Vyb DES engine | **PASS** - `results/phase2/vyb_des_check.json` verdict PASS, spike-for-spike vs reference |
| M3 | CUDA DES | **PASS** - CPU~GPU equivalence on the 512-neuron fixture: 5 spikes = reference, per-step identical, per-neuron counts identical, membrane trace identical, 0 launch errors. Required the [#270](https://github.com/rickenator/Vyb/issues/270) fix (PR #273) *and* a double-launch bug of mine in the runner; both kernels also proven individually. Whole-connectome GPU scale not yet attempted: `results/phase3/SUMMARY.txt` |
| M4 | Energy instrumentation | **done** - GPU 151.6 W mean, 6799.7 J above idle, 161.9 J/bio-s, 53.1 uJ/spike, 581 nJ/synaptic event (GPU lower bound; RAPL unreadable) |
| M5 | Latent geometry | **done** - spectral_32 AUC 0.9519 > degree-only 0.8739 > hyperbolic_2d 0.8646 > anatomical_xyz 0.8514; hyperbolic-vs-degree-baseline unresolved (run spread 0.037) |
| M6 | Downscaling 0.5x/0.25x replicas | **done** - 0.5x composite 0.179, 0.25x 0.267, 0.1x 0.276 |
| M7 | Renormalization validation (downscaled dynamics) | **done** - jaccard 0.909/0.950/0.853, final-fraction ratio 0.993/1.092/1.007, `results/phase5/dynamics.json` |
| M8 | 2x inverse scaling | **done** - N=278,510, E=5.19M (1.92x), synapses 2.00x, mean deg 18.62, strength 13.19 |
| M9 | Closure: R(G2) ~ G1 | **done - strong** - composite 0.0184, normalized degree Wasserstein 0.0074, community ARI 0.912 (`results/phase7/closure.json`) |
| M10 | 10x graph (~1.4M neurons) | **done** - N=1,392,550, E=27.84M, synapses 346.9M, mean deg 19.99, strength 12.46; 5x closure 0.236 |
| M11 | Plasticity (associative learning) | **done** - MB subgraph (5177 KC/96 MBON/331 DAN/56 glomeruli); AUC 0.51->1.00, 58 trials to 0.9, capacity 8; geometry gives +0.000 vs shuffled (real finding), the DAN gate does matter |
| M12 | Scaling benchmarks (capability vs size) | **done** - 4 scales, identical protocol fingerprint; PR ~ N^0.318, MI@delta=0.1 alpha=1.014, capacity 8->96 (censored); temporal/sequence depth 0 at every scale |
| M13 | Energy curves (power vs size) | **done** - ~438/500/946/2022 J for 1x/2x/5x/10x (~5% run-to-run; see `results/energy/CURVES-NOTE.md`); J per spike falls 8.9->4.2 uJ |
| M14 | Capability/watt | **done (lower bound)** - 0.070/0.615/0.612/0.526 capacity-units per GPU watt; numerator censored at the 96-class grid from 2x up |
| M15 | 100x attempt (~14M neurons) | not attempted; the doc makes it conditional on prior results |

## Phases

* **Phase 0 - source data and baseline** (§6). Done. Canonical dataset, published-value gate,
  threshold-robustness sweep, provenance in `data/processed/canonical_v783/meta.json`.
* **Phase 1 - executable biological baseline** (§7). LIF with refractory state, synaptic
  delays, excitatory/inhibitory signs from neurotransmitter, synapse-count weights; dense
  timestep reference plus sparse event engine, mutually validated.
* **Phase 2 - Vyb discrete-event runtime** (§8). Generic DES in Vyb (bucketed scheduler,
  event sources/sinks, recorder, replication) validated spike-for-spike against an independent
  Python reference on a 640-neuron fixture.
* **Phase 3 - event-driven GPU backend** (§9). GPU sparse event propagation with the CPU
  reference as the comparison target; dense timestep retained only as validation.
* **Phase 4 - latent geometry** (§10). Anatomical 3-D vs 2-D hyperbolic vs spectral
  embeddings scored by held-out connection prediction, against a degree-only baseline, with a
  fitted geometric connection law P(connect | distance).
* **Phase 5 - geometric renormalization baseline** (§11). Downscaling by latent-geometry
  agglomeration with a sparsity-preserving edge threshold, validated structurally and
  dynamically (identical normalized stimuli through G1, G0.5, G0.25).
* **Phase 6 - inverse geometric renormalization** (§12). Node subdivision in the latent
  geometry; children inherit the parent's partner set with the parent's synapse weights and the
  target child is drawn by the geometric law, so counts scale linearly with N and mean degree
  and mean connection strength stay bounded. Sibling (intra-group) edges added by the law at a
  damped rate.
* **Phase 7 - renormalization closure test** (§13). Lineage coarse-graining of each generated
  scale back to the parent neurons, compared with G1 across degree/weighted-degree Wasserstein
  distance, triad-census divergence, spectral distance, rich-club distance, connectivity-matrix
  divergence, community agreement and latent-distance distribution; composite score for
  internal comparison only, raw metrics preserved.
* **Phase 8 - plasticity and learning** (§8/§14). Mushroom-body associative benchmark with a
  dopamine-gated rule, shuffled-connectivity and no-plasticity controls, and an explicit
  measurement of how much performance comes from the rule versus the architecture.
* **Phase 9 - capability scaling tests** (§9/§15). Discrimination, memory capacity, temporal
  depth, sequence learning, generalization, robustness, dynamical complexity, each measured
  under one normalized protocol across scales, with power-law fits and honest reporting of
  whatever fails to scale.
* **Energy** (§17-§19). Two separate tracks, never conflated: biological-equivalent
  `P_bio = P0 * N/N0` anchored to cited biology, and measured hardware joules per simulated
  biological second / spike / synaptic event.

## Infrastructure built for the scaling phases (verified)

* `src/flyscale/renorm.py` - geometric coarse-graining (latent nearest-neighbor grouping plus a
  sparsity-preserving edge-threshold calibration that reports the unreachable degree range
  instead of silently missing the target), node subdivision for 2x-100x, lineage coarse-graining
  for closure, and `fit_connection_law` (logistic fit of P(connect|distance) -> R, T, AUC).
* `src/flyscale/closure.py` - the §13 metric suite with both correspondence styles (lineage and
  partition membership), composite score kept separate from the raw metrics.
* `src/flyscale/propagation.py` - normalized-stimulus cascade for the §11 dynamic validation.
* `tests/test_renorm.py` **passes**: 2x and 5x subdivision scale E linearly with N
  (45,284 -> 91,366 -> 234,386 edges), mean degree and mean connection strength stay bounded, and
  lineage closure of a 2x graph back to G1 is exact (composite 0.0, ARI 1.0).
* `scripts/assess_success_criteria.py` - assembles the §18 power-scaling table and the §25
  minimum/strong/exceptional verdicts from the phase artifacts; currently reports everything
  missing, which is correct until the runs land.

## Vyb defects found while doing this work (probes saved, not yet filed)

* Parenthesised operand after `*` inside an addition mis-types the multiplication's left operand
  as a pointer type - reproduces outside kernel mode (`src/vyb_kernels/probes/`).
* Device code: `tid_x`/`blk_x`/`dim_x`/`ld_i32` return `CInt`, which does not implicitly convert
  to `Int` in an assignment - every such site needs an explicit `as Int`.
* Earlier: raw io byte buffer corrupts across a module boundary, optional-`Vec` return rejected,
  `Vec` parameters not mutated for the caller, and ~29 us/element decode throughput (the
  project-level blocker for whole-connectome Vyb simulation).

## Geometry-driven scaling: anatomical vs learned hyperbolic (measured)

The same downscaling pipeline, run against both embeddings, so the doc's latent-geometry
hypothesis is tested rather than assumed. Structural closure against G1 (no correspondence for
these, partition membership used):

| factor | anatomical composite / degW / ARI | learned hyperbolic composite / degW / ARI |
|---|---|---|
| 0.5x | 0.179 / 0.129 / 0.516 | 0.214 / 0.104 / 0.331 |
| 0.25x | 0.267 / 0.308 / 0.557 | 0.311 / 0.303 / 0.198 |
| 0.1x | 0.276 / 0.146 / 0.521 | 0.320 / 0.242 / 0.240 |

Propagation dynamics (identical normalized stimulus): anatomical Jaccard 0.909/0.950/0.853,
hyperbolic 0.916/0.821/0.902.

Honest reading: the learned hyperbolic embedding fits the connection law *better* on this
protocol (AUC 0.888 vs 0.803 for anatomical xyz) and preserves weighted degree slightly better at
0.5x (degW 0.104 vs 0.129), but it does **not** rescue the renormalization - community structure
is clearly worse (ARI 0.33 vs 0.52) and the composite is worse at every factor. The scaling
results are therefore reported as geometry-driven with the trade-off stated, not as evidence that
the hyperbolic geometry is the right latent space.

## Explicit non-goals (§26) - standing

No consciousness claims, no transformer mapping, no language capability, no replacing the
connectome with a generic SNN, no treating synapse count as a learned weight, and no claiming
biological power efficiency from GPU measurements.
