# FlyScale

### 📄 **[Read the paper (PDF) →](paper.pdf)**

A map of a whole fruit-fly brain, redrawn bigger — and an honest test of whether "bigger" still
behaves like a brain.

---

## What is this

A fruit fly has about **139,000 neurons** and roughly **500 million synapses** — the tiny
connection points where one neuron talks to another. In 2024 a large collaboration called
**FlyWire** mapped *every one* of those neurons and connections in an adult fly brain. It is the
first complete wiring diagram of a brain that anyone has ever had.

This project takes that map and asks a simple question:

> **Can you make the map bigger and have it still work?**

We grow the brain's wiring from 1× to **2×, 5× and 10×** — up to **1.4 million neurons and 27.8
million connections** — then check, carefully, whether the enlarged versions still look and behave
like the original: the same kind of wiring, the same way a signal spreads through it, the same
abilities, and what it costs in energy.

The interesting part is not that we could. It is *where it worked, where it didn't, and what that
tells you* — including one result that goes against the obvious guess.

## Why "geometric"?

Instead of rewiring things at random, we first learn a **hidden geometry** for the brain: a
coordinate system in which neurons that are "close" are likely to be connected. Adding neurons
then means placing new points near old ones in that geometry and wiring them by the learned rule.
This is what keeps connections growing in step with neurons, instead of letting the brain become an
all-to-all tangle.

## The headline results

**It works, structurally.** The 10× brain keeps the same sparsity as the real one — connections grow
as N<sup>1.02</sup>, essentially in step with neuron count rather than N² — mean degree stays flat at
~19 partners per neuron, and connection strength stays ~12.6 synapses. Take the 10× brain and shrink
it back the way it was grown, and you recover **exactly** the original number of connections, with a
degree-distribution difference of 0 and community structure agreeing perfectly (ARI 1.0).

| scale | neurons | connections | synapses | mean partners |
|---|---|---|---|---|
| 1× (real brain) | 139,255 | 2,700,513 | 34,153,566 | 19.4 |
| 2× | 278,510 | 5,186,025 | 68,428,042 | 18.6 |
| 5× | 696,275 | 13,500,829 | 171,970,190 | 19.4 |
| 10× | 1,392,550 | 27,842,102 | 346,934,550 | 20.0 |

**Smaller brains still behave like the big one.** Compress the brain to half, a quarter, a tenth of
its neurons, send the same signal through, and the pattern of activity overlaps the original by
85–95% — the replicas reproduce the *behavior*, not just the shape.

**Bigger brains get more able — up to a point.** Stimulus information rises in proportion to size,
and memory capacity goes from 8 associations to at least 96. But nothing we built does *time*: tests
about delayed or ordered stimuli never beat chance, at any scale. Some curves had to be marked
"censored" — our test grid ran out before the system did, so we cannot honestly say how far capacity
grows.

**A result that goes the other way.** We trained a model of the fly's learning center (the mushroom
body) on odor–reward association. It learns well — from chance to perfect in 58 trials. But replace
the brain's actual wiring with a randomly shuffled version that has the same statistics, and **it
learns exactly as well** (+0.000 ± 0.000). What mattered was the *anatomy of the dopamine gating*,
not the specific wiring. A negative result, and a useful one.

**The energy has two answers, and we never mix them.** Measured on the GPU: 438 → 2,022 joules across
the ladder, with cost per spike *falling* from 8.9 to 4.2 µJ (the device stops being idle-bound). By
biology's own accounting — calorimetry on real fly brains — the equivalent figure is linear in neuron
count *by construction*. Those two numbers answer different questions, so they are never added
together, and the paper says so on every page where they appear.

**The verdict**, against the success criteria written down before the work started:

| criterion | verdict |
|---|---|
| minimum: a 2× graph that preserves the statistics and renormalizes back | **supported** |
| strong: a 10× graph retaining the invariants and renormalizing back | **supported** |
| exceptional: capability growing *faster* than neuron count | **not supported** |

## Two implementations, on purpose

The project runs the same thing twice: a **Python reference** (the oracle every number is checked
against) and a **Vyb-native production path** — a loader, a discrete-event simulation engine, and GPU
kernels written in [Vyb](https://github.com/rickenator/Vyb).

Building the second one turned out to be the most informative part of the work: it surfaced **twelve
compiler defects**, including a blocking one where a kernel-mode store wrote eight bytes into a
four-byte slot and silently zeroed a neighbor. That one is
[filed, fixed upstream, and re-verified](https://github.com/rickenator/Vyb/issues/270) by this
project.

The GPU kernels now agree with the CPU reference tick for tick. Honest scope: that equivalence is
proven on a 512-neuron fixture, not yet on the whole 139k-neuron connectome.

## Jargon decoder

| term | what it means here |
|---|---|
| connectome | the full wiring map: which neuron connects to which, and how strongly |
| synapse / connection | a synapse is one contact point; a "connection" means 5+ synapses on the same neuron pair, the convention the published papers use |
| graph, node, edge | a network: neurons are nodes, connections are edges |
| geometric renormalization | rescaling a network by moving it in a hidden geometry, then checking you get the original back |
| coarse-graining (downscaling) | merging nearby neurons into one to get a smaller replica |
| subdivision (upscaling) | replacing one neuron with *c* nearby children to get a bigger graph |
| LIF model | "leaky integrate-and-fire", the simplest standard neuron model: voltage builds up, fires past a threshold, then resets |
| discrete-event simulation | simulate only the moments when something actually happens, instead of stepping through every millisecond |
| AUC | a 0.5–1.0 score for "can this tell connected pairs from unconnected ones?"; 0.5 is a coin flip |
| hyperbolic embedding | coordinates on a curved (Poincaré) disk, in which distance predicts connection probability better than raw 3-D position does |

## Run it yourself

```sh
cd ~/Projects/VybFly          # paths in this repo are written relative to your home directory
uv venv .venv --python 3.12
.venv/bin/python scripts/fetch_data.py      # downloads the FlyWire release, verifies md5
.venv/bin/python scripts/phase0_build.py    # builds the canonical graph
.venv/bin/python scripts/phase0_gate.py     # checks it against published values (11/11 required)
```

Then any phase — for example the scaling ladder and its closure test:

```sh
.venv/bin/python scripts/phase4_geometry.py                                   # fit the geometries
.venv/bin/python scripts/phase6_upscale.py     --geometry auto --factors 2 5 10 --save
.venv/bin/python scripts/phase7_closure.py     --from-saved --factors 2 5 10
.venv/bin/python scripts/phase9_capability.py  --stage all                    # capability battery
.venv/bin/python scripts/assess_success_criteria.py                           # the verdicts
```

The Vyb side needs the Vyb toolchain (`VYB` and `VYB_STDLIB` env vars; defaults assume
`~/Projects/Vyb`):

```sh
.venv/bin/python scripts/vyb_des_check.py       # Vyb discrete-event engine vs the Python reference
.venv/bin/python scripts/phase3_gpu_check.py    # Vyb GPU kernels vs the CPU reference
```

The paper is built from the artifacts rather than typed by hand:

```sh
.venv/bin/python scripts/paper_collect_data.py  # every number -> results/paper/data.json
.venv/bin/python scripts/paper_make_figures.py
.venv/bin/python scripts/paper_make_tables.py
pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex
```

## What's in here, and what isn't

```
PROJECT-VYBFLY.md   the scope document, written before the work started, including the non-goals
paper.tex paper.pdf the paper, plus the scripts that generate its figures and tables
src/flyscale/       the Python reference: dataset, metrics, renormalization, dynamics, capability
src/vyb/            Vyb-native loader and discrete-event engine
src/vyb_kernels/    Vyb GPU kernels (NVPTX) and the probes used to isolate compiler defects
scripts/            one runnable script per phase, plus data fetch and paper generation
results/            the artifacts every claim traces to (JSON, PTX/cubin, summaries, notes)
tests/              the checks that gate the work (canonical dataset, LIF engines, renormalization)
docs/               DATASET.md (sources, conventions, gate), VYB-PORT.md, ROADMAP.md
```

**Not in the repo, on purpose:** the FlyWire source data and every derived array (replica graphs,
feature matrices, large CSVs). They are hundreds of MB and fully regenerable — the fetch script
verifies the download by md5 and the phase scripts rebuild everything from it. Keeping them out
makes the repo small and the pipeline the source of truth. In the run commands recorded inside
`results/`, paths are written as `~/Projects/...`.

Dataset sources, the 5-synapse threshold convention and the full Phase 0 gate table:
**[docs/DATASET.md](docs/DATASET.md)**.

## What we do not claim

- **No consciousness, no "it's thinking."** This is a graph and a simulator, and the paper says so.
- **No efficiency claims from GPU numbers.** Comparing a GPU's joules to a brain's is exactly the
  category error we avoid; hardware and biological energy are separate tracks, never combined.
- **The closure test proves invertibility, not self-similarity.** Children inherit their parent's
  partners, so shrinking the 10× graph back is close to an identity *for that reason*. The stronger
  test — letting an inferred geometry, rather than the generator, decide what collapses — has not
  been run.
- **The geometry question is unresolved at the top.** A learned embedding beats raw anatomy, but it
  does not beat a degree-only baseline by more than its own run-to-run spread; and the geometry that
  *predicts* connections best is not the one that *renormalizes* best.
- **Capability curves are censored** at the test grid's ceiling from 2× upward, which is why the
  exceptional criterion is undecided rather than refuted.
- **The whole-connectome GPU claim is still fixture-scale** (512 neurons), as noted above.

More of these — including corrections we made to our own earlier claims — are in the paper's
limitations section and its honesty-log appendix.

## Status

| milestone | state |
|---|---|
| M0 canonical FlyWire graph + published-value gate | **done** — 11/11 required checks |
| V1 Vyb-native loader | **done** — 26/26 invariants match the Python reference |
| M1 whole-connectome LIF baseline | **done** — dense and sparse engines agree exactly; 1 s of brain in 120 s of CPU |
| M2 Vyb discrete-event engine | **done** — 4/4 replications match the reference exactly |
| M3 Vyb GPU kernels | **done** — tick-for-tick agreement with the CPU reference (512-neuron fixture) |
| M4 energy instrumentation | **done** — GPU device power measured; CPU power unavailable on this machine |
| M5–M8 geometry, downscaling, upscaling, closure | **done** — see section 5 of the paper |
| M9 dynamic validation of the replicas | **done** — 85–95% active-set overlap |
| M11–M14 learning, capability, energy scaling, capability per watt | **done** — including the negative control result |
| M15 100× graph (~14M neurons) | **gated, not attempted** — needs the sibling-edge placement vectorized first |

## Credits and data

Built on the **FlyWire** whole-brain connectome of adult *Drosophila*:

- Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 634 (2024) —
  [10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)
- Lin et al., *Network statistics of the whole-brain connectome of Drosophila*, Nature 634 (2024) —
  [10.1038/s41586-024-07968-y](https://doi.org/10.1038/s41586-024-07968-y) — source of the published
  values the gate checks against
- Data release v783 — [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866)
- Conceptual basis for the scaling method: García-Pérez, Boguñá & Serrano, *Multiscale unfolding of
  real networks by geometric renormalization*, Nature Physics 14 (2018) —
  [10.1038/s41567-018-0072-5](https://doi.org/10.1038/s41567-018-0072-5)
- Biological energy anchor: Panda et al., calorimetry of individual live fly brains, Cell Reports
  Methods (2026) — [10.1016/j.crmeth.2026.101501](https://doi.org/10.1016/j.crmeth.2026.101501)

The full reference list, including the per-spike and per-vesicle biology sources, is in the paper.

The Vyb language and compiler used for the production path:
[rickenator/Vyb](https://github.com/rickenator/Vyb).

---

Built and executed with an autonomous agent harness ([hermes-agent](https://hermes-agent.nousresearch.com)).
Every number in the paper is generated from an artifact in `results/` — there are no illustrative
figures, and nothing in this README that isn't measured somewhere on disk.
