# Project Scope: Geometric Upscaling of the Drosophila Connectome

## Working Title

**FlyScale: Geometrically Scaled Connectome Simulation and Energy Characterization**

Potential downstream Vyb names:

* `flybrain`
* `FlyDES`
* `Watchfly`
* `Vyb Neural DES`

---

## 1. Project Objective

Develop a biologically grounded, computationally executable model of the adult Drosophila brain and determine whether its network architecture can be algorithmically **scaled upward by factors of approximately 10× to 100× while preserving the structural and dynamical properties of the original connectome**.

The principal experimental objective is to determine how:

* neuron count,
* synapse count,
* event activity,
* learning capacity,
* behavioral complexity,
* dynamical richness,
* and energy consumption

scale as the connectome is enlarged.

The energy baseline should be grounded in empirical biological evidence rather than optimistic computational assumptions.

Existing mammalian measurements indicate that whole-brain glucose consumption is approximately linear in total neuron count over large differences in brain size, with average whole-brain glucose expenditure per neuron remaining surprisingly stable across rodents and primates. The initial null hypothesis is therefore:

$$
P(N) \propto N
$$

where:

* \(N\) = number of neurons
* \(P\) = biological-equivalent power requirement

This project will investigate whether capability grows faster than energy expenditure even when energy itself scales approximately linearly.

The central research question is:

> **Can a Drosophila-derived brain architecture be enlarged geometrically while preserving its native organizational principles, yielding increased computational and learning capability at approximately linear energy cost?**

---

## 2. Scientific Baseline

The primary anatomical reference shall initially be the FlyWire adult female Drosophila connectome.

Current FlyWire data contains approximately:

* 139,255 neurons
* ~54.5 million synapses
* more than 8,400 annotated cell types
* neurotransmitter annotations
* brain-region / neuropil assignments
* neuron morphology and connectivity information

The reconstruction is anatomical rather than physiological: it contains the physical wiring diagram but not a recording of the dynamic state of the original animal.

The FlyWire brain exhibits nontrivial whole-network organization including:

* rich-club connectivity,
* recurrent motifs,
* feed-forward motifs,
* regional specialization,
* highly connected integrator/broadcaster populations,
* sparse local specialization combined with global connectivity.

These properties must be considered invariants or validation targets during network scaling rather than discarded in favor of simple random graph expansion.

Recent work further indicates that the Drosophila synaptic network possesses a useful latent geometry. A 2026 study found that a two-dimensional hyperbolic embedding represented the connectome's connectivity structure better by several metrics than its original three-dimensional anatomical neuron coordinates, with higher-dimensional Euclidean embeddings improving further.

This provides a plausible mathematical basis for geometric enlargement.

---

## 3. Core Hypotheses

### H1 — Structural Scale Preservation

The Drosophila connectome has sufficient geometric and topological regularity that larger synthetic networks can be constructed while preserving important statistics of the source connectome.

For enlarged graph \(G_s\) at scale factor \(s\):

$$
R(G_s) \approx G_1
$$

where \(R\) is an appropriate renormalization/coarse-graining operator.

A successful 10× network should coarse-grain back toward the original fly network rather than merely resemble it superficially.

---

### H2 — Approximately Linear Biological Energy Scaling

The first-order biological energy model shall assume:

$$
P_s \approx sP_1
$$

unless empirical or simulation evidence demonstrates otherwise.

This follows from mammalian measurements showing approximately fixed whole-brain metabolic expenditure per neuron across substantial variation in brain size.

No logarithmic or sublinear power claim shall be assumed in advance.

Better-than-linear behavior is an experimental result, not a design assumption.

---

### H3 — Capability May Scale Faster Than Power

Although total metabolic expenditure may grow roughly linearly with neuron count, useful capability may not.

Candidate quantities include:

* discrimination capacity,
* number of stable representations,
* memory capacity,
* temporal depth,
* task complexity,
* sensory resolution,
* robustness,
* number of separable internal states,
* transfer-learning capability,
* behavioral repertoire.

The important quantity may therefore be:

$$
\eta = \frac{\text{capability}}{\text{energy}}
$$

rather than raw watts.

---

### H4 — Geometric Enlargement May Increase Representational Resolution

A scaled network may possess a finer-grained internal representation of sensory and temporal state.

This is the operational replacement for phrases such as “higher-resolution consciousness.”

The project shall **not assume consciousness**.

Instead it shall measure phenomena that could plausibly correspond to richer internal representation:

* finer stimulus discrimination,
* greater state separability,
* larger memory capacity,
* longer temporal dependencies,
* more complex learned mappings,
* richer recurrent dynamics,
* larger effective dimensionality,
* improved generalization.

If those increase with network scale, the result stands independently of any claim about subjective experience.

---

## 4. Critical Scientific Distinction: Connectome vs Learned Brain

The FlyWire connectome does **not** contain a preserved snapshot of everything an individual fly had learned.

It provides:

* structural connectivity,
* synaptic locations,
* connection multiplicity,
* neuron identity,
* cell type,
* neurotransmitter information,
* morphology,
* anatomical organization.

It generally does not provide the complete physiological state required to reconstruct the individual animal's exact memories, including:

* instantaneous membrane potentials,
* neuromodulator concentrations,
* all receptor states,
* intracellular biochemical state,
* complete synaptic efficacy,
* short-term plasticity state,
* long-term molecular plasticity state.

Therefore the phrase **native capability** shall mean:

> Functional capability arising from the biologically observed architecture and known physiological rules, before task-specific artificial restructuring.

The project should preserve the topology that evolution built while allowing the resulting simulated brain to learn new tasks.

---

## 5. Architecture

The project should be divided into five independent layers.

```text
FlyWire / biological data
          │
          ▼
    Connectome Model
          │
          ▼
 Geometric Scale Engine
          │
          ▼
 Neural Dynamics Engine
          │
          ▼
 DES / CUDA Runtime
          │
          ▼
Experiment + Energy Harness
```

Each layer must be separately testable.

---

# 6. Phase 0 — Source Data and Reproducible Baseline

## Goal

Construct a canonical representation of the real FlyWire connectome and establish reproducible graph metrics.

## Tasks

Import:

* neuron IDs,
* directed connections,
* synapse counts,
* cell types,
* neurotransmitters,
* neuropils,
* available morphology,
* anatomical coordinates,
* connectivity annotations.

Represent the graph using efficient structures such as:

```text
Neuron[N]

CSR outgoing edges
CSR incoming edges

Edge:
    destination
    synapse_count
    neurotransmitter
    optional delay
    optional weight
```

Record canonical graph metrics:

* N neurons
* number of directed edges
* number of individual synapses
* in-degree distribution
* out-degree distribution
* weighted degree
* clustering coefficient
* path-length distribution
* strongly connected components
* rich-club coefficient
* motif distributions
* community structure
* neuropil connectivity matrix
* cell-type mixing matrix
* neurotransmitter distribution.

## Deliverable

`flywire_baseline.json`

plus reproducible analysis scripts.

## Gate

No simulation work proceeds until source graph metrics reproduce published values within acceptable tolerances.

---

# 7. Phase 1 — Executable Biological Baseline

## Goal

Create the smallest defensible whole-connectome dynamical model.

Do not optimize prematurely.

Start CPU-first if necessary.

## Initial neuron models

Implement at least:

1. Leaky integrate-and-fire
2. Configurable inhibitory/excitatory behavior
3. refractory state
4. synaptic delay
5. configurable weight mapping from anatomical synapse counts.

Later candidates:

* adaptive LIF,
* Izhikevich-type dynamics,
* conductance-based models,
* graded signaling where biologically appropriate,
* neuromodulation.

The initial objective is reproducibility and comparability, not maximal biological realism.

---

# 8. Phase 2 — Vyb Discrete-Event Runtime

## Objective

Build a general Vyb discrete-event engine suitable for biological networks.

Conceptual lineage:

```text
Simkit / Viskit
      ↓
modern Vyb DES
      ↓
GPU-capable event execution
```

## Required abstractions

```text
Entity
State
Event
EventType
Timestamp
Scheduler
EventSource
EventSink
Recorder
Replication
Experiment
```

A neuron need not be an individually allocated object.

The logical entity abstraction may compile into structure-of-arrays GPU storage.

Example:

```text
NeuronState[N]:
    Vm
    last_update
    refractory_until
    tau
    threshold
    type
```

---

# 9. Phase 3 — Event-Driven GPU Backend

Dense timestep simulation should be retained only as a validation reference.

Existing Drosophila simulation benchmarks show that sparse event-driven execution can dramatically reduce computational energy relative to dense timestep matrix operations because only a minute fraction of neurons may be active on a given step. One published software benchmark reported a reduction from roughly 5,362 to 772 joules per simulated biological second when switching from dense PyTorch execution to sparse event processing on its tested hardware. This is implementation-specific but strongly supports an event-centric design.

## Target architecture

Avoid a single global heap on GPU.

Use:

* time buckets,
* calendar queue,
* timing wheel,
* hierarchical timing wheel,
* or equivalent parallel scheduler.

Concept:

```text
event bucket T
      │
      ▼
group by destination
      │
      ▼
parallel synaptic accumulation
      │
      ▼
update affected neuron state
      │
      ▼
threshold / firing detection
      │
      ▼
scatter new events into future buckets
```

Experiment with:

* CSR/CSC layouts,
* destination sorting,
* fanout sorting,
* warp-cooperative edge traversal,
* segmented reduction,
* atomic accumulation,
* local event buffers,
* global overflow queues.

## Validation

For deterministic configurations:

```text
CPU DES ≈ GPU DES ≈ timestep reference
```

within stated numerical tolerances.

---

# 10. Phase 4 — Latent Geometry Discovery

## Objective

Determine the network geometry appropriate for scaling.

Do not assume physical XYZ coordinates are the correct scale space.

Evaluate:

* anatomical Euclidean space,
* 2-D hyperbolic embedding,
* higher-dimensional Euclidean embeddings,
* Node2Vec-like representations,
* spectral embeddings,
* diffusion geometry,
* graph Laplacian coordinates.

The 2026 Drosophila network-geometry work should serve as a direct starting reference rather than reinventing this analysis.

## Output

Every neuron receives a latent coordinate:

```text
Neuron:
    latent_position
    popularity / degree parameter
    cell_type
    neuropil
    physiological class
```

---

# 11. Phase 5 — Geometric Renormalization Baseline

Before scaling upward, demonstrate that scaling downward works.

Construct:

```text
1.0×
0.5×
0.25×
0.1×
```

connectomes.

Geometric network renormalization already provides mathematical frameworks for producing reduced replicas of real weighted networks while attempting to preserve multiscale topology.

Validate preservation of:

* degree distribution,
* weighted degree distribution,
* clustering,
* motifs,
* rich clubs,
* regional organization,
* shortest-path distribution,
* cell-type proportions,
* transmitter balance,
* modularity.

## Dynamic validation

Run identical normalized stimuli through:

```text
G1
G0.5
G0.25
```

and compare:

* propagation patterns,
* firing statistics,
* attractors,
* state dimensionality,
* latency,
* output distributions.

If downscaled networks do not preserve dynamics reasonably, inverse scaling should not proceed.

---

# 12. Phase 6 — Inverse Geometric Renormalization

This is the key research component.

Construct larger networks:

```text
G1
G2
G5
G10
G25
G50
G100
```

Target sizes based on 139,255 original neurons:

```text
1×       ~139 K
10×      ~1.39 M
100×     ~13.9 M
```

Synapse count must **not automatically scale quadratically**.

The connectivity law should preserve biological sparsity.

If average degree remains approximately bounded, then:

$$
E \propto N
$$

rather than:

$$
E \propto N^2
$$

This is crucial.

## Node subdivision concept

A source neuron or latent region may generate multiple synthetic descendants:

```text
parent
 ├─ child_1
 ├─ child_2
 ├─ child_3
 └─ ...
```

Children should:

* remain near the parent in latent geometry,
* inherit or appropriately diversify cell type,
* preserve transmitter class,
* redistribute parent connectivity,
* introduce local microstructure,
* establish intra-group connectivity,
* generate long-range connections using the inferred geometric connectivity law.

Do NOT simply duplicate neurons and edges.

---

# 13. Phase 7 — Renormalization Closure Test

This is the primary structural acceptance criterion.

For every generated scale:

```text
R(G10)   ≈ G1
R(G100)  ≈ G1
```

Measure distance between coarse-grained synthetic graph and biological source graph.

Potential comparison metrics:

* Wasserstein distance of degree distributions,
* motif divergence,
* spectral distance,
* Laplacian eigenvalue distribution,
* community similarity,
* rich-club similarity,
* connectivity-matrix divergence,
* graphlet distribution,
* embedding-distance distribution.

Generate a single composite score only for internal optimization; preserve raw metrics.

---

# 14. Phase 8 — Plasticity and Learning

Scaling structure alone is insufficient.

Implement known biologically relevant learning mechanisms gradually.

Prioritize the mushroom body because Drosophila mushroom-body circuits are strongly associated with associative learning and memory, and their connectomes are extensively studied. Kenyon cells encode stimuli sparsely, and dopaminergic circuits participate in reward/valence-dependent learning.

Candidates:

* STDP,
* reward-modulated STDP,
* Hebbian plasticity,
* homeostatic plasticity,
* dopamine-gated plasticity,
* synaptic normalization,
* structural plasticity later.

Do not permit learning rules to destroy the source architecture without explicit experimental reason.

---

# 15. Phase 9 — Capability Scaling Tests

The project must test whether larger brains become meaningfully more capable.

Use benchmark families that can scale continuously.

## Sensory discrimination

Present increasingly similar stimuli.

Measure minimum separable stimulus distance.

Hypothesis:

```text
larger network
→ finer internal separation
→ better discrimination
```

---

## Memory capacity

Train associations:

```text
stimulus_i → response_i
```

Increase the number until accuracy collapses.

Measure:

$$
M(N)
$$

---

## Temporal depth

Train tasks requiring increasingly long history.

Measure longest reliable dependency interval.

---

## Sequence learning

Train progressively longer:

```text
A → B → C → D ...
```

or sensorimotor trajectories.

---

## Generalization

Train incomplete stimulus classes.

Measure performance on novel variants.

---

## Robustness

Randomly remove:

* neurons,
* synapses,
* regions.

Measure graceful degradation.

---

## Dynamical complexity

Record:

* attractor count,
* effective dimensionality,
* entropy,
* mutual information,
* criticality indicators,
* avalanche statistics,
* state-space coverage.

---

# 16. "Resolution of Reality" Hypothesis

Treat this carefully and experimentally.

Do not attempt to prove subjective consciousness.

Instead define **representational resolution**.

Possible measurable proxies:

### Sensory resolution

How small a change in input can be reliably distinguished?

### Internal state resolution

How many distinct stable or metastable internal representations can coexist?

### Temporal resolution

How finely can sequences or delays be distinguished?

### World-model complexity

How many independent environmental variables can influence behavior simultaneously?

### Behavioral resolution

How many distinct actions or policies can be reliably conditioned?

If these scale favorably with network size, then it is reasonable to describe the larger network as possessing a higher-resolution internal representation.

Whether that corresponds to richer subjective experience remains scientifically unresolved.

---

# 17. Energy Model

Maintain two separate energy measurements.

## A. Biological-equivalent energy estimate

Use metabolic models anchored to experimental biology.

Initial null model:

$$
P_{bio}(N) = P_0\frac{N}{N_0}
$$

with more sophisticated corrections later for:

* firing rate,
* neuron class,
* synaptic activity,
* axonal length,
* transmitter type.

The mammalian linear-per-neuron relationship is evidence for the null model but must not be presented as direct Drosophila-specific measurement.

---

## B. Actual hardware energy

Measure directly:

```text
wall power
GPU power
CPU power
DRAM power if available
```

Record:

```text
joules / simulated biological second
joules / spike
joules / synaptic event
joules / trained task
joules / successful inference
```

Hardware efficiency and biological efficiency must never be conflated.

---

# 18. Required Power Scaling Experiment

For each scale:

```text
1×
2×
5×
10×
25×
50×
100×
```

run standardized workloads.

Record:

```text
N neurons
E synapses
active neurons / biological second
spikes / biological second
synaptic events / biological second

biological-equivalent watts
actual hardware watts

wall seconds / biological second

task performance
memory capacity
discrimination threshold
state-space metrics
```

Fit:

$$
P \sim aN
$$

$$
P \sim aN^\alpha
$$

and other candidate relationships.

Do not select the functional form in advance.

---

# 19. Capability/Energy Scaling

The central result should eventually be represented as:

$$
C(N)
$$

versus

$$
P(N)
$$

and:

$$
\eta(N)=\frac{C(N)}{P(N)}
$$

where \(C\) is not a single arbitrary intelligence score.

Maintain multiple capability curves.

Examples:

```text
memory capacity / watt
discrimination classes / watt
temporal depth / watt
successful task complexity / watt
effective state dimension / watt
```

---

# 20. Simulation Environments

Do not initially train on language.

Start with closed sensorimotor worlds.

Preferred progression:

### Level 1

Synthetic binary/light/odor stimuli.

### Level 2

2-D navigation environment.

Inputs:

* visual,
* proximity,
* reward,
* orientation.

Outputs:

* movement,
* turn,
* stop.

### Level 3

Complex navigation.

Add:

* memory,
* changing goals,
* delayed reward,
* hidden-state environments.

### Level 4

Abstract tasks.

Only after biological learning is demonstrated.

---

# 21. Baseline Behavioral Preservation

Before claiming the enlarged network is “better,” establish that the original simulation can reproduce known or plausible fly-level circuit functions.

Prioritize:

* olfactory discrimination,
* associative conditioning,
* visual response,
* navigation,
* action selection,
* memory persistence.

The enlarged brain should inherit these capacities before additional complexity is introduced.

---

# 22. Vyb Implementation Strategy

Keep the scientific model independent of Vyb initially where useful for validation.

Reference implementation may exist in Python/C++.

Production architecture should move toward:

```text
Vyb
 ├── connectome loader
 ├── graph analysis
 ├── DES
 ├── neural dynamics
 ├── plasticity
 ├── scale engine
 ├── experiment runner
 └── CUDA/NVPTX backend
```

Long-term objective:

```text
FlyWire data
   ↓
Vyb
   ↓
CUDA/NVPTX
```

with no Python required for production simulation.

---

# 23. GPU Scaling Strategy

The first large target should be **1–2 million neurons**, not 14 million.

Demonstrate:

```text
1×
2×
5×
10×
```

before attempting 100×.

Primary bottlenecks likely include:

* sparse graph storage,
* event queue pressure,
* atomic accumulation,
* fanout variance,
* memory locality,
* spike bursts,
* plasticity bookkeeping.

The simulation architecture must remain sparse.

---

# 24. Reproducibility Requirements

Every experiment records:

```text
git commit
dataset version
graph scale
random seed
embedding version
renormalization parameters
neuron model
plasticity parameters
simulation timestep/event precision
compiler version
CUDA version
GPU model
power sampling method
```

All experimental results must be reproducible from a manifest.

---

# 25. Success Criteria

## Minimum success

Demonstrate a synthetic 2× connectome that:

* preserves major source graph statistics,
* renormalizes back toward the real graph,
* executes stably,
* retains baseline circuit behavior.

## Strong success

Demonstrate a 10× connectome that:

* retains structural invariants,
* learns the same tasks,
* exceeds baseline capacity on at least one scaling benchmark,
* consumes approximately proportional biological-equivalent energy.

## Exceptional result

Demonstrate that:

$$
C(N)
$$

grows significantly faster than:

$$
P(N)
$$

over multiple scales.

Example:

```text
100× neurons
~100× biological energy
>>100× measurable task capacity
```

That would indicate improving capability per unit energy with scale.

---

# 26. Explicit Non-Goals

Do not initially attempt to:

* reproduce human intelligence,
* claim artificial consciousness,
* map fly neurons directly onto transformer concepts,
* create language capability,
* optimize benchmarks at the expense of biological structure,
* replace the connectome with a generic SNN,
* assume synapse count is equivalent to learned weight,
* claim biological power efficiency from GPU power measurements.

---

# 27. Research Risks

### Structural scaling may fail

The fly graph may not possess sufficient self-similarity for meaningful inverse renormalization.

Result is still scientifically valuable.

### Dynamics may not scale with topology

Graph statistics may survive while activity does not.

This is a primary experiment, not a failure of implementation.

### Missing biological variables

Connectome structure alone may be insufficient to reproduce useful behavior.

Add physiological detail incrementally.

### Training may dominate architecture

If task performance mostly depends on imposed learning rules rather than connectome geometry, quantify that explicitly.

### Larger may not mean better

The larger graph may become unstable, redundant, slower, or harder to train.

That is a valid experimental result.

---

# 28. Milestone Sequence

### M0 — Dataset

Reproduce FlyWire graph statistics.

### M1 — CPU neural baseline

Executable whole-fly network.

### M2 — DES engine

Correct event-driven Vyb runtime.

### M3 — CUDA DES

GPU execution matches CPU.

### M4 — Energy instrumentation

Hardware joules reproducibly measured.

### M5 — Latent geometry

Reproduce meaningful fly embedding.

### M6 — Downscaling

0.5× / 0.25× graph replicas.

### M7 — Renormalization validation

Downscaled networks preserve selected dynamics.

### M8 — 2× inverse scaling

First enlarged synthetic fly.

### M9 — Closure

2× graph renormalizes toward original.

### M10 — 10× graph

~1.4M-neuron simulation.

### M11 — Plasticity

Associative learning demonstrated.

### M12 — Scaling benchmarks

Capability vs size curves.

### M13 — Energy curves

Power vs size curves.

### M14 — Capability/watt

Combined scaling analysis.

### M15 — 100× attempt

~14M neurons if prior results justify it.

---

# 29. Immediate Agent Tasks

## Agent A — Connectome

Produce the canonical FlyWire graph dataset and reproduce published topology statistics.

## Agent B — Geometry

Reproduce current Drosophila hyperbolic/network embeddings and determine candidate geometric models for renormalization.

## Agent C — Neural Dynamics

Build a reproducible LIF-based whole-connectome simulator.

## Agent D — Vyb DES

Design a generic Vyb discrete-event runtime suitable for sparse neural events.

## Agent E — CUDA

Prototype bucketed GPU event scheduling and sparse synaptic propagation.

## Agent F — Energy

Build repeatable GPU/node power instrumentation and biological-equivalent energy accounting.

## Agent G — Learning

Build a biologically informed mushroom-body associative-learning benchmark.

These agents should work against common graph and experiment formats rather than inventing incompatible representations.

---

# 30. Guiding Principle

Do not design the answer into the experiment.

Start with the real fly.

Measure what is there.

Infer the geometry.

Scale the geometry.

Preserve what survives.

Run the same tasks.

Measure the energy.

Then determine whether a larger fly-derived brain actually buys more capability per watt.

The desired result is interesting.

The measured result is the project.

