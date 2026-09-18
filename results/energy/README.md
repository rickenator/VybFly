# M4 — Energy instrumentation (PROJECT-VYBFLY.md §17, §18)

Two energy tracks, measured and modeled **separately**, never conflated:

| track | file | what it is |
|---|---|---|
| **A. biological-equivalent** | `biological_model.json` | `P_bio(N) = P0 · N/N0` anchored to published metabolic measurements of real nervous tissue. Watts of *biology*, not of this machine. |
| **B. actual hardware** | `hardware_energy.json` | Joules integrated from real power sampling during a real instrumented workload on this box, plus the workload's own spike / synaptic-event counters. |

§17 of the scope doc is explicit that the mammalian linear-per-neuron relationship must not be
presented as a Drosophila measurement, and §26 forbids claiming biological power efficiency from
GPU measurements. Both rules are enforced as *flags and caveats in the JSON*, not just prose:
every anchor carries `citation_*`, `is_drosophila_measurement` and
`not_a_drosophila_measurement`, and every hardware/biology ratio carries a `caveat`.

---

## 1. Reproduce it on this machine (godzilla, RTX 3090 + i9-10850K, Ubuntu)

```bash
cd ~/Projects/VybFly
.venv/bin/python scripts/phase_energy_demo.py            # ~2 minutes end to end
```

That single command: loads the canonical FlyWire CSR, builds signed E/I weights, writes a
deterministic stimulus schedule, compiles `scripts/cuda/sparse_prop.cu` with `nvcc`, runs three
probe/timing runs, then samples power through nine phases and writes both JSONs plus
`results/energy/build/` artifacts.

Useful switches: `--cpu-seconds`, `--gpu-seconds`, `--idle-seconds`, `--idle-post-seconds`,
`--cooldown-seconds`, `--interval` (sampling period, default 0.2 s = 5 Hz), `--skip-cuda`,
`--results-dir`.

Individual pieces, if you want to see them work on their own:

```bash
# 1. what this box can and cannot measure (writes nothing)
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); \
  from flyscale.energy import probe_devices; import json; print(json.dumps(probe_devices(), indent=1))"

# 2. GPU power, two independent ways (NVML in-process vs nvidia-smi)
gcc -O2 -o /tmp/nvml_power scripts/cuda/nvml_power.c -lnvidia-ml && /tmp/nvml_power 5 1000
nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.sm --format=csv

# 3. compile and run the instrumented workload by itself (~50 s for 45 bio-seconds)
nvcc -O3 -arch=sm_86 -o results/energy/build/sparse_prop scripts/cuda/sparse_prop.cu
./results/energy/build/sparse_prop --dir data/processed/canonical_v783 \
  --weights results/energy/build/weights.f32 \
  --stim-file results/energy/build/stimulus_schedule.bin \
  --steps 1000 --epochs 84 --dt-ms 0.5 --vth 1.0 --decay 0.9 --refr-steps 20 --max-active 300000
```

Everything is deterministic given `--seed` (default 1337): the same weight file and stimulus
schedule hashes appear in every report (`reproducibility.files`), and the bytes of both inputs are
sha256-fingerprinted in the JSON.

## 2. What was measured on this box in the reference run

Run: 2026-09-17, 575 samples at **5.00 Hz** (intervals min 0.1912 s / median 0.2000 s /
max 0.2088 s — `sampling.interval_stats_s`), 114.9 s of sampling.

| device | measured? | how |
|---|---|---|
| `gpu:0` RTX 3090 | **yes** — min 22.1 W, median 113.0 W, max 158.7 W, mean 89.9 W, 10 325 J total | NVML `nvmlDeviceGetPowerUsage` in-process, 5 Hz |
| CPU package / cores (`intel-rapl:0`, `:0:0`) | **no** | `energy_uj` is mode `0400 root:root`; unreadable as uid 1000, `sudo -n` requires a password |
| CPU DRAM domain (`intel-rapl:0:1`) | **no** | same permissions |
| wall / PSU power | **no** | no `hwmon power*_input`, no `power_supply power_now`, no PDU/meter attached; `powerstat` absent; `turbostat` present but needs root |

Exact strings (not paraphrases) live in `device_availability.unavailable_devices` and
`unavailable_metrics` of `hardware_energy.json`. **Consequence: every per-spike and per-event
joule figure here is a GPU-device lower bound, not whole-machine energy.** `cpu_joules_per_spike`
and `whole_machine_joules` are recorded as unavailable rather than estimated.

### Phases and the workload's own counters (reference run)

| phase | wall | bio-s | spikes | synaptic events | GPU mean W | GPU J above idle |
|---|---|---|---|---|---|---|
| `settle` | 5.5 s | – | – | – | – | – (waits for the GPU to come back down after the probes) |
| `idle_baseline` | 8.0 s | – | – | – | 22.46 W (median) | – |
| `workload_cpu` (numpy) | 28.9 s | 2.50 | 7 333 295 | 668 988 775 | 22.24 | **−5.8 J** (noise floor: the numpy path never touches the GPU) |
| `workload_gpu` (CUDA) | 52.5 s | 42.0 | 128 146 295 | 11 710 877 981 | **151.64** (max 158.68) | **6 799.7 J** |
| `cooldown_gpu` | 8.0 s | – | – | – | 62.7 | 318.4 J (sensor/post-load decay; see §5) |
| `parity_check` | 0.45 s | 0.10 | 5 052 | 382 090 | 22.4 | ~0 |
| `idle_baseline_post` | 6.0 s | – | – | – | 22.44 W (median) | – |

Derived hardware-track numbers for `workload_gpu` (`measurement.per_phase.workload_gpu`):

```
wall_seconds_per_bio_second                     1.25      (CPU path: 11.55 -> 9.2x slower)
steps_per_wall_second                           1599      (CPU path: 173)
spikes_per_bio_second                           3.05e6    (network spike rate 21.9 Hz)
synaptic_events_per_bio_second                  2.79e8
synaptic_events_per_spike                       91.4
gpu_joules_above_idle_baseline                  6 799.7 J (7 118.1 J incl. cooldown window)
joules_per_bio_second_gpu_device                161.9 J   (169.5 J with the upper-bound attribution)
joules_per_spike_gpu_device_only                53.1 uJ   (55.5 uJ upper bound)
joules_per_synaptic_event_gpu_device_only       581 nJ    (608 nJ upper bound)
```

Cross-checks that the measurement is real, all inside `instrumentation_verification`:

* **Power tracks the load**: idle 22.49 W → 151.64 W during the GPU workload (+129.2 W,
  `tracks_load: true`); the GPU sat at 22.2 W during the CPU phase, i.e. the sampler is not
  reporting a constant.
* **Independent backend agreement**: 51 `nvidia-smi --query-gpu=power.draw` readings taken from
  the harness's main thread while the NVML sampler ran: mean 150.72 W vs the sampler's 151.64 W
  → 0.61 % apart.
* **Baseline audit**: the pre-run and post-run idle windows agree to 0.07 %
  (22.46 W vs 22.44 W, `baseline_stability.stable: true`); the lower-median window is used for
  attribution, so a window contaminated by the GPU's post-probe decay cannot inflate the baseline.
* **CPU vs GPU counters agree**: identical weight file + identical stimulus schedule, 200 steps,
  CPU 5 052 spikes / 382 090 events, GPU 5 052 / 382 090 → relative difference 0.0.
* **Epoch reproducibility**: `epochs_identical: false` — the 84 per-epoch counter deltas vary by
  ±0.9 % (1 506 836–1 533 191 spikes) because the CUDA `atomicAdd` accumulation order changes
  near-threshold decisions. Reported rather than hidden (`workload_execution.phases[…].epoch_deltas`).

## 3. The workload (what the joules were spent on)

* **Graph**: the canonical dataset `data/processed/canonical_v783` — 139 255 neurons,
  15 091 983 directed edges, 54 492 922 synapses (weight = published synapse count per pair).
* **Signs from published transmitter calls**: `bin/pairs.nt.i8 == 0` (gaba, per `meta.json`
  `nt_order`) marks the 3 233 367 inhibitory edges (21.42 %).
* **Model** (mirrored exactly in `NumpySparseLIF` and in `sparse_prop.cu`): LIF with a dense leak
  `V *= 0.9`, sparse scatter of `synapse-count-normalized` weights into the fired neurons' targets,
  fire when `V >= 1.0` and the 20-step (10 ms) refractory has expired. Epoch = 1000 steps ×
  0.5 ms = 0.5 s of simulated biological time. Synaptic events are counted as *edges walked*
  (fan-out of every active neuron), spikes as threshold crossings plus 500 Poisson-driven
  stimulus neurons at 5 Hz.
* **Activity regime**: 21.9 Hz mean network rate at `gain_e = 6`, `gain_i = 4`, i.e. below the
  100 Hz refractory ceiling but **above** the ~1–10 Hz usually assumed for fly spontaneous
  activity. This is a **stress workload chosen for a bounded, reproducible load, not a
  physiological model** (`workload.regime.caveat`). While tuning the E/I ratio the model showed
  classic bistability — silent or fully ignited with no stable middle — which is recorded with the
  probe numbers in `workload.regime.regime_probes`.
* The neuron model here is deliberately *not* the project's biological baseline (that is another
  phase's job); it exists to be a real, counter-rich sparse workload over the real graph.

## 4. `biological_model.json` — track A

`P_bio(N) = P0 · N / N0`, `N0 = 139 255` (FlyWire v783 whole-brain count).

| anchor | P0 | per neuron | citation | flags |
|---|---|---|---|---|
| **Drosophila whole-brain heat output (primary)** | **256 nW per brain** | **1.838 pW** | Panda K. *et al.*, "Direct quantification of the metabolic heat output of individual Drosophila brains", *Cell Reports Methods* 2026, DOI `10.1016/j.crmeth.2026.101501` (preprint: bioRxiv `10.1101/2025.08.08.669302`) | `is_drosophila_measurement: true`, `not_a_drosophila_measurement: false` |
| mammalian fixed budget per neuron (§17 null model) | 6 kcal/day per billion neurons = 2.906e-10 W per neuron | 290.6 pW | Herculano-Houzel S., *PLoS ONE* 6(3):e17514, 2011, DOI `10.1371/journal.pone.0017514` | **`not_a_drosophila_measurement: true`** |
| published *estimate* of fly nervous-system power (cross-check) | ~120 nW | 0.862 pW | Scheffer L.K., ISPD '21, DOI `10.1145/3439706.3446898` (derived from oxygen consumption, **not** a measurement) | `is_drosophila_measurement: false` |

Caveats attached to the primary anchor (verbatim in `anchor.caveats`): the brain is **explanted and
buffer-perfused**, one genotype (`y sc v`), one sex (female), one age (10 days); the calorimeter's
~40 s time constant and ~7.6 nW resolution **cannot see spike-driven transients**, so the value is a
slow/steady-state metabolic rate; pairing the measured organ with the FlyWire count is an
assumption; and the figure is total tissue metabolism (neurons + glia + housekeeping), not a
measure of spiking computation. The mammalian anchor is 158× larger per neuron than the measured
fly value, which is exactly why §17 insists the two be labeled differently.

Secondary, clearly-derived per-event biology (`mammalian_cortex_per_event`, all
`not_a_drosophila_measurement: true`, Attwell & Laughlin 2001, DOI
`10.1097/00004647-200110000-00001`): 7.1e8 ATP per neuron per spike → **58.9 pJ per spike**;
1.64e5 ATP per vesicle released → **13.6 fJ per release event**. ATP→joule uses 50 kJ/mol
(`8.30e-20 J`), with the ±40 % assumption band recorded in `provenance.atp_hydrolysis_assumption`.
`derived_fly_per_spike` (256 nW ÷ 139 255 ÷ assumed 1 Hz = 1.84 pJ) is flagged
`is_derived_not_measured: true`.

`scaling_table` gives P_bio at the §18 scales (1×→100×): 256 nW → 25.6 µW at 100×.

## 5. Known limitations, stated rather than hidden

1. **Only the GPU is measurable.** No CPU package power, no DRAM power, no wall power (reasons in
   §2). All hardware joules are device-side lower bounds; the "J per spike" of a whole machine is
   not available on this box.
2. **Shared GPU.** Another process (`llama-server`, ~22 GB) coexists on this GPU; any of its
   activity lands in the phase it happens during. The idle-baseline subtraction plus the two idle
   windows bound this, but a co-tenant burst during the GPU phase would inflate the numbers.
3. **Post-load power decay / attribution range.** The GPU reading falls from ~150 W to ~22 W over
   roughly ten seconds after the kernels stop. Whether that is sensor lag or genuine post-load
   power draw cannot be separated here, so workload energy is reported as a range: phase window
   only (lower bound) plus the `cooldown_gpu` window (upper bound) — e.g. 53.1–55.5 µJ per spike.
4. **`nvidia-smi` polling from a thread goes stale on this box.** Reproduced repeatedly: a sampler
   thread spawning `nvidia-smi --query-gpu=power.draw` returned a *constant* idle value (≈21 W)
   for the entire run while GPU utilization sat at 96 %; the same call from the shell, or NVML from
   the same thread, tracked 21 W → 155 W correctly. The harness therefore reads **NVML through
   `ctypes`** and keeps `nvidia-smi` as the cross-check. `probe_gpu()` records both backends and
   that note.
5. **Attribution noise floor** ≈ ±1 J per phase: a phase drawing exactly idle power integrates to
   within a few joules of zero (−5.8 J for the CPU phase, i.e. < 0.1 % of the GPU workload energy).
6. **Not measured here:** joules per trained task and per successful inference (§17 lists them;
   M4 runs no learning or inference task).

## 6. Field guide

`hardware_energy.json`

* `measurement.unavailable` — every metric that could not be measured, with the reason.
* `measurement.per_phase.<phase>` — work units (`steps`, `spikes_total`, `spikes_network`,
  `spikes_stimulus`, `synaptic_events`), rates per simulated biological second, wall-second cost,
  device watts, and the derived joules per bio-second / spike / synaptic event (plus their
  upper bounds where a cooldown window applies).
* `sampling`, `devices`, `phases`, `series` — raw evidence: sample count, interval statistics,
  achieved Hz, and the full per-sample series (`t_rel_s`, `phase`, `gpu:0`, `gpu:0:util_pct`).
* `instrumentation_verification` — proof the measurement is real (load tracking, backend
  agreement, baseline audit, CPU/GPU counter parity).
* `baseline_stability`, `settle_before_baseline` — how the power baseline was chosen and settled.
* `workload` — graph/weights/stimulus provenance (with sha256 heads), the model, the activity
  regime and its probes, and the CPU-vs-GPU parity result.
* `workload_execution` — exact commands, epoch counts, kernel seconds, per-epoch counter deltas,
  child CPU seconds.
* `reproducibility` — dataset version + hashes, driver/CUDA/nvcc versions, GPU identity and
  co-tenants, sampling method, host CPU, load average, source-file hashes.
* `cross_track_comparison` — the two track A/B numbers side by side, each with an explicit
  "this is not an efficiency claim" caveat.

`biological_model.json` — `anchor` (value, method, citation, caveats, flags), `P0_watts`,
`N0_neurons`, `per_neuron_watts`, `not_a_drosophila_measurement`, `comparison_anchors`,
`scaling_table`, ATP provenance and the per-event biology.

## 7. Files

```
src/flyscale/energy.py                     PowerSampler (NVML / nvidia-smi / RAPL / wall), JSON writer,
                                           MetabolicAnchor + BiologicalEnergyModel
scripts/phase_energy_demo.py               the runnable phase: probes, workloads, sampling, report
scripts/cuda/sparse_prop.cu                the instrumented sparse-propagation workload (sm_86)
scripts/cuda/nvml_power.c                  standalone NVML power probe used to characterize the sensor
results/energy/hardware_energy.json        track B + verification + reproducibility  (this run)
results/energy/biological_model.json       track A (constants, citations, flags, scaling table)
results/energy/README.md                   this file
results/energy/build/                      compiled binary + weights/stimulus + CUDA epoch log
                                           (gitignored; regenerate with the command in §1)
logs/phase_energy_m4.log                   stdout of the reference run
```
