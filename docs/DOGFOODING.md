# Dogfooding Vyb: what the GPU workload found

## Intent

VybFly treats the Vyb compiler and its NVPTX backend as a real dependency, not as something to adopt
after a Python prototype: production compute is Vyb + CUDA, with Python kept as a reference oracle.
The workload is large - millions of elements, buffers up to 380 MB - and reaches device paths small
unit tests rarely do: bulk DMA, array atomics, f32/f64 stores, descriptor-based kernel arguments.

## What acts as the harness

- `src/vyb_kernels/upscale_kernel.vyb` - the 100x upscale kernel and the probes used to bisect faults.
- `scripts/verify_upscale.py` - checks the artifact the kernels wrote against the canonical somata.

## What it surfaced

Two device faults appeared only under this workload, and turned out to be one defect: 32-bit stores
and atomics were lowered as 64-bit operations.

1. A 32-bit f32 store (`st_f32`) faulted the device: launch returned 0, the next synchronize returned
   716, and the context was poisoned for every later call (`probe_stf32`).
2. An array `atomic_add_i32` with a computed index faulted once the index was in range, while a
   scalar `atomic_add_i32` and an array `atomic_add_f64` both worked (repros in `probe_width.vyb`).

The same defect explained a quieter symptom that had looked like a language bug: a per-neuron i32
array came back with all but its first entry reading zero, collapsing most of the 100x children onto
their parents. One further finding was not a compiler defect: a 4-byte scalar readback through an
8-byte host slot returns a value whose upper half is garbage.

## Resolution

The lowering defect was reported upstream with the emitted PTX quoted inline, and has since been
fixed: commit efc8cfe, contained in v0.7.6. The compiler repository now covers it with a regression
fixture for the five intrinsic forms and an on-silicon verifier. The consuming project carried a
workaround while the defect was open - a pinned placement spread - and removed it once the fix
landed, so the placement path is back to the form the code was written in. The workload still checks
its own output every run.

## Honest limitations

Finding this took a real workload, purpose-built probes, and a comparison of launch codes against
synchronize codes; hours went into separating a device fault from a wrong stride, a unit error, or a
chunking bug, and no routine unit test would have hit it. The repros are committed so the claims can
be re-checked or refuted.
