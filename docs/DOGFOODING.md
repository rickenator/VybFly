# Dogfooding Vyb: what the GPU workload found

## Intent

VybFly treats the Vyb compiler and its NVPTX backend as a real dependency, not as something to
adopt after a Python prototype. The README states production compute is Vyb + CUDA, with Python
kept as a reference oracle. Because of that, every GPU milestone here doubles as a compiler
exercise: the workload is large (millions of elements, 60 MB to 380 MB buffers) and it uses device
paths that small unit tests rarely reach - bulk DMA of multi-megabyte buffers, array atomics,
f32/f64 stores, descriptor-based kernel arguments.

## What acts as the harness

- `src/vyb_kernels/upscale_kernel.vyb` - the 100x geometric upscale kernel, plus the bisecting
  probes used to localize faults.
- `scripts/verify_upscale.py` - reads back the artifact the kernels wrote and checks the child
  counts and displacements against the canonical somata they came from.
- `docs/VYB-PORT.md` - the record of the port and of each constraint discovered while doing it.

## What it surfaced

Two device-code faults appeared only under this workload:

1. A 32-bit f32 device store (`st_f32`) faults the device. The launch returns 0, the next
   synchronize returns 716, and the context is poisoned for every later call. The repro is
   `probe_stf32`.
2. An array `atomic_add_i32` with a computed index faults once the index is actually in range,
   while a scalar `atomic_add_i32` and an array `atomic_add_f64` both work. The repro is
   `probe_width.vyb (with probe_loop_f64 still in upscale_kernel.vyb)`.

Both repros are committed in `src/vyb_kernels/upscale_kernel.vyb`, and `docs/VYB-PORT.md` records
them.

A third finding was not a compiler bug and is recorded as such: a 4-byte scalar readback through
an 8-byte host slot silently returns a value whose upper half is garbage - a counter of 2,700,513
printed as -4,292,266,783. That is a usage hazard, not a defect.

## Honest limitations

Finding these required running a real workload, bisecting with purpose-built probe kernels, and
comparing launch return codes against device synchronize return codes. Several hours went into
distinguishing a device fault from a wrong stride, a unit error, or a chunking bug. A routine unit
test would not have hit any of them. Two of the findings could not be reproduced outside the
kernel-mode plus large-array setting, so they are recorded as workload observations rather than as
minimal language bugs.

This is evidence that the workload reaches paths the test suite does not. The repros are in the
repository so the claims can be re-checked or refuted.
