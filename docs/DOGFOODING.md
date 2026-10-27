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

## CPU ladder engines (perf/ursela-fly, Oct 2026)

Porting the FlyScale 1x-10x `sparse_prop` ladder to multithreaded Vyb on the CPU (i9-10850K, 10C/20T)
gave three engines in `src/vyb/` - `ladder_prop` (per-thread/per-block edge bins), `ladder_seg`
(static per-row target segments) and `ladder_desc` (segment descriptors built once per step). All
three reproduce the deterministic C++ reference's epoch-0 spike digest bit for bit at every scale and
thread count tried (1, 4, 8, 10, 12, 16, 20). Everything below came up while writing them.

### Compiler / runtime (fixed or patched on perf/ursela-fly-vyb, local only)

- **Every function is NoInline, even at -O2/-O3.** Small helpers (`ld64`, `st64`, `stat_add`) stayed
  real calls in hot loops. Patch: strip NoInline from defined functions before the -O2+ pipeline
  (`VYB_KEEP_NOINLINE=1` restores the old behaviour).
- **The JIT optimizes for the generic x86-64 CPU.** Patch: use the host CPU name and features
  (`VYB_JIT_GENERIC=1` restores generic). Neither patch moved the ladder's numbers measurably: its
  hot loops were already hand-inlined `freedom` blocks and are memory bound. They are kept because
  they are correct defaults for any compute-bound Vyb code.
- **`__vyb_runtime_push/pop_call_frame` surrounds every call**, including `time_nanos()` and trivial
  helpers (65 pairs in one scatter function). That is acceptable for debugging, but it should be
  elided at -O2 or for leaf calls.

### Language / stdlib friction

- Storing a literal through a narrower `loc<T>` (for example an Int literal into `loc<Int32>`) only
  warns, and it corrupted memory. It should be an error, or the literal should be narrowed.
- Some reported errors do not stop execution, so later output can look valid after an earlier failure.
- Generic inference fails on Float32 struct-literal fields.
- `for x in vec` over a `Vec` fails, so the code falls back to index loops.
- A user-declared `free` clashes with the builtin, and `ptr` is a reserved word. Neither is documented.
- The `Vec(n)` constructor docs do not match the implementation.
- There is no bulk typed read: loading a 200 MB binary CSR means `read_all` into a String plus `addr()`.
- There is no safe shared-slice threading. Every multithreaded engine is raw `freedom` pointer
  arithmetic plus a hand-rolled atomic spin barrier.
- There is no prefetch intrinsic (`llvm.prefetch`). The descriptor engine emulates one with a
  "touch the first record of the next 64 segments" pass. That pass bought about 30% on the pull phase.
- It was unclear whether `&&` short-circuits in `freedom` code, so bounds-guarded scans use explicit
  `go` flags rather than risk an out-of-bounds read.
- There is no Bool-to-Int conversion for branch-free counting; it is written as `((a - b) >> 63) & 1`.

### Tooling

- `perf` is unavailable (kernel.perf_event_paranoid=4), so profiling meant in-engine phase timers
  plus reading the `-c x.o -O2` object with `objdump`.
- The repo has no Python test runner (pytest is missing), so the equivalence checks are digest
  comparisons in the RESULT lines.
