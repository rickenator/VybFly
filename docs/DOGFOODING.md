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

### Compiler / runtime (status as of 2026-10-26)

The first two items below are in the Vyb PR for JIT inlining and host-CPU tuning. The
narrow-store fix, the `prefetch` intrinsic and the `-O2` call-frame change are committed and
tested on the `perf/ursela-fly-vyb` branch, and are proposed separately. The engines in this
repo do not depend on any of them: they produce the same digests on stock Vyb 0.7.7.

- **Every function is NoInline, even at -O2/-O3.** Small helpers (`ld64`, `st64`, `stat_add`) stayed
  real calls in hot loops. Patch: strip NoInline from defined functions before the -O2+ pipeline
  (`VYB_KEEP_NOINLINE=1` restores the old behavior).
- **The JIT optimizes for the generic x86-64 CPU.** Patch: use the host CPU name and features
  (`VYB_JIT_GENERIC=1` restores generic). Neither patch moved the ladder's numbers measurably: its
  hot loops were already hand-inlined `freedom` blocks and are memory bound. A clean A/B
  (2 repetitions, idle machine, builds interleaved) put the best engine at each scale within
  ±2% of stock; only `ladder_seg` gained, by 2–3%. They are kept because
  they are correct defaults for any compute-bound Vyb code.
- **`__vyb_runtime_push/pop_call_frame` surrounds every call**, including `time_nanos()` and trivial
  helpers (65 pairs in one scatter function, 169 sites in `ladder_desc`). On the branch, these calls
  are now skipped at -O2 and above (`VYB_CALL_FRAMES=1` keeps them). With frames on, `fib(35)` takes
  0.32 s; with them off it takes 0.10 s. `ladder_desc` did not change measurably. The same work fixed
  a bug in the runtime call stack: recursion deeper than 256 frames printed two warnings per frame,
  and pushes and pops went out of balance.

### Language / stdlib friction

- Storing a literal through a narrower `loc<T>` (for example an Int literal into `loc<Int32>`) only
  warned, and it did an 8-byte store that corrupted the neighboring memory. **Fixed on the branch:**
  an integer store now converts to the location's width. It truncates (with a warning for an
  out-of-range constant), or it sign- or zero-extends. There is a regression test that clobbers the
  neighbor on the old compiler.
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
  **Added on the branch:** `prefetch(addr[, rw[, locality]])` lowers to `llvm.prefetch` and checks
  its arguments at compile time. The engine here still uses the touch pass until the intrinsic
  lands upstream.
- It was unclear whether `&&` short-circuits in `freedom` code, so bounds-guarded scans use explicit
  `go` flags rather than risk an out-of-bounds read.
- There is no Bool-to-Int conversion for branch-free counting; it is written as `((a - b) >> 63) & 1`.

### Tooling

- `perf` is unavailable (kernel.perf_event_paranoid=4), so profiling meant in-engine phase timers
  plus reading the `-c x.o -O2` object with `objdump`.
- pytest is not in the system Python or the project `.venv`. It is only an optional `dev`
  dependency. With it installed (`pip install pytest` plus the project dependencies), `tests/`
  passes 30/30. The ladder equivalence checks are digest comparisons of the `RESULT` line against
  `sparse_prop_cpu`.

## GPU ladder engine (perf/ursela-fly, Oct 2026)

The ladder engine `src/vyb/ladder_gpu.vyb` and its kernels `src/vyb_kernels/ladder_gpu_kernel.vyb`
are bit-exact at 1x to 40x, and run 10x in 3.94 s on a GB10 (2.6x the CUDA kernel on the same
GPU) and in 2.46 s on an RTX 3090 (4.3x). Writing them turned up the following.

### Compiler (fixed on Vyb branch `perf/kernel-atomics-bits`, PR open)

- **Miscompile: `x<Int> = f() as Int` with `f` returning `CInt` kept stale high bits.** `CInt`,
  `CUInt`, `CLong`, `CSize` and the other C aliases were missing from the integer-class table.
  As a result, `as` between them and `Int` became an identity, and an i32 was stored into an
  i64 slot with only a "Type mismatch in assignment" warning. It hit host FFI code (CUDA driver
  return codes) and kernel intrinsics alike: for example, `-5 as Int` read back as
  1236950581243.
  - **Fixed:** the aliases are classified, and an int-to-int `as` now always extends or truncates,
    with zero extension for unsigned sources. Covered by `test/types/cint_as_int_widen_test.vyb`.
- **Missing kernel intrinsics** for anything exact. Without them there was no way to set a bit
  atomically, count spikes per warp, or reduce a 64-bit digest.
  - **Added:** `atomic_{or,and,xor}_u32`, `atomic_add_i64`, `atomic_xor_i64`, `atomic_add_f32`,
    `atomic_max_i32`, `popc_u32`, `ctz_u32` and `ballot_u32`. Each lowers to
    `atom`/`red.global` or `vote.sync.ballot`. Covered by `test/kernel/test_atomics_bits.vyb`
    plus an on-GPU check (`check_atomics_bits.py`, PASS on GB10).
- Stock Vyb cannot compile the kernel until that PR lands, so the PTX is committed next to the
  source. The host runner itself runs on stock 0.7.7.

### Fixed on Vyb branch `perf/kernel-friction` (stacked on `perf/kernel-atomics-bits`, PR open)

- **No 64-bit warp shuffle.** The digest butterfly moved i64 bits through
  `shfl_xor_f64(f64_from_bits(x))`. **Added:** `shfl_{xor,down,up,idx}_i64`, lowered to a pair
  of 32-bit `shfl.sync`. The kernel now uses `shfl_xor_i64`. Covered by
  `test/kernel/test_shfl_i64.vyb` plus an on-GPU check (`check_shfl_i64.py`, PASS on GB10).
- **`var<Int> fail = 0` gave a parse error that did not name the keyword.** It now says the name
  is a reserved word, for variables and parameters alike
  (`test/parser/test_reserved_word_{variable,param}_name.vyb`).
- **Kernel launch parameters were built by hand** as an `[Int; N]` of `addr()` cells, where an
  argument-order slip is silent. **Added:** `cuda_launch_args(f, grid, block, a, b, ...)`, which
  has the compiler pack `kernelParams` from typed arguments. It must be called in `freedom`, and
  it rejects `String` arguments and a bad arity at compile time (4 tests under `test/ffi/`;
  `test_cuda_launch_args.vyb` prints `LAUNCH_ARGS OK` on a GB10).

### Still open

- The NVPTX path neither unrolls nor software-pipelines a data-dependent `ctz` loop, so each
  set bit paid a full memory latency. Hand-grouping 8 weight loads per round, summed in order,
  took 10x from 5.98 s to 4.19 s. A `#pragma unroll`-style hint, or an independent-load
  scheduling pass, would make this automatic.
- There are no GPU timing hooks (events) in the stdlib; times are host wall clock around one
  final `cuCtxSynchronize`.
- The `network` stdlib only sends and receives `String`s. The two-Spark split moves raw
  `[i64 count][i32 ids]` buffers, so it declares libc `send`/`recv`/`setsockopt`/`close`/`usleep`
  as externs and hard-codes `MSG_DONTWAIT`, `MSG_NOSIGNAL`, `TCP_NODELAY` and `SO_REUSEADDR`.
  **Being fixed** on Vyb branch `perf/net-bytes` (PR pending): `socket_send_bytes`,
  `socket_recv_bytes`/`_exact` and a framed full-duplex `socket_exchange_bytes`/`_range` over
  `Vec<UInt8>`, with two tests. The split keeps its externs on `main` until that merges.
  `VYBFLY_SPLIT_NET=bytes` on branch `perf/split-net-bytes` uses it and is bit-exact.
- **A by-value `Vec` parameter is deep-copied on every call.** The first byte API took
  `data<Vec<UInt8>>`, so the split's 11 MB reused send buffer was copied on each exchange. Passed
  through one helper function, that was 9.2 ms per exchange: 91.9 s for a 20x run, against
  12.0 s over the extern path. The send side now takes `their<Vec<UInt8> const>`, passed as
  `view(buf)`, and an exchange costs 3 us. The guide documents the copy, but nothing warns when a
  large `Vec` goes by value.
- Smaller items from the same work: a `match` binding of an optional `Vec` has no known type for
  `addr()` ("Cannot take address of expression with unknown type", reported without a file or
  line), so it has to be rebound to a typed local first. A user `extern` `memcpy` with `Int`
  arguments clashes with the stdlib's `ptr` declaration ("Redefinition of function 'memcpy'"),
  so the engine uses `memmove`. And on aarch64 (GB10),
  `test/agents/test_agent_payloads.vyb` hangs at 0% CPU after its Int-payload cases, with every
  build tried, so the full suite there reports 1322 passed, 1 failed.

### RDMA from plain externs (no root)

The split's RDMA exchange (`VYBFLY_SPLIT_NET=rdma`) is ibverbs called from Vyb with no C shim.
It works, but it surfaced:

- **The JIT binds every extern when it loads a module, and only dlopens a fixed list of
  libraries** (ssl, sqlite3, git2, cuda, SDL3). There is no flag or environment variable to add
  one. Plain `ibv_*` externs failed with "Symbols not found" unless the run used
  `LD_PRELOAD=libibverbs.so.1`, and they broke the TCP path too. The engine now resolves the
  verbs with `dlopen`/`dlsym` and calls them through function pointers. A `--link`-style library
  list for the JIT, or lazy extern binding, would make plain externs usable.
- **Calling a `loc<fn(...)>` pointer:** `f(...) as Int` fails ("'as' on a wildcard error requires
  an error pointer"), and assigning the call to an existing `var` fails ("could not determine
  type of LHS or RHS"). Only a fresh typed declaration (`rc<CInt> = f(...)`) works.
- **`CInt as Int` warns** "Storing i32 into location of type i64" (see the `CInt as` miscompile
  above), so negative return codes are tested before the cast.
- **No C struct layouts.** `ibv_post_send`, `ibv_post_recv` and `ibv_poll_cq` are `static inline`
  in `verbs.h`, so they are called through `ibv_context.ops` at byte offsets. Every work request,
  SGE, completion and QP attribute is also filled in by offset, about 40 constants taken from
  rdma-core 50 on aarch64. A header-derived struct import would remove them.
