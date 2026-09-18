# Vyb-native implementation plan (FlyScale)

Director's steer: this project should end up in Vyb, not Python. PROJECT-VYBFLY.md §22 allows
a Python/C++ reference implementation for validation and requires production execution to be
Vyb + CUDA/NVPTX with no Python in the loop. This document is the bridge, and it now carries
verified findings from the first working Vyb loader.

## Division of labor

| Layer | Reference (Python) | Production (Vyb) |
|---|---|---|
| Source ingest (feather/tsv parsing) | yes, `scripts/fetch_data.py` + `flyscale.io` | no - ingest emits a flat binary that Vyb reads directly |
| Canonical graph representation | yes, `flyscale.connectome` | `src/vyb/flyload.vyb` (loader over the same flat binary) |
| Graph metrics (degrees, components, paths, motifs) | yes, `flyscale.metrics` (the oracle) | `metrics.vyb` - must match the oracle numerically |
| Neural dynamics (LIF) | Phase 1 reference | Phase 1/2 Vyb |
| Discrete-event runtime | no | Vyb DES (the real target) |
| GPU event execution | no | Vyb CUDA/NVPTX |

The Python layer keeps its value as a *golden reference*: every Vyb metric is checked against
the oracle's numbers on the same canonical dataset.

## Canonical binary format (the interface)

Header-less little-endian arrays, so Vyb needs nothing from Python at runtime:

```
data/processed/canonical_v783/bin/
  meta.json                  counts, element types, thresholds, provenance
  neurons.root_id.i64        N           neuron root ids, index order
  neurons.<attr>.txt         N lines     super_class, cell_class, cell_type, supertype, top_nt, side
  neurons.coords.f32         6N floats   annotation/soma coordinates (geometry phase)
  pairs.pre.i32              M           connection source neuron index
  pairs.post.i32             M           connection target neuron index
  pairs.syn.i32              M           synapse count of the connection
  pairs.nt.i8                M           dominant transmitter code (0..5, NT_TYPES order)
  csr.out.indptr.i64         N+1
  csr.out.indices.i32        M
  csr.out.syn.i32            M
  csr.in.indptr.i64          N+1
  csr.in.indices.i32         M
  csr.in.syn.i32             M
  neuropils.txt              K lines     neuropil label per code
```

Total 449.6 MB for v783 (M = 15,091,983 pairs at threshold 1). The threshold-5 view used for
all published comparisons is a filter on `pairs.syn.i32` - no extra files.

## Verified Vyb constraints (2026-09-17, build `vyb 0.7.5`, probes in `src/vyb/probes/`)

These cost real debugging time and must be respected by any Vyb code that reads the dataset:

1. **A raw io byte buffer cannot cross a module boundary.** `io.read_bytes_at` returns a
   `Vec<UInt8>` that is correct when used in the same function, but a module function that
   *returns* it hands the caller a buffer of the right length with **garbage contents**
   (probe: inline read gave `0, 62`; the same 16 bytes returned through a module function gave
   `101106528132824, 0`). Passing a `File` into a helper fails the same way. Decode inside the
   function that reads, and return `Vec<Int>` built with `push`.
2. **A function returning `Vec<UInt8>?` cannot `return` a `Vec<UInt8>`.** The compiler emits
   `Unsupported or invalid cast from type { ptr, i64, i64 } to { { ptr, i64, i64 }, i1 }`.
   An optional Vec only comes straight out of an io intrinsic; signal failure with a short
   buffer instead (`probe_b`).
3. **A `Vec` parameter is not mutated for the caller** ("Vec assignment deep-copies on
   borrow"), so out-parameters do not work: `load_i64(out<Vec<Int>>)` returned an empty Vec
   (`probe_d`). Return results by value.
4. **Byte-at-a-time decoding is ~29 us per element** (~34k elements/s): `summarise_i32` over
   1M elements took 30.7 s, over 5M took 148.1 s, i.e. ~29 us/element, roughly three orders
   of magnitude off a native loop (`probe_e`). A full pass over the 15.1M-pair arrays costs
   ~7 minutes. Consequence for the port: **the dataset should be read with a bulk path** (a
   runtime/binding helper that maps file bytes to an integer view, or a `.vyb` array-typed
   bulk read) rather than element-wise `Vec.get` in Vyb code. Until that exists, V1 validation
   runs single-pass reductions and accepts the runtime; production (DES/CUDA) must not depend
   on element-wise decoding.

### Verified Vyb constraints found while building the M2 DES (2026-09-17, same build)

Probes: `src/vyb/probes/probe_m2_*.vyb`, indexed with outcomes in `src/vyb/probes/README.md`.

5. **`Vec<Vec<T>>` is unusable.** get -> push -> set write-back dies two ways (LLVM
   `ICmpInst::AssertOK` core dump; `free(): double free detected in tcache 2` at exit for the
   minimal slice), `Vec<Vec<String>>` segfaults, `outer.get(i).get(0)` returns **garbage** (a silent
   wrong answer), `outer.get(i).push(x)` silently writes into a temporary, and `outer[i].push(x)` is
   a semantic error. Every bucketed structure must therefore be a structure-of-arrays: per-bucket
   chain heads (`Vec<Int>`) over one flat event arena. This is what `des.vyb`'s calendar queue does,
   and it is also the layout the CUDA backend needs, so the constraint cost us nothing in the end.
6. **Caller state is only mutable through a `their<T>` borrow** (confirmed: `Vec` push/set and
   struct scalar fields all reach the caller, including across a module boundary for a module-defined
   `share(all)` struct). A bare `Vec` parameter is a copy; passing an owned value to a `their<T>`
   parameter requires `borrow(...)` at the owning call site.
7. **Re-borrowing an existing `their<T>` segfaults**: `f(borrow(v), x)` where `v<their<T>>` is
   already a borrow (probe_m2_i, rc=139); passing `v` straight through works (probe_m2_k). Rule:
   borrow once at the owner, propagate afterwards.
8. **`ptr` is reserved and is rejected as a struct field name** ("Expected field name in struct ..."),
   with no mention of the offending token (probe_m2_h). Struct fields must also be
   **comma-separated** (probe_m2_g).
9. **`String.to_int` / `to_float` do not exist on this build** despite `docs/refman/language.md`
   (probe_m2_c); `split`, `trim`, `starts_with`, `contains` do work. Applications ship their own
   integer parser.
10. A `while (a && b)` condition with an early-exit body (the `lif_leak` shape) compiles and runs
    correctly (probe_m2_j), and `time::time_mono_millis` + `io::open_write/write_str/read_all` are a
    complete timing/result-emission surface (probe_m2_d).

## Milestones and state

* **V0** - `scripts/export_bin.py` emits the flat binary. **DONE** (449.6 MB, 18 arrays).
* **V1** - Vyb loads the dataset and reproduces the loader-level invariants
  (`src/vyb/phase0_counts.vyb`), checked against the Python reference by
  `scripts/vyb_v1_check.py`. **DONE**: 26/26 invariants match exactly (neuron and pair
  counts, synapse total, the threshold-5 counts, CSR row pointers and degree statistics, and
  whole-array sums for the pre/post/root-id arrays - the root-id sum matches even though it
  overflows int64, so both sides wrap identically). Result: `results/phase0/vyb_v1_check.json`.
  Runtime ~28 min for the full scan, dominated by the decode throughput in constraint 4.
* **V2** - Vyb computes components (SCC/WCC), reciprocity, clustering, sampled BFS path
  statistics. Gate: match the oracle within stated tolerances.
* **V3** - Vyb DES engine (§8): Entity/State/Event/Scheduler, LIF neuron state in
  structure-of-arrays, validated against the Phase 1 Python LIF reference. **DONE** (M2):
  `src/vyb/des.vyb` (bucketed calendar-queue runtime) + `src/vyb/phase2_des.vyb` (LIF application),
  gate `scripts/vyb_des_check.py` against the fixture from `scripts/make_tiny_net.py`.
  Result: `results/phase2/des_summary.json` and `results/phase2/vyb_des_check.json` - 4/4
  replications match spike-for-spike (771,379 spikes over 22,766,285 events in the primary
  replication, 0 mismatched entities, identical FNV hashes), 6.6-7.0M events/s measured.
* **V4** - bucketed CUDA/NVPTX event scheduling (§9, §23), validated against V3.

Phase 1 (LIF baseline) and Phase 2 (Vyb DES) can run in parallel with V1/V2: they consume the
same canonical dataset.

## Reproducing the Vyb side

```
cd ~/Projects/VybFly
VYB_STDLIB=~/Projects/Vyb/stdlib ~/Projects/Vyb/build/vyb \
    src/vyb/phase0_counts.vyb --module-path src/vyb > results/phase0/vyb_counts.txt
python scripts/vyb_v1_check.py
```

M2 (DES) gate - fixture, runtime, application, independent oracle:

```
cd ~/Projects/VybFly
python3 scripts/make_tiny_net.py                     # data/processed/tiny_net/*.csv + meta.json
VYB_STDLIB=~/Projects/Vyb/stdlib ~/Projects/Vyb/build/vyb \
    src/vyb/probes/probe_m2_f_des_smoke.vyb --module-path src/vyb     # 42 unit checks
VYB_STDLIB=~/Projects/Vyb/stdlib ~/Projects/Vyb/build/vyb \
    src/vyb/phase2_des.vyb --module-path src/vyb > results/phase2/vyb_des_stdout.txt
python3 scripts/vyb_des_check.py                     # -> results/phase2/vyb_des_check.json
```

`scripts/vyb_des_check.py --reference-only` runs the oracle alone (fast, useful when tuning the
fixture with `scripts/make_tiny_net.py --neurons/--ticks/--e-weight/...`).


Probes (each is a standalone program; `probe_c` needs `--module-path src/vyb/probes`):

```
VYB_STDLIB=~/Projects/Vyb/stdlib ~/Projects/Vyb/build/vyb src/vyb/probes/probe_a_inline_read.vyb
VYB_STDLIB=~/Projects/Vyb/stdlib ~/Projects/Vyb/build/vyb src/vyb/probes/probe_c_module_buffer_boundary.vyb --module-path src/vyb/probes
```

## Status on the current upstream build (re-checked 2026-09-17)

Every constraint above was re-run against a fresh build of `main@51ced31` (Vyb 0.7.5, Debug) in a
scratch checkout, and the ones that still reproduce are now filed on the Vyb tracker. Three probes had
gone stale and were replaced (`probe_m3_*`), so the table below reflects what actually happens today,
not what happened on the 2026-09-12 build:

| constraint | probe | on main@51ced31 | filed |
|---|---|---|---|
| 1 module-boundary `Vec<UInt8>` | `probe_c_module_buffer_boundary.vyb` | **reproduces** (right length, garbage contents) | #281 |
| 2 optional `Vec` return | `probe_m3_vec_return_shapes.vyb` (new) | **reproduces** (cast error, then core dump) | #282 |
| 3 non-mutating `Vec` parameter | `probe_m3_vec_param_copy.vyb` (new) | **reproduces** (silent no-op) | #283 |
| 4 decode throughput | `probe_e_throughput.vyb` | not re-measured (see below) | pending |
| 5 `Vec<Vec<T>>` | `probe_m2_b1/b4/b6/b7` | **reproduces** (LLVM assert, double free, silent garbage, SIGSEGV) | #284 |
| 6 caller state via `their<T>` | `probe_m2_a`, `probe_m2_e*` | still correct (no defect) | - |
| 7 re-borrowing a `their<T>` | `probe_m2_i_nested_borrow.vyb` | **reproduces** (segfault, rc=139) | #285 |
| 8 `ptr` as a field name | `probe_m2_h_struct_field_ptr.vyb` | **reproduces** (message names no token) | #286 |
| 9 `String.to_int`/`to_float` | `probe_m2_c_struct_vec_parse.vyb` | **reproduces** (documented, not implemented) | #287 |
| 10 timing/io surface, `while (a && b)` | `probe_m2_d`, `probe_m2_j` | still correct (no defect) | - |

Also re-checked: the parenthesised-operand front-end bug (`X + Y * (Z)` mis-typing the left operand,
`src/vyb_kernels/probes/paren_operand_bug.vyb`) **no longer reproduces** on main@51ced31 — the probe now
compiles and runs on the same shape matrix that used to fail, so it is recorded as fixed upstream.

## Open items to raise upstream (Vyb repo)

Filed 2026-09-17, each with a minimal probe and the observed output:

* #281 - module-boundary `Vec<UInt8>` returns garbage contents (silent data corruption);
* #282 - an optional `Vec` return type cannot return a `Vec` (internal cast error, then a core dump);
* #283 - mutating a plain `Vec` parameter is a silent no-op for the caller;
* #284 - `Vec<Vec<T>>` is unusable (silent garbage on a chained get, SIGSEGV, heap corruption);
* #285 - re-borrowing an existing `their<T>` (`f(borrow(v), x)`) segfaults;
* #286 - `ptr` is rejected as a struct field name without naming the offending token;
* #287 - refman documents `String.to_int`/`to_float`, which the build does not implement.

Still unfiled, deliberately: the element-wise decode throughput (constraint 4). The 2026-09-12
measurement (~29 us/element, ~34k elements/s) is what makes a full 15.1M-element pass cost ~7 minutes,
but upstream `main` is moving fast right now and the number should be re-measured on the next build
before it is filed as an issue — an out-of-date performance claim is worse than none.


## 100x generation on the GPU (Vyb + NVPTX, 2026-09-18)

`src/vyb_kernels/upscale_kernel.vyb` (kernels) and `src/vyb_kernels/upscale100.vyb` (host runner)
generate the 100x connectome entirely on the device: occupancy grid -> per-neuron local spacing
(Newton cube root, unrolled) -> child placement -> density raster -> coordinate streaming. Python is
not in the loop; `scripts/verify_upscale.py` only *checks* what the kernels wrote.

### What the kernels found (all reproduced with a one-kernel probe)

| Device feature | Verdict on this box (RTX 3090, Vyb 0.7.5) |
| --- | --- |
| `st_f32` | **faults the device.** Launch returns 0, the next `cuCtxSynchronize` returns 716 (`INVALID_PC`), and the context is poisoned for every later call. `probe_stf32` in the kernel file is the repro. Use `st_f64` (proven by the phase-3 LIF kernel) or `st_i32`. |
| array `atomic_add_i32` (`cell_count + idx*4`) | **faults the device** once the index is actually in range. Scalar `atomic_add_i32` (issue #273's neighbour) is fine, and array `atomic_add_f64` works, so the histogram cell counts are f64. |
| array `atomic_add_f64` | fine (same pattern as the phase-3 synaptic accumulator). |
| rolled `while` loops with f64 math and a large integer `%` | fine (probe_loop_f64 sums 4096 x 12 uniforms to 24,567.5 against an expected 24,576). |
| `ld_f32` on a bulk buffer | fine. |
| bulk `cuMemcpyHtoD_v2`/`DtoH_v2` | fine and fast: read a 60 MB CSR file with `fread` into `malloc`, one call per array. The per-element `cuda_write_i32` loop the phase-3 runner uses would take hours at this size. |
| 4-byte scalar readback | **misleading.** A 4-byte `cuMemcpyDtoH_v2` into an 8-byte slot leaves garbage in the upper half: a counter of 2,700,513 printed as -4,292,266,783. Allocate 8 bytes for the device counter and copy 8 bytes back. |
| rolled 12-step loop seeded by `% 4294967296` | fine, but note the value must be *stored* with the same width it is read back with. |
| `from<loc<CVoid>>(0)` / `addr()` / `loc()` | only valid inside `freedom {}`; nesting a helper that itself contains `freedom` fails, so bulk IO is inlined in the runner. |
| extern declarations | need `share(all)` on the line before, and a `extern "C" { ... }` block; `fopen`/`fread`/`fwrite`/`malloc` keep the executable free of `io`'s `open` symbol (importing `io` in a `--build` executable segfaults inside `cuInit`). |

### Unit conventions that cost real time

* The canonical `neurons.coords.f32` records are **6 floats per neuron** (`pos_xyz`, `soma_xyz`), so
  the stride is **24 bytes** and positions are the first three. Reading with stride 12 walks into the
  soma triples and off the end of the buffer: that produced 7.5% NaN families and families anchored
  to the wrong cells before it was caught.
* Coordinates are in **nanometres** (x 21,906..225,720), not micrometres. A "sanity" bound of 1e5 nm
  silently classified 62% of the brain as invalid.
* Descriptor floats travel as **micro-units** (value x 1e6), so a 28.076 nm pixel is passed as
  28,076,000 - passing 28,076 puts every soma millions of pixels off the canvas, and the raster then
  sums to zero while every launch reports success.

### Verification without a Python oracle

`threshold_count` reproduces 2,700,513 pairs at the 5-synapse rule from the raw CSR (the published
convention), `coord_checksum` counts 139,241 usable somata and rejects exactly the 14 without a
position, and each scale's raster must sum to `n_parents * c` exactly. Those are checks against
values the project already published, not a second implementation.

### Root cause (found on upstream main 5167646, 2026-09-18)

32-bit device stores and atomics are **lowered as 64-bit** operations. Minimal repros and the exact
PTX are in `src/vyb_kernels/probes/probe_width.vyb`; filed upstream as
[Vyb #301](https://github.com/rickenator/Vyb/issues/301), because the consuming project exists to
find exactly this class of defect:

```
st_f32(out + i*4, 1.5)      ->  mov.u64 %rd21, 4609434218613702656 ; st.global.u64 [%rd23], %rd21
atomic_add_i32(buf + i*4, 1) -> atom.global.add.u64 %rd28, [%rd27], 1
st_i32(out + i*4, 7)         ->  two st.global.u32 halves (one 8-byte store split in two)
```

Expected: `st.global.f32` / `atom.global.add.u32` / one `st.global.u32`. The 64-bit lowering is
harmless when the destination really is an 8-byte slot (which is why a *scalar* `atomic_add_i32`
counter survives) and destructive when it is not: an array atomic writes past the last cell, and an
`i32` array written element-wise is clobbered by its own left neighbour - which is precisely what
turned every sigma entry except the first into zero.

Controls that lower correctly and stay green: `st_f64` -> `st.global.u64`, `atomic_add_f64` ->
`atom.global.add.f64`.

### The one that actually blocked the deliverable

| Device feature | Verdict |
| --- | --- |
| `1.0 * <Int loaded from memory>` inside a kernel (descriptor word, or an element of an `i32` array) | **evaluates to 0 in the arithmetic.** The load itself is fine and the host sees the right value, but the Int-to-Float conversion in the device expression does not take effect: `s = 1.0 * sigma_const / 1000.0` gave `s = 0`, so 98.6% of the 13.9M children landed exactly on their parents while every launch reported success. Substituting a source literal (`s = 510.3`) fixed it instantly and the placement statistics became textbook (mean displacement 816 nm = 1.6 sigma). The same pattern explains an earlier failure in the per-cell spacing path, which also converted a loaded `i32`. |

The placement therefore carries a pinned literal for the spread. That value is not hand-waved: the
runner derives it on the host from the cloud's second moments (spacing 850.6 nm, sigma 510.3 nm) and
asserts the pin matches within 1 nm on every run, so it cannot drift silently.

### Open item: the sibling-damping edit in the Python reference

`src/flyscale/renorm.py` scales the sibling-damping rate beyond the calibrated factor
(`prob_scale * min(1, 9 / (c - 1))`), so the local-synapse budget stays constant at c=100 instead of
inflating mean degree from ~19 to ~30. By construction the multiplier is exactly 1 at c = 2, 5 and
10, so the published ladder must be unchanged - but that has **not** been demonstrated by a run yet:
two verification attempts produced an empty `scales` block (one run killed early, one exited without
computing its scale). The 100x path does not use this code at all (generation is on the GPU), so
nothing shipped depends on it. Verify or revert before the Python upscale is used again.
