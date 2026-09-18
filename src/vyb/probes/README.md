# Vyb probes (FlyScale)

Minimal reproductions of compiler/runtime behaviour, run against the pinned build
(`Vyb 0.7.5 (build=Debug, sanitize=none)`, `~/Projects/Vyb/build/vyb`, build dated 2026-09-12).
Each file is a standalone program; the header comment states the question it answers and the exact
command. Keep them: several encode behaviour that costs hours to rediscover.

Run one:

```
VYB_STDLIB=$HOME/Projects/Vyb/stdlib $HOME/Projects/Vyb/build/vyb <probe>.vyb
# probe_m2_e_module_borrow needs --module-path src/vyb/probes
# probe_m2_f_des_smoke   needs --module-path src/vyb
```

## Phase 0 / dataset probes (V1)

| Probe | Question | Result |
|---|---|---|
| `probe_a_inline_read.vyb` | is a byte buffer valid inside the function that read it? | yes, inline decode is correct |
| `probe_b_module_read_chunk.vyb` | chunked reads through a helper | failure mode copy of (c) |
| `probe_c_module_buffer_boundary.vyb` | can a `Vec<UInt8>` cross a module boundary? | **no** - same length, garbage contents |
| `probe_d_loader.vyb` | do `Vec` out-parameters work? | **no** - the callee mutates a copy |
| `probe_e_throughput.vyb` | byte-at-a-time decode cost | ~29 us/element (~34k elements/s) |
| `probemod.vyb` | helper module for the boundary probes | - |

## M2 (DES) probes

| Probe | Question | Result | Consequence for `des.vyb` |
|---|---|---|---|
| `probe_m2_a_borrow_mut.vyb` | can a function mutate the caller's `Vec`/struct fields through a `their<>` borrow? | **yes** (push, set, scalar field writes all reach the caller) | the whole mutating API takes `their<...>` borrows |
| `probe_m2_e_mod.vyb` + `probe_m2_e_module_borrow.vyb` | does that hold for a module-defined `share(all)` struct across a module boundary? | **yes** | `des.vyb` can own `Sched`/`EntityState`/`Recorder` types and mutate caller state |
| `probe_m2_i_nested_borrow.vyb` | is `f(borrow(v), x)` legal when `v` is already a `their<T>` parameter? | **no - segfault (rc=139)** | never re-borrow: propagate the existing borrow (`f(v, x)`) |
| `probe_m2_k_borrow_propagation.vyb` | is passing an existing `their<T>` straight to another `their<T>` parameter OK? | **yes** (both `Vec` and struct-field mutations reach the caller) | the rule is "wrap once at the owning call site, propagate after" |
| `probe_m2_h_struct_field_ptr.vyb` | is `ptr` usable as a struct field name? | **no** - parse error "Expected field name in struct 'B'" | `Recorder.row_ptr`, not `Recorder.ptr` |
| `probe_m2_g_struct_field_names.vyb` | are ordinary field names (`ticks`, `count`, `total`) fine? | yes (also proves struct fields are **comma-separated**) | every struct field ends with `,` |
| `probe_m2_b1_nested_vec_writeback.vyb` | `Vec<Vec<Int>>` get -> push -> set write-back | **LLVM `ICmpInst::AssertOK` assert, core dump (rc=134)** | nested `Vec`s are unusable; buckets are per-bucket chain heads over one flat arena |
| `probe_m2_b4_nested_get_push_set_len.vyb` | minimal slice of the above (no chained call) | values read back correctly, then **`free(): double free detected in tcache 2` at exit (rc=134)** | as above: heap corruption on destruction |
| `probe_m2_b6_nested_chained_get.vyb` | `outer.get(i).get(0)` | **garbage value** (100383503685497 instead of 100) plus the double free | never chain a member call onto a `Vec`-returning call |
| `probe_m2_b7_nested_vec_of_string.vyb` | is `Vec<Vec<String>>` write-back any better? | **no - SIGSEGV (rc=139)** | same conclusion |
| `probe_m2_b3_nested_vec_chained.vyb` | `outer.get(i).push(x)` | no crash, **silently pushes into a temporary** (`inner1_len=0`) | silent-wrong-answer mode; avoid |
| `probe_m2_b2_nested_vec_index_lvalue.vyb` | `outer[i].push(x)` (index lvalue into a nested `Vec`) | **semantic error**: "Cannot determine type of object in member expression" | - |
| `probe_m2_b5_nested_empty_inner.vyb` | seeding `Vec<Vec<Int>>` with empty inner vectors and only reading | clean | a nested `Vec` is only safe while it is never written through |
| `probe_m2_c_struct_vec_parse.vyb` | parse helpers: `String.split/trim/contains`, `to_int`/`to_float` | `split`/`trim`/`contains` fine; **`Method 'to_int' not found for type 'String'`** (and `to_float`) | `phase2_des.vyb` ships its own integer parser |
| `probe_m2_d_clock_fileio.vyb` | monotonic clock + write/read a result file | works (`time_mono_millis`, `open_write`/`write_str`/`read_all`) | throughput timing and JSON emission |
| `probe_m2_j_while_conjunction.vyb` | `while (a && b)` with an early-exit body (the `lif_leak` shape) | works; hand-computed leak values reproduced exactly | `lif_leak` uses the conjunction form |
| `probe_m2_f_des_smoke.vyb` | unit smoke test of the M2 runtime with hand-computed expectations | **42 checks, 0 failures** (`probe-m2-f-des-smoke-PASS`) | run this before `phase2_des.vyb` when touching `des.vyb` |

### Open items for the compiler repo

1. `Vec<Vec<T>>` write-back is memory-unsafe in three different ways (LLVM assert, double free,
   SIGSEGV) and one silent-wrong-answer path (chained `get(i).push`). Highest priority: the silent
   cases, because they produce plausible-looking results.
2. Re-borrowing an existing `their<T>` (`f(borrow(v))` where `v<their<T>>`) segfaults instead of
   being accepted or rejected (probe_m2_i).
3. `ptr` is a reserved identifier but is rejected as a *field name* with a message that does not
   name the offending token (probe_m2_h).
4. `String.to_int` / `to_float` are advertised in `docs/refman/language.md` but missing from the
   build's semantic table (probe_m2_c).
