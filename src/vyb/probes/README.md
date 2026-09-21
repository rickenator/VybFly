# Vyb probes (FlyScale)

Minimal reproductions of compiler/runtime behavior, run against the pinned build.
Each file is a standalone program; the header comment states the question it answers and the exact
command. Keep them: several encode behavior that costs hours to rediscover.

Run one:

```
VYB_STDLIB=$HOME/Projects/Vyb/stdlib $HOME/Projects/Vyb/build/vyb <probe>.vyb
# probe_m2_e_module_borrow needs --module-path src/vyb/probes
# probe_m2_f_des_smoke   needs --module-path src/vyb
```

## ⚠️ Status change: re-verified against Vyb 0.7.6 (2026-09-20)

**Every item in the old "Open items for the compiler repo" list is now FIXED upstream.**
The tables below are kept as a record of what was true on the older builds, but if you
are learning Vyb today, read this section first — several of those probes now teach the
*opposite* of the correct lesson.

| Probe | Was (0.7.5 / main@51ced31) | On 0.7.6 (`b758b686`) |
|---|---|---|
| `probe_m2_b1_nested_vec_writeback` | LLVM `ICmpInst::AssertOK` assert, core dump (rc=134) | **clean — rc=0, correct values** |
| `probe_m2_b4_nested_get_push_set_len` | `free(): double free detected in tcache 2` (rc=134) | **clean — rc=0, no double free** |
| `probe_m2_b7_nested_vec_of_string` | SIGSEGV (rc=139) | **clean — rc=0, correct contents** |
| `probe_m2_b3_nested_vec_chained` | *silent* wrong answer (pushed into a temporary) | **now a compile-time error**, with a message that names the fix |
| `probe_m2_b2_nested_vec_index_lvalue` | semantic error, "Cannot determine type…" | **works** — `Vec<T>` subscripts now resolve their element type |
| `probe_m2_b6_nested_chained_get` | garbage value (`100383503685497`) + double free | **correct (`chained_get=101`), rc=0** |
| `probe_m2_i_nested_borrow` | **segfault (rc=139)** | **compile-time error** with a teaching message (see below) |
| `probe_m2_h_struct_field_ptr` | rejection whose message did not name `ptr` | rejected **by name**: `'ptr' is a reserved word and cannot be used as a field name in struct 'B'` |
| `probe_m2_c_struct_vec_parse` (item 4) | `String.to_int` / `to_float` advertised but missing | **present** — `to_int()` works |

### The two diagnostics worth knowing as a learner

These replaced crashes, and they are the single most useful thing in this directory.
Memorize them — they tell you what to type instead:

**Re-borrowing** (`f(borrow(v))` where `v` is already a `their<T>` parameter):

```
'v' is already a borrow: borrowing it again would borrow a borrow.
Borrow once at the owner and propagate -- pass 'v' directly instead.
```

The rule: **wrap once at the owning call site, propagate after.**

**Writing through a by-value temporary** (`outer.get(i).push(x)`):

```
cannot call 'push' on a by-value temporary: the receiver is a copy, so the write
cannot reach stored state; bind it to a local and write it back with
set(index, value).
```

Read that literally: `get()` hands you a *copy*, so mutating it can never reach your
data. Bind it, mutate it, `set()` it back.

## Phase 0 / dataset probes (V1)

| Probe | Question | Result |
|---|---|---|
| `probe_a_inline_read.vyb` | is a byte buffer valid inside the function that read it? | yes, inline decode is correct |
| `probe_b_module_read_chunk.vyb` | chunked reads through a helper | failure mode copy of (c) |
| `probe_c_module_buffer_boundary.vyb` | can a `Vec<UInt8>` cross a module boundary? | **no** - same length, garbage contents |
| `probe_d_loader.vyb` | do `Vec` out-parameters work? | **no** - the callee mutates a copy |
| `probe_e_throughput.vyb` | byte-at-a-time decode cost | ~29 us/element (~34k elements/s) |
| `probemod.vyb` | helper module for the boundary probes | - |

> Note on (c)/(d): passing a `Vec` to a plain (non-borrow) parameter still hands over a
> copy — that part is language design, not a bug, and `probe_m3_vec_param_copy` documents
> it (`#283`). Use `their<Vec<T>>` when the callee must mutate the caller's vector.

## M2 (DES) probes

| Probe | Question | Result | Consequence for `des.vyb` |
|---|---|---|---|
| `probe_m2_a_borrow_mut.vyb` | can a function mutate the caller's `Vec`/struct fields through a `their<>` borrow? | **yes** (push, set, scalar field writes all reach the caller) | the whole mutating API takes `their<...>` borrows |
| `probe_m2_e_mod.vyb` + `probe_m2_e_module_borrow.vyb` | does that hold for a module-defined `share(all)` struct across a module boundary? | **yes** | `des.vyb` can own `Sched`/`EntityState`/`Recorder` types and mutate caller state |
| `probe_m2_i_nested_borrow.vyb` | is `f(borrow(v), x)` legal when `v` is already a `their<T>` parameter? | ~~no - segfault (rc=139)~~ **0.7.6: rejected at compile time with a clear message** | never re-borrow: propagate the existing borrow (`f(v, x)`) |
| `probe_m2_k_borrow_propagation.vyb` | is passing an existing `their<T>` straight to another `their<T>` parameter OK? | **yes** (both `Vec` and struct-field mutations reach the caller) | the rule is "wrap once at the owning call site, propagate after" |
| `probe_m2_h_struct_field_ptr.vyb` | is `ptr` usable as a struct field name? | ~~no - parse error "Expected field name in struct 'B'"~~ **0.7.6: rejected by name** | `Recorder.row_ptr`, not `Recorder.ptr` (still true) |
| `probe_m2_g_struct_field_names.vyb` | are ordinary field names (`ticks`, `count`, `total`) fine? | yes (also proves struct fields are **comma-separated**) | every struct field ends with `,` |
| `probe_m2_b1_nested_vec_writeback.vyb` | `Vec<Vec<Int>>` get -> push -> set write-back | ~~**LLVM assert, core dump (rc=134)**~~ **0.7.6: works, rc=0** | nested `Vec`s are usable; buckets are per-bucket chain heads over one flat arena |
| `probe_m2_b4_nested_get_push_set_len.vyb` | minimal slice of the above (no chained call) | ~~values read back correctly, then **double free at exit (rc=134)**~~ **0.7.6: clean** | as above |
| `probe_m2_b6_nested_chained_get.vyb` | `outer.get(i).get(0)` | ~~garbage value, plus the double free~~ **0.7.6: correct (`chained_get=101`), rc=0** | chained reads are now safe; note `get()` still yields a **copy**, so chaining a *mutation* is still wrong (see b3) |
| `probe_m2_b7_nested_vec_of_string.vyb` | is `Vec<Vec<String>>` write-back any better? | ~~**no - SIGSEGV (rc=139)**~~ **0.7.6: works, rc=0** | same conclusion as b1 |
| `probe_m2_b3_nested_vec_chained.vyb` | `outer.get(i).push(x)` | ~~no crash, **silently pushes into a temporary**~~ **0.7.6: compile-time error naming the fix** | silent-wrong-answer mode is closed |
| `probe_m2_b2_nested_vec_index_lvalue.vyb` | `outer[i].push(x)` (index lvalue into a nested `Vec`) | ~~semantic error~~ **0.7.6: works** | - |
| `probe_m2_b5_nested_empty_inner.vyb` | seeding `Vec<Vec<Int>>` with empty inner vectors and only reading | clean | - |
| `probe_m2_c_struct_vec_parse.vyb` | parse helpers: `String.split/trim/contains`, `to_int`/`to_float` | `split`/`trim`/`contains` fine; ~~`to_int` not found~~ **0.7.6: `to_int` works** | `phase2_des.vyb` still ships its own parser for the pinned build |
| `probe_m2_d_clock_fileio.vyb` | monotonic clock + write/read a result file | works (`time_mono_millis`, `open_write`/`write_str`/`read_all`) | throughput timing and JSON emission |
| `probe_m2_j_while_conjunction.vyb` | `while (a && b)` with an early-exit body (the `lif_leak` shape) | works; hand-computed leak values reproduced exactly | `lif_leak` uses the conjunction form |
| `probe_m2_f_des_smoke.vyb` | unit smoke test of the M2 runtime with hand-computed expectations | **42 checks, 0 failures** (`probe-m2-f-des-smoke-PASS`) | run this before `phase2_des.vyb` when touching `des.vyb` |

## Re-check probes (2026-09-17, against upstream `main@51ced31`)

Written because three original probes had gone stale (the module or its exports had changed underneath
them). Each isolates one claim from `docs/VYB-PORT.md`, and each was run against a fresh build of
upstream main in a scratch checkout:

| Probe | Question | Result on main@51ced31 | Issue |
|---|---|---|---|
| `probe_m3_vec_return_shapes.vyb` | can a function return `Vec<Int>?` / `Vec<UInt8>?`? | no — internal cast error, then a core dump | #282 (**now CLOSED**) |
| `probe_m3_vec_param_copy.vyb` | does mutating a plain `Vec` parameter reach the caller? | no — silent no-op (`caller_len=0`) | #283 (**now CLOSED**) |
| `probe_m3_optional_vec_return.vyb` | the same shape in the smallest form we could write | `UInt8` unresolved in a return type | (rolled into #282) |

Run them from the repository root:

```
VYB_STDLIB=$HOME/Projects/Vyb/stdlib $HOME/Projects/Vyb/build/vyb src/vyb/probes/probe_m3_vec_return_shapes.vyb
VYB_STDLIB=$HOME/Projects/Vyb/stdlib $HOME/Projects/Vyb/build/vyb src/vyb/probes/probe_m3_vec_param_copy.vyb
```

`probe_b_module_read_chunk.vyb` and `probe_d_loader.vyb` are kept for the record but no longer test what
they were written for: `flyload` no longer exports `read_chunk`, and `probe_d` was rewritten to use a
return value instead of an out-parameter.

## Open items for the compiler repo

**None of the four original items remain open on 0.7.6.** All were re-verified fixed on
2026-09-20 (see the status table at the top). If you hit a new one, add it to this list
*with the build hash*, so the next reader can tell whether it still applies.
