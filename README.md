# FlyScale

### 📄 **[Read the paper → aniviza.com/paper.pdf](https://aniviza.com/paper.pdf)** &nbsp;·&nbsp; **[Landing page](https://aniviza.com/paper.html)**

*25 pages, 5.7 MB. GitHub's in-page preview can't display this PDF, so the links above serve the
file directly. You can also browse [`paper.pdf`](paper.pdf) in the repo, or open it after
`git clone`. A counted mirror, which logs how the paper is read, lives at
[`vybfly-paper-relay.aniviza.workers.dev/paper.pdf`](https://vybfly-paper-relay.aniviza.workers.dev/paper.pdf).*

A 25-page study of the adult *Drosophila* whole-brain connectome: **discrete event simulation** of
the FlyWire map on a Vyb-native runtime, scaled across a 10× ladder — and an honest test of whether
"bigger" still behaves like a brain.

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

### The Vyb CPU ladder: whole connectome, 1x to 10x, bit-exact

`src/vyb/` also contains three multithreaded CPU engines that run the full `sparse_prop` LIF
workload on the 1x–10x ladder graphs. The workload is dt 0.5 ms, with 8 epochs × 1000 steps
= 4 s of biological time.

- `ladder_prop` uses per-(thread, block) edge bins. It is the fastest engine at 1x and 2x.
- `ladder_seg` pulls along static per-row target segments.
- `ladder_desc` streams 4-byte segment descriptors that are built once per step. It is the
  fastest engine at 5x and 10x.

Each engine reproduces the spike-set digest of the deterministic C++ reference
(`scripts/cpu/sparse_prop_cpu.cpp`) **bit for bit**, at every scale and at every thread count
tried (1, 4, 8, 10, 12, 16, 20). All three were re-verified on stock Vyb 0.7.7.

The table below was measured on an i9-10850K (10C/20T) at 16 threads, using simulation time and
the best engine per scale. The baseline is the first Vyb engine (`1928e58`). The full numbers
are in `results/energy/vyb_ladder.json`.

| scale | synaptic events | baseline | now | events/s | wall s per bio-s | RSS | C++ ref, 1 thread | GPU kernel (RTX 3090) |
|---|---|---|---|---|---|---|---|---|
| 1x  | 1.05 G  | 1.55 s  | **1.21 s** (prop)  | 870 M | 0.30 | 146 MB | 8.32 s  | 3.75 s  |
| 2x  | 2.00 G  | 4.16 s  | **2.60 s** (prop)  | 767 M | 0.65 | 193 MB | 14.97 s | 3.15 s  |
| 5x  | 5.22 G  | 13.11 s | **7.50 s** (desc) | 696 M | 1.88 | 363 MB | 43.45 s | 5.74 s  |
| 10x | 10.69 G | 31.03 s | **16.59 s** (desc) | 644 M | 4.15 | 667 MB | 88.06 s | 10.68 s |

The 1x and 2x "now" values were measured with a Vyb build that includes the JIT changes in the
open Vyb PR (inlining at `-O2`, a host-CPU target machine). The 5x and 10x values were measured
on stock Vyb 0.7.7, after `ladder_desc` moved to 4-byte descriptors. The JIT changes do not
matter here.
A clean back-to-back rerun on an idle machine (2 repetitions per engine and scale, both builds
interleaved) gave identical digests on stock Vyb 0.7.7, and the best engine at each scale ran
within about 2% of the PR build: 1.22 s at 1x, 2.60 s at 2x, 7.72 s at 5x and 17.97 s at 10x
(those last two are with the old 8-byte descriptors).
Only `ladder_seg` gained, by 2–3%. The compiler friction this work turned up is in
`docs/DOGFOODING.md`.

### The Vyb GPU engine: bit-exact, 2.4x to 4.8x the CUDA kernel on the same GPU

`src/vyb/ladder_gpu.vyb` runs the same ladder workload on an NVIDIA GPU through the CUDA driver
API, with device kernels written in Vyb (`src/vyb_kernels/ladder_gpu_kernel.vyb`). It matches
the C++ reference **bit for bit** at every scale: the spike count, synaptic events, epoch-0
spikes and both digests. The existing CUDA kernel (`sparse_prop.cu`) does not, because it
accumulates with float `atomicAdd`.

The engine is exact without float atomics:

- The push kernel only sets bits. A warp per active source sets one bit per out-edge in a
  bitmap over CSC positions, which are each target's in-edges sorted by source.
- The update kernel uses one thread per neuron. It walks its CSC range in ascending order and
  sums the weights in exactly the reference order, keeping 8 weight loads in flight per round.
  Most weights come from a 1-byte code instead of a 4-byte float (see below). It then
  integrates and fires, and appends spikes warp-aggregated.

| scale | Vyb GPU engine (RTX 3090) | GPU kernel (RTX 3090) | Vyb GPU engine (GB10) | `sparse_prop.cu` (same GB10) | Vyb CPU, 20 Arm cores (GB10 host) | C++ ref, 1 thread |
|---|---|---|---|---|---|---|
| 1x  | **0.98 s** | 3.75 s  | **0.77 s** | 3.67 s (inexact)  | 0.96 s (prop, 16 threads) | 8.32 s |
| 2x  | **0.90 s** | 3.15 s  | **0.93 s** | 3.04 s (inexact)  | 1.88 s (prop, 16 threads) | 14.97 s |
| 5x  | **1.71 s** | 5.74 s  | **2.22 s** | 5.29 s (inexact)  | 4.25 s (prop) | 43.45 s |
| 10x | **2.46 s** | 10.68 s | **4.17 s** | 10.29 s (inexact) | 7.52 s (prop) | 88.06 s |
| 20x | **4.04 s** | | **7.45 s** | | | 174.4 s |
| 40x | **8.72 s** | | **14.71 s** | | | 320.6 s |

Every Vyb GPU number above matches the C++ reference digests exactly. The 3090 times are the
best of 3 runs on an idle GPU (Godzilla). The GB10 is a DGX Spark with 273 GB/s of unified
memory. Its 1x to 10x times are the best of 3 with the GPU otherwise idle, and 20x and 40x are
the best of 2. Earlier runs beside a resident LLM server were no slower (10x best 3.94 s), so
the GB10 engine is bandwidth-bound, not contention-bound. At 10x, nsys puts the step at about
410 us of update plus 110 us of push, close to the memory-bandwidth floor of about 75 MB per
step. On the 3090 (936 GB/s), 1x is slower than 2x because the 8,000 steps of two launches
each are launch-latency-bound at that size. The 20x and 40x GB10 times here predate the
u8 weight codes and the stimulus-flag bitmap: with them the same 8-epoch 40x run is
**12.77 s** on one GB10 and **6.59 s** on the two-Spark RDMA split (see the u8-codes
section below), still bit-exact. These GB10 times are from before the u8 weight codes
described below; with them the GB10 takes 0.85, 0.90, 1.84, 3.34, 6.45 and 12.77 s at 1x to 40x.

The 20x and 40x graphs come from `scripts/` the same way as the ladder (n = 2,785,100 and
5,570,200; 56.0M and 112.3M edges). Their C++ reference runs `sparse_prop_cpu --max-active N`
with N = n, because the default 300,000-entry active list overflows at 20x and the reference
then keeps the lowest ids, which no parallel engine reproduces. Set `VYBFLY_MAX_ACTIVE` to n
for the same uncapped run in Vyb.

#### Faster GB10 kernels: u8 weight codes

nsys on a GB10 at 40x put a step at 1.31 ms of update and 0.59 ms of push. Update is the hot
kernel. For every touched in-edge it reads a 4-byte f32 weight from a scattered place in the
CSC. The weight is f32(gain * syn / t), where syn is the edge's synapse count and t is the
target's total input. Now each edge also has a 1-byte code c, which is its syn (1 to 255),
and each target has an f32 pair (hi, lo) with hi + lo close to gain / t. The kernel computes
`fma(c, hi, c * lo)` in f32. The host keeps a code only when it has proved, edge by edge,
that this gives exactly the reference f32 weight:

- in double, c * hi is exact, and so is c * lo before it is rounded to f32;
- a TwoSum check shows the double sum is exact;
- the result equals the stored weight.

Otherwise the code is 0, and the kernel reads the f32 weight as before. 1.4% of the edges
escape. The weights, the summation order and every digest are unchanged.

Push now also sets one bit per stimulated target in a small flag bitmap. Update then reads
one 32-bit word per warp instead of each neuron's stimulus count. Times are 8 epochs, best of
2, old and new kernels interleaved in one session, and every run matches the C++ reference
digests:

| 8 epochs, GB10 | before | u8 codes | change |
|---|---|---|---|
| 1x | 0.81 s | 0.85 s | +5% |
| 2x | 0.93 s | **0.90 s** | -3% |
| 5x | 2.21 s | **1.84 s** | -17% |
| 10x | 4.26 s | **3.34 s** | -22% |
| 20x | 7.49 s | **6.45 s** | -14% |
| 40x | 14.59 s | **12.77 s** (12.51 s in an earlier run) | -12% |
| 40x, two GB10s, RDMA + zero-copy | 7.87 s | **6.59 s** | -16% |

At 40x, nsys shows update falling from 1.31 to 1.01 ms per step, while push stays at
0.58 ms. At 1x the working set fits in L2, so the extra arithmetic costs more than the bytes
it saves. A few other changes did not help:

- Push is bound by about 10.8M random bit-ORs per step. Giving each source 16 or 8 lanes
  instead of a warp was slower.
- Sweeping the block size (128 to 512) and push blocks (1024 to 4096) gained 1.5% at best,
  so the defaults stay.
- Fusing push into update would need a grid-wide sync between them.
- Computing the weights in f64, which is exact by construction, made update 2.5x slower,
  because the GB10's FP64 rate is low. So everything stays f32.

Proving the codes adds host load time (40x: 1.8 s to 2.9 s). `VYBFLY_GPU_W8=0` turns the
codes off: every edge escapes, which is exact but slower than the old kernel.

#### Two Sparks, one network: a bit-exact split

`src/vyb/ladder_gpu_split.vyb` runs one connectome across both Sparks. Neurons are cut by
target at half the in-edges. Each rank keeps every source row, but only the edges into its own
neurons, so a step needs nothing from the peer except the list of neurons that fired. After
each step the ranks swap fired-id lists on the direct 200 GbE link, over TCP or with RDMA
writes (`VYBFLY_SPLIT_NET=rdma`), then push the peer's spikes into their own targets. It is bit-exact at 10x, 20x and 40x (digests equal to the C++
reference), and with `VYBFLY_SPLIT_PARTS=1` it is the single-GPU engine.

| 20x, 8 epochs | sim | GPU time per rank | network |
|---|---|---|---|
| one GB10 (`ladder_gpu`) | 7.45 s | 7.45 s | none |
| two GB10s, serialized exchange | 8.85 s | ~3.7 to 4.1 s | ~3.4 s |
| two GB10s, full-duplex exchange | 7.51 s (profiled), 6.79 s | ~3.7 to 4.1 s | ~3.1 to 3.5 s |
| two GB10s, full-duplex, own push overlapped with the exchange | 6.64 s, 7.30 s, 7.59 s | ~3.7 to 4.1 s | ~2.7 to 3.2 s (sim minus GPU) |
| two GB10s, RDMA exchange (`VYBFLY_SPLIT_NET=rdma`), overlap on | **4.54 s**, 4.57 s | ~3.3 to 3.7 s (GPU plus copy-back) | ~0.24 to 1.08 s |

At 40x the split wins clearly. Two GB10s (duplex and overlap on) take **11.35 s** (best of 2;
the other run took 12.29 s) against 14.71 s on one GB10. That is 1.30x faster, and again
bit-exact. The GPU work per step doubles from 20x, but the exchange cost grows less.

The split halves the GPU time, but the exchange costs about 0.35 ms per step. A ping-pong test
over the same link shows why: 36 us for 8 bytes with busy-polling, but about 475 us for 240 KB
each way, and about 800 us with blocking sockets. Socket buffers are capped at 212 KB
(`net.core.wmem_max`) without root. So over TCP two Sparks only just beat one at 20x, beat it by 1.30x at 40x, and lose at
10x (6.09 s against about 4.2 s, measured with the serialized exchange). The overlap's gain is within
run-to-run noise, since the GPU push is short next to the exchange. The overlap and the duplex
exchange are on by default (`VYBFLY_SPLIT_OVERLAP`, `VYBFLY_SPLIT_DUPLEX`). Both only reorder
bit-ORs and integer sums, so they cannot change a result.

RDMA removes most of the exchange cost, and it needs no root. The ConnectX ports expose RoCE v2
devices that an ordinary user can open (`/dev/infiniband/uverbs*` is world-writable and the
memlock limit is unlimited). With `VYBFLY_SPLIT_NET=rdma` each rank registers its send buffer and
a two-slot receive buffer, swaps queue-pair details over the TCP connection, then posts one RDMA
write-with-immediate per step into the peer's slot, with the id count in the immediate.
libibverbs is opened with `dlopen` at run time, so a TCP run needs nothing new. At 20x this takes
**4.54 s** (best of 2; the other run took 4.57 s), against 7.59 s over TCP in the same session and
7.45 s on one GB10. That is 1.64x faster than one GB10, and bit-exact. A ping-pong written in Vyb
over the same verbs takes 2.3 us for 16 bytes and 87.5 us for 1 MB each way (12 GB/s), against
100 to 230 us and 355 to 454 us over TCP. Most of the remaining exchange time is the copy of the
fired list back to the host.

At 10x, 40x and 80x RDMA pays off as well. These are 8 epochs, two runs each, all bit-exact (10x
42322697548985024/188222298830040, 40x 239142599791732164/1733953483748172, 80x
508755035752662158/103693891554914; 80x is 11.1M neurons and 225M edges, every neuron allowed active):

| 8 epochs | one GB10 | two GB10s, TCP | two GB10s, RDMA | RDMA vs one GB10 |
|---|---|---|---|---|
| 10x | 4.19 s | 6.09 s (serialized) | **2.61 s**, 2.63 s | 1.60x |
| 20x | 7.45 s | 7.59 s | **4.54 s**, 4.57 s | 1.64x |
| 40x | 14.71 s | 11.35 s | **8.05 s**, 8.18 s | 1.83x |
| 80x | 34.05 s, 34.10 s | not run | **15.60 s**, 15.62 s | 2.18x |

So with RDMA two Sparks now beat one at every scale tried, including 10x, where TCP lost, and the
gain grows with scale. At 80x the wire itself is 0.9 to 1.7 s of the 15.6 s per rank. An earlier
version of this note put most of the rest down to a 12.3 to 12.6 s fired-list copy back to the host.
That was wrong: the copy ran on the legacy stream, so `dtoh_ms` also counted the wait for that
step's GPU work. The copy itself is about 2 to 3% of the step.

The zero-copy exchange (`VYBFLY_SPLIT_ZC`, default 1, used with two ranks and the overlapped push)
removes even that. The GB10's GPU shares the host's memory, so the send and receive buffers are pinned in place
with `cuMemHostRegister` and the kernels use them directly. The update kernel writes the fired list
straight into the RDMA send buffer, and the push kernel reads the peer's list straight from the
receive buffer. Only an 8-byte count comes back per step. GPUDirect RDMA from device memory is not
available here: the driver reports no dma-buf and no GPUDirect RDMA support, and registering a device
pointer with ibverbs fails without root. On GB10, though, pinned host memory is the same physical
memory, so registering it has the same effect. If pinning fails the engine warns and falls back to
copies (`VYBFLY_SPLIT_ZC=0` forces that). Both modes are bit-exact (8 epochs):

| 8 epochs | ZC off | ZC on |
|---|---|---|
| 10x, RDMA | 2.61 s | **2.51 s** |
| 10x, TCP | 4.75 s | **4.10 s** |
| 40x, RDMA | 8.06 s | **7.81 s** |
| 80x, RDMA | 15.66 s | **15.28 s**, 15.38 s |

At 40x with ZC on, rank 1 spends 7.27 s waiting on the GPU, 0.04 s on copies and 0.48 s on the
network. Against an ideal of 7.35 s, what is left is about 0.46 s of per-step exchange and barrier
latency. Sending a bitmap instead of an id list would cut the bytes only about 1.4x at this firing
rate. At 80x ZC on removes the copies (`dtoh_ms` falls from 12.3-12.5 s to 0.04 s per rank) but
gains only about 2%: each rank then waits 13.3-14.4 s on the GPU, so the 80x step is bound by the
kernels themselves, not by the transfer. The three 80x runs above were back to back and all
bit-exact.

Run it, rank 0 first, on two machines that share the graph files:

```sh
# on the rank-0 host (it listens on VYBFLY_SPLIT_PORT)
VYBFLY_SPLIT_PARTS=2 VYBFLY_SPLIT_RANK=0 scripts/gpu/run_split_rank.sh <ladder>/s20 <ladder>/stim_s20.bin
# on the rank-1 host
VYBFLY_SPLIT_PARTS=2 VYBFLY_SPLIT_RANK=1 VYBFLY_SPLIT_PEER=<rank-0 ip> \
  scripts/gpu/run_split_rank.sh <ladder>/s20 <ladder>/stim_s20.bin
# for the RDMA exchange, set VYBFLY_SPLIT_NET=rdma on both hosts (VYBFLY_RDMA_DEV and
# VYBFLY_RDMA_GID pick the RoCE device and GID index; defaults rocep1s0f0 and 3)
```

On the Arm host, `ladder_prop` also beats `ladder_desc` at 5x and 10x, the opposite of the i9. `scripts/cpu/run_ladder.sh` therefore picks `ladder_prop` on aarch64.

Run it:

```sh
VYBFLY_ENGINE=gpu VYBFLY_LADDER_DIR=<ladder>/s10 VYBFLY_LADDER_STIM=<ladder>/stim_s10.bin \
  VYBFLY_EPOCHS=8 VYB=vyb scripts/cpu/run_ladder.sh --module-path "$VYB_STDLIB"
```

This works on stock Vyb 0.7.7, because the kernel is committed as PTX (sm_86, which the driver
JIT-compiles for newer GPUs). Regenerating the PTX needs the kernel intrinsics in Vyb PRs
`perf/kernel-atomics-bits` and `perf/kernel-friction` (which adds `shfl_xor_i64`):

```sh
vyb --kernel src/vyb_kernels/ladder_gpu_kernel.vyb --gpu sm_86 --ptx src/vyb_kernels/ladder_gpu_kernel.ptx
```

Further knobs are `VYBFLY_GPU_BLOCK` (256), `VYBFLY_GPU_PUSH_BLOCKS` (2048),
`VYBFLY_GPU_W8` (1, the u8 weight codes), `VYBFLY_GPU_DEVICE` (0) and `VYBFLY_MAX_ACTIVE` (300000, the active-list capacity).

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

The CPU ladder engines read a ladder graph's CSR (`results/energy/curves/s<N>/`, which the
energy phase produces) and a stimulus file that the C++ reference writes:

```sh
g++ -O3 -march=native -ffp-contract=off -o sparse_prop_cpu scripts/cpu/sparse_prop_cpu.cpp
./sparse_prop_cpu --dir results/energy/curves/s10 --epochs 8 --steps 1000 --syn-gain 6 \
    --refr-steps 20 --dump-stim stim_s10.bin                    # prints the reference digest
VYBFLY_LADDER_DIR=results/energy/curves/s10 VYBFLY_LADDER_STIM=stim_s10.bin \
VYBFLY_THREADS=16 VYBFLY_EPOCHS=8 \
    $VYB --module-path $VYB_STDLIB src/vyb/ladder_desc.vyb      # or ladder_prop / ladder_seg
```

`scripts/cpu/run_ladder.sh` picks the engine for you with the same environment variables. It
chooses `ladder_prop` below 10 M edges (1x and 2x) and `ladder_desc` at or above that (5x and
10x), which matches the measured crossover. Set `VYBFLY_ENGINE` to force a particular engine:

```sh
VYBFLY_LADDER_DIR=results/energy/curves/s10 VYBFLY_LADDER_STIM=stim_s10.bin \
VYBFLY_THREADS=16 VYBFLY_EPOCHS=8 VYB=$VYB \
    scripts/cpu/run_ladder.sh --module-path $VYB_STDLIB
```

Compare the `digest_sum` and `digest_xor` values on the engine's `RESULT` line against the
reference's output; they must be equal.

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
| M2b Vyb multithreaded CPU ladder (1x–10x) | **done** — bit-exact digests vs the C++ reference; 10x runs at 4.15 s wall per bio-s |
| M3 Vyb GPU engine (ladder workload) | **done** — bit-exact digests, spike counts and synaptic events against the C++ reference at 1x–40x on an RTX 3090 and 1x–10x on a GB10; 2.4x–4.8x the CUDA `sparse_prop.cu` kernel on the same GPU, without float atomics. (The older 512-neuron fixture agreement is the M3 harness, not the ceiling.) |
| M3b two-Spark split | **done** — one connectome across both Sparks, bit-exact at 10x, 20x, 40x and 80x; 2.18x over one GB10 at 80x with RDMA (15.60 s vs 34.05 s), 15.28 s with the zero-copy exchange, which is then bound by the kernels rather than the wire |
| M4 energy instrumentation | **done** — GPU device power measured; CPU power unavailable on this machine |
| M5–M8 geometry, downscaling, upscaling, closure | **done** — see section 5 of the paper |
| M9 dynamic validation of the replicas | **done** — 85–95% active-set overlap |
| M11–M14 learning, capability, energy scaling, capability per watt | **done** — including the negative control result |
| M15 100× ~14M neurons | **graph done, propagation not attempted** — the 100× graph itself exists and is verified (`results/upscale100/`: 13,925,500 child records, 0.000000% on their parent, placement re-run after #301 with the workaround pin removed — `docs/VYB-PORT.md` §100x). No ladder run at 100× yet: 80× (11.1M neurons, 225M edges) on two Sparks is the largest simulated scale |

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


## The Vyb family

Vyb is a language with more than one consumer; each project below is written in
it and is a real, running thing rather than a demo.

| project | what it is |
|---|---|
| **[Vyb](https://github.com/rickenator/Vyb)** | the language, the LLVM JIT/AOT compiler and the standard library — [vyb-lang.org](https://vyb-lang.org/) |
| **[VybOS](https://github.com/rickenator/VybOS)** | a Linux-based OS whose declarative, content-addressed configuration and control framework are Vyb; PID 1 is Vyb |
| **VybWin** *(private for now)* | the window environment for VybOS: apps are cards summoned into an arena, with the ledger one flip away |
| **[VybForge](https://github.com/rickenator/VybForge)** | the shop floor: a desired-state interviewer for VybOS, a Vyb-native on-GPU inference substrate, and the self-contained voice loop |
