#!/usr/bin/env python3
"""Generate the deterministic "tiny net" fixture for the FlyScale Phase 2 DES gate (PROJECT-VYBFLY.md §8).

The fixture is deliberately small, plain-text and integer-valued so that a Vyb program and an
independent Python reference can both read it and agree *exactly* -- no floating point crosses the
gate, so there is no tolerance to hide behind.

Outputs (default data/processed/tiny_net/):

  neurons.csv        one row per entity, index order == entity id order
  edges.csv          one row per directed synapse, sorted by (pre, post)
  stimulus_s0.csv    a stimulus (external event source) schedule, sorted by tick
  stimulus_s1.csv    a second, independent schedule (used by a different Replication)
  meta.json          counts, units, seeds, parameters, sha256 of every file

Units and semantics (authoritative; mirrored by src/vyb/des.vyb and scripts/vyb_des_check.py):

  time      integer ticks, 1 tick = TICK_MS = 0.1 ms simulated
  voltage   integer micro-units, 1 unit = 1e-6 mV  (so -65000 == -65.0 mV)
  weight    same micro-units; positive == excitatory, negative == inhibitory
  decay_k   per-tick leak factor, k/1000 = exp(-1/tau_ticks), precomputed HERE so that both
            implementations perform identical integer arithmetic (no libm on either side)

Columns are documented in a `#` header block at the top of every CSV, followed by one column-name
line. Blank lines and `#` lines are ignored by both readers.

Determinism: all randomness comes from a hand-rolled 64-bit LCG (not `random`/`numpy`), so the
fixture is reproducible across Python versions. The fixture is the *only* input to both
implementations; neither side runs an RNG of its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixture parameters (the "biology" of the tiny net)
# ---------------------------------------------------------------------------

TICK_MS = 0.1  # simulated ms per tick
GENERATOR_VERSION = "m2-tiny-net-1"

DEFAULTS = dict(
    neurons=640,
    e_fraction=0.80,          # 80 % excitatory, 20 % inhibitory
    min_out_degree=12,
    max_out_degree=48,
    min_delay=1,              # ticks; >= 1 keeps a spike from re-entering the same tick
    max_delay=8,              # ticks
    e_weight=1100,            # micro-units, mean EPSP peak (tuned: recurrent propagation ~8x)
    e_weight_jitter=0.20,
    i_weight=-1500,           # micro-units, mean IPSP peak
    i_weight_jitter=0.20,
    tau_ticks=20,             # 2.0 ms membrane time constant
    refr_ticks=5,             # 0.5 ms absolute refractory
    vrest=-65000,             # -65.0 mV
    vth=-57000,               # -57.0 mV  (8.0 mV above rest)
    vreset=-70000,            # -70.0 mV
    stim_fraction=0.25,       # fraction of neurons that receive external drive
    stim_period_ticks=50,     # one evoked spike per stimulated neuron every 5.0 ms
    stim_jitter_ticks=4,
    stim_weight=30000,        # micro-units: supra-threshold on its own from rest
    # primary run horizon; the throughput run uses this many ticks (20000 = 2000 ms simulated,
    # ~22.8M events in the reference). Keep this equal to the committed fixture so that a plain
    # `python3 scripts/make_tiny_net.py` reproduces data/processed/tiny_net byte for byte.
    ticks=20000,
    short_ticks=500,          # 50.0 ms, used for the determinism-check replications
    seeds=dict(network=0x5EED1, stimulus_s0=0x5EED2, stimulus_s1=0x5EED3),
)

# ---------------------------------------------------------------------------
# Deterministic RNG: 64-bit LCG (Numerical Recipes constants), no library state.
# ---------------------------------------------------------------------------


class LCG:
    A = 6364136223846793005
    C = 1442695040888963407
    M = 1 << 64

    def __init__(self, seed: int) -> None:
        self.state = seed & (self.M - 1)

    def next_u64(self) -> int:
        self.state = (self.A * self.state + self.C) % self.M
        return self.state

    def below(self, n: int) -> int:
        """Uniform integer in [0, n). Rejection-free: n is small relative to 2**64."""
        return self.next_u64() % n

    def between(self, lo: int, hi: int) -> int:
        """Inclusive integer range [lo, hi]."""
        return lo + self.below(hi - lo + 1)

    def scaled(self, base: int, jitter: float) -> int:
        """base * U[1-jitter, 1+jitter], rounded, with the sign of base preserved."""
        if jitter <= 0:
            return base
        span = int(round(abs(base) * jitter))
        delta = self.between(-span, span)
        mag = abs(base) + delta
        if mag < 0:
            mag = 0
        return mag if base >= 0 else -mag


def lcg_gauss(rand: LCG, n: int = 3) -> int:
    """Approximate normal by summing uniforms; returns a value in [-n, n] (mean 0).

    Only used for jitter-shaped quantities, so an approximate distribution is fine and keeps the
    generator dependency-free and exactly reproducible.
    """
    total = 0
    for _ in range(n):
        total += rand.below(2001) - 1000  # [-1000, 1000]
    return int(round(total / 1000.0))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_neurons(cfg: dict) -> list[dict]:
    rand = LCG(cfg["seeds"]["network"])
    n = cfg["neurons"]
    tau = cfg["tau_ticks"]
    decay_k = int(round(1000.0 * math.exp(-1.0 / tau)))
    if decay_k >= 1000:
        decay_k = 999
    if decay_k < 1:
        decay_k = 1
    neurons = []
    for i in range(n):
        is_e = (i % 100) < int(round(100.0 * cfg["e_fraction"]))
        neurons.append(
            dict(
                index=i,
                label="n%04d" % i,
                type="E" if is_e else "I",
                tau_ticks=tau,
                decay_k=decay_k,
                refr_ticks=cfg["refr_ticks"],
                vth=cfg["vth"],
                vrest=cfg["vrest"],
                vreset=cfg["vreset"],
            )
        )
    return neurons


def build_edges(cfg: dict, neurons: list[dict]) -> list[dict]:
    rand = LCG(cfg["seeds"]["network"] ^ 0x9E3779B97F4A7C15)
    n = len(neurons)
    lo_d, hi_d = cfg["min_delay"], cfg["max_delay"]
    edges = []
    for pre in range(n):
        span = cfg["max_out_degree"] - cfg["min_out_degree"]
        deg = cfg["min_out_degree"] + rand.below(span + 1)
        if deg > n - 1:
            deg = n - 1
        # sample `deg` distinct posts via partial Fisher-Yates over a window of candidates
        chosen: list[int] = []
        seen: set[int] = set()
        guard = 0
        while len(chosen) < deg and guard < 100 * deg + 100:
            guard += 1
            post = rand.below(n)
            if post == pre or post in seen:
                continue
            seen.add(post)
            chosen.append(post)
        src = neurons[pre]
        for post in sorted(chosen):
            delay = rand.between(lo_d, hi_d)
            if src["type"] == "E":
                weight = rand.scaled(cfg["e_weight"], cfg["e_weight_jitter"])
                if weight <= 0:
                    weight = 1
            else:
                weight = rand.scaled(cfg["i_weight"], cfg["i_weight_jitter"])
                if weight >= 0:
                    weight = -1
            edges.append(dict(pre=pre, post=post, delay=delay, weight=weight, nt=src["type"]))
    return edges


def build_stimulus(cfg: dict, seed_key: str, stream: int) -> list[dict]:
    rand = LCG(cfg["seeds"][seed_key])
    n = cfg["neurons"]
    n_stim = max(1, int(round(cfg["stim_fraction"] * n)))
    # deterministic selection of stimulated entities
    chosen: list[int] = []
    seen: set[int] = set()
    while len(chosen) < n_stim:
        k = rand.below(n)
        if k in seen:
            continue
        seen.add(k)
        chosen.append(k)
    chosen.sort()
    period = cfg["stim_period_ticks"]
    jit = cfg["stim_jitter_ticks"]
    events = []
    tick = 0
    while tick < cfg["ticks"]:
        for target in chosen:
            # stream 1 fires the same population on a different phase so the two schedules differ
            if stream == 1 and (target + tick // period) % 2 == 0:
                continue
            off = rand.between(-jit, jit)
            t = tick + off
            if t < 0:
                t = 0
            events.append(dict(tick=t, target=target, weight=cfg["stim_weight"]))
        tick += period
    events.sort(key=lambda e: (e["tick"], e["target"]))
    return events


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def write_neurons(path: Path, neurons: list[dict]) -> None:
    lines = [
        "# FlyScale tiny LIF network - neurons (PROJECT-VYBFLY.md Phase 2 / M2 DES gate)",
        "# generated by scripts/make_tiny_net.py (%s); do not hand-edit" % GENERATOR_VERSION,
        "# units: time = ticks (1 tick = %g ms simulated); voltage = integer micro-units (1e-6 mV)" % TICK_MS,
        "# columns: index,label,type,tau_ticks,decay_k,refr_ticks,vth,vrest,vreset",
        "# decay_k: per-tick leak factor, k/1000 = exp(-1/tau_ticks), rounded (integer arithmetic)",
        "index,label,type,tau_ticks,decay_k,refr_ticks,vth,vrest,vreset",
    ]
    for r in neurons:
        lines.append(
            "%d,%s,%s,%d,%d,%d,%d,%d,%d"
            % (r["index"], r["label"], r["type"], r["tau_ticks"], r["decay_k"],
               r["refr_ticks"], r["vth"], r["vrest"], r["vreset"])
        )
    path.write_text("\n".join(lines) + "\n")


def write_edges(path: Path, edges: list[dict]) -> None:
    lines = [
        "# FlyScale tiny LIF network - synapses (PROJECT-VYBFLY.md Phase 2 / M2 DES gate)",
        "# generated by scripts/make_tiny_net.py (%s); do not hand-edit" % GENERATOR_VERSION,
        "# units: delay = ticks (>= 1, so a spike cannot re-enter its own tick); weight = micro-units",
        "# sorted by (pre, post); CSR row pointers are derivable by counting rows per pre",
        "# columns: edge,pre,post,delay_ticks,weight,nt",
        "edge,pre,post,delay_ticks,weight,nt",
    ]
    for k, e in enumerate(edges):
        lines.append(
            "%d,%d,%d,%d,%d,%s" % (k, e["pre"], e["post"], e["delay"], e["weight"], e["nt"])
        )
    path.write_text("\n".join(lines) + "\n")


def write_stimulus(path: Path, events: list[dict], stream: int, cfg: dict) -> None:
    lines = [
        "# FlyScale tiny LIF network - external event source (stimulus schedule), stream %d" % stream,
        "# generated by scripts/make_tiny_net.py (%s); do not hand-edit" % GENERATOR_VERSION,
        "# semantics: each row schedules one event of kind STIMULUS at `tick`, delivered to",
        "# entity `target` and integrated exactly like a synaptic event of `weight` micro-units.",
        "# columns: tick,target,weight",
        "tick,target,weight",
    ]
    for e in events:
        lines.append("%d,%d,%d" % (e["tick"], e["target"], e["weight"]))
    path.write_text("\n".join(lines) + "\n")


def write_replications(path: Path, cfg: dict) -> list[dict]:
    reps = [
        dict(name="main", stream=0, ticks=cfg["ticks"]),
        dict(name="det_a", stream=0, ticks=cfg["short_ticks"]),
        dict(name="det_b", stream=0, ticks=cfg["short_ticks"]),
        dict(name="alt", stream=1, ticks=cfg["short_ticks"]),
    ]
    lines = [
        "# FlyScale tiny LIF network - Replication table (Experiment config) for the M2 DES gate",
        "# generated by scripts/make_tiny_net.py (%s); do not hand-edit" % GENERATOR_VERSION,
        "# Both implementations read this so that they run exactly the same set of replications.",
        "# columns: name,stimulus_stream,horizon_ticks",
        "# `main` is the throughput replication; det_a and det_b are the same configuration (they must",
        "# produce bit-identical output, which is the DES determinism check); alt uses the other stream.",
        "name,stimulus_stream,horizon_ticks",
    ]
    for r in reps:
        lines.append("%s,%d,%d" % (r["name"], r["stream"], r["ticks"]))
    path.write_text("\n".join(lines) + "\n")
    return reps


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/processed/tiny_net", help="output directory")
    ap.add_argument("--neurons", type=int, default=DEFAULTS["neurons"])
    ap.add_argument("--ticks", type=int, default=DEFAULTS["ticks"], help="primary-run horizon in ticks")
    ap.add_argument("--short-ticks", type=int, default=DEFAULTS["short_ticks"])
    ap.add_argument("--e-weight", type=int, default=DEFAULTS["e_weight"])
    ap.add_argument("--i-weight", type=int, default=DEFAULTS["i_weight"])
    ap.add_argument("--stim-weight", type=int, default=DEFAULTS["stim_weight"])
    ap.add_argument("--stim-period", type=int, default=DEFAULTS["stim_period_ticks"])
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg["neurons"] = args.neurons
    cfg["ticks"] = args.ticks
    cfg["short_ticks"] = args.short_ticks
    cfg["e_weight"] = args.e_weight
    cfg["i_weight"] = args.i_weight
    cfg["stim_weight"] = args.stim_weight
    cfg["stim_period_ticks"] = args.stim_period

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    neurons = build_neurons(cfg)
    edges = build_edges(cfg, neurons)
    stim_s0 = build_stimulus(cfg, "stimulus_s0", 0)
    stim_s1 = build_stimulus(cfg, "stimulus_s1", 1)

    write_neurons(out / "neurons.csv", neurons)
    write_edges(out / "edges.csv", edges)
    write_stimulus(out / "stimulus_s0.csv", stim_s0, 0, cfg)
    write_stimulus(out / "stimulus_s1.csv", stim_s1, 1, cfg)
    reps = write_replications(out / "replications.csv", cfg)

    out_degs: dict[int, int] = {}
    for e in edges:
        out_degs[e["pre"]] = out_degs.get(e["pre"], 0) + 1
    in_degs: dict[int, int] = {}
    for e in edges:
        in_degs[e["post"]] = in_degs.get(e["post"], 0) + 1
    n_e = sum(1 for x in neurons if x["type"] == "E")
    weights = [e["weight"] for e in edges]
    delays = [e["delay"] for e in edges]

    meta = dict(
        generator="scripts/make_tiny_net.py",
        generator_version=GENERATOR_VERSION,
        units=dict(tick_ms=TICK_MS, voltage="1e-6 mV (integer micro-units)",
                   decay="k/1000 per tick, iterated"),
        counts=dict(
            n_neurons=len(neurons),
            n_excitatory=n_e,
            n_inhibitory=len(neurons) - n_e,
            n_edges=len(edges),
            n_stimulus_s0=len(stim_s0),
            n_stimulus_s1=len(stim_s1),
            ticks=cfg["ticks"],
            short_ticks=cfg["short_ticks"],
        ),
        degree=dict(
            mean_out_degree=round(len(edges) / len(neurons), 4),
            max_out_degree=max(out_degs.values()) if out_degs else 0,
            min_out_degree=min(out_degs.values()) if out_degs else 0,
            max_in_degree=max(in_degs.values()) if in_degs else 0,
        ),
        weights=dict(mean=round(sum(weights) / len(weights), 2), min=min(weights), max=max(weights)),
        delays=dict(min=min(delays), max=max(delays)),
        seeds=cfg["seeds"],
        params=cfg,
        replications=reps,
        files={},
    )
    for name in ("neurons.csv", "edges.csv", "stimulus_s0.csv", "stimulus_s1.csv",
                 "replications.csv"):
        p = out / name
        meta["files"][name] = dict(bytes=p.stat().st_size, sha256=sha256(p))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print("wrote %s" % out)
    print("  neurons=%d (E=%d I=%d) edges=%d mean_out=%.2f delay=[%d..%d] weight=[%d..%d]"
          % (len(neurons), n_e, len(neurons) - n_e, len(edges),
             len(edges) / len(neurons), min(delays), max(delays), min(weights), max(weights)))
    print("  stimulus s0=%d s1=%d events; primary horizon %d ticks (%g ms)"
          % (len(stim_s0), len(stim_s1), cfg["ticks"], cfg["ticks"] * TICK_MS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
