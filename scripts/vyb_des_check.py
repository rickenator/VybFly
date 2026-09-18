#!/usr/bin/env python3
"""Independent Python reference + gate checker for the FlyScale Phase 2 (M2) Vyb DES runtime.

This program is the *oracle*.  It re-implements the LIF / discrete-event semantics from scratch
against the same fixture files that src/vyb/phase2_des.vyb reads, using different data structures
(the reference buckets events with a plain dict tick -> list plus a heapq of pending ticks, rather
than the Vyb calendar queue), and then compares the two runs spike for spike:

  * per-entity spike counts                    (exact integers)
  * per-entity spike-tick sequences            (exact, canonical: entity-major, tick-ascending)
  * the global (tick, entity) spike multiset   (exact after canonical sorting)
  * per-tick spike histograms                  (exact)
  * scheduled / stimulus / synaptic event counts and processed ticks per replication
  * a 64-bit FNV-style rolling hash over the canonical record, computed independently here and
    compared against the hash the Vyb runtime recorded in results/phase2/des_summary.json

Nothing here is allowed to license a tolerance: every quantity crossing the gate is an integer and
both implementations perform identical integer arithmetic (the per-tick leak factor `decay_k` is
precomputed in the fixture by scripts/make_tiny_net.py so that no libm call and no floating point
exists on either side).  Division truncates toward zero on both sides; the reference uses an
explicit magnitude-based helper so Python's floor division cannot silently differ from C.

Canonical semantics (mirrored in src/vyb/des.vyb's header; this is the contract):

  time          integer ticks, 1 tick = 0.1 ms; a replication's horizon H means events with
                tick >= H are never created or processed
  entity state  vm (micro-units), last_update (tick), refractory_until (tick)
  integration   lazy: an entity is touched only when an event arrives for it
                1. leak from last_update to t, one tick at a time, truncating toward zero:
                       vm <- vrest + trunc((vm - vrest) * decay_k / 1000)
                2. add the batch sum of the arriving weights
                3. if t >= refractory_until and vm >= threshold: fire --
                       record (t, entity); vm <- vreset; refractory_until <- t + refr_ticks;
                       schedule every out-edge at t + delay with its weight
                   else: vm keeps the integrated value
                4. last_update <- t
  refractory    arrivals with t < refractory_until are DROPPED (absolute clamp), but they still
                advance last_update
  batching      within one tick, all arrivals for an entity are summed before the single
                integration above ("group by destination, then update")
  ordering      events are processed in ascending tick order.  Targets within a tick may be updated
                in any order: a fired entity only schedules events at t + delay with delay >= 1
                (asserted below), so no within-tick causal chain can exist.  The gate therefore
                compares canonical entity-major / tick-ascending records (and the sorted
                (tick, entity) multiset), which are order-independent by construction.

Usage:
  python3 scripts/vyb_des_check.py --reference-only     # run the oracle, print stats (tuning)
  python3 scripts/vyb_des_check.py                      # oracle + full comparison vs the Vyb run
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

FIXTURE = Path("data/processed/tiny_net")
RESULTS = Path("results/phase2")
TICK_MS = 0.1

# Event kinds (must match src/vyb/des.vyb)
EV_STIMULUS = 1
EV_SYNAPTIC = 2

MASK64 = (1 << 64) - 1
FNV_SEED = 0x00000100000001B3
FNV_PRIME = 0x100000001B3

VYB_STDLIB = os.environ.get("VYB_STDLIB", str(Path.home() / "Projects" / "Vyb" / "stdlib"))
VYB_BIN = os.environ.get("VYB", str(Path.home() / "Projects" / "Vyb" / "build" / "vyb"))


# ---------------------------------------------------------------------------
# Fixture IO (the same files the Vyb program reads)
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> list[list[str]]:
    """Read a `#`-commented CSV: skip blank lines, comment lines and the column-name row."""
    rows: list[list[str]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if parts[0] in ("index", "edge", "tick", "name"):
            continue  # column-name row
        rows.append(parts)
    return rows


class Fixture:
    def __init__(self, root: Path = FIXTURE) -> None:
        self.root = root
        nrows = read_rows(root / "neurons.csv")
        self.n = len(nrows)
        self.label: list[str] = []
        self.ntype: list[str] = []
        self.tau: list[int] = []
        self.decay_k: list[int] = []
        self.refr_ticks: list[int] = []
        self.vth: list[int] = []
        self.vrest: list[int] = []
        self.vreset: list[int] = []
        for r in nrows:
            idx = int(r[0])
            assert idx == len(self.label), "neurons.csv must be in index order"
            self.label.append(r[1])
            self.ntype.append(r[2])
            self.tau.append(int(r[3]))
            self.decay_k.append(int(r[4]))
            self.refr_ticks.append(int(r[5]))
            self.vth.append(int(r[6]))
            self.vrest.append(int(r[7]))
            self.vreset.append(int(r[8]))

        erows = read_rows(root / "edges.csv")
        self.n_edges = len(erows)
        pre = [0] * self.n_edges
        post = [0] * self.n_edges
        delay = [0] * self.n_edges
        weight = [0] * self.n_edges
        for k, r in enumerate(erows):
            pre[k] = int(r[1])
            post[k] = int(r[2])
            delay[k] = int(r[3])
            weight[k] = int(r[4])
        # CSR by pre (the file is pre-sorted; the Vyb loader relies on the same guarantee)
        indptr = [0] * (self.n + 1)
        for p in pre:
            indptr[p + 1] += 1
        for i in range(self.n):
            indptr[i + 1] += indptr[i]
        assert indptr[self.n] == self.n_edges
        cursor = list(indptr)
        self.out_post = [0] * self.n_edges
        self.out_delay = [0] * self.n_edges
        self.out_weight = [0] * self.n_edges
        for k in range(self.n_edges):
            c = cursor[pre[k]]
            self.out_post[c] = post[k]
            self.out_delay[c] = delay[k]
            self.out_weight[c] = weight[k]
            cursor[pre[k]] = c + 1
        self.out_indptr = indptr
        assert min(delay) >= 1, "min synaptic delay must be >= 1 tick (no same-tick re-entry)"
        assert min(self.refr_ticks) >= 1, "refractory period must be >= 1 tick"

        self.stim: dict[int, list[tuple[int, int, int]]] = {}
        for s in (0, 1):
            rows = read_rows(root / ("stimulus_s%d.csv" % s))
            self.stim[s] = [(int(r[0]), int(r[1]), int(r[2])) for r in rows]

        self.meta = json.loads((root / "meta.json").read_text())

    def replications(self) -> list[tuple[str, int, int]]:
        return [(r[0], int(r[1]), int(r[2])) for r in read_rows(self.root / "replications.csv")]


# ---------------------------------------------------------------------------
# Integer helpers (must match the Vyb side exactly)
# ---------------------------------------------------------------------------


def trunc_div(a: int, b: int) -> int:
    """Truncate toward zero (C semantics), independent of Python's floor division."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q


def leak(vm: int, vrest: int, decay_k: int, ticks: int) -> int:
    """Iterated integer leak, one step per tick, truncating toward zero each step."""
    d = vm - vrest
    for _ in range(ticks):
        if d == 0:
            break
        d = trunc_div(d * decay_k, 1000)
    return vrest + d


def signed64(v: int) -> int:
    v &= MASK64
    return v - (1 << 64) if v >= (1 << 63) else v


def fnv_canonical(per_entity: list[list[int]]) -> int:
    """FNV-style 64-bit rolling hash over the canonical entity-major record (see des.vyb)."""
    h = FNV_SEED
    for i, ticks in enumerate(per_entity):
        if ticks:
            h = ((h ^ i) * FNV_PRIME) & MASK64
            for t in ticks:
                h = ((h ^ t) * FNV_PRIME) & MASK64
    return signed64(h)


# ---------------------------------------------------------------------------
# The reference kernel
# ---------------------------------------------------------------------------


class Replication:
    def __init__(self, name: str, stream: int, ticks: int) -> None:
        self.name = name
        self.stream = stream
        self.ticks = ticks
        self.spikes: list[tuple[int, int]] = []      # emission order
        self.counts: list[int] = []
        self.events_scheduled = 0
        self.stimulus_events = 0
        self.synaptic_events = 0
        self.processed_ticks = 0
        self.max_batch = 0
        self.wall_ms = 0

    def stats(self) -> dict:
        return dict(
            name=self.name,
            stream=self.stream,
            horizon_ticks=self.ticks,
            spikes=len(self.spikes),
            events_scheduled=self.events_scheduled,
            stimulus_events=self.stimulus_events,
            synaptic_events=self.synaptic_events,
            ticks_processed=self.processed_ticks,
            max_batch=self.max_batch,
            wall_ms=self.wall_ms,
        )


def run_reference(fx: Fixture, name: str, stream: int, horizon: int) -> Replication:
    t_start = time.time()
    rep = Replication(name, stream, horizon)
    n = fx.n
    vm = list(fx.vrest)
    last = [0] * n
    refr_until = [0] * n
    counts = [0] * n
    spikes = rep.spikes

    buckets: dict[int, list[tuple[int, int, int]]] = {}
    pending: list[int] = []  # heap of distinct ticks with work

    def schedule(tick: int, target: int, kind: int, value: int) -> None:
        if tick >= horizon or tick < 0:
            return
        lst = buckets.get(tick)
        if lst is None:
            lst = []
            buckets[tick] = lst
            heapq.heappush(pending, tick)
        lst.append((target, kind, value))
        rep.events_scheduled += 1
        if kind == EV_STIMULUS:
            rep.stimulus_events += 1
        else:
            rep.synaptic_events += 1

    for (tick, target, weight) in fx.stim[stream]:
        schedule(tick, target, EV_STIMULUS, weight)

    out_post, out_delay, out_weight, out_indptr = (
        fx.out_post, fx.out_delay, fx.out_weight, fx.out_indptr)

    while pending:
        t = heapq.heappop(pending)
        batch = buckets.pop(t)
        rep.processed_ticks += 1
        if len(batch) > rep.max_batch:
            rep.max_batch = len(batch)
        # 1. group by destination (accumulate)
        acc: dict[int, int] = {}
        for (target, _kind, value) in batch:
            acc[target] = acc.get(target, 0) + value
        # 2. update each affected entity exactly once
        for target in sorted(acc):          # ascending entity: canonical, though order-irrelevant
            w = acc[target]
            if t < refr_until[target]:
                last[target] = t            # absolute refractory: arrival dropped, clock advances
                continue
            v = leak(vm[target], fx.vrest[target], fx.decay_k[target], t - last[target])
            v += w
            vm[target] = v
            last[target] = t
            if v >= fx.vth[target]:
                spikes.append((t, target))
                counts[target] += 1
                vm[target] = fx.vreset[target]
                refr_until[target] = t + fx.refr_ticks[target]
                for e in range(out_indptr[target], out_indptr[target + 1]):
                    schedule(t + out_delay[e], out_post[e], EV_SYNAPTIC, out_weight[e])
    rep.counts = counts
    rep.spikes.sort(key=lambda p: (p[0], p[1]))
    rep.wall_ms = int(round((time.time() - t_start) * 1000.0))
    return rep


# ---------------------------------------------------------------------------
# Canonical forms
# ---------------------------------------------------------------------------


def canonical_by_entity(spikes: list[tuple[int, int]], n: int) -> list[list[int]]:
    """entity-major, tick-ascending view -- canonical regardless of within-tick emission order."""
    per: list[list[int]] = [[] for _ in range(n)]
    for (t, i) in spikes:
        per[i].append(t)
    for lst in per:
        lst.sort()
    return per


def write_spikes_csv(path: Path, per_entity: list[list[int]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# canonical spike record: entity-major, tick-ascending. columns: entity,tick\n")
        fh.write("entity,tick\n")
        for i, ticks in enumerate(per_entity):
            for t in ticks:
                fh.write("%d,%d\n" % (i, t))
    return sum(len(x) for x in per_entity)


def read_vyb_spikes(path: Path, n: int) -> tuple[list[list[int]], dict]:
    """Read the Vyb spike record (one `entity,tick` line per spike) into the canonical view.

    Verified while reading: entities appear in ascending order (entity-major) and each entity's
    ticks are non-decreasing (tick-ascending) -- i.e. the file really is the canonical record.
    """
    per: list[list[int]] = [[] for _ in range(n)]
    prev_entity = -1
    ordered_ok = True
    ascending_ok = True
    total = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("entity"):
            continue
        e_s, t_s = line.split(",")
        e = int(e_s)
        t = int(t_s)
        if e != prev_entity:
            if e <= prev_entity:
                ordered_ok = False
            prev_entity = e
        if per[e] and t < per[e][-1]:
            ascending_ok = False
        per[e].append(t)
        total += 1
    return per, dict(entities=len(per), spikes=total, entity_major_ascending=ordered_ok,
                     tick_ascending_per_entity=ascending_ok)


def read_vyb_counts(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("entity,"):
            continue
        p = line.split(",")
        out[int(p[0])] = dict(type=p[1], spikes=int(p[2]), first_tick=int(p[3]),
                              last_tick=int(p[4]), tick_sum=int(p[5]))
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(rep: Replication, ref_per: list[list[int]], vyb_per: list[list[int]],
            vyb_counts: dict[int, dict], readonly: dict) -> dict:
    n = max(len(ref_per), len(vyb_per))
    mismatches: list[dict] = []
    count_file_mismatches: list[dict] = []
    entities_with_spikes = 0
    total_ref = sum(len(x) for x in ref_per)
    total_vyb = sum(len(x) for x in vyb_per)
    for i in range(n):
        rt = ref_per[i] if i < len(ref_per) else []
        vt = vyb_per[i] if i < len(vyb_per) else []
        if rt or vt:
            entities_with_spikes += 1
        if len(rt) != len(vt) or rt != vt:
            first = None
            for k in range(max(len(rt), len(vt))):
                a = rt[k] if k < len(rt) else None
                b = vt[k] if k < len(vt) else None
                if a != b:
                    first = dict(position=k, reference=a, vyb=b)
                    break
            if len(mismatches) < 25:
                mismatches.append(dict(entity=i, reference_count=len(rt), vyb_count=len(vt),
                                       first_divergence=first,
                                       reference_ticks_head=rt[:12], vyb_ticks_head=vt[:12]))
        c = vyb_counts.get(i)
        if c is not None and (c["spikes"] != len(vt) or c["tick_sum"] != sum(vt)
                              or (vt and (c["first_tick"] != vt[0] or c["last_tick"] != vt[-1]))):
            if len(count_file_mismatches) < 25:
                count_file_mismatches.append(dict(entity=i, counts_file=c, spike_file=vt[:12]))
    ref_pairs = sorted((t, i) for i, ticks in enumerate(ref_per) for t in ticks)
    vyb_pairs = sorted((t, i) for i, ticks in enumerate(vyb_per) for t in ticks)
    hist_ref: dict[int, int] = {}
    for (t, _i) in ref_pairs:
        hist_ref[t] = hist_ref.get(t, 0) + 1
    hist_vyb: dict[int, int] = {}
    for (t, _i) in vyb_pairs:
        hist_vyb[t] = hist_vyb.get(t, 0) + 1
    hist_mismatch = [dict(tick=t, reference=hist_ref[t], vyb=hist_vyb.get(t, 0))
                     for t in sorted(set(hist_ref) | set(hist_vyb))
                     if hist_ref.get(t, 0) != hist_vyb.get(t, 0)]
    ref_stats = rep.stats()
    return dict(
        replication=rep.name,
        horizon_ticks=rep.ticks,
        stimulus_stream=rep.stream,
        entities=len(ref_per),
        entities_with_spikes=entities_with_spikes,
        reference_spikes=total_ref,
        vyb_spikes=total_vyb,
        spike_counts_match=total_ref == total_vyb and not mismatches,
        spike_tick_sequences_identical=not mismatches,
        sorted_tick_entity_lists_identical=ref_pairs == vyb_pairs,
        per_tick_histogram_identical=not hist_mismatch,
        counts_file_consistent=not count_file_mismatches,
        vyb_file_structure=readonly,
        reference_fnv_hash=fnv_canonical(ref_per),
        vyb_fnv_hash=fnv_canonical(vyb_per),
        n_mismatched_entities=len(mismatches),
        mismatches=mismatches[:25],
        histogram_mismatches=hist_mismatch[:25],
        counts_file_mismatches=count_file_mismatches[:25],
        reference_stats=ref_stats,
    )


def compare_summary_fields(comp: dict, vyb_rep: dict) -> dict:
    """Compare the event-level counters the Vyb runtime reported with the reference's."""
    fields = ["spikes", "events_scheduled", "stimulus_events", "synaptic_events", "ticks_processed"]
    out = {}
    for f in fields:
        ref_v = comp["reference_stats"].get(f)
        vyb_v = vyb_rep.get(f)
        out[f] = dict(reference=ref_v, vyb=vyb_v, match=(ref_v == vyb_v))
    vh = comp["vyb_fnv_hash"]
    out["spike_hash"] = dict(reference=comp["reference_fnv_hash"],
                             vyb=vyb_rep.get("spike_hash"),
                             vyb_recomputed_from_spike_file=vh,
                             match=(comp["reference_fnv_hash"] == vyb_rep.get("spike_hash") == vh))
    out["all_match"] = all(v["match"] for v in out.values())
    return out


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vyb_build_info() -> dict:
    p = Path(VYB_BIN)
    info = dict(binary=str(p), exists=p.exists())
    if p.exists():
        st = p.stat()
        info["mtime_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime))
        info["size_bytes"] = st.st_size
        for flag in ("--version", "-v"):
            try:
                out = subprocess.run([str(p), flag], capture_output=True, text=True, timeout=60)
                txt = (out.stdout + out.stderr).strip()
                if txt:
                    info["version"] = txt.splitlines()[0]
                    break
            except Exception as exc:  # pragma: no cover
                info["version_error"] = repr(exc)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--reference-only", action="store_true",
                    help="run the oracle and print stats; do not look for Vyb output")
    ap.add_argument("--only", default=None, help="restrict to one replication name")
    args = ap.parse_args()

    fx = Fixture(Path(args.fixture))
    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = fx.replications()
    if args.only:
        reps = [r for r in reps if r[0] == args.only]
        if not reps:
            print("no such replication: %s" % args.only, file=sys.stderr)
            return 2

    print("fixture %s: %d entities, %d edges, mean out-degree %.2f, stimulus s0=%d s1=%d events"
          % (args.fixture, fx.n, fx.n_edges, fx.n_edges / fx.n,
             len(fx.stim[0]), len(fx.stim[1])))
    print("replications: %s" % ", ".join("%s(stream %d, %d ticks)" % r for r in reps))

    summary_path = out_dir / "des_summary.json"
    vyb_summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    vyb_reps = {}
    if vyb_summary:
        for r in vyb_summary.get("replications", []):
            vyb_reps[r["name"]] = r

    comparisons = []
    reference_stats = []
    for (name, stream, ticks) in reps:
        rep = run_reference(fx, name, stream, ticks)
        ref_per = canonical_by_entity(rep.spikes, fx.n)
        ref_n = write_spikes_csv(out_dir / ("ref_spikes_%s.csv" % name), ref_per)
        st = rep.stats()
        st["bio_ms"] = ticks * TICK_MS
        st["spikes_per_bio_second"] = round(len(rep.spikes) / (ticks * TICK_MS / 1000.0), 3)
        st["events_per_bio_second"] = round(rep.events_scheduled / (ticks * TICK_MS / 1000.0), 1)
        st["fnv_hash"] = fnv_canonical(ref_per)
        st["spike_file_lines"] = ref_n
        reference_stats.append(st)
        print("  [ref] %-6s spikes=%-8d events=%-9d (stim %d / syn %d) ticks_processed=%-6d "
              "bio=%g ms wall=%d ms"
              % (name, len(rep.spikes), rep.events_scheduled, rep.stimulus_events,
                 rep.synaptic_events, rep.processed_ticks, ticks * TICK_MS, rep.wall_ms))

        if args.reference_only:
            continue
        vyb_path = out_dir / ("vyb_spikes_%s.csv" % name)
        if not vyb_path.exists():
            print("  [vyb] missing %s -- run phase2_des.vyb first" % vyb_path, file=sys.stderr)
            continue
        vyb_per, structure = read_vyb_spikes(vyb_path, fx.n)
        vyb_counts = read_vyb_counts(out_dir / ("vyb_counts_%s.csv" % name))
        comp = compare(rep, ref_per, vyb_per, vyb_counts, structure)
        comp["vyb_output"] = dict(path=str(vyb_path), sha256=sha256(vyb_path))
        comp["reference_output"] = dict(path=str(out_dir / ("ref_spikes_%s.csv" % name)),
                                        sha256=sha256(out_dir / ("ref_spikes_%s.csv" % name)))
        if name in vyb_reps:
            comp["vyb_reported"] = vyb_reps[name]
            comp["field_comparison"] = compare_summary_fields(comp, vyb_reps[name])
        comparisons.append(comp)
        ok = (comp["spike_counts_match"] and comp["spike_tick_sequences_identical"]
              and comp["sorted_tick_entity_lists_identical"] and comp["per_tick_histogram_identical"]
              and comp["counts_file_consistent"] and structure["entity_major_ascending"]
              and structure["tick_ascending_per_entity"]
              and comp.get("field_comparison", {}).get("all_match", True))
        print("  [cmp] %-6s %s: ref=%d vyb=%d spikes, mismatched entities=%d, "
              "hash ref=%d vyb=%d, counters_match=%s"
              % (name, "MATCH" if ok else "MISMATCH", comp["reference_spikes"], comp["vyb_spikes"],
                 comp["n_mismatched_entities"], comp["reference_fnv_hash"], comp["vyb_fnv_hash"],
                 comp.get("field_comparison", {}).get("all_match")))

    det = {}
    for a, b in (("det_a", "det_b"),):
        for tag in ("ref", "vyb"):
            pa = out_dir / ("%s_spikes_%s.csv" % (tag, a))
            pb = out_dir / ("%s_spikes_%s.csv" % (tag, b))
            if pa.exists() and pb.exists():
                det["%s_%s_eq_%s_byte_identical" % (tag, a, b)] = pa.read_text() == pb.read_text()
    if vyb_summary:
        det["vyb_summary_determinism_ok"] = vyb_summary.get("determinism_ok")
        det["vyb_summary_det_a_hash"] = vyb_summary.get("det_a_spike_hash")
        det["vyb_summary_det_b_hash"] = vyb_summary.get("det_b_spike_hash")
        det["vyb_det_hashes_equal"] = (vyb_summary.get("det_a_spike_hash")
                                       == vyb_summary.get("det_b_spike_hash"))
        ref_hashes = {s["name"]: s["fnv_hash"] for s in reference_stats}
        det["reference_det_hashes_equal"] = ref_hashes.get("det_a") == ref_hashes.get("det_b")
        det["vyb_hash_matches_reference_hash_det_a"] = (
            vyb_summary.get("det_a_spike_hash") == ref_hashes.get("det_a"))

    vyb_det_ok = bool(det.get("vyb_det_a_eq_det_b_byte_identical"))
    comparisons_pass = vyb_det_ok and bool(comparisons) and all(
        c["spike_counts_match"] and c["spike_tick_sequences_identical"]
        and c["sorted_tick_entity_lists_identical"] and c["per_tick_histogram_identical"]
        and c["counts_file_consistent"]
        and c["vyb_file_structure"]["entity_major_ascending"]
        and c["vyb_file_structure"]["tick_ascending_per_entity"]
        and c.get("field_comparison", {}).get("all_match", True)
        for c in comparisons)
    residual = [dict(replication=c["replication"], entity=m["entity"],
                     reference_count=m["reference_count"], vyb_count=m["vyb_count"],
                     first_divergence=m["first_divergence"]) for c in comparisons for m in c["mismatches"]]

    report = dict(
        gate="PROJECT-VYBFLY.md 8 / milestone M2 - Vyb discrete-event runtime vs independent reference",
        verdict=("PASS" if comparisons_pass else
                 ("NO_VYB_OUTPUT" if not comparisons else "FAIL")),
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=dict(hostname=platform.node(), platform=platform.platform(),
                  python=sys.version.split()[0], cpu_count=os.cpu_count()),
        vyb=dict(binary=vyb_build_info(), stdlib=VYB_STDLIB, module_path=str(Path("src/vyb").resolve()),
                 invocation=(vyb_summary or {}).get("vyb_invocation"),
                 runtime_module="src/vyb/des.vyb (main-less share(all) DES runtime)",
                 application="src/vyb/phase2_des.vyb (LIF application)"),
        fixture=dict(root=str(Path(args.fixture).resolve()), meta=fx.meta,
                     files={name: dict(sha256=sha256(Path(args.fixture) / name))
                            for name in ("neurons.csv", "edges.csv", "stimulus_s0.csv",
                                         "stimulus_s1.csv", "replications.csv")
                            if (Path(args.fixture) / name).exists()}),
        semantics_contract=dict(
            time="integer ticks, 1 tick = 0.1 ms simulated; events with tick >= horizon are never created or processed",
            voltages="integer micro-units (1e-6 mV); weights in the same scale, sign = excitatory/inhibitory",
            leak="per-tick integer leak, truncating toward zero each step: vm <- vrest + trunc((vm-vrest)*decay_k/1000)",
            integration="lazy: an entity integrates only when an event arrives for it",
            batching="within a tick all arrivals for one entity are summed before the single integration (order-independent)",
            refractory="arrivals with t < refractory_until are dropped and still advance last_update",
            firing="fire iff t >= refractory_until and vm >= threshold; then vm <- vreset, refractory_until <- t + refr_ticks",
            delay_rule="every synaptic delay >= 1 tick, so a spike can never re-enter its own tick",
            tie_breaking=("events are processed in ascending tick order; targets within a tick may be updated in any "
                          "order because a fired entity only schedules events at t+delay (delay>=1), so no within-tick "
                          "causal chain exists, and all arithmetic is exact integer arithmetic, so accumulation order "
                          "cannot change a result. The gate compares the canonical entity-major / tick-ascending spike "
                          "record, the sorted (tick, entity) multiset, the per-tick histogram and a 64-bit rolling hash, "
                          "all of which are order-independent by construction."),
            tolerance="none - integer arithmetic end to end; any difference is a real mismatch",
        ),
        reference=[s for s in reference_stats],
        reference_cross_replication_determinism=det,
        comparisons=comparisons,
        residual_mismatches=residual,
        vyb_summary=vyb_summary,
        vyb_summary_consistency=(dict(
            spikes_match=all(
                vyb_reps.get(c["replication"], {}).get("spikes") == c["reference_spikes"]
                for c in comparisons),
            hashes_match=all(
                vyb_reps.get(c["replication"], {}).get("spike_hash") == c["reference_fnv_hash"]
                for c in comparisons),
            determinism_ok=bool(vyb_summary.get("determinism_ok")) if vyb_summary else None,
            late_or_far_events_zero=(
                (vyb_summary.get("total_late_events") == 0
                 and vyb_summary.get("total_far_events") == 0) if vyb_summary else None),
        ) if vyb_summary else None),
    )

    (out_dir / "vyb_des_check.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("verdict: %s" % report["verdict"])
    print("wrote %s" % (out_dir / "vyb_des_check.json"))
    return 0 if report["verdict"] in ("PASS", "NO_VYB_OUTPUT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
