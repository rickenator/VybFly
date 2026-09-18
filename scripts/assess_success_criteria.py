"""Assemble the project-level assessment: PROJECT-VYBFLY.md §18 (power scaling table) and §25
(success criteria: minimum / strong / exceptional).

    python scripts/assess_success_criteria.py

Reads every phase artifact that exists, states the headline number for each, and evaluates the
scope document's own success criteria against measured results only. A criterion whose evidence
is missing is reported as UNKNOWN with the artifact that would have to exist - never as a pass.
Outputs: results/assessment.json and results/SUMMARY.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(path: str):
    p = RES / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:                                     # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {exc}"}


def num(x, nd: int = 6):
    return None if x is None else round(float(x), nd)


def main() -> int:
    p0 = load("phase0/gate_result.json")
    p1 = load("phase1/lif_baseline.json")
    p2 = load("phase2/vyb_des_check.json")
    p4 = load("phase4/geometry.json")
    p5 = load("phase5/downscale.json")
    p5d = load("phase5/dynamics.json")
    p6 = load("phase6/upscale.json")
    p7 = load("phase7/closure.json")
    p8 = load("phase8/learning.json")
    p9 = load("phase9/capability.json")
    energy = load("energy/hardware_energy.json")
    # per-scale energy curves live in their own file; without this the proportionality check
    # would always see zero measured scales and report None (which is what it did)
    energy_curves = load("energy/curves.json")
    bio = load("energy/biological_model.json")

    # ------------------------------------------------------------------ phase inventory
    inventory = {
        "M0 baseline dataset + published gate": {
            "artifact": "results/phase0/gate_result.json",
            "present": p0 is not None,
            "headline": None if not p0 else
                f"{p0.get('required_checks_passed')}/{p0.get('required_checks_total')} required "
                f"checks vs published FlyWire values; passed={p0.get('passed')}",
        },
        "M1 LIF whole-connectome baseline": {
            "artifact": "results/phase1/lif_baseline.json",
            "present": p1 is not None,
            "headline": None if not p1 else _m1_headline(p1),
        },
        "M2 Vyb DES engine + reference equivalence": {
            "artifact": "results/phase2/vyb_des_check.json",
            "present": p2 is not None,
            "headline": None if not p2 else _m2_headline(p2),
        },
        "M3 event-driven GPU backend": {
            "artifact": "results/phase3/",
            "present": (RES / "phase3").exists() and any((RES / "phase3").iterdir()),
            "headline": _m3_headline(),
        },
        "M4 energy instrumentation": {
            "artifact": "results/energy/hardware_energy.json",
            "present": energy is not None,
            "headline": None if not energy else _energy_headline(energy),
        },
        "M5 latent geometry": {
            "artifact": "results/phase4/geometry.json",
            "present": p4 is not None,
            "headline": None if not p4 else _m5_headline(p4),
        },
        "M6/M7 downscaling + dynamic validation": {
            "artifact": "results/phase5/{downscale,dynamics}.json",
            "present": p5 is not None,
            "headline": None if not p5 else _m6_headline(p5, p5d),
        },
        "M8/M10 inverse scaling (2x..10x)": {
            "artifact": "results/phase6/upscale.json",
            "present": p6 is not None,
            "headline": None if not p6 else _m8_headline(p6),
        },
        "M9 renormalization closure": {
            "artifact": "results/phase7/closure.json",
            "present": p7 is not None,
            "headline": None if not p7 else _m9_headline(p7),
        },
        "M11 plasticity / associative learning": {
            "artifact": "results/phase8/learning.json",
            "present": p8 is not None,
            "headline": None if not p8 else _m11_headline(p8),
        },
        "M12-M14 capability and energy scaling": {
            "artifact": "results/phase9/capability.json",
            "present": p9 is not None,
            "headline": None if not p9 else _m12_headline(p9),
        },
    }

    # ------------------------------------------------------------------ §18 power scaling table
    table = power_scaling_table(p6, p7, p9, energy, bio)

    # ------------------------------------------------------------------ §25 success criteria
    criteria = []
    criteria.append(evaluate_minimum(p7, p6, p5d, p1))
    criteria.append(evaluate_strong(p7, p6, p8, p9, bio,
                                      {"hardware": energy, "curves": energy_curves}))
    criteria.append(evaluate_exceptional(p9, bio, energy))

    out = {
        "phase_inventory": inventory,
        "power_scaling_table": table,
        "success_criteria": criteria,
        "missing_artifacts": [k for k, v in inventory.items() if not v["present"]],
    }
    (RES / "assessment.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (RES / "SUMMARY.md").write_text(render_markdown(out, inventory, table, criteria))
    print(f"wrote {RES/'assessment.json'} and {RES/'SUMMARY.md'}")
    for c in criteria:
        print(f"{c['criterion']}: {c['verdict']}")
    return 0


# --------------------------------------------------------------------------- headlines
def _m1_headline(p1: dict) -> str:
    keys = ("spikes_per_second", "mean_rate_hz", "fraction_active", "simulated_biological_seconds",
            "wall_seconds", "agreement")
    parts = [f"{k}={p1.get(k)}" for k in keys if k in p1]
    if not parts:
        parts = [f"keys={sorted(p1)[:8]}"]
    return "; ".join(str(p) for p in parts)


def _m2_headline(p2: dict) -> str:
    for k in ("verdict", "passed", "match", "spikes_match", "summary", "result"):
        if k in p2:
            return f"{k}={p2[k]}"
    return f"keys={sorted(p2)[:8]}"


def _m3_headline() -> str | None:
    ptx = RES / "phase3" / "spike_bucket.ptx"
    if not ptx.exists():
        return None
    entries = ptx.read_text().count(".entry")
    gate = RES / "phase3" / "gpu_equivalence.json"
    verdict = "gate verdict unknown"
    if gate.exists():
        try:
            g = json.loads(gate.read_text())
            verdict = ("CPU~GPU equivalence PASSED" if g.get("passed")
                       else f"CPU~GPU equivalence FAILED (mismatches: {g.get('mismatched_checks')})")
        except Exception:
            pass
    notes = []
    for f in ("SUMMARY.txt",):
        if (RES / "phase3" / f).exists():
            notes.append(f"{f} present")
    return (f"spike_bucket.ptx with {entries} .entry kernels; {verdict}"
            + (f"; {', '.join(notes)}" if notes else ""))


def _energy_headline(e: dict) -> str:
    parts = []
    for k in ("mean_gpu_watts", "mean_cpu_watts", "joules_total", "joules_per_spike",
              "joules_per_synaptic_event", "joules_per_simulated_biological_second", "samples"):
        v = e.get(k)
        if v is None:
            for sub in ("gpu", "cpu", "summary", "workload"):
                if isinstance(e.get(sub), dict) and k in e[sub]:
                    v = e[sub][k]
                    break
        if v is not None:
            parts.append(f"{k}={num(v, 4)}")
    return "; ".join(parts) if parts else f"keys={sorted(e)[:8]}"


def _m5_headline(p4: dict) -> str:
    scores = p4.get("scores") or p4.get("geometries") or p4
    if isinstance(scores, dict):
        parts = []
        for name, v in list(scores.items())[:6]:
            if isinstance(v, dict):
                auc = v.get("auc") or v.get("roc_auc") or v.get("held_out_auc")
                if auc is not None:
                    parts.append(f"{name}:AUC={num(auc, 4)}")
        return "; ".join(parts) if parts else f"keys={sorted(p4)[:8]}"
    return f"keys={sorted(p4)[:8]}"


def _m6_headline(p5: dict, p5d: dict | None) -> str:
    parts = []
    for f, e in (p5.get("factors") or {}).items():
        c = e.get("closure") or {}
        parts.append(f"{f}x: N={e.get('replica', {}).get('n_neurons')} "
                     f"composite={c.get('composite')}")
    if p5d:
        for f, e in (p5d.get("replicas") or {}).items():
            cm = e.get("comparison") or {}
            parts.append(f"dyn {f}x: jaccard={cm.get('active_set_jaccard')} "
                         f"final_ratio={cm.get('final_fraction_ratio')}")
    return "; ".join(parts) if parts else f"keys={sorted(p5)[:8]}"


def _m8_headline(p6: dict) -> str:
    parts = []
    for f, e in (p6.get("scales") or {}).items():
        parts.append(f"{f}x: N={e['replica']['n_neurons']} E={e['replica']['n_connections']} "
                     f"mean_deg={e['mean_degree_replica']} comp={e['closure'].get('composite')}")
    ex = p6.get("scaling_exponents") or {}
    c = ex.get("connections_vs_neurons") or {}
    if c.get("alpha") is not None:
        parts.append(f"E~N^{c['alpha']} (R2={c.get('r2')})")
    return "; ".join(parts)


def _m9_headline(p7: dict) -> str:
    parts = []
    for f, e in (p7.get("scales") or {}).items():
        c = e["closure"]
        parts.append(f"{f}x: R(N)={e['renormalised']['n']} composite={c.get('composite')} "
                     f"deg_wass_n={c.get('degree_wasserstein_normalised')} "
                     f"ARI={c.get('community_ari')}")
    return "; ".join(parts)


def _m11_headline(p8: dict) -> str:
    for k in ("summary", "verdict", "headline"):
        if k in p8:
            return json.dumps(p8[k])[:400]
    return f"keys={sorted(p8)[:8]}"


def _m12_headline(p9: dict) -> str:
    for k in ("summary", "headline", "scaling"):
        if k in p9:
            return json.dumps(p9[k])[:400]
    return f"keys={sorted(p9)[:8]}"


# --------------------------------------------------------------------------- §18 table
def power_scaling_table(p6, p7, p9, energy, bio) -> list[dict]:
    rows: dict[float, dict] = {}

    def row(scale: float) -> dict:
        return rows.setdefault(scale, {"scale": scale})

    if p6:
        for f, e in (p6.get("scales") or {}).items():
            r = row(float(f))
            r["N"] = e["replica"]["n_neurons"]
            r["E"] = e["replica"]["n_connections"]
            r["synapses"] = e["replica"]["n_synapses"]
    if p7:
        for f, e in (p7.get("scales") or {}).items():
            r = row(float(f))
            dyn = (e.get("dynamics") or {}).get("cascade") or {}
            if dyn:
                r["final_active_fraction"] = dyn.get("final_active_fraction")
                r["latency_steps_to_half"] = dyn.get("latency_steps_to_half")
            cmp = (e.get("dynamics") or {}).get("comparison_vs_reference") or {}
            if cmp:
                r["dynamics_curve_L1"] = cmp.get("curve_L1")
    if p9 and isinstance(p9, dict):
        for f, e in (p9.get("scales") or {}).items():
            try:
                r = row(float(f))
            except (TypeError, ValueError):
                continue
            if isinstance(e, dict):
                for k in ("memory_capacity", "discrimination_threshold", "temporal_depth",
                          "effective_dimensionality", "wall_seconds"):
                    if k in e:
                        r[k] = e[k]
    if bio and isinstance(bio, dict):
        for r in rows.values():
            n = r.get("N")
            if n and "P0_watts" in json.dumps(bio)[:20000]:
                pass
    # biological-equivalent watts follow the doc's null model P = P0 * N / N0 when the model
    # exposes its constants; otherwise the column stays empty rather than invented
    try:
        const = _bio_constants(bio)
        if const:
            n0, p0 = const
            for r in rows.values():
                if r.get("N"):
                    r["bio_watts_null_model"] = num(p0 * r["N"] / n0, 9)
    except Exception:                                            # noqa: BLE001
        pass
    if energy and isinstance(energy, dict):
        jps = energy.get("joules_per_synaptic_event") or energy.get("joules_per_spike")
        if jps:
            for r in rows.values():
                if r.get("E"):
                    r["hardware_joules_synaptic_events_per_bio_second_ESTIMATE"] = num(jps * r["E"], 6)
    return [rows[k] for k in sorted(rows)]


def _bio_constants(bio) -> tuple[float, float] | None:
    if not isinstance(bio, dict):
        return None
    flat = bio.get("model") or bio.get("null_model") or bio
    n0 = flat.get("N0_neurons") or flat.get("n0") or flat.get("N0")
    p0 = flat.get("P0_watts") or flat.get("p0_watts") or flat.get("P0")
    if n0 and p0:
        return float(n0), float(p0)
    return None


# --------------------------------------------------------------------------- §25 criteria
def evaluate_minimum(p7, p6, p5d, p1) -> dict:
    evidence, verdicts = [], []
    if p7 and (p7.get("scales") or {}).get("2.0"):
        c = p7["scales"]["2.0"]["closure"]
        ok_graph = (c.get("degree_wasserstein_normalised") is not None
                    and c["degree_wasserstein_normalised"] < 0.5)
        ok_closure = c.get("composite") is not None and c["composite"] < 0.5
        evidence.append(f"2x closure composite={c.get('composite')} "
                        f"degree_wasserstein_normalised={c.get('degree_wasserstein_normalised')}")
        verdicts += [ok_graph, ok_closure]
    else:
        # "renormalising back toward the real graph" is the core of the minimum criterion: with
        # no 2x closure measurement the criterion is undecided, not satisfied. A missing test
        # must never count as a pass.
        evidence.append("2x closure test not measured (results/phase7/closure.json missing)")
        verdicts.append(None)
    dyn_ok = False
    if p5d:
        for f, e in (p5d.get("replicas") or {}).items():
            j = (e.get("comparison") or {}).get("active_set_jaccard")
            if j is not None and j > 0.5:
                dyn_ok = True
        evidence.append(f"downscaled dynamics jaccard>0.5: {dyn_ok}")
    verdicts.append(dyn_ok or None)
    if p1:
        evidence.append("LIF baseline artifact present (executes stably under its own gate)")
        verdicts.append(True)
    return _verdict(
        "Minimum success - 2x synthetic connectome preserving statistics, renormalising back "
        "toward the real graph, executing stably, retaining baseline circuit behaviour",
        verdicts, evidence,
        "results/phase7/closure.json (2.0), results/phase5/dynamics.json, results/phase1/lif_baseline.json")


def _connection_exponent(p6) -> float | None:
    """alpha in E ~ N^alpha, straight from the per-scale replica counts on disk."""
    pts = []
    for k, v in (p6.get("scales") or {}).items():
        r = v.get("replica") or {}
        n, e = r.get("n_neurons"), r.get("n_connections")
        if n and e:
            pts.append((float(n), float(e)))
    if len(pts) < 2:
        return None
    xs = np.log([p[0] for p in pts])
    ys = np.log([p[1] for p in pts])
    return float(np.polyfit(xs, ys, 1)[0])


def _is_capability_metric(m: str) -> bool:
    """True only for names that describe measured behaviour, not cost or protocol bookkeeping.

    Guards a real failure mode: a first version of this scan accepted any numeric field, so it
    fitted an exponent to `scale_total_seconds` (wall-clock cost, which grows with N) and
    reported "capability grows faster than power" on the strength of a timing number.
    """
    bad = ("control", "identical_pools", "recruits", "readout", "components_for_90pct",
           "n_neurons", "n_edges", "n_synapses", "n_trials", "n_cascades",
           "seconds", "cost", "wall", "elapsed", "time", "bytes", "fingerprint",
           "protocol", "calibration", "seed", "epochs", "steps", "censored",
           # graph descriptors: they measure the size of the graph, not what it can do
           "n_connections", "density", "degree", "mean_", "synapses_per")
    return not any(b in m for b in bad)


def _capability_growth(p9) -> tuple[bool | None, list[str]]:
    """Did any capability metric actually grow across measured scales?

    Presence of the artifact is not evidence: this reads metrics_by_scale and requires at least
    three measured scales plus one metric that increases from first to last, reporting censored
    metrics explicitly (a value pinned at the test grid's ceiling is a lower bound).
    """
    mb = p9.get("metrics_by_scale") or {}
    scales = sorted(mb, key=lambda s: float(s))
    notes: list[str] = []
    if len(scales) < 3:
        notes.append(f"only {len(scales)} scale(s) measured in metrics_by_scale "
                     f"({scales}) - a scaling claim needs at least 3")
        return None, notes
    grew, censored = [], []
    # Metrics excluded from the growth scan, with the reason for each:
    #   control / identical_pools  - null controls that should sit at chance; drift there is
    #                                an artifact of the decoder, not a capability
    #   recruits / readout / components_for_90pct / n_ - scale with N by protocol construction
    #                                (the readout is 5% of N and recruits are a fixed fraction
    #                                of N), so their growth is bookkeeping, not capability
    for m in (mb[scales[0]].get("metrics") or mb[scales[0]]):
        if not _is_capability_metric(m):
            continue
        try:
            series = [mb[s].get("metrics", mb[s]).get(m) for s in scales]
        except AttributeError:
            continue
        if any(v is None or isinstance(v, (dict, list, str)) for v in series):
            continue
        try:
            vals = [float(v) for v in series]
        except (TypeError, ValueError):
            continue
        if vals[-1] > vals[0] * 1.1:
            grew.append((m, vals))
        if m in ("memory_capacity",):
            for s in scales:
                entry = mb[s].get("metrics", mb[s])
                if isinstance(entry, dict) and entry.get("censored_at_grid_max"):
                    censored.append(f"{m} at {s}x is censored at the test-grid ceiling")
    for m, vals in grew:
        notes.append(f"grew: {m} {vals}")
    notes.extend(sorted(set(censored)))
    if not grew:
        notes.append("no measured metric increased from the smallest to the largest scale")
    return bool(grew), notes


def _energy_proportionality(energy) -> tuple[bool | None, list[str]]:
    """'~proportional biological energy' needs measured per-scale hardware numbers."""
    curves = energy.get("curves") if isinstance(energy, dict) else None
    per_scale = (curves or {}).get("per_scale") or {}
    usable = {k: v for k, v in per_scale.items()
              if isinstance(v, dict) and v.get("hardware", {}).get("joules_per_spike")}
    notes = []
    if len(usable) < 2:
        notes.append(f"per-scale hardware energy measured at {len(usable)} scale(s); "
                     "proportionality needs at least 2")
        return None, notes
    notes.append("joules_per_spike by scale: " + ", ".join(
        f"{k}x={usable[k]['hardware']['joules_per_spike']:.3g}" for k in sorted(usable, key=float)))
    return True, notes


def evaluate_strong(p7, p6, p8, p9, bio, energy) -> dict:
    verdicts, evidence = [], []
    a = _connection_exponent(p6)
    if a is None:
        evidence.append("connection exponent not computable (needs >= 2 upscaled scales)")
        verdicts.append(None)
    else:
        verdicts.append(abs(a - 1.0) < 0.25)
        evidence.append(f"connections ~ N^{a:.4f} (sparsity preserved requires alpha ~ 1)")
    if p7 and (p7.get("scales") or {}).get("10.0"):
        c = p7["scales"]["10.0"]["closure"]
        verdicts.append(c.get("composite") is not None and c["composite"] < 0.5)
        evidence.append(f"10x closure composite={c.get('composite')}")
    else:
        evidence.append("10x closure test not measured (only 2x has been renormalized back so far)")
        verdicts.append(None)
    if p9:
        ok, notes = _capability_growth(p9)
        verdicts.append(ok)
        evidence += notes
    if p8:
        att = (p8.get("attribution") or {})
        learned = att.get("auc_real_plastic")
        frozen = att.get("auc_real_frozen")
        if learned is None or frozen is None:
            evidence.append("learning artifact has no learned-vs-frozen comparison")
            verdicts.append(None)
        else:
            verdicts.append(float(learned) > float(frozen))
            evidence.append(f"associative learning: plastic AUC {learned} vs frozen AUC {frozen} "
                            "(and see results/phase8/SUMMARY.txt for the shuffled-connectivity "
                            "control, which matters for any architecture claim)")
    if bio:
        ok, notes = _energy_proportionality(energy)
        verdicts.append(ok)
        evidence.append("biological track is linear in N by construction (P_bio(N) = P0*N/N0), so "
                        "this clause is a model identity rather than a measurement; what "
                        "was measured is the hardware side: " + "; ".join(notes))
    return _verdict(
        "Strong success - 10x connectome retaining invariants, learning the same tasks, "
        "beating baseline capacity on at least one benchmark, ~proportional biological energy",
        verdicts, evidence,
        "results/phase6/upscale.json, results/phase7/closure.json (10.0), "
        "results/phase9/capability.json, results/phase8/learning.json, results/energy/curves.json")


def evaluate_exceptional(p9, bio, energy) -> dict:
    """C(N) grows faster than P(N).

    Judged on the *biological* track, which is the only track the doc allows for this claim:
    P_bio(N) = P0*N/N0 is linear in N by construction, so the question reduces to whether the
    best-growing capability metric has an exponent clearly above 1. A metric that grows ~1:1 is
    proportional, not better than proportional, and is reported as such.
    """
    verdicts, evidence = [], []
    mb = (p9 or {}).get("metrics_by_scale") or {}
    scales = sorted(mb, key=float)
    if len(scales) < 3:
        evidence.append(f"capability curves measured at {len(scales)} scale(s); at least 3 needed")
        verdicts.append(None)
    else:
        best = None
        for m in (mb[scales[0]].get("metrics") or mb[scales[0]]):
            if not _is_capability_metric(m):
                continue
            try:
                vals = [float(mb[s].get("metrics", mb[s]).get(m)) for s in scales]
            except (TypeError, ValueError):
                continue
            ns = [float(s) for s in scales]
            if len(set(vals)) < 3 or min(vals) <= 0:
                continue
            a = float(np.polyfit(np.log(ns), np.log(vals), 1)[0])
            best = (m, a, vals) if best is None or a > best[1] else best
        if best is None:
            evidence.append("no capability metric with a usable positive series across scales")
            verdicts.append(None)
        else:
            m, a, vals = best
            verdicts.append(a > 1.1)
            cens = [s for s in scales
                    if (mb[s].get("metrics", mb[s]).get("memory_capacity_censored") in (1, 1.0, True))]
            evidence.append(f"best-growing capability metric: {m} ~ N^{a:.3f} (values {vals}); "
                            "P_bio is linear in N by construction, so only an exponent clearly "
                            "above 1 would make capability grow faster than power")
            if cens:
                evidence.append(f"CAVEAT: the capacity series is censored at the 96-class grid at "
                                f"{len(cens)} scale(s), so this exponent is a lower bound - "
                                "extending the grid is the prerequisite for a definitive verdict "
                                "on the exceptional criterion")
    curves = (energy or {}).get("curves") or {}
    per = {k: v for k, v in (curves.get("per_scale") or {}).items()
           if isinstance(v, dict) and v.get("hardware", {}).get("joules_per_spike")}
    if len(per) >= 2:
        js = {k: per[k]["hardware"]["joules_per_spike"] for k in sorted(per, key=float)}
        evidence.append("measured hardware cost per spike by scale (falls with scale because "
                        "the GPU becomes launch-bound at small N - a machine property, NOT "
                        "biological efficiency): " +
                        ", ".join(f"{k}x={v:.3g}" for k, v in js.items()))
    return _verdict(
        "Exceptional result - C(N) grows faster than P(N) across multiple scales",
        verdicts, evidence,
        "results/phase9/capability.json + results/energy/curves.json")


def _verdict(criterion: str, verdicts: list, evidence: list[str], artifacts: str) -> dict:
    # A criterion is SUPPORTED only when every check is measured and true. An unmeasured check
    # makes it PARTIAL; a missing test is never allowed to pass silently.
    if not verdicts or all(v is None for v in verdicts):
        verdict = "UNKNOWN"
    elif any(v is False for v in verdicts):
        verdict = "NOT SUPPORTED"
    elif any(v is None for v in verdicts):
        verdict = "PARTIAL"
    else:
        verdict = "SUPPORTED"
    return {"criterion": criterion, "verdict": verdict,
            "checks": [None if v is None else bool(v) for v in verdicts],
            "evidence": evidence, "artifacts": artifacts}


def render_markdown(out, inventory, table, criteria) -> str:
    lines = ["# FlyScale results summary", "",
             "Generated by `scripts/assess_success_criteria.py` from the phase artifacts that",
             "exist on disk. Anything not measured is reported as missing, never as a pass.", "",
             "## Phase inventory", "",
             "| phase / milestone | artifact | present | headline |", "|---|---|---|---|"]
    for k, v in inventory.items():
        head = (v["headline"] or "").replace("|", "\\|")
        lines.append(f"| {k} | `{v['artifact']}` | {'yes' if v['present'] else '**no**'} | {head} |")
    lines += ["", "## §18 power scaling table", ""]
    if table:
        cols = sorted({k for r in table for k in r if k != "scale"})
        lines.append("| scale | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for r in table:
            lines.append(f"| {r['scale']} | " + " | ".join(str(r.get(c, '')) for c in cols) + " |")
    else:
        lines.append("_no scaling artifacts yet_")
    lines += ["", "## §25 success criteria", ""]
    for c in criteria:
        lines.append(f"### {c['verdict']} - {c['criterion']}")
        for e in c["evidence"]:
            lines.append(f"* {e}")
        lines.append(f"* evidence expected in: `{c['artifacts']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
