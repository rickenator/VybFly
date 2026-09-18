"""Collect every number the paper needs from the phase artifacts into one JSON.

Rationale: the paper must not contain a number that cannot be traced to an artifact. Each field
records the file it came from; anything missing is recorded as None with the expected path, so the
paper can state "not measured" instead of inventing a value.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "paper" / "data.json"
SOURCES: dict[str, str] = {}


def load(rel: str):
    p = RES / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def pick(d, path, default=None):
    cur = d
    for part in path.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


data: dict = {"_sources": SOURCES, "_missing": []}


def get(key: str, rel: str, path: str, default=None):
    d = load(rel)
    if d is None:
        data[key] = None
        data["_missing"].append(f"{key} (file {rel} missing)")
        return None
    v = pick(d, path, default)
    if v is None:
        data["_missing"].append(f"{key} (path {rel}:{path} missing)")
    data[key] = v
    SOURCES[key] = f"results/{rel}:{path}"
    return v


# ---------------------------------------------------------------- Phase 0 / dataset
get("p0_gate", "phase0/gate_result.json", "passed")
get("p0_required_total", "phase0/gate_result.json", "required_checks_total")
get("p0_required_passed", "phase0/gate_result.json", "required_checks_passed")
get("p0_advisory_total", "phase0/gate_result.json", "advisory_checks_total")
get("p0_advisory_passed", "phase0/gate_result.json", "advisory_checks_passed")
get("p0_version", "phase0/gate_result.json", "canonical_version")
get("p0_derived", "phase0/gate_result.json", "derived")
d0 = load("phase0/flywire_baseline.json")
if d0:
    cc = d0.get("canonical_counts") or {}
    data["dataset"] = {**cc, "synapse_threshold_for_scaling": d0.get("synapse_threshold"),
                       "canonical_version": d0.get("canonical_version"),
                       "threshold_note": d0.get("threshold_note")}
    SOURCES["dataset"] = "results/phase0/flywire_baseline.json:canonical_counts"
    data["p0_gate_checks"] = [
        {"name": c.get("name"), "advisory": c.get("advisory"), "passed": c.get("passed"),
         "comment": c.get("comment")} for c in (d0.get("gate") or []) if isinstance(c, dict)]
    if not data["p0_gate_checks"]:
        # the gate result file carries the checks; the baseline file may not
        g0 = load("phase0/gate_result.json") or {}
        data["p0_gate_checks"] = [
            {"name": c.get("name"), "advisory": c.get("advisory"), "passed": c.get("passed"),
             "comment": c.get("comment")} for c in (g0.get("checks") or [])]
        SOURCES["p0_gate_checks"] = "results/phase0/gate_result.json:checks"
    else:
        SOURCES["p0_gate_checks"] = "results/phase0/flywire_baseline.json:gate"
    data["p0_blocks"] = sorted((d0.get("blocks") or {}).keys())
    data["phase0_variants"] = sorted((d0.get("variants") or {}).keys())
else:
    data["dataset"] = None
    data["_missing"].append("dataset (phase0/flywire_baseline.json missing)")


# ---------------------------------------------------------------- Phase 4 / geometry
g4 = load("phase4/geometry.json")
if g4:
    rows = []
    for r in (g4.get("comparison_table") or []):
        rows.append({
            "geometry": r.get("geometry"),
            "auc": r.get("auc"),
            "ap": r.get("ap"),
            "loglik": r.get("log_likelihood_mean"),
            "R": r.get("R"),
            "T": r.get("T"),
        })
    dob = g4.get("degree_only_baseline") or {}
    doh = dob.get("heldout") or {}
    if doh:
        rows.append({"geometry": "degree_only_baseline", "auc": doh.get("auc"),
                     "ap": doh.get("average_precision"),
                     "loglik": doh.get("log_likelihood_mean"), "R": None, "T": None})
    rows.sort(key=lambda r: -(r["auc"] or 0))
    data["geometry_rows"] = rows
    SOURCES["geometry_rows"] = "results/phase4/geometry.json:comparison_table"
    data["geometry_headline"] = g4.get("headline")
    data["geometry_protocol"] = g4.get("protocol")
    data["geometry_config"] = g4.get("config")
    data["geometry_sensitivity"] = g4.get("hyperbolic_budget_sensitivity")
    data["geometry_ranking"] = g4.get("ranking_by_heldout_auc")
    for k in ("geometry_headline", "geometry_protocol", "geometry_config", "geometry_sensitivity",
              "geometry_ranking"):
        SOURCES[k] = f"results/phase4/geometry.json:{k.replace('geometry_', '')}"
    if not rows:
        data["_missing"].append("geometry_rows (comparison_table empty)")
else:
    data["geometry_rows"] = None
    data["_missing"].append("geometry_rows (phase4/geometry.json missing)")


# ---------------------------------------------------------------- Phase 5 / downscale
for tag, rel in (("downscale_anatomical", "phase5/downscale.json"),
                 ("downscale_hyperbolic", "phase5/downscale_hyperbolic.json")):
    d = load(rel)
    rows = []
    if d:
        for k, v in sorted((d.get("factors") or {}).items(), key=lambda kv: -float(kv[0])):
            r, c = v.get("replica") or {}, v.get("closure") or {}
            rows.append({"factor": float(k), "n_neurons": r.get("n_neurons"),
                         "n_connections": r.get("n_connections"),
                         "mean_out_degree": r.get("mean_out_degree"),
                         "mean_strength": r.get("mean_synapses_per_connection"),
                         "composite": c.get("composite"),
                         "degree_wasserstein": c.get("degree_wasserstein_normalised"),
                         "ari": c.get("community_ari")})
        SOURCES[tag] = f"results/{rel}:factors"
    data[tag] = rows or None
    if not rows:
        data["_missing"].append(f"{tag} ({rel})")

for tag, rel in (("dynamics_anatomical", "phase5/dynamics.json"),
                 ("dynamics_hyperbolic", "phase5/dynamics_hyperbolic.json")):
    d = load(rel)
    rows = []
    if d:
        for k, v in sorted((d.get("replicas") or {}).items(), key=lambda kv: -float(kv[0])):
            row = {"factor": float(k), **(v.get("comparison") or {})}
            casc = v.get("cascade") or {}
            row["final_active_fraction"] = casc.get("final_active_fraction")
            row["latency_steps_to_half"] = casc.get("latency_steps_to_half")
            row["active_curve"] = casc.get("active_curve")
            row["active_fraction_curve"] = casc.get("active_fraction_curve")
            row["n_neurons"] = casc.get("n_neurons")
            rows.append(row)
        SOURCES[tag] = f"results/{rel}:replicas"
        ref = d.get("reference") or {}
        data[tag.replace("dynamics", "cascade_ref")] = {
            "final_active_fraction": ref.get("final_active_fraction"),
            "latency_steps_to_half": ref.get("latency_steps_to_half"),
            "active_curve": ref.get("active_curve"),
            "n_seeds": ref.get("n_seeds"),
            "relative_threshold": ref.get("relative_threshold")}
    data[tag] = rows or None

# ---------------------------------------------------------------- Phase 6 / upscale
d6 = load("phase6/upscale.json")
rows = []
if d6:
    for k, v in sorted((d6.get("scales") or {}).items(), key=lambda kv: float(kv[0])):
        r, c = v.get("replica") or {}, v.get("closure") or {}
        rows.append({"factor": float(k), "n_neurons": r.get("n_neurons"),
                     "n_connections": r.get("n_connections"), "n_synapses": r.get("n_synapses"),
                     "mean_out_degree": r.get("mean_out_degree"),
                     "mean_strength": r.get("mean_synapses_per_connection"),
                     "composite": c.get("composite"),
                     "degree_wasserstein": c.get("degree_wasserstein_normalised"),
                     "ari": c.get("community_ari")})
    data["upscale"] = rows
    SOURCES["upscale"] = "results/phase6/upscale.json:scales"
    data["upscale_geometry"] = (d6.get("geometry") or {}).get("source")
    data["upscale_subdivision"] = d6.get("node_subdivision")
else:
    data["upscale"] = None
    data["_missing"].append("upscale (phase6/upscale.json missing)")

# reference counts for the 1x graph, taken from whatever the pipeline recorded
d5 = load("phase5/downscale.json")
data["reference"] = (d5 or {}).get("reference")
if data.get("reference"):
    SOURCES["reference"] = "results/phase5/downscale.json:reference"

# ---------------------------------------------------------------- Phase 7 / closure
d7 = load("phase7/closure.json")
rows = []
if d7:
    for k, v in sorted((d7.get("scales") or {}).items(), key=lambda kv: float(kv[0])):
        c = v.get("closure") or {}
        r = v.get("renormalised") or v.get("renormalized") or {}
        rows.append({"factor": float(k), "n_neurons": r.get("n") or r.get("n_neurons"),
                     "n_connections": r.get("edges") or r.get("n_connections"),
                     "n_synapses": r.get("synapses"),
                     "source_n": (v.get("source_graph") or {}).get("n"),
                     "composite": c.get("composite"),
                     "degree_wasserstein": c.get("degree_wasserstein_normalised"),
                     "motif_l1": c.get("motif_divergence_L1"),
                     "ari": c.get("community_ari"),
                     "spectral": c.get("spectral_distance")})
    data["closure"] = rows
    SOURCES["closure"] = "results/phase7/closure.json:scales"
    data["closure_geometry"] = (d7.get("geometry") or {}).get("source")
else:
    data["closure"] = None
    data["_missing"].append("closure (phase7/closure.json missing)")

# ---------------------------------------------------------------- Phase 1 / LIF baseline
d1 = load("phase1/lif_baseline.json")
if d1:
    data["lif_top_keys"] = sorted(d1)[:40]
    SOURCES["lif_top_keys"] = "results/phase1/lif_baseline.json"
    data["lif_graph"] = d1.get("graph")
    data["lif_gate"] = d1.get("engine_agreement_gate")
    data["lif_scan"] = d1.get("operating_point_scan")
    data["lif_levels"] = d1.get("level1_stimulus_sets")
else:
    data["lif_top_keys"] = None
    data["_missing"].append("lif_baseline.json missing")

# ---------------------------------------------------------------- Phase 2 / Vyb DES
d2 = load("phase2/vyb_des_check.json")
data["des_verdict"] = pick(d2, "verdict") if d2 else None
data["des_fixture"] = pick(d2, "fixture") if d2 else None
data["des_vyb_summary"] = pick(d2, "vyb_summary") if d2 else None
data["des_comparisons"] = pick(d2, "comparisons") if d2 else None
SOURCES["des_verdict"] = "results/phase2/vyb_des_check.json:verdict"

# ---------------------------------------------------------------- Phase 3 / GPU
d3 = load("phase3/gpu_equivalence.json")
data["gpu_gate"] = {"passed": pick(d3, "passed"), "checks": pick(d3, "checks"),
                    "launch_errors": pick(d3, "launch_errors_reported_by_gpu")} if d3 else None
SOURCES["gpu_gate"] = "results/phase3/gpu_equivalence.json"

# ---------------------------------------------------------------- Phase 8 / plasticity
d8 = load("phase8/learning.json")
if d8:
    data["mb_attribution"] = d8.get("attribution")
    data["mb_aggregate_keys"] = sorted(d8.get("aggregate") or {})
    data["mb_annotations"] = d8.get("annotations")
    SOURCES["mb_attribution"] = "results/phase8/learning.json:attribution"
else:
    data["mb_attribution"] = None
    data["_missing"].append("phase8/learning.json missing")

# ---------------------------------------------------------------- Phase 9 / capability
d9 = load("phase9/capability.json")
if d9:
    mb = d9.get("metrics_by_scale") or {}
    rows = []
    for s in sorted(mb, key=float):
        m = mb[s].get("metrics", mb[s])
        rows.append({"scale": float(s), "n_neurons": m.get("n_neurons"),
                     "min_delta": m.get("min_separable_delta"),
                     "mi_delta_0_1": m.get("discrimination_bits_at_overlap_0_9"),
                     "memory_capacity": m.get("memory_capacity"),
                     "memory_censored": m.get("memory_capacity_censored"),
                     "temporal_depth": m.get("temporal_depth_steps"),
                     "sequence_depth": m.get("sequence_depth"),
                     "participation_ratio": m.get("participation_ratio"),
                     "gen_gap": m.get("generalization_gap"),
                     "robust_syn": m.get("robustness_half_degradation_rate_synapses"),
                     "total_seconds": m.get("scale_total_seconds")})
    data["capability"] = rows
    SOURCES["capability"] = "results/phase9/capability.json:metrics_by_scale"
    data["capability_fits"] = d9.get("fits")
    data["capability_protocol"] = d9.get("protocol")
else:
    data["capability"] = None
    data["_missing"].append("phase9/capability.json missing")

# ---------------------------------------------------------------- energy
dhe = load("energy/hardware_energy.json")
dbio = load("energy/biological_model.json")
dcur = load("energy/curves.json")
data["energy_hardware"] = {
    "devices": pick(dhe, "devices"),
    "measurement": pick(dhe, "measurement"),
    "unavailable": pick(dhe, "unavailable_metrics"),
} if dhe else None
SOURCES["energy_hardware"] = "results/energy/hardware_energy.json"
data["energy_biological"] = {
    "model": pick(dbio, "model"), "anchor": pick(dbio, "anchor"),
    "P0_watts": pick(dbio, "P0_watts"), "N0_neurons": pick(dbio, "N0_neurons"),
    "per_neuron_picowatts": pick(dbio, "per_neuron_picowatts"),
    "mammalian": pick(dbio, "mammalian_anchor") or pick(dbio, "cross_check"),
} if dbio else None
SOURCES["energy_biological"] = "results/energy/biological_model.json"
if dcur:
    rows = []
    for k, v in sorted((dcur.get("per_scale") or {}).items(), key=lambda kv: float(kv[0])):
        h, b = v.get("hardware") or {}, v.get("biological") or {}
        rows.append({"scale": float(k), "n_neurons": v.get("n_neurons"),
                     "gpu_joules": h.get("gpu_joules"), "j_per_spike": h.get("joules_per_spike"),
                     "j_per_event": h.get("joules_per_synaptic_event"),
                     "gpu_mean_watts": h.get("gpu_mean_watts"),
                     "wall_s": v.get("wall_seconds"),
                     "bio_watts": b.get("power_watts"),
                     "cap_per_watt": (v.get("capability_per_watt") or {}).get("memory_capacity_per_watt")})
    data["energy_curves"] = rows
    data["energy_curves_protocol"] = dcur.get("protocol")
    SOURCES["energy_curves"] = "results/energy/curves.json:per_scale"
else:
    data["energy_curves"] = None

# ---------------------------------------------------------------- assessment (verdicts)
da = load("assessment.json")
if da:
    crit = []
    for c in (da.get("criteria") if isinstance(da.get("criteria"), list) else []):
        crit.append(c)
    data["criteria"] = crit or da.get("success_criteria")
    data["milestones"] = da.get("milestones") or da.get("inventory")
    SOURCES["criteria"] = "results/assessment.json"
else:
    data["criteria"] = None

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=1, default=str) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size} bytes)")
print(f"missing fields: {len(data['_missing'])}")
for m in data["_missing"][:12]:
    print("  -", m)
