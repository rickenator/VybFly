"""Phase 9 / M12-M14: capability scaling benchmarks across connectome scales (1x, 2x, 5x, 10x).

    python scripts/phase9_capability.py --stage all
    python scripts/phase9_capability.py --stage replicas
    python scripts/phase9_capability.py --stage calibrate
    python scripts/phase9_capability.py --stage run --scales 1 2 5 10
    python scripts/phase9_capability.py --stage aggregate

Stages
    replicas    build the reference graph (canonical v783, 5-synapse rule) and the subdivided
                replicas with flyscale.renorm.upscale, cache them under results/phase9/replicas/,
                and record the geometry law, seeds and timings used to build them.
    calibrate   fix the cascade's relative threshold on the 1x graph by the rule declared in
                flyscale.capability.calibrate_relative_threshold. Run once; the chosen value is
                stored in results/phase9/calibration.json and applied at every scale.
    run         run the full battery at the requested scales and write one JSON + response-matrix
                artifact per scale. Each scale is a separate process invocation so a slow 10x run
                cannot lose the 1x-5x results.
    aggregate   fit capability-vs-N and capability-vs-synapses curves across the scales that ran,
                merge everything into results/phase9/capability.json, and write
                results/phase9/SUMMARY.txt.

Everything is seeded (--seed), every protocol number is recorded in the output, and every failure
is written as an explicit error field rather than being skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from flyscale import capability as cap
from flyscale import renorm, synthetic
from flyscale.connectome import Connectome
from flyscale.renorm import GeometryLaw, fit_connection_law
from flyscale.synthetic import GraphView, from_connectome

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "processed" / "canonical_v783"
PHASE9 = ROOT / "results" / "phase9"
REPLICAS = PHASE9 / "replicas"
ARTIFACTS = PHASE9 / "artifacts"
SCALES_DIR = PHASE9 / "scales"
DEFAULT_SCALES = (1.0, 2.0, 5.0, 10.0)


# --------------------------------------------------------------------------- io helpers
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_strip(payload), indent=2, sort_keys=True, default=str) + "\n")


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [_strip(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def module_hash() -> str:
    """sha256 of src/flyscale/capability.py - identifies the code that produced a result."""
    p = ROOT / "src" / "flyscale" / "capability.py"
    if not p.exists():
        return "(src/flyscale/capability.py not found)"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def environment_note() -> dict:
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, timeout=10).stdout.strip()
    except Exception as exc:                                  # pragma: no cover
        commit, dirty = f"unavailable ({exc})", ""
    meta = read_json(CANON / "meta.json", {}) or {}
    commit = commit.strip()
    if not commit or commit in ("HEAD", "master"):
        commit = "(repository has no commits yet - working tree only)"
    return {
        "dataset": str(CANON.relative_to(ROOT)),
        "dataset_version": meta.get("canonical_version"),
        "n_dataset_neurons": (meta.get("counts") or {}).get("n_neurons"),
        "git_commit": commit or "(no commits in repository)",
        "git_working_tree_dirty": bool(dirty),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def dataset_hash() -> str:
    h = hashlib.sha256()
    for name in ("meta.json",):
        p = CANON / name
        if p.exists():
            h.update(p.read_bytes())
    for name in ("pairs.parquet", "neurons.parquet"):
        p = CANON / name
        if p.exists():
            h.update(f"{name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- geometry
def anatomical_coords(scale: bool = True) -> np.ndarray:
    """Annotation coordinates (xyz, standardised). The Phase 4 embedding directory is empty, so the
    verified anatomical geometry is the only geometry law available; the choice is recorded."""
    import pandas as pd
    ann = pd.read_parquet(CANON / "neurons.parquet",
                          columns=["idx", "pos_x", "pos_y", "pos_z"]).sort_values("idx")
    xyz = np.array(ann[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64), copy=True)
    bad = ~np.isfinite(xyz).all(axis=1)
    if bad.any():
        xyz[bad] = np.nanmean(xyz[~bad], axis=0)
    if scale:
        xyz = (xyz - xyz.mean(axis=0)) / np.maximum(xyz.std(axis=0), 1e-9)
    return xyz


def load_geometry(g: GraphView, seed: int = 0, refit: bool = False) -> GeometryLaw:
    """Geometry law with the fitted connection law cached in results/phase9/geometry.json."""
    cache = PHASE9 / "geometry.json"
    coords = anatomical_coords()
    if coords.shape[0] != g.n:
        raise SystemExit(f"coordinate count {coords.shape[0]} != graph n {g.n}")
    entry = read_json(cache, {}) or {}
    if entry.get("R") is not None and not refit:
        law = GeometryLaw(coords=coords, kind="euclidean", R=float(entry["R"]),
                          T=float(entry["T"]), source=entry.get("source", "annotation_xyz"))
        return law
    t0 = time.time()
    fit = fit_connection_law(coords, g.pre, g.post, kind="euclidean", seed=seed)
    law = GeometryLaw(coords=coords, kind="euclidean", R=float(fit["R"]), T=float(fit["T"]),
                      source="annotation_xyz+law_fit")
    write_json(cache, {"kind": "euclidean", "source": law.source, "R": law.R, "T": law.T,
                       "fit": {k: v for k, v in fit.items() if k != "confusion"},
                       "seconds": round(time.time() - t0, 2),
                       "note": ("P(connect)=1/(1+exp((d-R)/T)) fitted by flyscale.renorm."
                                "fit_connection_law on standardised anatomical xyz; no Phase 4 "
                                "embedding exists in results/phase4/artifacts")})
    return law


def build_replicas(args) -> dict:
    """Build (or load) the 1x reference and the upscaled replicas; record what was used."""
    REPLICAS.mkdir(parents=True, exist_ok=True)
    summary_path = PHASE9 / "replicas.json"
    existing = read_json(summary_path, {}) or {}
    result: dict = {"reference_threshold": args.threshold, "seed": args.seed,
                    "scales": existing.get("scales", {}), "errors": existing.get("errors", {})}
    wanted = [float(s) for s in args.scales]
    t_all = time.time()
    g1 = None
    for scale in wanted:
        key = f"{scale:g}"
        tag = f"g{scale:g}"
        path = REPLICAS / tag
        need_build = args.force_replicas or not (path / "graph.npz").exists()
        t0 = time.time()
        if not need_build:
            try:
                g = synthetic.load_graph(path)
                result["scales"][key] = {"source": "cached", "path": str(path.relative_to(ROOT)),
                                         "graph": g.summary(), "seconds_load": round(time.time() - t0, 2)}
                print(f"[replicas] {key}x loaded from cache: {json.dumps(g.summary())}")
                continue
            except Exception as exc:
                result["errors"][key] = f"cached replica unreadable: {exc}"
                need_build = True
        if scale == 1.0:
            c = Connectome(CANON)
            c5 = c.thresholded(args.threshold)
            g = from_connectome(c5, provenance={
                "kind": f"canonical_v783_thr{args.threshold}", "n_source": int(c.n),
                "source_pairs": int(c.pairs.shape[0]), "threshold": args.threshold,
                "operation": "identity (1x reference)"})
            prov_extra = {"operation": "identity"}
            seconds_build = time.time() - t0
        else:
            if g1 is None:
                g1 = from_connectome(Connectome(CANON).thresholded(args.threshold),
                                     provenance={"kind": f"canonical_v783_thr{args.threshold}"})
            law = load_geometry(g1, seed=args.seed, refit=args.refit_geometry)
            t_build = time.time()
            g = renorm.upscale(g1, law, scale, seed=args.seed,
                               rewire_fraction=args.rewire_fraction,
                               sibling_edges=not args.no_sibling_edges,
                               sibling_prob_scale=args.sibling_prob_scale)
            seconds_build = time.time() - t_build
            prov_extra = {"operation": "renorm.upscale", "factor": scale,
                          "rewire_fraction": args.rewire_fraction,
                          "sibling_edges": not args.no_sibling_edges,
                          "sibling_prob_scale": args.sibling_prob_scale,
                          "geometry": law.note(), "geometry_R": law.R, "geometry_T": law.T}
        synthetic.save_graph(g, path)
        result["scales"][key] = {
            "source": "built", "path": str(path.relative_to(ROOT)), "graph": g.summary(),
            "seconds_build": round(seconds_build, 1),
            "seconds_total": round(time.time() - t0, 1), "parameters": prov_extra,
            "provenance": _strip(g.provenance)}
        print(f"[replicas] {key}x built in {seconds_build:.1f}s: {json.dumps(g.summary())}")
        write_json(summary_path, result)
    result["seconds_total"] = round(time.time() - t_all, 1)
    write_json(summary_path, result)
    return result


def load_scale_graph(scale: float, args) -> tuple[GraphView, dict]:
    """Load a replica from the cache, or build it in-process if the cache is missing."""
    path = REPLICAS / f"g{scale:g}"
    meta = (read_json(PHASE9 / "replicas.json", {}) or {}).get("scales", {}).get(f"{scale:g}", {})
    if (path / "graph.npz").exists():
        t0 = time.time()
        g = synthetic.load_graph(path)
        info = {"source": "cache", "path": str(path.relative_to(ROOT)),
                "seconds_load": round(time.time() - t0, 2)}
        g.provenance = dict(g.provenance or {})
        return g, info
    if scale == 1.0:
        t0 = time.time()
        c = Connectome(CANON)
        g = from_connectome(c.thresholded(args.threshold),
                            provenance={"kind": f"canonical_v783_thr{args.threshold}",
                                        "operation": "identity"})
        return g, {"source": "built_in_process", "seconds_load": round(time.time() - t0, 2)}
    t0 = time.time()
    g1 = from_connectome(Connectome(CANON).thresholded(args.threshold),
                         provenance={"kind": f"canonical_v783_thr{args.threshold}"})
    law = load_geometry(g1, seed=args.seed, refit=args.refit_geometry)
    g = renorm.upscale(g1, law, scale, seed=args.seed, rewire_fraction=args.rewire_fraction,
                       sibling_edges=not args.no_sibling_edges,
                       sibling_prob_scale=args.sibling_prob_scale)
    return g, {"source": "built_in_process", "seconds_load": round(time.time() - t0, 1),
               "warning": "replica cache was missing; rebuilt with the recorded parameters"}


# --------------------------------------------------------------------------- stage: run
class Budget:
    """Stop adding batteries when the per-scale wall budget is exhausted; record the truncation."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def ok(self) -> bool:
        return self.elapsed() < self.seconds


def run_scale(scale: float, args) -> dict:
    t_scale = time.time()
    out_path = SCALES_DIR / f"scale_{scale:g}.json"
    g, load_info = load_scale_graph(scale, args)
    n_edges = int(g.pre.size)
    n_syn = int(g.syn.sum())
    proto = cap.Protocol(seed=args.seed, relative_threshold=args.relative_threshold,
                         n_classes=args.n_classes,
                         readout_fraction=args.readout_fraction,
                         pool_fraction=args.pool_fraction,
                         sample_fraction=args.sample_fraction,
                         reference_overlap=args.reference_overlap)
    budget = Budget(args.time_budget_s)
    readout = cap.readout_indices(g, proto)
    print(f"[run] scale {scale:g}x: N={g.n} E={n_edges} syn={n_syn} readout={readout.size} "
          f"pools={proto.n_classes} RT={proto.relative_threshold}", flush=True)

    entry: dict = {
        "scale": float(scale),
        "graph": g.summary(),
        "n_neurons": int(g.n),
        "n_connections": n_edges,
        "n_synapses": n_syn,
        "mean_synapses_per_connection": float(g.syn.mean()) if n_syn else None,
        "synapse_scale_threshold_unit": float(cap.synapse_scale(g)),
        "absolute_threshold": float(cap.normalised_threshold(g, proto.relative_threshold)),
        "readout_dim": int(readout.size),
        "readout_fraction_actual": float(readout.size / g.n),
        "load_info": load_info,
        "raw_replica_provenance": _strip(g.provenance),
        "capability": {},
        "cost": {},
        "errors": {},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = f"s{scale:g}"

    def _progress(name):
        state = {"done": 0, "total": 0, "last": 0.0}

        def cb(done, total):
            state["done"], state["total"] = done, total
            now = time.time()
            if done == total or now - state["last"] > 30:
                state["last"] = now
                print(f"[run] {tag} {name}: {done}/{total} cascades "
                      f"({budget.elapsed():.0f}s elapsed)", flush=True)
        return cb

    def _save_features(bat: dict) -> dict:
        """Persist the response matrix (uint8) and its trial metadata; return the metadata."""
        keep = {k: v for k, v in bat.items() if k != "features"}
        keep["features_shape"] = list(bat["features"].shape)
        if "pools" in keep:                                   # index arrays: not JSON friendly
            keep["pools"] = [{"overlap": p["overlap"], "delta": p["delta"], "pair": p["pair"],
                              "pool_size": int(p["A"].size)} for p in keep["pools"]]
        fpath = ARTIFACTS / f"{tag}_{bat['name']}_features.npy"
        np.save(fpath, bat["features"])
        tpath = ARTIFACTS / f"{tag}_{bat['name']}_meta.json"
        write_json(tpath, keep)
        keep["features_path"] = str(fpath.relative_to(ROOT))
        keep["meta_path"] = str(tpath.relative_to(ROOT))
        return keep

    def _battery(name: str, fn, cost_key: str, error_key: str):
        """Run one battery with budget guard + error recording; returns the battery or None."""
        if not budget.ok():
            entry.setdefault("skipped_for_budget", []).append(name)
            print(f"[run] {tag} {name}: SKIPPED, per-scale budget of {budget.seconds:.0f}s reached",
                  flush=True)
            return None
        try:
            t0 = time.time()
            bat = fn()
            entry["cost"][f"battery_{cost_key}_seconds"] = round(time.time() - t0, 1)
            return bat
        except Exception as exc:                               # noqa: BLE001 - recorded, not hidden
            entry["errors"][error_key] = f"{type(exc).__name__}: {exc}"
            print(f"[run] {tag} {name} battery FAILED: {exc}", flush=True)
            return None

    # ---- battery 1: stimulus classes (memory capacity + generalization + complexity)
    bat_classes = _battery(
        "classes", lambda: cap.run_class_battery(g, proto, readout,
                                                 on_trial=_progress("classes")),
        "classes", "class_battery")
    if bat_classes is not None:
        entry["battery_classes"] = _save_features(bat_classes)
        entry["capability"]["memory_capacity"] = cap.memory_capacity(bat_classes, proto)
        entry["capability"]["generalization"] = cap.generalization(bat_classes, proto)
        entry["capability"]["dynamical_complexity"] = cap.dynamical_complexity(
            bat_classes, proto, bat_classes["union_fraction"])

    # ---- battery 2: sensory discrimination
    bat_disc = _battery(
        "discrimination",
        lambda: cap.run_discrimination_battery(g, proto, readout,
                                               on_trial=_progress("discrimination")),
        "discrimination", "discrimination_battery")
    if bat_disc is not None:
        entry["capability"]["sensory_discrimination"] = cap.discrimination(bat_disc, proto)
        meta = _save_features(bat_disc)
        entry["battery_discrimination"] = {k: v for k, v in meta.items() if k != "pools"}

    # ---- battery 3: temporal depth
    bat_temp = _battery(
        "temporal",
        lambda: cap.run_temporal_battery(g, proto, readout, on_trial=_progress("temporal")),
        "temporal", "temporal_battery")
    if bat_temp is not None:
        entry["capability"]["temporal_depth"] = cap.temporal_depth(bat_temp, proto)
        entry["battery_temporal"] = _save_features(bat_temp)

    # ---- battery 4: sequence learning
    bat_seq = _battery(
        "sequence",
        lambda: cap.run_sequence_battery(g, proto, readout, on_trial=_progress("sequence")),
        "sequence", "sequence_battery")
    if bat_seq is not None:
        entry["capability"]["sequence_learning"] = cap.sequence_learning(bat_seq, proto)
        entry["battery_sequence"] = _save_features(bat_seq)

    # ---- battery 5: robustness
    bat_rob = _battery(
        "robustness",
        lambda: cap.run_robustness_battery(g, proto, readout, on_trial=_progress("robustness")),
        "robustness", "robustness_battery")
    if bat_rob is not None:
        entry["capability"]["robustness"] = cap.robustness(bat_rob, proto)
        entry["battery_robustness"] = _save_features(bat_rob)

    # ---- activity / cost bookkeeping (§18 asks for these alongside capability)
    newly = entry.get("battery_classes", {}).get("newly_active")
    if newly is not None:
        arr = np.asarray(newly, dtype=np.float64)
        steps = arr.shape[1]
        entry["activity"] = {
            "steps_per_stimulus": int(steps),
            "mean_active_fraction_per_trial": float(np.mean(
                entry["battery_classes"]["final_active_fraction"])),
            "mean_neurons_recruited_per_trial": float(np.mean(arr.sum(axis=1))),
            "mean_newly_active_per_step": float(arr.mean()),
            "neurons_active_at_last_step": float(np.mean(arr[:, -1])) if steps else None,
            "note": ("the cascade is a discrete-step model: a step is one synchronous update, not "
                     "a declared duration, so no spikes-per-biological-second is asserted here"),
        }
    entry["cost"]["scale_total_seconds"] = round(time.time() - t_scale, 1)
    battery_seconds = sum(v for k, v in entry["cost"].items() if k.startswith("battery_"))
    cascades = sum(int((entry.get(f"battery_{b}") or {}).get("n_trials") or 0)
                   for b in ("classes", "discrimination", "temporal", "sequence", "robustness"))
    entry["cost"]["cascades_total"] = int(cascades)
    entry["cost"]["seconds_per_cascade"] = round(battery_seconds / max(cascades, 1), 4)
    entry["cost"]["wall_seconds_per_1000_stimuli"] = round(
        1000.0 * entry["cost"]["scale_total_seconds"] / max(cascades, 1), 3)
    entry["cost"]["cpu_count"] = args.cpu_count_note
    entry["cost"]["shared_machine_note"] = (
        "wall seconds were measured while other phase jobs of this project were running on the "
        "same 20-core host, so they are an upper bound on an idle machine's cost; cascades_total "
        "and seconds_per_cascade are reported so the cost can be re-derived from work done rather "
        "than from a contended clock")
    entry["cost"]["energy"] = read_energy_note()
    entry["protocol"] = proto.note()
    entry["protocol_fingerprint"] = cap.fingerprint(proto)
    entry["truncated_by_budget"] = not budget.ok()
    write_json(out_path, entry)
    print(f"[run] scale {scale:g}x done in {entry['cost']['scale_total_seconds']:.0f}s -> {out_path}",
          flush=True)
    return entry


def read_energy_note() -> dict:
    """Per-scale energy numbers if any work-stream has produced them; never a dependency."""
    candidates = sorted((ROOT / "results").glob("energy/**/*.json"))
    found = [str(p.relative_to(ROOT)) for p in candidates]
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for key in ("per_scale", "scales", "curves", "summary"):
            block = data.get(key) if isinstance(data, dict) else None
            if isinstance(block, dict) and any(str(s) in block for s in ("1", "1.0", "2", "5", "10")):
                return {"available": True, "source": str(p.relative_to(ROOT)), "field": key,
                        "data": _strip(block), "caveat": "produced by another work-stream; not used "
                                                         "to derive any capability number here"}
    return {"available": False, "files_seen": found,
            "note": "no per-scale energy results exist under results/energy; capability/energy "
                    "(§19, M14) cannot be computed from this work-stream's outputs alone"}


# --------------------------------------------------------------------------- stage: aggregate
def capability_metrics(scales: dict) -> dict:
    """Flat dictionary of scalar capability numbers per scale, for fitting."""
    out: dict = {}
    for key, entry in sorted(scales.items(), key=lambda kv: float(kv[0])):
        c = entry.get("capability", {})
        row: dict = {}
        dd = c.get("sensory_discrimination") or {}
        row["min_separable_delta"] = dd.get("min_separable_delta")
        curve_dd = dd.get("curve") or {}
        deltas_tested = sorted(float(k) for k in curve_dd)
        positive_deltas = [d for d in deltas_tested if d > 0]
        finest = positive_deltas[0] if positive_deltas else None
        row["discrimination_accuracy_at_finest_delta"] = (
            (curve_dd.get(f"{finest:.4f}") or {}).get("accuracy") if finest is not None else None)
        row["discrimination_bits_at_finest_delta"] = (
            (curve_dd.get(f"{finest:.4f}") or {}).get("mutual_information_bits")
            if finest is not None else None)
        row["discrimination_finest_delta_tested"] = finest
        row["min_separable_delta_at_grid_minimum"] = bool(
            dd.get("min_separable_delta") is not None and finest is not None
            and abs(dd["min_separable_delta"] - finest) < 1e-9)
        # MI at the identical-pool control (Delta = 0, overlap 1.0) and at two fixed overlaps;
        # computed from the stored curve so they do not depend on the level count
        row["discrimination_bits_identical_pools_control"] = _mi_at(dd, 1.0)
        row["discrimination_bits_at_overlap_0.95"] = _mi_at(dd, 0.95)
        row["discrimination_bits_at_overlap_0.9"] = _mi_at(dd, 0.9)
        row["discrimination_bits_at_overlap_0.6"] = _mi_at(dd, 0.6)
        row["accuracy_at_full_stimulus_difference"] = dd.get("accuracy_at_full_difference")
        mc = c.get("memory_capacity") or {}
        row["memory_capacity"] = mc.get("capacity")
        row["memory_capacity_censored"] = mc.get("censored_at_grid_max")
        row["memory_capacity_above_chance_only"] = mc.get("capacity_above_chance_only")
        _ct = mc.get("capacity_at_thresholds") or {}
        row["memory_capacity_at_0.95"] = (_ct.get("0.95") or {}).get("capacity")
        row["memory_capacity_at_0.99"] = (_ct.get("0.99") or {}).get("capacity")
        row["memory_capacity_accuracy_at_grid_max"] = mc.get("accuracy_at_grid_max")
        td = c.get("temporal_depth") or {}
        row["temporal_depth_steps"] = td.get("temporal_depth_steps")
        row["temporal_depth_delay_zero_accuracy"] = td.get("delay_zero_accuracy")
        sl = c.get("sequence_learning") or {}
        row["sequence_depth"] = sl.get("sequence_depth")
        gn = c.get("generalization") or {}
        row["generalization_novel_accuracy"] = (gn.get("novel") or {}).get("accuracy")
        row["generalization_matched_accuracy"] = (gn.get("matched") or {}).get("accuracy")
        row["generalization_gap"] = gn.get("generalization_gap")
        rb = c.get("robustness") or {}
        row["robustness_baseline_accuracy"] = rb.get("baseline_accuracy")
        row["robustness_half_degradation_rate_neurons"] = rb.get("half_degradation_rate_neurons")
        row["robustness_half_degradation_rate_synapses"] = rb.get("half_degradation_rate_synapses")
        row["robustness_accuracy_at_max_neuron_lesion"] = _last_accuracy(rb, "neurons")
        row["robustness_lowest_significant_drop_rate_neurons"] = rb.get(
            "lowest_rate_with_significant_drop_neurons")
        row["robustness_lowest_significant_drop_rate_synapses"] = rb.get(
            "lowest_rate_with_significant_drop_synapses")
        row["robustness_max_tolerated_rate_neurons"] = rb.get("maximum_tolerated_rate_neurons")
        row["robustness_max_tolerated_rate_synapses"] = rb.get("maximum_tolerated_rate_synapses")
        # MI-based degradation, recomputed here from the stored curves so it is available even for
        # runs produced before the MI fields were added to the battery output
        for kind in ("neurons", "synapses"):
            rows = (rb.get("curves") or {}).get(kind) or []
            if rows:
                row[f"robustness_accuracy_at_max_{kind}_lesion"] = rows[-1]["accuracy"]
            base_mi = rb.get("baseline_mutual_information_bits")
            if rows and base_mi:
                row[f"robustness_mi_retention_at_largest_rate_{kind}"] = float(
                    rows[-1]["mutual_information_bits"] / base_mi)
                row[f"robustness_mi_at_largest_rate_{kind}"] = float(
                    rows[-1]["mutual_information_bits"])
                row[f"robustness_mi_half_degradation_rate_{kind}"] = cap.half_degradation_rate(
                    [r["rate"] for r in rows],
                    [base_mi] + [r["mutual_information_bits"] for r in rows], floor=0.0)
        dx = c.get("dynamical_complexity") or {}
        ed = dx.get("effective_dimensionality") or {}
        row["participation_ratio"] = ed.get("participation_ratio")
        row["components_for_90pct_variance"] = ed.get("components_for_90pct_variance")
        re_ = dx.get("response_entropy") or {}
        row["mean_unit_entropy_bits"] = re_.get("mean_unit_binary_entropy_bits")
        row["distinct_pattern_ratio"] = re_.get("distinct_pattern_ratio")
        row["readout_unit_coverage"] = re_.get("readout_unit_coverage")
        sr = dx.get("stimulus_response_information") or {}
        row["stimulus_response_mi_bits"] = sr.get("mutual_information_bits")
        row["stimulus_response_normalised_mi"] = sr.get("normalised_mutual_information")
        row["stimulus_response_mi_at_ceiling"] = sr.get("at_or_above_ceiling")
        row["stimulus_response_mi_ceiling_bits"] = sr.get("mutual_information_ceiling_bits")
        rs = dx.get("recruitment_statistics") or {}
        row["branching_ratio_ratio_of_sums"] = rs.get("branching_ratio_ratio_of_sums")
        row["avalanche_size_exponent"] = rs.get("size_distribution_exponent")
        row["avalanche_size_exponent_r2"] = rs.get("size_distribution_r2")
        row["mean_recruits_per_step"] = rs.get("mean_recruits_per_step")
        ss = dx.get("state_space_coverage") or {}
        row["neurons_recruited_fraction"] = ss.get("neurons_recruited_fraction")
        row["mean_final_active_fraction"] = ss.get("mean_final_active_fraction")
        row["n_neurons"] = entry.get("n_neurons")
        row["n_synapses"] = entry.get("n_synapses")
        row["n_connections"] = entry.get("n_connections")
        row["readout_dim"] = entry.get("readout_dim")
        row["scale_total_seconds"] = (entry.get("cost") or {}).get("scale_total_seconds")
        row["errors"] = entry.get("errors") or {}
        row["protocol_fingerprint"] = entry.get("protocol_fingerprint")
        out[key] = row
    return out


def _mi_at(disc: dict, overlap: float) -> float | None:
    for v in (disc.get("curve") or {}).values():
        if abs(v.get("overlap", -1) - overlap) < 1e-9:
            return v.get("mutual_information_bits")
    return None


def _last_accuracy(rob: dict, kind: str) -> float | None:
    rows = (rob.get("curves") or {}).get(kind) or []
    return rows[-1]["accuracy"] if rows else None


#: metric -> (direction, bounded?) ; direction 1 = more is better, -1 = smaller is better
METRIC_DIRECTION = {
    "min_separable_delta": (-1, True),
    "accuracy_at_full_stimulus_difference": (1, True),
    "discrimination_bits_at_overlap_0.9": (1, False),
    "discrimination_bits_at_overlap_0.6": (1, False),
    "discrimination_bits_at_overlap_0.95": (1, False),
    "discrimination_bits_identical_pools_control": (0, True),
    "discrimination_bits_at_finest_delta": (1, False),
    "discrimination_accuracy_at_finest_delta": (1, True),
    "discrimination_finest_delta_tested": (0, False),  # protocol constant, not a capability
    "min_separable_delta_at_grid_minimum": (0, True),
    "robustness_accuracy_at_max_synapses_lesion": (1, True),
    "memory_capacity_at_0.95": (1, False),
    "memory_capacity_at_0.99": (1, False),
    "memory_capacity": (1, False),
    "memory_capacity_at_0.95": (1, False),
    "memory_capacity_at_0.99": (1, False),
    "memory_capacity_above_chance_only": (1, False),
    "memory_capacity_accuracy_at_grid_max": (1, True),
    "temporal_depth_steps": (1, False),
    "temporal_depth_delay_zero_accuracy": (1, True),
    "sequence_depth": (1, False),
    "generalization_novel_accuracy": (1, True),
    "generalization_matched_accuracy": (1, True),
    "generalization_gap": (-1, True),
    "robustness_baseline_accuracy": (1, True),
    "robustness_half_degradation_rate_neurons": (1, True),
    "robustness_half_degradation_rate_synapses": (1, True),
    "robustness_lowest_significant_drop_rate_neurons": (1, True),
    "robustness_lowest_significant_drop_rate_synapses": (1, True),
    "robustness_max_tolerated_rate_neurons": (1, True),
    "robustness_max_tolerated_rate_synapses": (1, True),
    "robustness_mi_retention_at_largest_rate_neurons": (1, True),
    "robustness_mi_retention_at_largest_rate_synapses": (1, True),
    "robustness_mi_half_degradation_rate_neurons": (1, True),
    "robustness_mi_half_degradation_rate_synapses": (1, True),
    "robustness_accuracy_at_max_neuron_lesion": (1, True),
    "participation_ratio": (1, False),
    "components_for_90pct_variance": (1, False),
    "mean_unit_entropy_bits": (1, True),
    "distinct_pattern_ratio": (1, True),
    "readout_unit_coverage": (1, True),
    "stimulus_response_mi_bits": (1, False),
    "stimulus_response_normalised_mi": (1, True),
    "stimulus_response_mi_at_ceiling": (0, True),
    "stimulus_response_mi_ceiling_bits": (0, False),   # = log2(n_classes), constant by design
    "branching_ratio_ratio_of_sums": (0, False),
    "avalanche_size_exponent": (0, False),
    "avalanche_size_exponent_r2": (0, True),
    "mean_recruits_per_step": (0, False),
    "neurons_recruited_fraction": (1, True),
    "mean_final_active_fraction": (0, True),
}

#: capability metrics carried into the capability-per-watt analysis of §19/M14
PER_WATT_METRICS = ("memory_capacity", "participation_ratio", "stimulus_response_mi_bits",
                    "discrimination_bits_at_overlap_0.9", "components_for_90pct_variance",
                    "generalization_novel_accuracy")


def aggregate(args) -> dict:
    scales = {}
    for p in sorted(SCALES_DIR.glob("scale_*.json")):
        data = read_json(p)
        if data:
            scales[f"{data['scale']:g}"] = data
    if not scales:
        raise SystemExit("no per-scale results in results/phase9/scales - run --stage run first")

    metrics = capability_metrics(scales)
    keys = sorted(metrics, key=lambda k: float(k))
    ns = [float(metrics[k]["n_neurons"]) for k in keys]
    syns = [float(metrics[k]["n_synapses"]) for k in keys]

    fits: dict = {}
    for name, (direction, bounded) in METRIC_DIRECTION.items():
        vals = [metrics[k].get(name) for k in keys]
        entry = {
            "direction": ("higher_is_better" if direction == 1 else
                          ("lower_is_better" if direction == -1 else "descriptive")),
            "bounded_like_probability": bounded,
            "values_by_scale": dict(zip(keys, vals)),
            "power_law_vs_neurons": _fit(cap.fit_power_law, ns, vals, name),
            "power_law_vs_synapses": _fit(cap.fit_power_law, syns, vals, name),
            "linear_vs_neurons": _fit(cap.fit_linear, ns, vals, name),
        }
        entry["verdict"] = _verdict(entry, keys)
        fits[name] = entry

    def _cascade_count(scale_entry: dict) -> int:
        return sum(int((scale_entry.get(f"battery_{b}") or {}).get("n_trials") or 0)
                   for b in ("classes", "discrimination", "temporal", "sequence", "robustness"))

    cost = {"per_scale": {}, "shared_machine_note": (
        "wall seconds were measured while other phase jobs of this project were running on the same "
        "20-core host, so they are an upper bound on an idle machine's cost; cascades_total and "
        "seconds_per_cascade are given so the cost can be re-derived from work done rather than "
        "from a contended clock")}
    for k in keys:
        sc = scales[k]
        cc = _cascade_count(sc)
        bsec = sum(v for kk, v in (sc.get("cost") or {}).items()
                   if kk.startswith("battery_") and isinstance(v, (int, float)))
        cost["per_scale"][k] = {
            "n_neurons": metrics[k]["n_neurons"], "n_synapses": metrics[k]["n_synapses"],
            "readout_dim": metrics[k]["readout_dim"],
            "total_seconds": metrics[k]["scale_total_seconds"],
            "battery_seconds": {b: (sc.get("cost") or {}).get(f"battery_{b}_seconds")
                                for b in ("classes", "discrimination", "temporal", "sequence",
                                          "robustness")},
            "cascades_total": cc,
            "seconds_per_cascade": round(bsec / max(cc, 1), 4) if bsec else None,
            "seconds_per_1000_stimuli": (sc.get("cost") or {}).get(
                "wall_seconds_per_1000_stimuli"),
            "energy": (sc.get("cost") or {}).get("energy"),
            "errors": metrics[k]["errors"],
            "truncated_by_budget": sc.get("truncated_by_budget"),
        }
    if len(keys) >= 3:
        cost["total_seconds_vs_neurons"] = _fit(
            cap.fit_power_law, ns, [metrics[k]["scale_total_seconds"] for k in keys],
            "scale_total_seconds")

    per_watt = capability_per_watt(metrics, keys, ns)

    payload = {
        "phase": "9 (capability scaling tests, §15; milestones M12 scaling benchmarks, M14 "
                 "capability/watt where energy data exists)",
        "deliverable": "results/phase9/capability.json",
        "environment": environment_note(),
        "dataset_hash": dataset_hash(),
        "scales_run": keys,
        "protocol": (scales[keys[0]].get("protocol") if scales.get(keys[0]) else None),
        "protocol_fingerprints": {k: metrics[k]["protocol_fingerprint"] for k in keys},
        "readout_dims": {k: metrics[k]["readout_dim"] for k in keys},
        "metrics_by_scale": metrics,
        "fits": fits,
        "capability_per_watt": per_watt,
        "cost": cost,
        "scaling_exponents_summary": _summary_table(fits, keys),
        "calibration": read_json(PHASE9 / "calibration.json"),
        "reproducibility_verification": read_json(PHASE9 / "verification.json"),
        "replicas": read_json(PHASE9 / "replicas.json"),
        "self_check": read_json(PHASE9 / "self_check.json"),
        "capability_module_sha256_16": module_hash(),
        "honesty": {
            "grounding_rule": ("measurements only: capabilities that do not grow, saturate, or "
                               "cannot be measured at a scale are reported as such with their "
                               "numbers (PROJECT-VYBFLY.md §30)"),
            "reported_errors": {k: metrics[k]["errors"] for k in keys if metrics[k]["errors"]},
            "censoring": _censor_note(scales, keys),
            "constants_at_all_scales": sorted(
                name for name, f in fits.items() if not f["verdict"].get("grows_with_scale")
                and "constant" in str(f["verdict"].get("reason"))),
            "measurement_limits": _measurement_limits(metrics, keys, per_watt),
        },
    }
    write_json(PHASE9 / "capability.json", payload)
    write_summary(payload)
    return payload


def _measurement_limits(metrics: dict, keys: list, per_watt: dict) -> dict:
    """What this work-stream cannot answer, derived from what actually exists on disk."""
    limits = {
        "not_measured": [],
        "grid_limited": {},
        "ceiling_limited": {},
    }
    # energy-dependent items: report only if the data is genuinely missing
    bio_ok = bool(((per_watt.get("biological_track") or {}) if per_watt else {}).get("available"))
    hw_ok = bool(((per_watt.get("hardware_track") or {}) if per_watt else {}).get("measured_power_w"))
    limits["not_measured"].append(
        "spikes per biological second (the cascade is step-based; no step duration is asserted)")
    if not bio_ok:
        limits["not_measured"].append(
            "capability per biological watt (results/energy/biological_model.json not found)")
    if not hw_ok:
        limits["not_measured"].append(
            "capability per measured hardware watt (no hardware power data found for a CPU "
            "workload; the energy work-stream's watts are for its own GPU workload)")
    if bio_ok:
        limits["not_measured"].append(
            "an independent power-vs-N exponent (the biological model is linear in N by "
            "construction, so it cannot test the energy hypothesis by itself)")
    # grid limits, computed from the values actually observed
    if any(metrics[k].get("min_separable_delta_at_grid_minimum") for k in keys):
        limits["grid_limited"]["min_separable_delta"] = (
            "the smallest stimulus distance tested (Delta = 0.05) is already separable at some "
            "scales, so the fitted exponent for Delta* is bounded below by the grid, not by the "
            "network; use the MI-at-Delta=0.05 column as the censoring-free discrimination "
            "measure")
    if all(isinstance(metrics[k].get("memory_capacity_above_chance_only"), (int, float))
           and metrics[k]["memory_capacity_above_chance_only"] == max(metrics[k].get(
               "memory_capacity_at_0.99") or 0, metrics[k].get("memory_capacity") or 0)
           for k in keys):
        limits["grid_limited"]["memory_capacity_above_chance_only"] = (
            "every level of the association grid is above chance at every scale; the informative "
            "number is the capacity at an accuracy threshold, not 'above chance'")
    if any(metrics[k].get("memory_capacity_censored") for k in keys):
        limits["grid_limited"]["memory_capacity"] = (
            "capacity is at the top of the tested grid (96 associations) at some scales, so the "
            "fitted exponent is a lower bound; the capacity at the stricter 0.99 threshold "
            "resolves further")
    if any(metrics[k].get("stimulus_response_mi_at_ceiling") for k in keys):
        limits["ceiling_limited"]["stimulus_response_mi_bits"] = (
            "the decoder's information is capped at log2(96) = 6.585 bits by the number of "
            "stimulus classes; the 5x and 10x values sit on that cap, so their difference is not "
            "a network property")
    for name, ceiling in (("generalization_novel_accuracy", 1.0),
                          ("robustness_accuracy_at_max_neuron_lesion", 1.0),
                          ("accuracy_at_full_stimulus_difference", 1.0),
                          ("distinct_pattern_ratio", 1.0),
                          ("robustness_baseline_accuracy", 1.0)):
        vals = [metrics[k].get(name) for k in keys if isinstance(metrics[k].get(name), (int, float))]
        if vals and min(vals) >= ceiling - 1e-9:
            limits["ceiling_limited"][name] = (
                f"at the {ceiling} measurement ceiling at every scale: no headroom to detect "
                "growth (a harder version of the task would be needed)")
    if any(isinstance(metrics[k].get("robustness_mi_retention_at_largest_rate_neurons"), (int, float))
           and metrics[k]["robustness_mi_retention_at_largest_rate_neurons"] > 1.0 for k in keys):
        limits["ceiling_limited"]["robustness_mi_retention"] = (
            "a retention above 1.0 is estimation noise from 10 test samples per lesion rate, not "
            "an improvement; the MI robustness numbers carry that caveat")
    return limits


def capability_per_watt(metrics: dict, keys: list, ns: list) -> dict:
    """§19/M14: capability per unit energy, using whatever the energy work-stream has published.

    The biological-equivalent track publishes a per-neuron watt figure and a per-scale table
    (results/energy/biological_model.json), so C(N)/P_bio(N) can be computed. That model is linear
    in N by construction (P_bio = per_neuron_watts x N), which the note records: capability per
    biological watt is then exactly capability per neuron rescaled, and its exponent is
    alpha_C - 1 by construction, not an independent measurement.

    The hardware track measured GPU power only (results/energy/hardware_energy.json); this
    work-stream's cascade is CPU numpy and the CPU RAPL counters are unreadable for this uid, so no
    hardware energy is claimed for these runs.
    """
    bio = read_json(ROOT / "results" / "energy" / "biological_model.json", {}) or {}
    table = {round(float(r["scale"]), 6): r for r in (bio.get("scaling_table") or [])
             if isinstance(r, dict) and r.get("scale") is not None}
    hw = read_json(ROOT / "results" / "energy" / "hardware_energy.json", {}) or {}
    curves = read_json(ROOT / "results" / "energy" / "curves.json", {}) or {}
    hw_curves = (curves.get("per_scale") or {}) if isinstance(curves, dict) else {}
    stale = {}
    for k in keys:
        their = ((hw_curves.get(k) or {}).get("capability") or {}).get("metrics") or {}
        for name in ("memory_capacity", "participation_ratio", "stimulus_response_mi_bits"):
            mine = metrics[k].get(name)
            if name in their and isinstance(mine, (int, float)):
                if abs(float(their[name]) - float(mine)) > 1e-9:
                    stale.setdefault(k, []).append(name)
    out: dict = {
        "biological_track": {
            "source": "results/energy/biological_model.json" if bio else None,
            "available": bool(bio),
            "per_neuron_watts": bio.get("per_neuron_watts"),
            "anchor": bio.get("anchor_key"),
            "caveat": ("P_bio(N) is linear in N by construction, so per-watt exponents below are "
                       "(alpha_C - 1) by construction; the useful independent number remains "
                       "alpha_C vs N"),
            "per_scale": {},
            "fits_per_neuron_watt": {},
        },
        "hardware_track": {
            "source": "results/energy/hardware_energy.json" if hw else None,
            "per_scale_source": "results/energy/curves.json" if hw_curves else None,
            "available_metrics": sorted((hw.get("devices") or {}).keys()),
            "unavailable": (hw.get("unavailable_metrics") or {}) if isinstance(
                hw.get("unavailable_metrics"), (dict, str)) else None,
            "measured_power_w": ((hw.get("devices") or {}).get("gpu:0") or {}).get("mean_w"),
            "baseline_power_w": next((p.get("mean_w") for p in (hw.get("phases") or [])
                                      if p.get("label") == "idle_baseline"), None),
            "per_scale": {k: {"hardware": (hw_curves.get(k) or {}).get("hardware"),
                              "wall_seconds": (hw_curves.get(k) or {}).get("wall_seconds"),
                              "capability_per_watt_computed_by_energy_stream": (
                                  hw_curves.get(k) or {}).get("capability_per_watt")}
                          for k in keys} if hw_curves else None,
            "note": ("hardware power was sampled by the energy work-stream for its own GPU/CUDA "
                     "workload, not for this CPU cascade, so no hardware energy is attributed to "
                     "the capability runs here; the §19/M14 combination over measured watts lives "
                     "in results/energy/curves.json"),
            "capability_numbers_the_energy_stream_read": {
                "file": "results/energy/curves.json",
                "stale_against_this_run": stale or False,
                "note": ("the energy stream copies capability values out of results/phase9/"
                         "capability.json; entries listed above were computed from an earlier "
                         "protocol revision and must be re-read from the current file for §19/M14"),
            } if hw_curves else None,
        },
    }
    if not table:
        out["biological_track"]["note"] = ("no biological scaling table found; capability per "
                                           "watt not computed")
        return out
    for k in keys:
        row = table.get(round(float(k), 6))
        if not row:
            continue
        p = float(row["p_bio_w"])
        entry = {"n_neurons": row.get("n_neurons"), "p_bio_w": p}
        for name in PER_WATT_METRICS:
            v = metrics[k].get(name)
            numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
            entry[f"{name}_per_bio_watt"] = (float(v) / p) if numeric else None
            entry[f"{name}_per_bio_nanowatt"] = (float(v) / (p * 1e9)) if numeric else None
        out["biological_track"]["per_scale"][k] = entry
    for name in PER_WATT_METRICS:
        vals = [metrics[k].get(name) for k in keys]
        pows = [table.get(round(float(k), 6), {}).get("p_bio_w") for k in keys]
        pairs = [(float(v) / float(pw), float(n)) for v, pw, n in zip(vals, pows, ns)
                 if isinstance(v, (int, float)) and pw]
        if len(pairs) >= 3:
            out["biological_track"]["fits_per_neuron_watt"][name] = cap.fit_power_law(
                [n for _, n in pairs], [x for x, _ in pairs],
                label=f"{name} per biological watt")
    return out


def _fit(fn, xs, ys, name):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if isinstance(y, (int, float, np.integer, np.floating)) and not isinstance(y, bool)
             and np.isfinite(float(y))]
    if not pairs:
        return {"label": name, "n_points": 0, "alpha": None, "r2": None,
                "note": "metric not available at any scale"}
    return fn([p[0] for p in pairs], [p[1] for p in pairs], label=name)


def _verdict(entry: dict, keys: list) -> dict:
    vals = [entry["values_by_scale"].get(k) for k in keys]
    numeric = [float(v) for v in vals
               if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)]
    if not numeric:
        return {"grows_with_scale": None, "reason": "not measured at any scale"}
    if entry["direction"] == "descriptive":
        return {"grows_with_scale": None,
                "reason": (f"descriptive diagnostic, no growth claim applies (values "
                           f"{min(numeric):.6g}..{max(numeric):.6g})")}
    if len(set(np.round(numeric, 9))) <= 1:
        return {"grows_with_scale": False,
                "reason": f"constant at {numeric[0]:.6g} across every measured scale"}
    pl = entry["power_law_vs_neurons"] or {}
    alpha, r2 = pl.get("alpha"), pl.get("r2")
    if alpha is None:
        return {"grows_with_scale": None, "reason": pl.get("note") or "not fitted"}
    direction = entry["direction"]
    sign = float(np.sign(alpha))
    if direction == "higher_is_better":
        grows: bool | None = sign > 0
    elif direction == "lower_is_better":
        grows = sign < 0
    else:
        grows = None
    weak = (r2 is None) or (r2 < 0.8)
    return {
        "grows_with_scale": (bool(grows) if grows is not None else None),
        "alpha_vs_neurons": float(alpha), "r2_vs_neurons": r2,
        "weak_fit_warning": bool(weak),
        "reason": (f"alpha={alpha:.3f} with R^2={'n/a' if r2 is None else round(r2, 3)}"
                   + ("; weak log-log fit (R^2<0.8), treat the exponent as indicative only"
                      if weak else "")),
    }


def _censor_note(scales: dict, keys: list) -> dict:
    """Per scale: the measurements whose resolution was exhausted by the tested grid."""
    out = {}
    for k in keys:
        c = (scales.get(k) or {}).get("capability", {})
        flags = []
        if (c.get("memory_capacity") or {}).get("censored_at_grid_max"):
            flags.append("memory capacity at the top of the tested grid")
        if (c.get("temporal_depth") or {}).get("censored_at_grid_max"):
            flags.append("temporal depth at the top of the tested delays")
        if (c.get("sequence_learning") or {}).get("censored_at_grid_max"):
            flags.append("sequence depth at the top of the tested lengths")
        if ((c.get("dynamical_complexity") or {}).get("effective_dimensionality") or {}).get(
                "censored_at_n_trials"):
            flags.append("participation ratio at the number of trials")
        dd = c.get("sensory_discrimination") or {}
        if dd and not dd.get("separable_within_tested_range", True):
            flags.append("stimulus distance never became separable on the tested grid")
        if flags:
            out[k] = flags
    return out


def _summary_table(fits: dict, keys: list) -> list:
    rows = []
    for name, f in sorted(fits.items()):
        pn = f["power_law_vs_neurons"]
        rows.append({
            "metric": name, "direction": f["direction"],
            "values": [f["values_by_scale"][k] for k in keys],
            "alpha_vs_N": pn.get("alpha"), "r2_vs_N": pn.get("r2"),
            "alpha_vs_synapses": f["power_law_vs_synapses"].get("alpha"),
            "r2_vs_synapses": f["power_law_vs_synapses"].get("r2"),
            "verdict": f["verdict"],
        })
    return rows


# --------------------------------------------------------------------------- summary text
def write_summary(payload: dict) -> None:
    keys = payload["scales_run"]
    met = payload["metrics_by_scale"]
    fits = payload["fits"]
    lines: list[str] = []
    add = lines.append

    add("PHASE 9 / M12-M14 - CAPABILITY SCALING TESTS - SUMMARY")
    add("=" * 78)
    add("")
    add("Deliverables: src/flyscale/capability.py, scripts/phase9_capability.py,")
    add("              results/phase9/capability.json, results/phase9/SUMMARY.txt,")
    add("              results/phase9/artifacts/ (response matrices per scale)")
    add("")
    add("DATASET AND ENVIRONMENT")
    env = payload["environment"]
    for k in ("dataset", "dataset_version", "git_commit", "git_working_tree_dirty", "python",
              "platform"):
        add(f"  {k}: {env.get(k)}")
    add(f"  dataset_hash: {payload['dataset_hash']}")
    add("")
    add("WHAT WAS RUN AT EVERY SCALE")
    add("  graphs:      canonical v783 thresholded at 5 synapses (1x), then renorm.upscale to")
    add("               2x/5x/10x with the recorded geometry (standardised anatomical xyz,")
    add("               P(connect)=1/(1+exp((d-R)/T)) fit on the graph itself)")
    for k in keys:
        if k in met:
            add(f"    {k:>4}x: N={met[k]['n_neurons']:,}  connections={met[k]['n_connections']:,}  "
                f"synapses={met[k]['n_synapses']:,}")
    add("  dynamics:    synchronous threshold cascade (flyscale.propagation), 8 steps, relative")
    add("               threshold fixed at %.2f x the graph's own mean synapses/connection"
        % (payload["protocol"]["relative_threshold"] if payload.get("protocol") else float("nan")))
    add("  stimulus:    %d classes, each trial seeds %.4f x N neurons drawn from a class pool of"
        % (payload["protocol"]["n_classes"] if payload.get("protocol") else 0,
           payload["protocol"]["sample_fraction"] if payload.get("protocol") else float("nan")))
    add("               %.4f x N stimulable neurons (sensory / visual_projection / optic)"
        % (payload["protocol"]["pool_fraction"] if payload.get("protocol") else float("nan")))
    add("  readout:     binary final activation mask over %.3f x N neurons (fixed seed);"
        % (payload["protocol"]["readout_fraction"] if payload.get("protocol") else float("nan")))
    add("               nearest-centroid / template-matching decoder (analytic, no training)")
    add("  statistics:  every threshold-based capability requires accuracy >= threshold AND a")
    add("               one-sided exact binomial p<0.05 vs chance; accuracies that clear the")
    add("               threshold but not the test are listed separately, never counted")
    add("  protocol revision: %s (superseded revisions kept under results/phase9/"
        % (payload["protocol"].get("protocol_revision") if payload.get("protocol") else "?"))
    add("               scales_archive/ and artifacts_archive/, both described in the JSON)")
    add("  readout dims: " + ", ".join(f"{k}x->{payload['readout_dims'][k]}" for k in keys))
    add("")
    add("CAPABILITY TABLE (one row per scale)")
    header = ("  metric", *[f"{k}x" for k in keys])
    add("  " + f"{'metric':<44}" + "".join(f"{k + 'x':>12}" for k in keys))
    order = [
        ("sensory discrimination, min separable delta", "min_separable_delta"),
        ("  discrimination accuracy at max delta", "accuracy_at_full_stimulus_difference"),
        ("  discrimination MI (bits) at overlap 0.95 (Delta=0.05)",
         "discrimination_bits_at_overlap_0.95"),
        ("  discrimination MI (bits) at overlap 0.9 (Delta=0.10)",
         "discrimination_bits_at_overlap_0.9"),
        ("  discrimination MI (bits) at overlap 0.6 (Delta=0.40)",
         "discrimination_bits_at_overlap_0.6"),
        ("  discrimination MI (bits), identical-pool control",
         "discrimination_bits_identical_pools_control"),
        ("  discrimination accuracy at the finest Delta",
         "discrimination_accuracy_at_finest_delta"),
        ("  min separable delta == grid minimum (censored)",
         "min_separable_delta_at_grid_minimum"),
        ("memory capacity M* (associations)", "memory_capacity"),
        ("  memory capacity censored at grid max", "memory_capacity_censored"),
        ("  memory capacity at accuracy>=0.95", "memory_capacity_at_0.95"),
        ("  memory capacity at accuracy>=0.99", "memory_capacity_at_0.99"),
        ("  memory capacity above chance (any level)", "memory_capacity_above_chance_only"),
        ("  accuracy at capacity grid max", "memory_capacity_accuracy_at_grid_max"),
        ("temporal depth (delay steps)", "temporal_depth_steps"),
        ("  delayed match accuracy at delay 0", "temporal_depth_delay_zero_accuracy"),
        ("sequence depth (order-reversal k)", "sequence_depth"),
        ("generalization: novel-variant accuracy", "generalization_novel_accuracy"),
        ("  generalization: matched accuracy", "generalization_matched_accuracy"),
        ("  generalization gap", "generalization_gap"),
        ("robustness: baseline accuracy", "robustness_baseline_accuracy"),
        ("  robustness: lowest rate with significant drop, neurons",
         "robustness_lowest_significant_drop_rate_neurons"),
        ("  robustness: lowest rate with significant drop, synapses",
         "robustness_lowest_significant_drop_rate_synapses"),
        ("  robustness: max tolerated rate, neurons", "robustness_max_tolerated_rate_neurons"),
        ("  robustness: MI retained at 90% neuron loss",
         "robustness_mi_retention_at_largest_rate_neurons"),
        ("  robustness: MI retained at 90% synapse loss",
         "robustness_mi_retention_at_largest_rate_synapses"),
        ("  robustness: MI half-degradation rate, neurons",
         "robustness_mi_half_degradation_rate_neurons"),
        ("  robustness: MI half-degradation rate, synapses",
         "robustness_mi_half_degradation_rate_synapses"),
        ("  robustness: half-degradation rate, neurons", "robustness_half_degradation_rate_neurons"),
        ("  robustness: half-degradation rate, synapses", "robustness_half_degradation_rate_synapses"),
        ("  robustness: accuracy at max neuron lesion", "robustness_accuracy_at_max_neuron_lesion"),
        ("  robustness: accuracy at max synapse lesion",
         "robustness_accuracy_at_max_synapses_lesion"),
        ("complexity: participation ratio", "participation_ratio"),
        ("  complexity: components for 90% variance", "components_for_90pct_variance"),
        ("  complexity: mean unit entropy (bits)", "mean_unit_entropy_bits"),
        ("  complexity: distinct-pattern ratio", "distinct_pattern_ratio"),
        ("  complexity: readout unit coverage", "readout_unit_coverage"),
        ("  complexity: stimulus-response MI (bits)", "stimulus_response_mi_bits"),
        ("  complexity: normalised MI", "stimulus_response_normalised_mi"),
        ("  complexity: MI at the class ceiling", "stimulus_response_mi_at_ceiling"),
        ("  complexity: avalanche size exponent", "avalanche_size_exponent"),
        ("  complexity: avalanche exponent R^2", "avalanche_size_exponent_r2"),
        ("  complexity: branching ratio (sum ratio)", "branching_ratio_ratio_of_sums"),
        ("  complexity: neurons recruited fraction", "neurons_recruited_fraction"),
        ("  complexity: mean final active fraction", "mean_final_active_fraction"),
    ]
    for label, name in order:
        cells = []
        for k in keys:
            v = met.get(k, {}).get(name)
            if isinstance(v, bool):
                cells.append(f"{str(v):>12}")
            elif isinstance(v, (int, float)):
                cells.append(f"{v:>12.4f}" if abs(v) < 1e4 else f"{v:>12.1f}")
            else:
                cells.append(f"{'-':>12}")
        add("  " + f"{label:<44}" + "".join(cells))
    add("")
    add("FITTED EXPONENTS (value ~ N^alpha and ~ synapses^alpha, least squares on log-log axes)")
    add("  " + f"{'metric':<44}{'alpha_N':>10}{'R2_N':>8}{'alpha_E':>10}{'R2_E':>8}  verdict")
    for row in payload["scaling_exponents_summary"]:
        an = row["alpha_vs_N"]
        rn = row["r2_vs_N"]
        ae = row["alpha_vs_synapses"]
        re_ = row["r2_vs_synapses"]
        vs = (row["verdict"] or {}).get("grows_with_scale")
        ver = "grows" if vs is True else ("flat/no growth" if vs is False else "descriptive")
        if (row["verdict"] or {}).get("weak_fit_warning"):
            ver += "*"
        add("  " + f"{row['metric']:<44}"
            + (f"{an:>10.3f}" if isinstance(an, (int, float)) else f"{'-':>10}")
            + (f"{rn:>8.3f}" if isinstance(rn, (int, float)) else f"{'-':>8}")
            + (f"{ae:>10.3f}" if isinstance(ae, (int, float)) else f"{'-':>10}")
            + (f"{re_:>8.3f}" if isinstance(re_, (int, float)) else f"{'-':>8}")
            + f"  {ver}")
    add("  (* weak log-log fit, R^2 < 0.8: the exponent is indicative only - the per-scale values")
    add("   above are the primary evidence)")
    add("")
    limits = payload["honesty"].get("measurement_limits") or {}
    add("  note: the 'identical-pool control' row is the discrimination MI estimator's noise floor")
    add("  (two identical pools must give 0 bits; anything above 0 is the 16-sample estimate's")
    add("  spread), so discrimination MI should be read against that column, not against zero")
    add("")
    add("BENCHMARK VERDICTS (§9 families, headline number per scale)")
    bench = [
        ("sensory discrimination: min separable Delta", "min_separable_delta", "lower is better"),
        ("discrimination MI (bits) at overlap 0.95 (Delta=0.05)",
         "discrimination_bits_at_overlap_0.95", "higher is better"),
        ("memory capacity M* (acc>=0.9)", "memory_capacity", "higher is better"),
        ("memory capacity (acc>=0.99)", "memory_capacity_at_0.99", "higher is better"),
        ("temporal depth (delay steps)", "temporal_depth_steps", "higher is better"),
        ("sequence depth (k, order vs reversal)", "sequence_depth", "higher is better"),
        ("generalization: novel-variant accuracy", "generalization_novel_accuracy",
         "higher is better"),
        ("robustness: max tolerated neuron loss", "robustness_max_tolerated_rate_neurons",
         "higher is better"),
        ("robustness: lowest significant drop, synapses",
         "robustness_lowest_significant_drop_rate_synapses", "lower = more fragile"),
        ("robustness: MI retained at 90% synapse loss",
         "robustness_mi_retention_at_largest_rate_synapses", "higher is more robust"),
        ("robustness: accuracy at 90% synapse loss",
         "robustness_accuracy_at_max_synapses_lesion", "higher is more robust"),
        ("complexity: participation ratio", "participation_ratio", "higher is better"),
        ("complexity: stimulus-response MI (bits)", "stimulus_response_mi_bits",
         "higher is better"),
    ]
    add("  " + f"{'benchmark':<48}" + "".join(f"{k + 'x':>12}" for k in keys) + "  verdict")
    for label, name, note in bench:
        cells = []
        for k in keys:
            v = met.get(k, {}).get(name)
            if isinstance(v, bool):
                cells.append(f"{str(v):>12}")
            elif isinstance(v, (int, float)):
                cells.append(f"{v:>12.4f}" if abs(v) < 1e4 else f"{v:>12.1f}")
            else:
                cells.append(f"{'-':>12}")
        fit = (payload["fits"].get(name) or {}).get("verdict") or {}
        vs = fit.get("grows_with_scale")
        verdict = ("grows" if vs is True else ("does not grow" if vs is False else "unresolved"))
        if name == "memory_capacity" and any(
                (met.get(k, {}).get("memory_capacity_censored") for k in keys)):
            verdict += " (censored at the grid maximum on some scales)"
        if name == "stimulus_response_mi_bits" and any(
                (met.get(k, {}).get("stimulus_response_mi_at_ceiling") for k in keys)):
            verdict += " (at the log2(n_classes) ceiling on some scales)"
        if name == "min_separable_delta" and any(
                (met.get(k, {}).get("min_separable_delta_at_grid_minimum") for k in keys)):
            verdict += " (at the grid floor on some scales - the exponent is a grid artifact)"
        if name in (limits.get("ceiling_limited") or {}):
            verdict += " (no headroom: the metric is at its ceiling at every scale)"
        add("  " + f"{label:<48}" + "".join(cells) + f"  {verdict} [{note}]")
    add("")
    add("DID CAPABILITY GROW WITH SCALE? (the honest answer, metric by metric)")
    grew, flat, censored, unmeasurable = [], [], [], []
    for row in payload["scaling_exponents_summary"]:
        vs = (row["verdict"] or {}).get("grows_with_scale")
        reason = (row["verdict"] or {}).get("reason", "")
        if vs is True:
            grew.append(f"{row['metric']} (alpha={row['alpha_vs_N']:.3f}, R2={row['r2_vs_N']})")
        elif vs is False:
            flat.append(f"{row['metric']} ({reason})")
        else:
            unmeasurable.append(f"{row['metric']} ({reason})")
    grid_names = set((payload["honesty"].get("measurement_limits") or {}).get("grid_limited") or {})
    grid_metrics = {"min_separable_delta": "min_separable_delta",
                    "memory_capacity": "memory_capacity",
                    "memory_capacity_at_0.95": "memory_capacity",
                    "memory_capacity_at_0.99": "memory_capacity",
                    "memory_capacity_above_chance_only": "memory_capacity_above_chance_only"}
    grew_clear, grew_marginal = [], []
    for x in grew:
        name = x.split(" ")[0]
        fit = (payload["fits"].get(name) or {}).get("verdict") or {}
        alpha = abs(float(fit.get("alpha_vs_neurons") or 0.0))
        r2 = fit.get("r2_vs_neurons")
        if name in grid_metrics and grid_metrics[name] in grid_names:
            grew_marginal.append(x + "  [grid-limited: see the limits below]")
        elif alpha >= 0.1 and r2 is not None and r2 >= 0.7:
            grew_clear.append(x)
        else:
            grew_marginal.append(x + "  [small exponent or weak fit]")
    add("  grew with scale (exponent >= 0.1 with R^2 >= 0.7):")
    for s in grew_clear or ["    (none)"]:
        add(f"    - {s}")
    add("  grew, but the evidence is marginal (small exponent, weak fit, or a grid/ceiling")
    add("  artefact - do not quote these as scaling laws without the caveat):")
    for s in grew_marginal or ["    (none)"]:
        add(f"    - {s}")
    add("  did NOT grow with scale (real negatives, headroom existed):")
    ceil_names = set((payload["honesty"].get("measurement_limits") or {}).get(
        "ceiling_limited") or {})
    flat_real = [x for x in flat if not any(x.startswith(c + " ") for c in ceil_names)]
    flat_ceiling = [x for x in flat if any(x.startswith(c + " ") for c in ceil_names)]
    for s in flat_real or ["    (none)"]:
        add(f"    - {s}")
    add("  no growth detectable, but the metric was already at its measurement ceiling:")
    for s in flat_ceiling or ["    (none)"]:
        add(f"    - {s}")
    descriptive = [x for x in unmeasurable
                   if METRIC_DIRECTION.get(x.split(" ")[0], (1, False))[0] == 0]
    unresolved = [x for x in unmeasurable if x not in descriptive]
    add("  descriptive diagnostics (no growth claim applies):")
    for s in descriptive or ["    (none)"]:
        add(f"    - {s}")
    for s in unresolved:
        add(f"    - unresolved: {s}")
    for k, flags in (payload["honesty"]["censoring"] or {}).items():
        add(f"  censored at {k}x: {', '.join(flags)}")
    add("")
    add("PER-SCALE COST (wall seconds; the cascade is CPU numpy, single process)")
    add("  " + f"{'scale':>6}{'N':>12}{'cascades':>10}{'classes':>9}{'discrim':>9}{'temporal':>9}"
               f"{'sequence':>9}{'robust':>9}{'total_s':>10}")
    for k in keys:
        c = payload["cost"]["per_scale"][k]
        b = c["battery_seconds"]

        def _s(x):
            v = b.get(x)
            return f"{v:>9.1f}" if isinstance(v, (int, float)) else f"{'-':>9}"

        add("  " + f"{k + 'x':>6}{c['n_neurons']:>12,}{c['cascades_total']:>10,}" + "".join(
            _s(x) for x in ("classes", "discrimination", "temporal", "sequence", "robustness"))
            + f"{c['total_seconds']:>10.0f}")
    add("  note: " + payload["cost"]["shared_machine_note"])
    tot = payload["cost"].get("total_seconds_vs_neurons")
    if isinstance(tot, dict) and tot.get("alpha") is not None:
        add(f"  wall seconds ~ N^{tot['alpha']:.3f} (R2={tot['r2']:.3f}) - a runtime exponent of "
            f"~1 means the cost is linear in the neuron count, as the CSR cascade should be")
    energy = payload["cost"]["per_scale"][keys[0]].get("energy") if keys else None
    if (energy or {}).get("available"):
        add(f"  energy: per-scale energy results exist at {(energy or {}).get('source')} "
            f"(field '{(energy or {}).get('field')}'); they are another work-stream's output and "
            f"are not used to derive any capability number here")
    else:
        add(f"  energy: NOT AVAILABLE - {(energy or {}).get('note', '')}")
    add("")
    pw = payload.get("capability_per_watt") or {}
    bt = pw.get("biological_track") or {}
    add("CAPABILITY PER WATT (§19, M14)")
    if bt.get("available"):
        add(f"  biological-equivalent power from {bt['source']} "
            f"(per-neuron {bt['per_neuron_watts']:.3g} W, anchor {bt['anchor']}):")
        add("  (capability unit per biological watt; alpha is the fit of that value against N,")
        add("   which equals alpha_C - 1 by construction of the linear biological model)")
        add(f"  {'capability':<40}" + "".join(f"{k + 'x':>14}" for k in keys)
            + f"{'alpha_N':>9}{'R2':>7}{'n':>4}")
        for name in PER_WATT_METRICS:
            row = bt["per_scale"]
            cells = []
            for k in keys:
                v = (row.get(k) or {}).get(f"{name}_per_bio_watt")
                cells.append(f"{v:>14.3g}" if isinstance(v, (int, float)) else f"{'-':>14}")
            fit = (bt["fits_per_neuron_watt"] or {}).get(name) or {}
            alpha, r2 = fit.get("alpha"), fit.get("r2")
            add(f"  {name:<40}" + "".join(cells)
                + (f"{alpha:>9.3f}" if isinstance(alpha, (int, float)) else f"{'-':>9}")
                + (f"{r2:>7.3f}" if isinstance(r2, (int, float)) else f"{'-':>7}")
                + f"{fit.get('n_points', 0):>4}")
        add(f"  caveat: {bt['caveat']}")
    else:
        add(f"  biological track unavailable ({(bt or {}).get('note')})")
    hwt = pw.get("hardware_track") or {}
    add(f"  hardware track: {hwt.get('note')}")
    if hwt.get("per_scale"):
        add("  hardware anchors per scale (measured by the energy work-stream for its own GPU "
            "workload):")
        for k in keys:
            row = (hwt["per_scale"] or {}).get(k) or {}
            hwd = row.get("hardware") or {}
            add(f"    {k}x: wall {row.get('wall_seconds')} s, gpu mean "
                f"{hwd.get('gpu_mean_watts')} W, gpu joules {hwd.get('gpu_joules')}")
    stale = (hwt.get("capability_numbers_the_energy_stream_read") or {}).get(
        "stale_against_this_run")
    if stale:
        detail = "; ".join(f"{k}x: {', '.join(v)}" for k, v in sorted(stale.items()))
        add("  NOTE: results/energy/curves.json carries capability values copied from an earlier "
            "protocol revision (" + detail + "); the M14 combination must be re-read from the "
            "current results/phase9/capability.json")
    elif hwt.get("capability_numbers_the_energy_stream_read"):
        add("  the capability values the energy stream copied from results/phase9/capability.json "
            "match this run")
    if hwt.get("measured_power_w") is not None:
        add(f"    (energy work-stream anchors: workload mean {hwt['measured_power_w']} W on "
            f"{hwt.get('available_metrics')}, idle baseline {hwt.get('baseline_power_w')} W)")
    add("")
    add("ERRORS AND THINGS THAT COULD NOT BE MEASURED")
    reported = payload["honesty"]["reported_errors"]
    if reported:
        for k, errs in reported.items():
            if errs:
                add(f"  {k}x: {json.dumps(errs)}")
    else:
        add("  no battery failed at any scale")
    lim = payload["honesty"].get("measurement_limits") or {}
    for item in lim.get("not_measured", []):
        add(f"  not measured: {item}")
    if lim.get("grid_limited"):
        add("  limited by the tested grid (the value is bounded by the protocol, not the network):")
        for name, note in lim["grid_limited"].items():
            add(f"    - {name}: {note}")
    if lim.get("ceiling_limited"):
        add("  limited by a measurement ceiling (no headroom left in the metric):")
        for name, note in lim["ceiling_limited"].items():
            add(f"    - {name}: {note}")
    add("")
    ver = payload.get("reproducibility_verification")
    add("REPRODUCIBILITY CHECK")
    sc = payload.get("self_check") or {}
    add(f"  self-checks (flyscale.capability.self_check): {json.dumps(sc, sort_keys=True)}")
    add("    the masked cascade used for lesioning is bit-identical to propagation.cascade, to")
    add("    propagation.cascade on an explicitly lesioned graph, and cascade_epochs reproduces")
    add("    cascade step for step")
    if ver:
        add(f"  re-run check (--stage verify, 1x re-run against the stored file): "
            f"{ver.get('verdict')}, {ver.get('n_common_capability_fields')} capability fields "
            f"compared, {ver.get('n_differing_fields')} differing; per-trial activation and")
        add(f"    recruitment arrays identical: "
            f"{ver.get('per_trial_activation_fractions_identical')}, "
            f"{ver.get('per_trial_recruitment_identical')}; protocol fingerprints match: "
            f"{ver.get('protocol_fingerprints_match')}")
    else:
        add("  re-run check: not performed (run --stage verify)")
    add("")
    add("HOW TO REPRODUCE")
    add("  cd %s" % ROOT)
    add("  .venv/bin/python scripts/phase9_capability.py --stage all")
    add("  stages, in order (each is resumable and writes its own output):")
    add("    --stage replicas    # build/cache the 1x-10x graphs under results/phase9/replicas/")
    add("    --stage calibrate   # fix the cascade threshold + robustness reference on 1x only")
    add("    --stage run --scales 1 2 5 10   # ~1 min / 2 min / 5 min / 11 min on this shared host")
    add("    --stage verify      # re-run 1x and compare every capability field")
    add("    --stage aggregate   # fit the exponents, write capability.json + SUMMARY.txt")
    add("  every protocol number, seed, geometry parameter, fit and error is recorded in")
    add("  results/phase9/capability.json; the per-scale response matrices are in")
    add("  results/phase9/artifacts/")
    (PHASE9 / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {PHASE9 / 'SUMMARY.txt'}")


# --------------------------------------------------------------------------- stage: calibration
def calibrate(args) -> dict:
    g, info = load_scale_graph(1.0, args)
    proto = cap.Protocol(seed=args.seed, relative_threshold=args.relative_threshold,
                         readout_fraction=args.readout_fraction,
                         pool_fraction=args.pool_fraction,
                         sample_fraction=args.sample_fraction,
                         n_classes=args.n_classes)
    t0 = time.time()
    result = cap.calibrate_relative_threshold(g, proto)
    result["seconds"] = round(time.time() - t0, 1)
    result["graph"] = {"source": info.get("source"), "n": int(g.n), "edges": int(g.pre.size)}
    result["protocol_before_calibration"] = {"relative_threshold": proto.relative_threshold}

    # second protocol setting fixed once, on the 1x graph: the robustness reference point
    t0 = time.time()
    proto_cal = cap.Protocol(**{**{k: getattr(proto, k) for k in
                                   ("seed", "readout_fraction", "pool_fraction", "sample_fraction",
                                    "n_classes", "cascade_steps", "max_active_fraction")},
                                "relative_threshold": float(result["chosen"])})
    readout = cap.readout_indices(g, proto_cal)
    bat = cap.run_discrimination_battery(g, proto_cal, readout)
    disc = cap.discrimination(bat, proto_cal)
    result["reference_overlap_choice"] = cap.choose_reference_overlap(disc)
    result["discrimination_at_1x_for_reference"] = {
        "curve": disc["curve"], "min_separable_delta": disc["min_separable_delta"],
        "seconds": round(time.time() - t0, 1)}
    write_json(PHASE9 / "calibration.json", result)
    print(f"[calibrate] chosen relative threshold {result['chosen']} "
          f"(activity {result['chosen_row']['mean_final_active_fraction']:.3f}); "
          f"reference overlap {result['reference_overlap_choice']['overlap']} "
          f"(1x accuracy {result['reference_overlap_choice'].get('accuracy_at_1x')})")
    return result


# --------------------------------------------------------------------------- main
def _flatten(obj, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        # sort_keys: the stored JSON is written with sorted keys, so an order-sensitive
        # comparison would report false differences between a run and its own file
        out[prefix] = json.dumps(obj, sort_keys=True, default=str)
    else:
        out[prefix] = obj
    return out


def _same_arrays(a, b) -> bool:
    """True when two (possibly JSON-loaded vs numpy) arrays/lists hold the same values."""
    try:
        return bool(np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object)))
    except Exception:                                          # pragma: no cover
        return a == b


def _different_entries(a, b) -> int:
    """Number of entries that differ (shape mismatches count all missing entries)."""
    try:
        aa, bb = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        if aa.shape != bb.shape:
            return int(max(aa.size, bb.size))
        return int(sum(1 for x, y in zip(aa.ravel(), bb.ravel()) if x != y))
    except Exception:                                          # pragma: no cover
        return -1


def verify(args) -> dict:
    """Re-run the smallest scale and compare every capability field with the stored run.

    This is the reproducibility check the project's §24 requires: same seeds, same protocol, same
    numbers. It reports the field-by-field comparison, including fields that only exist in the
    newer run (a module that changed between runs must be visible, not hidden).
    """
    path = SCALES_DIR / "scale_1.json"
    before = read_json(path)
    if before is None:
        raise SystemExit("no results/phase9/scales/scale_1.json to verify against - run --scales 1")
    entry = run_scale(1.0, args)
    fa, fb = _flatten(before.get("capability", {})), _flatten(entry.get("capability", {}))
    common = set(fa).intersection(set(fb))
    differing = sorted(k for k in common if fa[k] != fb[k])
    result = {
        "scale": 1.0,
        "n_common_capability_fields": len(common),
        "n_differing_fields": len(differing),
        "differing_fields": differing[:50],
        "fields_only_in_the_earlier_run": sorted(set(fa).difference(set(fb)))[:50],
        "fields_only_in_the_rerun": sorted(set(fb).difference(set(fa)))[:50],
        "protocol_fingerprints_match": (before.get("protocol_fingerprint") ==
                                        entry.get("protocol_fingerprint")),
        "per_trial_activation_fractions_identical": _same_arrays(
            (before.get("battery_classes") or {}).get("final_active_fraction"),
            (entry.get("battery_classes") or {}).get("final_active_fraction")),
        "n_trials_with_different_activation": _different_entries(
            (before.get("battery_classes") or {}).get("final_active_fraction"),
            (entry.get("battery_classes") or {}).get("final_active_fraction")),
        "per_trial_recruitment_identical": _same_arrays(
            (before.get("battery_classes") or {}).get("newly_active"),
            (entry.get("battery_classes") or {}).get("newly_active")),
        "n_trials_with_different_recruitment": _different_entries(
            (before.get("battery_classes") or {}).get("newly_active"),
            (entry.get("battery_classes") or {}).get("newly_active")),
        "verdict": ("identical" if not differing else "DIFFERENT - the capability numbers are not "
                                                     "reproducible from the recorded seeds"),
        "capability_module_sha256_16_this_run": module_hash(),
        "note": ("a DIFFERENT verdict can come from two very different causes and the field list "
                 "distinguishes them: fields 'only in the rerun' mean the module gained fields "
                 "after the stored run (a version artefact, not irreproducibility), whereas a "
                 "field in 'differing_fields' whose value also appears in the stored run is a real "
                 "reproducibility failure"),
    }
    write_json(PHASE9 / "verification.json", result)
    print(f"[verify] {result['verdict']}: {len(common)} compared fields, {len(differing)} differing",
          flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "replicas", "calibrate", "run", "aggregate", "self-check",
                             "verify"])
    ap.add_argument("--scales", nargs="*", type=float, default=list(DEFAULT_SCALES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--relative-threshold", type=float, default=None,
                    help="cascade threshold in units of mean synapses/connection; default = the "
                         "calibrated value, or 3.0 if no calibration exists")
    ap.add_argument("--reference-overlap", type=float, default=None,
                    help="pool overlap of the robustness reference task; default = the value "
                         "chosen from the 1x discrimination curve (see --stage calibrate)")
    ap.add_argument("--n-classes", type=int, default=96)
    ap.add_argument("--readout-fraction", type=float, default=0.05)
    ap.add_argument("--pool-fraction", type=float, default=0.004)
    ap.add_argument("--sample-fraction", type=float, default=0.002)
    ap.add_argument("--rewire-fraction", type=float, default=0.1)
    ap.add_argument("--sibling-prob-scale", type=float, default=0.1)
    ap.add_argument("--no-sibling-edges", action="store_true")
    ap.add_argument("--force-replicas", action="store_true")
    ap.add_argument("--refit-geometry", action="store_true")
    ap.add_argument("--time-budget-s", type=float, default=840.0,
                    help="per-scale wall budget; batteries after it are skipped and recorded")
    ap.add_argument("--cpu-count-note", type=int, default=0)
    args = ap.parse_args()

    import os
    args.cpu_count_note = args.cpu_count_note or (os.cpu_count() or 0)

    if args.relative_threshold is None:
        cal = read_json(PHASE9 / "calibration.json", {}) or {}
        args.relative_threshold = float(cal.get("chosen", 3.0))
    if args.reference_overlap is None:
        cal = read_json(PHASE9 / "calibration.json", {}) or {}
        args.reference_overlap = float(
            (cal.get("reference_overlap_choice") or {}).get("overlap") or 0.0)
    print(f"[phase9] stage={args.stage} scales={args.scales} seed={args.seed} "
          f"RT={args.relative_threshold} threshold={args.threshold} "
          f"reference_overlap={args.reference_overlap}")

    if args.stage == "self-check":
        res = cap.self_check()
        write_json(PHASE9 / "self_check.json", res)
        print("[self-check]", json.dumps(res))
        return 0

    if args.stage in ("all", "self-check"):
        res = cap.self_check()
        write_json(PHASE9 / "self_check.json", res)
        print("[self-check]", json.dumps(res))

    if args.stage in ("all", "replicas"):
        build_replicas(args)
    if args.stage in ("all", "calibrate"):
        calibrate(args)
        cal = read_json(PHASE9 / "calibration.json", {}) or {}
        args.relative_threshold = float(cal.get("chosen", args.relative_threshold))
    if args.stage in ("all", "run"):
        for scale in args.scales:
            run_scale(scale, args)
    if args.stage == "verify":
        verify(args)
    if args.stage in ("all", "aggregate"):
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
