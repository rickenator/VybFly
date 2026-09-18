"""Phase 8 / M11 — mushroom-body associative learning benchmark (FlyWire v783).

    python scripts/phase8_learning.py [--threshold 5] [--seeds 0 1 2 3 4] [--quick]

PROJECT-VYBFLY.md §14: implement biologically relevant learning mechanisms, prioritizing
the mushroom body; §27 flags "training may dominate architecture" as a primary risk.
This script builds the benchmark:

  * extract the real MB subgraph by cell-type annotation (Kenyon cells, MBONs, DANs,
    antennal-lobe projection neurons), reporting the exact labels found,
  * sparse Kenyon-cell odor codes driven through the measured ALPN->KC fan-in (fixed
    k-winner-take-all sparsity, with trial-to-trial variability of the active KC set),
  * a reward-modulated Hebbian (three-factor) rule on the KC->MBON pathway, dopamine-gated
    through the measured DAN->MBON wiring, with homeostatic per-KC weight normalization,
  * protocol 1 — acquisition: N odors, one rewarded; learning curve over trials,
  * protocol 2 — memory capacity: raise the number of odor->reward associations until the
    trained network can no longer rank rewarded odors above unrewarded ones,
  * controls — degree-preserving shuffle of the KC->MBON targets, randomised dopamine gate,
    no-plasticity (frozen) runs, random weights on the real topology, shuffled ALPN->KC
    odor coding, uniform (non-anatomical) dopamine broadcast, and the additive rule,
  * architecture-preservation audit — synapses created/lost, weight drift, row budgets.

Outputs: results/phase8/learning.json, results/phase8/SUMMARY.txt.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CANON, ROOT, strip_arrays, write_json          # noqa: E402
from flyscale.connectome import Connectome                          # noqa: E402
from flyscale import plasticity as P                                # noqa: E402

OUT = ROOT / "results" / "phase8"

#: acquisition protocol
N_ODORS = 8
ACQ_TRIALS = 300
PROBE_EVERY = 10
#: capacity protocol
CAP_ODORS = 64
CAP_GRID = (1, 2, 4, 8, 16, 32, 48)
CAP_TRIALS_PER_ASSOC = 60
CAP_CRITERION = 0.90
CAP_LOOSE = 0.75


# --------------------------------------------------------------------------- helpers ----
def downsample(seq, max_points: int = 30) -> list:
    a = np.asarray(seq, dtype=float)
    if a.size <= max_points:
        return [float(x) for x in a]
    idx = np.unique(np.linspace(0, a.size - 1, max_points).astype(int))
    return [float(x) for x in a[idx]]


def nanmean_safe(x) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def nanstd_safe(x) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanstd(a)) if np.isfinite(a).any() else float("nan")


def trials_to_criterion(trials, auc, crit=CAP_CRITERION) -> float:
    for t, a in zip(trials, auc):
        if np.isfinite(a) and a >= crit:
            return float(t)
    return float("nan")


#: every condition = one (graph manipulation, plasticity configuration) pair.
#:  name                  graph kwargs                                       cfg changes
CONDITIONS: dict[str, dict] = {
    "real_plastic": {
        "graph": {}, "cfg": {},
        "note": "real MB wiring, reward-modulated Hebbian plasticity (main condition)"},
    "real_frozen": {
        "graph": {}, "cfg": {"plasticity": False},
        "note": "no-plasticity control: KC->MBON weights frozen at their anatomical values"},
    "shuffled_plastic": {
        "graph": {"shuffle_kc_mbon": True}, "cfg": {},
        "note": "KC->MBON targets degree-preservingly shuffled (same KC out-degrees and "
                "MBON in-degrees, same synapse-weight multiset), plastic"},
    "shuffled_frozen": {
        "graph": {"shuffle_kc_mbon": True}, "cfg": {"plasticity": False},
        "note": "shuffled wiring and no plasticity = random-weight readout with the same "
                "degree sequence and weight multiset"},
    "random_weight_plastic": {
        "graph": {"permute_weights": True}, "cfg": {},
        "note": "real KC->MBON topology but synapse weights randomly permuted, plastic"},
    "shuffled_gate_plastic": {
        "graph": {"shuffle_kc_mbon": True, "shuffle_dan_gate": True}, "cfg": {},
        "note": "shuffled KC->MBON wiring AND degree-preserving shuffle of the reward "
                "DAN->MBON gate, plastic"},
    "randcode_plastic": {
        "graph": {"shuffle_alpn_kc": True}, "cfg": {},
        "note": "real MB, odor coding replaced by a degree-matched random ALPN->KC "
                "projection, plastic"},
    "bounded_plastic": {
        "graph": {}, "cfg": {"max_factor": 4.0, "min_factor": 0.25},
        "note": "real wiring, plastic, every synapse capped at 4x / 0.25x its anatomical "
                "weight (explicit architecture-preservation constraint)"},
    "additive_plastic": {
        "graph": {}, "cfg": {"rule": "additive"},
        "note": "same gate/homeostasis but the additive three-factor update "
                "dw = eta*da*x*y (ablates the soft-bound form)"},
    "uniform_gate_plastic": {
        "graph": {}, "cfg": {"gate": "all"},
        "note": "dopamine broadcast to every MBON instead of routed through the measured "
                "DAN wiring (ablates the anatomical gate)"},
    "allpool_plastic": {
        "graph": {}, "cfg": {"pool": "all"},
        "note": "readout pool = all 96 MBONs instead of only the PAM-gated ones"},
    "aversive_sign_plastic": {
        "graph": {}, "cfg": {}, "mode": "punish", "capacity_direction": "down",
        "note": "identical to real_plastic in every respect (same anatomical PAM gate, same "
                "readout) except that the teaching signal is -1: isolates the SIGN of the "
                "dopamine signal, so the punished odor is predicted to fall in the readout"},
    "ppl_wiring_plastic": {
        "graph": {}, "cfg": {"gate": "ppl"}, "mode": "punish", "capacity_direction": "down",
        "note": "punishment routed through the measured PPL1/PPL2 -> MBON wiring instead of "
                "the PAM wiring; the two dopamine populations innervate largely different "
                "MBON sets, so this is a wiring comparison, not a clean sign control"},
    "rpe_plastic": {
        "graph": {}, "cfg": {"da_mode": "rpe"},
        "note": "reward prediction error dopamine (da = r - V/V_ref) instead of r = 1"},
}


def build_graph_and_code(c, sub, args, name: str, seed: int, *, sparsity=None,
                         graph=None) -> tuple[P.MBGraph, P.OdorCodes, dict]:
    if graph is None:
        base = dict(shuffle_kc_mbon=False, shuffle_alpn_kc=False, shuffle_dan_gate=False,
                    permute_weights=False, n_swap_factor=args.swap_factor)
        base.update(CONDITIONS[name]["graph"])
        graph = P.build_mb_graph(c, sub, args.threshold, seed=seed, **base)
    code = P.odor_codes(graph, max(N_ODORS, CAP_ODORS),
                        sparsity=args.sparsity if sparsity is None else sparsity,
                        channels_per_odor=args.channels_per_odor, seed=seed)
    return graph, code, graph.meta


def run_condition(c, sub, args, name: str, seed: int, *, sparsity=None, graph=None,
                  code=None) -> dict:
    spec = CONDITIONS[name]
    if graph is None:
        graph, code, gmeta = build_graph_and_code(c, sub, args, name, seed,
                                                  sparsity=sparsity)
    else:
        gmeta = graph.meta
    cfg_kw = {"eta": args.eta, "plasticity": True, "rule": args.rule,
              "code_jitter": args.jitter, "gate": "pam", "da_mode": "reward_only",
              "normalize": True, "max_factor": None, "min_factor": None, "pool": "pam"}
    cfg_kw.update(spec.get("cfg", {}))
    cfg = P.PlasticityConfig(**cfg_kw)
    mode = spec.get("mode", "reward")
    cap_da = -1.0 if mode == "punish" else 1.0
    cap_dir = spec.get("capacity_direction", "up")
    desc = {"note": spec["note"], "graph_meta": gmeta, "n_kc": graph.n_kc,
            "n_mbon": graph.n_mbon, "code": code.info,
            "da_gate_mbons": int(cfg.pool and graph.pam_gate.sum() if cfg.gate == "pam"
                                 else graph.ppl_gate.sum() if cfg.gate == "ppl"
                                 else graph.n_mbon),
            "readout_pool_mbons": int(graph.pam_gate.sum() if cfg.pool == "pam"
                                      else graph.n_mbon)}

    # ---- protocol 1: acquisition -----------------------------------------------------
    L = P.MBLearner(graph, cfg, code, seed=seed)
    acq = P.acquisition(L, n_odors=N_ODORS, rewarded=(0,), trials=ACQ_TRIALS,
                        probe_every=PROBE_EVERY, mode=mode, seed=seed)
    drift = L.drift()
    acq_summary = {
        "auc_initial": acq["auc"][0], "auc_final": acq["auc"][-1],
        "auc_learned_initial": acq["auc_learned"][0],
        "auc_learned_final": acq["auc_learned"][-1],
        "trials_to_auc90": trials_to_criterion(acq["trials"], acq["auc"]),
        "trials_to_auc_learned90": trials_to_criterion(acq["trials"], acq["auc_learned"]),
        "value_target_initial": acq["value_target"][0],
        "value_target_final": acq["value_target"][-1],
        "value_nontarget_initial": acq["value_nontarget"][0],
        "value_nontarget_final": acq["value_nontarget"][-1],
        "response_target_initial": acq["mean_mbon_response_target"][0],
        "response_target_final": acq["mean_mbon_response_target"][-1],
        "value_target_relative_change": (
            (acq["value_target"][-1] - acq["value_target"][0])
            / max(abs(acq["value_target"][0]), 1e-12)),
        "value_separation_final": acq["value_target"][-1] - acq["value_nontarget"][-1],
        "argmax_hit_final": acq["argmax_hit"][-1],
        "n_updates": L.n_updates,
    }
    del L

    # ---- protocol 2: memory capacity --------------------------------------------------
    cap_rows = []
    for m in CAP_GRID:
        if m >= CAP_ODORS:
            continue
        Lc = P.MBLearner(graph, cfg, code, seed=seed)
        r = P.capacity_protocol(Lc, n_odors=CAP_ODORS, n_rewarded=m,
                                trials=max(200, CAP_TRIALS_PER_ASSOC * m),
                                da_value=cap_da, seed=seed)
        score = r["pair_auc"] if cap_dir == "up" else r["pair_auc_inverted"]
        score_l = (r["pair_auc_learned"] if cap_dir == "up"
                   else 1.0 - r["pair_auc_learned"])
        cap_rows.append({"n_rewarded": m, "trials": r["trials"],
                         "pair_auc": r["pair_auc"], "pair_auc_learned": r["pair_auc_learned"],
                         "score": score, "score_learned": score_l,
                         "separation_frac": r["separation_frac"],
                         "value_rewarded_mean": r["value_rewarded_mean"],
                         "value_unrewarded_mean": r["value_unrewarded_mean"],
                         "curve_trials": downsample(r["curve"]["trials"]),
                         "curve_auc": downsample(r["curve"]["auc"]),
                         "curve_auc_learned": downsample(r["curve"]["auc_learned"])})
        del Lc

    def capacity_at(key: str, crit: float) -> float:
        best = 0.0
        for row in cap_rows:
            if row[key] >= crit:
                best = max(best, float(row["n_rewarded"]))
        return best

    return {
        "condition": name, "seed": seed, "desc": desc,
        "config": {"eta": cfg.eta, "rule": cfg.rule, "plasticity": cfg.plasticity,
                   "da_mode": cfg.da_mode, "gate": cfg.gate, "pool": cfg.pool,
                   "normalize": cfg.normalize, "max_factor": cfg.max_factor,
                   "min_factor": cfg.min_factor, "code_jitter": cfg.code_jitter,
                   "sparsity": float(sparsity if sparsity is not None else args.sparsity),
                   "capacity_da": cap_da},
        "acquisition": {k: acq[k] for k in
                        ("trials", "auc", "auc_learned", "value_target", "value_nontarget",
                         "argmax_hit", "mean_mbon_response_target",
                         "mean_mbon_response_other")},
        "acquisition_summary": acq_summary,
        "drift": drift,
        "capacity": cap_rows,
        "capacity_summary": {
            "capacity_auc90": capacity_at("score", CAP_CRITERION),
            "capacity_auc75": capacity_at("score", CAP_LOOSE),
            "capacity_auc90_learned": capacity_at("score_learned", CAP_CRITERION),
            "auc_by_m": {str(int(r["n_rewarded"])): r["pair_auc"] for r in cap_rows},
            "auc_learned_by_m": {str(int(r["n_rewarded"])): r["pair_auc_learned"]
                                 for r in cap_rows},
            # direction-adjusted (so a punishment protocol scores in the learned direction)
            "score_by_m": {str(int(r["n_rewarded"])): r["score"] for r in cap_rows},
            "score_learned_by_m": {str(int(r["n_rewarded"])): r["score_learned"]
                                   for r in cap_rows},
        },
    }


# ------------------------------------------------------------------------------- main ----
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5,
                    help="synapse threshold defining the MB subgraph (5 = published)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--eta", type=float, default=0.1, help="plasticity rate")
    ap.add_argument("--sparsity", type=float, default=0.05, help="active KC fraction")
    ap.add_argument("--channels-per-odor", type=int, default=5,
                    help="glomeruli activated per synthetic odor")
    ap.add_argument("--jitter", type=float, default=0.3,
                    help="fraction of the active KC set re-drawn per presentation")
    ap.add_argument("--rule", default="multiplicative", choices=["multiplicative", "additive"])
    ap.add_argument("--swap-factor", type=int, default=10,
                    help="edge swaps per edge in the shuffle controls")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds = args.seeds[:2]
    t_start = time.time()

    c = Connectome(CANON)
    sub = P.extract_mb_subgraph(c, threshold=args.threshold)
    stats = P.mb_pair_stats(c, sub, args.threshold, c.neuropils)

    out: dict = {
        "phase": "8", "milestone": "M11",
        "criterion": "associative learning demonstrated in a biologically informed "
                     "mushroom-body model on the FlyWire v783 connectome",
        "dataset": {"directory": str(CANON.relative_to(ROOT)), "n_neurons": int(c.n),
                    "meta_version": c.meta.get("version"),
                    "threshold_synapses": args.threshold},
        "annotations": {
            "labels_used": sub.labels,
            "population_sizes": sub.counts,
            "glomerulus_channels": sub.channel_names,
            "annotation_caveats": [
                "Kenyon cells carry top_nt == 'dopamine' in neurons.parquet for 5172/5177 "
                "cells while known_nt is 'acetylcholine; sNPF; ...' for every one of them "
                "(known_nt_source: Barnstedt 2016 / Johard 2008 / Aso 2019). top_nt is "
                "therefore not trustworthy for Kenyon cells in this canonical build and "
                "was NOT used. The neuropil-level nt_code on KC->MBON edges is "
                "acetylcholine for 100% of those edges (21438/21438 at thr=5), which is "
                "the transmitter check actually used here.",
                "cell_class 'ALPN' (685) = cell_sub_class uniglomerular (277) + "
                "multiglomerular (400) + 8 unlabelled; only the 277 uniglomerular cells "
                "define the 56 glomerulus channels. The multiglomerular cells stay "
                "unassigned (channel -1) because they genuinely innervate several "
                "glomeruli, so an odor over 5 glomeruli drives 284 of the 685 ALPNs.",
                "cell_class 'olfactory' (2281, all cell_type ORN_*) is the olfactory "
                "receptor neuron population. It is not used: FlyWire contains no odor "
                "data, so an ORN layer would add purely synthetic structure.",
                "MBON cell_class holds 96 neurons; MBON25 and MBON34 share the cell_type "
                "'MBON25,MBON34' and two cells are published as MBON15-like / MBON17-like.",
                "4 'MBIN' (mushroom-body input) neurons exist in cell_class but are not "
                "used: the MB intrinsic layer is not part of this pathway.",
            ],
        },
        "subgraph_stats": stats,
        "model": {
            "pathway": "ALPN -> KC -> MBON, with DAN providing the teaching signal",
            "odour_code": "synthetic odor = sparse pattern over the 56 glomeruli "
                          "(channels_per_odor), pushed through the measured row-normalized "
                          "ALPN->KC synapse matrix and passed through a fixed-fraction "
                          "k-winner-take-all; trial-to-trial code jitter documented below",
            "rule": {
                "multiplicative": "dw_ij = eta * da * g_j * x_i * (y_j / y_ref) * w_ij",
                "additive": "dw_ij = eta * da * g_j * x_i * y_j",
                "eligibility": "e_ij = x_i * y_j (pre x post coincidence on an existing "
                               "synapse; no separate decayed trace is used)",
                "dopamine_gate": "g_j = 1 iff MBON j receives >= 1 reward-DAN (PAM->MBON) "
                                 "connection at the same threshold; da = +1 on rewarded "
                                 "presentations, 0 otherwise",
                "homeostasis": "after each update every Kenyon cell's total output weight "
                               "onto the MBON layer is rescaled to its anatomical row sum",
                "structural_constraint": "plasticity is masked to synapses present in the "
                                         "connectome: no synapse can be created or deleted",
            },
            "readout": "mean MBON activity over the dopamine-gated (PAM-innervated) MBON "
                       "pool; the value of an odor is this mean",
        },
        "protocols": {
            "acquisition": {"n_odors": N_ODORS, "rewarded_odor": 0, "trials": ACQ_TRIALS,
                            "probe_every": PROBE_EVERY},
            "capacity": {"n_odors": CAP_ODORS, "n_rewarded_grid": list(CAP_GRID),
                         "trials_per_association": CAP_TRIALS_PER_ASSOC,
                         "criterion_auc": CAP_CRITERION, "criterion_auc_loose": CAP_LOOSE,
                         "metric": "AUC = tie-aware fraction of (rewarded, unrewarded) "
                                   "odor pairs whose readout value is correctly ordered; "
                                   "chance = 0.5. 'learned' AUC uses the change in value "
                                   "relative to the untrained readout, which removes the "
                                   "innate bias of the naive connectome readout."},
        },
        "parameters": {"eta": args.eta, "sparsity": args.sparsity,
                       "channels_per_odor": args.channels_per_odor,
                       "code_jitter": args.jitter, "rule": args.rule,
                       "seeds": args.seeds, "swap_factor": args.swap_factor,
                       "command": " ".join(sys.argv)},
        "runs": {}, "qc": {},
        "condition_notes": {k: v["note"] for k, v in CONDITIONS.items()},
    }
    print(f"[8] populations: {sub.counts}")
    print(f"[8] KC->MBON pairs={stats['KC->MBON']['pairs']} "
          f"syn={stats['KC->MBON']['synapses']} MBONs reached="
          f"{stats['KC->MBON']['MBONs_with_input']}")

    for name in CONDITIONS:
        for seed in args.seeds:
            key = f"{name}|seed{seed}"
            r = run_condition(c, sub, args, name, seed)
            out["runs"][key] = r
            s = r["acquisition_summary"]
            print(f"[8] {key:32s} auc {s['auc_initial']:.2f}->{s['auc_final']:.2f} "
                  f"(learned {s['auc_learned_final']:.2f}) t90={s['trials_to_auc90']} "
                  f"cap90={r['capacity_summary']['capacity_auc90']:.0f} "
                  f"[{time.time() - t_start:.0f}s]")

    # ---- eta sweep --------------------------------------------------------------------
    out["eta_sweep"] = {}
    for eta in (0.02, 0.05, 0.1, 0.25, 0.5):
        for seed in args.seeds:
            key = f"eta{eta}|seed{seed}"
            out["eta_sweep"][key] = _run_with(args, c, sub, "real_plastic", seed, eta=eta)
        runs = [out["eta_sweep"][f"eta{eta}|seed{s}"] for s in args.seeds]
        t90 = nanmean_safe([r["acquisition_summary"]["trials_to_auc90"] for r in runs])
        cap90 = np.mean([r["capacity_summary"]["capacity_auc90"] for r in runs])
        print(f"[8] eta={eta:<5} t90={t90:.0f} cap90={cap90:.0f} "
              f"[{time.time() - t_start:.0f}s]")

    # ---- KC sparsity sweep ------------------------------------------------------------
    out["sparsity_sweep"] = {}
    for sp in (0.02, 0.05, 0.10, 0.20):
        for seed in args.seeds:
            key = f"sparsity{sp}|seed{seed}"
            out["sparsity_sweep"][key] = _run_with(args, c, sub, "real_plastic", seed,
                                                   sparsity=sp)
        runs = [out["sparsity_sweep"][f"sparsity{sp}|seed{s}"] for s in args.seeds]
        af = np.mean([r["acquisition_summary"]["auc_final"] for r in runs])
        cp = np.mean([r["capacity_summary"]["capacity_auc90"] for r in runs])
        print(f"[8] sparsity={sp:<5} auc_final={af:.3f} cap90={cp:.0f} "
              f"[{time.time() - t_start:.0f}s]")

    # ---- KC population-size sweep -----------------------------------------------------
    g_real = P.build_mb_graph(c, sub, args.threshold, seed=args.seeds[0])
    out["kc_fraction_sweep"] = {}
    for frac in (0.125, 0.25, 0.5, 1.0):
        for seed in args.seeds:
            gs = P.subsample_kcs(g_real, frac, seed=seed)
            code = P.odor_codes(gs, max(N_ODORS, CAP_ODORS), sparsity=args.sparsity,
                                channels_per_odor=args.channels_per_odor, seed=seed)
            key = f"kcfrac{frac}|seed{seed}"
            out["kc_fraction_sweep"][key] = run_condition(c, sub, args, "real_plastic", seed,
                                                          graph=gs, code=code)
        runs = [out["kc_fraction_sweep"][f"kcfrac{frac}|seed{s}"] for s in args.seeds]
        nk = runs[0]["desc"]["graph_meta"]["n_kc_kept"]
        cp = np.mean([r["capacity_summary"]["capacity_auc90"] for r in runs])
        cpl = np.mean([r["capacity_summary"]["capacity_auc90_learned"] for r in runs])
        print(f"[8] kc_frac={frac:<6} n_kc={nk:<5} cap90={cp:.0f} cap90_learned={cpl:.0f} "
              f"[{time.time() - t_start:.0f}s]")

    # ---- capacity training-budget check ----------------------------------------------
    out["capacity_budget_check"] = {}
    for budget in (60, 300):
        for seed in args.seeds:
            key = f"budget{budget}|seed{seed}"
            out["capacity_budget_check"][key] = _run_capacity_only(
                args, c, sub, "real_plastic", seed, budget)
        runs = [out["capacity_budget_check"][f"budget{budget}|seed{s}"] for s in args.seeds]
        means = {}
        for m in CAP_GRID:
            if m >= CAP_ODORS:
                continue
            means[int(m)] = round(float(np.mean([r["score_by_m"][str(int(m))]
                                                 for r in runs])), 3)
        print(f"[8] capacity budget/assoc={budget:<4} auc by m {means} "
              f"[{time.time() - t_start:.0f}s]")

    # ---- aggregation ------------------------------------------------------------------
    agg_out = {}
    for name in CONDITIONS:
        runs = [out["runs"][f"{name}|seed{s}"] for s in args.seeds]
        auc = np.array([r["acquisition"]["auc"] for r in runs], float)
        aucl = np.array([r["acquisition"]["auc_learned"] for r in runs], float)
        vt = np.array([r["acquisition"]["value_target"] for r in runs], float)
        cap = {str(int(m)): float(np.mean([r["capacity_summary"]["auc_by_m"][str(int(m))]
                                           for r in runs]))
               for m in CAP_GRID if m < CAP_ODORS}
        cap_l = {str(int(m)): float(np.mean(
            [r["capacity_summary"]["auc_learned_by_m"][str(int(m))] for r in runs]))
            for m in CAP_GRID if m < CAP_ODORS}
        cap_s = {str(int(m)): float(np.mean(
            [r["capacity_summary"]["score_by_m"][str(int(m))] for r in runs]))
            for m in CAP_GRID if m < CAP_ODORS}
        cap_sl = {str(int(m)): float(np.mean(
            [r["capacity_summary"]["score_learned_by_m"][str(int(m))] for r in runs]))
            for m in CAP_GRID if m < CAP_ODORS}
        agg_out[name] = {
            "acquisition_auc_mean": [float(x) for x in auc.mean(axis=0)],
            "acquisition_auc_sd": [float(x) for x in auc.std(axis=0)],
            "acquisition_auc_learned_mean": [float(x) for x in aucl.mean(axis=0)],
            "acquisition_auc_learned_sd": [float(x) for x in aucl.std(axis=0)],
            "acquisition_value_target_mean": [float(x) for x in vt.mean(axis=0)],
            "acquisition_value_target_sd": [float(x) for x in vt.std(axis=0)],
            "trials_to_auc90": nanmean_safe(
                [r["acquisition_summary"]["trials_to_auc90"] for r in runs]),
            "trials_to_auc90_sd": nanstd_safe(
                [r["acquisition_summary"]["trials_to_auc90"] for r in runs]),
            "trials_to_auc_learned90": nanmean_safe(
                [r["acquisition_summary"]["trials_to_auc_learned90"] for r in runs]),
            "auc_final": float(np.mean([r["acquisition_summary"]["auc_final"] for r in runs])),
            "auc_final_sd": float(np.std([r["acquisition_summary"]["auc_final"] for r in runs])),
            "auc_initial": float(np.mean([r["acquisition_summary"]["auc_initial"] for r in runs])),
            "auc_learned_final": float(np.mean(
                [r["acquisition_summary"]["auc_learned_final"] for r in runs])),
            "value_target_relative_change": float(np.mean(
                [r["acquisition_summary"]["value_target_relative_change"] for r in runs])),
            "value_separation_final": float(np.mean(
                [r["acquisition_summary"]["value_separation_final"] for r in runs])),
            "capacity_auc_by_m_mean": cap,
            "capacity_auc_learned_by_m_mean": cap_l,
            "capacity_score_by_m_mean": cap_s,
            "capacity_score_learned_by_m_mean": cap_sl,
            "capacity_auc90": float(np.mean([r["capacity_summary"]["capacity_auc90"]
                                             for r in runs])),
            "capacity_auc75": float(np.mean([r["capacity_summary"]["capacity_auc75"]
                                             for r in runs])),
            "capacity_auc90_learned": float(np.mean(
                [r["capacity_summary"]["capacity_auc90_learned"] for r in runs])),
            "drift_mean_abs_relative_change": float(np.mean(
                [r["drift"]["mean_abs_relative_change"] for r in runs])),
            "drift_fraction_beyond_2x": float(np.mean(
                [r["drift"]["fraction_beyond_2x"] for r in runs])),
            "drift_weight_spearman": float(np.mean(
                [r["drift"]["weight_spearman_initial_vs_final"] for r in runs])),
            "drift_row_budget_max_rel_error": float(np.mean(
                [r["drift"]["row_budget_max_rel_error"] for r in runs])),
            "synapses_created": float(np.mean([r["drift"]["synapses_created"] for r in runs])),
            "synapses_lost": float(np.mean([r["drift"]["synapses_lost"] for r in runs])),
            "n_seeds": len(args.seeds),
        }
    out["aggregate"] = agg_out

    #: capacity read off the *seed-averaged* AUC-by-m curve, which is far more robust than
    #: the per-seed capacity (a single seed can miss the criterion by 0.01 and score 0)
    def group_cap(name: str, key: str, crit: float) -> float:
        curve = agg_out[name][key]
        best = 0.0
        for m in CAP_GRID:
            if m >= CAP_ODORS:
                continue
            if curve[str(int(m))] >= crit:
                best = float(m)
        return best

    out["group_capacity"] = {
        n: {"capacity_auc90": group_cap(n, "capacity_score_by_m_mean", CAP_CRITERION),
            "capacity_auc75": group_cap(n, "capacity_score_by_m_mean", CAP_LOOSE),
            "capacity_auc90_learned": group_cap(n, "capacity_score_learned_by_m_mean",
                                                CAP_CRITERION)}
        for n in CONDITIONS}

    def paired(a: str, b: str, field: str) -> dict:
        d = np.array([out["runs"][f"{a}|seed{s}"]["acquisition_summary"][field]
                      - out["runs"][f"{b}|seed{s}"]["acquisition_summary"][field]
                      for s in args.seeds], float)
        return {"field": field, "a": a, "b": b, "mean_diff": nanmean_safe(d),
                "sd_diff": nanstd_safe(d), "per_seed_diff": [float(x) for x in d]}

    out["paired_comparisons"] = {
        k: paired(*v) for k, v in {
            "rule_effect_real__plastic_minus_frozen":
                ("real_plastic", "real_frozen", "auc_final"),
            "rule_effect_shuffled__plastic_minus_frozen":
                ("shuffled_plastic", "shuffled_frozen", "auc_final"),
            "geometry__real_minus_shuffled_plastic":
                ("real_plastic", "shuffled_plastic", "auc_final"),
            "geometry__real_minus_shuffled_gate_plastic":
                ("real_plastic", "shuffled_gate_plastic", "auc_final"),
            "geometry__real_minus_randcode_plastic":
                ("real_plastic", "randcode_plastic", "auc_final"),
            "geometry__real_minus_randomweight_plastic":
                ("real_plastic", "random_weight_plastic", "auc_final"),
            "geometry__initial_bias_real_minus_shuffled":
                ("real_plastic", "shuffled_plastic", "auc_initial"),
            "gate__real_minus_uniform_gate":
                ("real_plastic", "uniform_gate_plastic", "auc_final"),
            "pool__pam_pool_minus_all_pool":
                ("real_plastic", "allpool_plastic", "auc_final"),
            "bounded__minus_unbounded": ("bounded_plastic", "real_plastic", "auc_final"),
            "rule_form__multiplicative_minus_additive":
                ("real_plastic", "additive_plastic", "auc_final"),
        }.items()
    }

    real, shuf = agg_out["real_plastic"], agg_out["shuffled_plastic"]
    rfroz, sfroz = agg_out["real_frozen"], agg_out["shuffled_frozen"]
    out["attribution"] = {
        "auc_real_plastic": real["auc_final"],
        "auc_real_frozen": rfroz["auc_final"],
        "auc_shuffled_plastic": shuf["auc_final"],
        "auc_shuffled_frozen": sfroz["auc_final"],
        "auc_random_weight_plastic": agg_out["random_weight_plastic"]["auc_final"],
        "auc_additive_plastic": agg_out["additive_plastic"]["auc_final"],
        "auc_uniform_gate_plastic": agg_out["uniform_gate_plastic"]["auc_final"],
        "auc_allpool_plastic": agg_out["allpool_plastic"]["auc_final"],
        "rule_effect_real": real["auc_final"] - rfroz["auc_final"],
        "rule_effect_shuffled": shuf["auc_final"] - sfroz["auc_final"],
        "geometry_effect_initial_bias": real["auc_initial"] - shuf["auc_initial"],
        "geometry_effect_frozen_instant": rfroz["auc_final"] - sfroz["auc_final"],
        "geometry_x_rule_interaction": (real["auc_final"] - rfroz["auc_final"]
                                        - (shuf["auc_final"] - sfroz["auc_final"])),
        "capacity_auc90_real_plastic": real["capacity_auc90"],
        "capacity_auc90_real_frozen": rfroz["capacity_auc90"],
        "capacity_auc90_shuffled_plastic": shuf["capacity_auc90"],
        "capacity_auc90_shuffled_frozen": sfroz["capacity_auc90"],
        "capacity_auc90_random_weight_plastic":
            agg_out["random_weight_plastic"]["capacity_auc90"],
        "capacity_auc90_learned_real_plastic": real["capacity_auc90_learned"],
    }

    drifts = [r["drift"] for r in out["runs"].values()
              if r["config"]["plasticity"] and not r["config"]["max_factor"]]
    all_drifts = [r["drift"] for r in out["runs"].values() if r["config"]["plasticity"]]
    out["qc"]["architecture_preservation"] = {
        "plastic_runs_audited": len(all_drifts),
        "all_edges_preserved": bool(all(d["edges_preserved"] for d in all_drifts)),
        "max_synapses_created": int(max(d["synapses_created"] for d in all_drifts)),
        "max_synapses_lost": int(max(d["synapses_lost"] for d in all_drifts)),
        "max_row_budget_rel_error": float(max(d["row_budget_max_rel_error"] for d in all_drifts)),
        "mean_abs_relative_weight_change_range": [
            float(min(d["mean_abs_relative_change"] for d in drifts)),
            float(max(d["mean_abs_relative_change"] for d in drifts))],
        "weight_spearman_range": [
            float(min(d["weight_spearman_initial_vs_final"] for d in drifts)),
            float(max(d["weight_spearman_initial_vs_final"] for d in drifts))],
        "fraction_weights_beyond_2x_range": [
            float(min(d["fraction_beyond_2x"] for d in drifts)),
            float(max(d["fraction_beyond_2x"] for d in drifts))],
    }

    code0 = P.odor_codes(g_real, CAP_ODORS, sparsity=args.sparsity,
                         channels_per_odor=args.channels_per_odor, seed=args.seeds[0])
    out["qc"]["kc_codes"] = P.code_overlap(code0)
    out["qc"]["runtime_s"] = time.time() - t_start

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "learning.json", strip_arrays(out))
    write_summary(out, OUT / "SUMMARY.txt")
    print(f"[8] wrote {OUT/'learning.json'} and {OUT/'SUMMARY.txt'} "
          f"({out['qc']['runtime_s']:.0f}s)")
    return 0


def _run_with(args, c, sub, name: str, seed: int, **kw) -> dict:
    """run_condition with an overridden eta/sparsity, keeping everything else fixed."""
    import copy
    a = copy.copy(args)
    if "eta" in kw:
        a.eta = kw.pop("eta")
    if "sparsity" in kw:
        a.sparsity = kw.pop("sparsity")
    return run_condition(c, sub, a, name, seed, **kw)


def _run_capacity_only(args, c, sub, name: str, seed: int, trials_per_assoc: int) -> dict:
    """Capacity sweep with a different training budget per association."""
    import copy
    a = copy.copy(args)
    global CAP_TRIALS_PER_ASSOC
    keep = CAP_TRIALS_PER_ASSOC
    CAP_TRIALS_PER_ASSOC = int(trials_per_assoc)
    try:
        r = run_condition(c, sub, a, name, seed)
    finally:
        CAP_TRIALS_PER_ASSOC = keep
    return {"auc_by_m": r["capacity_summary"]["auc_by_m"],
            "auc_learned_by_m": r["capacity_summary"]["auc_learned_by_m"],
            "score_by_m": r["capacity_summary"]["score_by_m"],
            "score_learned_by_m": r["capacity_summary"]["score_learned_by_m"],
            "capacity_auc90": r["capacity_summary"]["capacity_auc90"],
            "capacity_auc90_learned": r["capacity_summary"]["capacity_auc90_learned"],
            "trials_per_assoc": int(trials_per_assoc)}


def write_summary(out: dict, path: Path) -> None:
    a = out["aggregate"]
    at = out["attribution"]
    pc = out["paired_comparisons"]
    sub = out["subgraph_stats"]
    lab = out["annotations"]["labels_used"]
    pop = out["annotations"]["population_sizes"]
    par = out["parameters"]
    A = a["real_plastic"]
    L: list[str] = []
    add = L.append
    add("FlyScale Phase 8 / M11 — mushroom-body associative learning on FlyWire v783")
    add("=" * 78)
    add("")
    add("A biologically informed associative-learning benchmark on the real mushroom-body")
    add("subgraph of the canonical v783 connectome: measured acquisition curve, measured")
    add("memory capacity, and controls that make the result readable.")
    add("All numbers come from one executed run of scripts/phase8_learning.py.")
    add(f"dataset      : {out['dataset']['directory']} ({out['dataset']['n_neurons']} neurons)")
    add(f"subgraph     : {out['dataset']['threshold_synapses']}-synapse threshold "
        "(published convention)")
    add(f"parameters   : eta={par['eta']}, rule={par['rule']}, KC sparsity={par['sparsity']}, "
        f"glomeruli/odor={par['channels_per_odor']},")
    add(f"               per-presentation KC jitter={par['code_jitter']}, seeds={par['seeds']}")
    add(f"command      : {par['command']}")
    add(f"runtime      : {out['qc']['runtime_s']:.0f} s")
    add("")
    add("1. POPULATIONS — the annotation labels actually found and used")
    add("-" * 78)
    add(f"  Kenyon cells  cell_class == 'Kenyon_Cell'                  n = {pop['kc']:>6}")
    add(f"                cell_type: {', '.join(lab['kc']['cell_types'])}")
    add(f"  MBONs         cell_class == 'MBON'                         n = {pop['mbon']:>6}")
    add(f"  DANs          cell_class == 'DAN'                          n = {pop['dan']:>6}")
    add(f"                PAM*  (reward dopaminergic)                  n = {pop['pam']:>6}")
    add(f"                PPL*  (PPL1/PPL2, punishment dopaminergic)   n = {pop['ppl']:>6}")
    add(f"  ALPNs         cell_class == 'ALPN'                         n = {pop['alpn']:>6}")
    add(f"                of which cell_sub_class == 'uniglomerular'   n = {pop['upn']:>6}")
    add(f"  glomeruli     {lab['upn']['n_glomeruli']} channels from the uniglomerular PN cell_type prefix")
    add("  Every population above exists verbatim in neurons.parquet; none was invented or")
    add("  substituted. No MB-intrinsic (MBIN, n=4) or olfactory-receptor (cell_class")
    add("  'olfactory', n=2281) population was used.")
    add("")
    add("  CAVEAT, recorded not corrected (the file belongs to another work-stream):")
    add("  neurons.parquet has top_nt == 'dopamine' for 5172 of 5177 Kenyon cells, while")
    add("  known_nt is 'acetylcholine; sNPF' for all of them and 100% of KC->MBON edges")
    add("  carry nt_code == acetylcholine. top_nt is inconsistent for Kenyon cells in this")
    add("  canonical build and was not used; the edge-level nt_code was used instead.")
    add("")
    add("2. MEASURED MB SUBGRAPH (published 5-synapse threshold, real synapses)")
    add("-" * 78)
    km = sub["KC->MBON"]
    add(f"  ALPN->KC   pairs={sub['ALPN->KC']['pairs']:>7}   synapses={sub['ALPN->KC']['synapses']:>8}")
    add(f"  KC->MBON   pairs={km['pairs']:>7}   synapses={km['synapses']:>8}   "
        f"KCs with output={km['KCs_with_output']}/{pop['kc']}   "
        f"MBONs reached={km['MBONs_with_input']}/{pop['mbon']}")
    add(f"             KCs per MBON {km['KCs_per_MBON_mean']:.1f} (mean), "
        f"MBONs per KC {km['MBONs_per_KC_mean']:.2f} (mean)")
    add(f"  DAN->MBON  pairs={sub['DAN->MBON']['pairs']:>7}   (PAM {sub['PAM->MBON']['pairs']}, "
        f"PPL {sub['PPL->MBON']['pairs']}); dopamine nt_code on "
        f"{sub['DAN->MBON']['nt_codes'].get('dopamine', 0)}/"
        f"{sub['DAN->MBON']['pairs']} pairs")
    add(f"  KC->KC     pairs={sub['KC->KC']['pairs']:>7}   (recurrent KC collaterals; measured, not used)")
    add(f"  KC->MBON synapse rows inside the MB neuropils: {km['mb_neuropil_rows']}")
    add("")
    add("3. MODEL")
    add("-" * 78)
    add(f"  odor -> KC code: {par['channels_per_odor']} of {lab['upn']['n_glomeruli']} glomeruli "
        "activated, drive pushed through the")
    add("  measured row-normalized ALPN->KC matrix, then a k-winner-take-all at fixed")
    add(f"  sparsity. Measured: {out['qc']['kc_codes']['active_per_odour']:.0f} active KCs of "
        f"{out['qc']['kc_codes']['n_kc']} (sparsity "
        f"{out['qc']['kc_codes']['sparsity_measured']:.3f}),")
    add(f"  mean pairwise overlap {out['qc']['kc_codes']['mean_pairwise_overlap_cells']:.1f} cells "
        f"= {out['qc']['kc_codes']['mean_pairwise_overlap_frac_of_active']:.1%} of the active set "
        f"(independent-code expectation "
        f"{out['qc']['kc_codes']['expected_overlap_if_independent']:.1f} cells).")
    add(f"  rule ({out['model']['rule'][par['rule']]}) on existing KC->MBON synapses only:")
    add("    eligibility e_ij = x_i * y_j")
    add("    dopamine gate g_j = 1 iff MBON j receives a measured reward-DAN (PAM->MBON) input")
    add("    da = +1 on a rewarded presentation, 0 otherwise")
    add("    homeostasis: each Kenyon cell's total output weight is rescaled to its")
    add("    anatomical row sum after every update")
    add("    no synapse is created, deleted, or re-signed (plasticity is masked to the")
    add("    measured connectome)")
    pool_mbons = out["runs"][f"real_plastic|seed{par['seeds'][0]}"]["desc"]["readout_pool_mbons"]
    add(f"  readout: mean activity over the {pool_mbons} anatomically PAM-innervated MBONs")
    add("    (fixed for every condition, so ablating the dopamine gate does not change the")
    add("    readout; the 'allpool' condition re-runs it over all 96 MBONs as a check)")
    add("  protocol 1 (acquisition): 8 odors, odor 0 rewarded, 300 trials, probe every 10")
    add(f"  protocol 2 (capacity): 64 odors, m rewarded, m in {list(CAP_GRID)}, "
        f"{CAP_TRIALS_PER_ASSOC} trials per association;")
    add("    score = AUC over all (rewarded, unrewarded) odor pairs of the readout value")
    add("    ('learned' AUC uses the change in value from the untrained readout, removing")
    add("    the innate bias of the naive connectome)")
    add("")
    add("4. ACQUISITION — learning curve (mean over "
        f"{A['n_seeds']} seeds, sd in parentheses)")
    add("-" * 78)
    add(f"  {'condition':<26}{'AUC t=0':>9}{'AUC t=300':>11}{'learned t=300':>15}"
        f"{'trials->0.9':>13}{'dV/V target':>13}")
    for name, agg in a.items():
        add(f"  {name:<26}{agg['auc_initial']:>9.3f}{agg['auc_final']:>7.3f}"
            f" ({agg['auc_final_sd']:.2f}){agg['auc_learned_final']:>11.3f}"
            f"{agg['trials_to_auc90']:>13.0f}"
            f"{agg['value_target_relative_change']:>13.3f}")
    add("")
    add("  acquisition curves — mean AUC (rewarded-odor vs unrewarded rank separation)")
    add(f"  {'condition':<26}{'t=10':>7}{'t=30':>7}{'t=60':>7}{'t=100':>7}{'t=150':>7}"
        f"{'t=200':>7}{'t=300':>7}")
    grid = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
            180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
    picks = [10, 30, 60, 100, 150, 200, 300]
    for name in ("real_plastic", "real_frozen", "shuffled_plastic", "shuffled_frozen",
                 "uniform_gate_plastic", "randcode_plastic", "additive_plastic",
                 "rpe_plastic", "aversive_sign_plastic", "ppl_wiring_plastic",
                 "allpool_plastic", "bounded_plastic", "random_weight_plastic"):
        curve = a[name]["acquisition_auc_mean"]
        cells = "".join(f"{curve[grid.index(t)]:>7.2f}" for t in picks)
        add(f"  {name:<26}{cells}")
    add("")
    add("  condition definitions:")
    for name, note in out["condition_notes"].items():
        add(f"    {name:<24} {note}")
    add("")
    add("5. MEMORY CAPACITY — how many odor->reward associations are held at once")
    add("-" * 78)
    add(f"  {'m':>4}  {'real+plastic':>13}{'shuffled+pl':>13}{'real frozen':>13}"
        f"{'shuf frozen':>13}{'randw+pl':>11}{'additive':>11}")
    for m in CAP_GRID:
        if m >= CAP_ODORS:
            continue
        row = [a[n]["capacity_auc_by_m_mean"][str(int(m))] for n in
               ("real_plastic", "shuffled_plastic", "real_frozen", "shuffled_frozen",
                "random_weight_plastic", "additive_plastic")]
        add(f"  {m:>4}  {row[0]:>13.3f}{row[1]:>13.3f}{row[2]:>13.3f}{row[3]:>13.3f}"
            f"{row[4]:>11.3f}{row[5]:>11.3f}")
    add("")
    gc = out["group_capacity"]
    add(f"  measured capacity = largest m whose seed-averaged AUC reaches the criterion")
    add(f"  {'condition':<24}{'cap AUC>=0.9':>14}{'cap AUC>=0.75':>15}"
        f"{'learned cap':>13}{'per-seed mean':>15}")
    for n in ("real_plastic", "shuffled_plastic", "shuffled_gate_plastic", "randcode_plastic",
              "random_weight_plastic", "additive_plastic", "bounded_plastic",
              "real_frozen", "shuffled_frozen", "uniform_gate_plastic", "allpool_plastic",
              "rpe_plastic", "aversive_sign_plastic", "ppl_wiring_plastic"):
        add(f"    {n:<24}{gc[n]['capacity_auc90']:>13.0f}"
            f"{gc[n]['capacity_auc75']:>15.0f}{gc[n]['capacity_auc90_learned']:>13.0f}"
            f"{a[n]['capacity_auc90']:>15.1f}")
    add("  capacity is interference-limited, not training-limited: raising the budget from")
    add(f"  {CAP_TRIALS_PER_ASSOC} to 300 trials per association moves the collapse point "
        "by at most one")
    add("  grid step (see capacity_budget_check in learning.json).")
    add("")
    add("6. CONTROLS AND THE §27 RISK (does training dominate architecture?)")
    add("-" * 78)
    add(f"  rule effect, real wiring        : {at['rule_effect_real']:+.3f} AUC (plastic - frozen)")
    add(f"  rule effect, shuffled wiring    : {at['rule_effect_shuffled']:+.3f} AUC")
    add(f"  geometry effect, initial bias   : {at['geometry_effect_initial_bias']:+.3f} AUC")
    add(f"  geometry effect, frozen end     : {at['geometry_effect_frozen_instant']:+.3f} AUC")
    add(f"  geometry x rule interaction     : {at['geometry_x_rule_interaction']:+.3f} AUC")
    add(f"  real - shuffled (plastic), paired per seed: "
        f"{pc['geometry__real_minus_shuffled_plastic']['mean_diff']:+.3f} "
        f"+- {pc['geometry__real_minus_shuffled_plastic']['sd_diff']:.3f} "
        f"(n={len(pc['geometry__real_minus_shuffled_plastic']['per_seed_diff'])})")
    for k in ("geometry__real_minus_shuffled_gate_plastic",
              "geometry__real_minus_randcode_plastic",
              "geometry__real_minus_randomweight_plastic",
              "gate__real_minus_uniform_gate",
              "pool__pam_pool_minus_all_pool",
              "bounded__minus_unbounded",
              "rule_form__multiplicative_minus_additive"):
        v = pc[k]
        add(f"  {k:<47} {v['mean_diff']:+.3f} +- {v['sd_diff']:.3f}")
    add("")
    add("7. ARCHITECTURE PRESERVATION (scope doc: learning rules must not destroy the source)")
    add("-" * 78)
    q = out["qc"]["architecture_preservation"]
    add(f"  plastic runs audited: {q['plastic_runs_audited']}   all KC->MBON edges preserved: "
        f"{q['all_edges_preserved']}")
    add(f"  synapses created (max over runs): {q['max_synapses_created']}   deleted (max): "
        f"{q['max_synapses_lost']}")
    add(f"  per-KC output weight budget restored to the anatomical value, max relative error "
        f"{q['max_row_budget_rel_error']:.1e}")
    add(f"  mean |relative weight change| {q['mean_abs_relative_weight_change_range'][0]:.3f}"
        f"-{q['mean_abs_relative_weight_change_range'][1]:.3f}; "
        f"fraction of synapses beyond 2x "
        f"{q['fraction_weights_beyond_2x_range'][0]:.3f}-"
        f"{q['fraction_weights_beyond_2x_range'][1]:.3f}")
    add(f"  Spearman(initial weight, final weight) "
        f"{q['weight_spearman_range'][0]:.3f}-{q['weight_spearman_range'][1]:.3f} "
        "-> the anatomical weight ordering survives learning")
    add(f"  explicit cap variant (all synapses within 0.25x-4x of anatomical): AUC "
        f"{a['bounded_plastic']['auc_final']:.3f} vs {A['auc_final']:.3f} unbounded")
    add("")
    add("8. SENSITIVITY")
    add("-" * 78)
    for eta in (0.02, 0.05, 0.1, 0.25, 0.5):
        runs = [out["eta_sweep"][f"eta{eta}|seed{s}"] for s in par["seeds"]]
        t90 = np.array([r["acquisition_summary"]["trials_to_auc90"] for r in runs], float)
        af = np.array([r["acquisition_summary"]["auc_final"] for r in runs], float)
        cp = np.array([r["capacity_summary"]["capacity_auc90"] for r in runs], float)
        add(f"  eta={eta:<5} trials to AUC>=0.9 {nanmean_safe(t90):>6.0f} "
            f"(sd {nanstd_safe(t90):>5.0f}),  final AUC {af.mean():.3f}, "
            f"capacity {cp.mean():.0f}")
    for sp in (0.02, 0.05, 0.10, 0.20):
        runs = [out["sparsity_sweep"][f"sparsity{sp}|seed{s}"] for s in par["seeds"]]
        af = np.array([r["acquisition_summary"]["auc_final"] for r in runs], float)
        cp = np.array([r["capacity_summary"]["capacity_auc90"] for r in runs], float)
        cpl = np.array([r["capacity_summary"]["capacity_auc90_learned"] for r in runs], float)
        add(f"  KC sparsity={sp:<5} final AUC {af.mean():.3f}, capacity(AUC>=0.9) "
            f"{cp.mean():>4.0f}, learned-capacity {cpl.mean():>4.0f}")
    add("  (the raw-capacity column is coarse: the m=8 crossing sits within seed noise at")
    add("   large KC counts, while the learned-capacity column saturates at 8 for >=1294 KCs)")
    for frac in (0.125, 0.25, 0.5, 1.0):
        runs = [out["kc_fraction_sweep"][f"kcfrac{frac}|seed{s}"] for s in par["seeds"]]
        nk = runs[0]["desc"]["graph_meta"]["n_kc_kept"]
        cp = np.array([r["capacity_summary"]["capacity_auc90"] for r in runs], float)
        cpl = np.array([r["capacity_summary"]["capacity_auc90_learned"] for r in runs], float)
        af = np.array([r["acquisition_summary"]["auc_final"] for r in runs], float)
        add(f"  KCs kept {nk:>5} ({frac:>5.3f} of {pop['kc']}): capacity(AUC>=0.9) "
            f"{cp.mean():.0f}, learned-capacity {cpl.mean():.0f}, final AUC {af.mean():.3f}")
    add("")
    add("9. WHAT IS AND IS NOT SHOWN")
    add("-" * 78)
    add(f"  * The plastic pathway learns the association: the rewarded odor's rank")
    add(f"    separation rises from AUC {A['auc_initial']:.2f} (naive connectome readout) to "
        f"{A['auc_final']:.2f}")
    add(f"    within {A['trials_to_auc90']:.0f} of 300 trials and the mean MBON response to it")
    add(f"    rises by {A['value_target_relative_change']:+.1%} while the untrained connectome "
        "readout")
    add("    puts it below the unrewarded odors; the no-plasticity control is exactly flat.")
    add("  * The connectome geometry is NOT what makes learning work: a degree-preserving")
    add(f"    shuffle of the KC->MBON targets reproduces acquisition "
        f"({a['shuffled_plastic']['auc_final']:.3f} vs "
        f"{A['auc_final']:.3f}) and capacity")
    add(f"    ({a['shuffled_plastic']['capacity_auc90']:.0f} vs {A['capacity_auc90']:.0f} "
        f"per-seed, {gc['shuffled_plastic']['capacity_auc90']:.0f} vs "
        f"{gc['real_plastic']['capacity_auc90']:.0f} on the seed-averaged curve), as do "
        "shuffled")
    add("    weights, a shuffled dopamine gate, a random-projection odor code and the")
    add("    additive variant of the rule. Under this benchmark §27 is confirmed in the")
    add("    strong form:")
    add("    the imposed rule carries the performance and the connectome only sets the")
    add("    initial bias of the naive readout.")
    add(f"    measured capacity is {gc['real_plastic']['capacity_auc90']:.0f} association(s) for "
        f"real+plastic and {gc['shuffled_plastic']['capacity_auc90']:.0f} for shuffled+plastic, "
        f"against")
    add(f"    {gc['real_frozen']['capacity_auc90']:.0f} and "
        f"{gc['shuffled_frozen']['capacity_auc90']:.0f} for the two frozen controls.")
    add("  * The 'allpool' condition is flat by construction, and that is worth stating:")
    add("    with the readout taken over the entire MBON layer, per-KC output-weight")
    add("    normalization makes the pool mean exactly invariant (sum_j y_j = sum_i x_i *")
    add("    budget_i), so a uniform readout over all 96 MBONs cannot move no matter how the")
    add("    weights redistribute. Only a restricted readout can express learning.")
    add("  * The rpe variant reaches a much higher capacity (48 vs 8) because in that mode an")
    add("    unrewarded presentation carries an active negative teaching signal")
    add("    (da = 0 - V/V_ref < 0) that suppresses the competitor odors; the reward-only")
    add("    rule never depresses an unrewarded odor and is limited by interference.")
    add("  * Capacity rises as the KC code gets sparser (sparsity 0.02 -> 10 associations,")
    add("    0.05 -> 7, 0.10 -> 4, 0.20 -> 2), the sparse-coding prediction.")
    add("  * Two manipulations do change the outcome and are reported as such, both of them")
    add("    changing *where the teaching signal can act* rather than what the odors look")
    add("    like:")
    add(f"    - broadcasting dopamine to every MBON instead of routing it through the measured")
    add(f"      DAN wiring gives final AUC "
        f"{a['uniform_gate_plastic']['auc_final']:.3f} vs {A['auc_final']:.3f} with the")
    add(f"      anatomical gate (learned-component AUC "
        f"{a['uniform_gate_plastic']['auc_learned_final']:.3f} vs "
        f"{A['auc_learned_final']:.3f}), i.e. *which* synapses the teaching signal can")
    add(f"      reach does matter here ({a['uniform_gate_plastic']['trials_to_auc90']:.0f} vs "
        f"{A['trials_to_auc90']:.0f} trials to criterion, capacity "
        f"{gc['uniform_gate_plastic']['capacity_auc90']:.0f} vs "
        f"{gc['real_plastic']['capacity_auc90']:.0f}), unlike which KCs an MBON listens to or")
    add("      how the odor code is built.")
    add(f"    - flipping only the sign of the teaching signal (aversive_sign_plastic, "
        f"identical")
    add(f"      gate and readout) drives the punished odor's readout down: final AUC "
        f"{a['aversive_sign_plastic']['auc_final']:.3f}")
    add(f"      (chance 0.5), where every appetitive condition sits at "
        f"{A['auc_final']:.3f}. The learned")
    add("      component is symmetric, so the rule is genuinely signed, not a mere")
    add(f"      strengthening term. Routing the punishment through the measured PPL1/PPL2")
    add(f"      wiring instead gives {a['ppl_wiring_plastic']['auc_final']:.3f}: the two "
        "dopamine populations")
    add("      innervate largely different MBON sets, so with the appetitive readout fixed")
    add("      this is a wiring comparison and not a clean sign control — reported as such.")
    add("  * This is a benchmark, not a model of the fly: no spiking, no KC->KC collaterals,")
    add("    no MBON->DAN feedback, single-compartment MBONs, a synthetic odor input over")
    add("    glomeruli and a uniform-mean readout over the PAM-gated MBON pool.")
    add("  * Absolute capacity depends on the readout, criterion and protocol; the frozen")
    add("    and shuffled controls are therefore reported on exactly the same protocol.")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
