"""Phase 0 gate: compare the canonical baseline against published FlyWire values.

    python scripts/phase0_gate.py [--baseline results/phase0/flywire_baseline.json]

Writes results/phase0/gate_result.json and prints a plain-text table. Every comparison
states its published source, its tolerance, and whether the published number came from
connectome version v630 or v783 (the canonical dataset here is v783, which has 139,255
neurons against v630's 127,978, so version-sensitive quantities carry an explicit caveat).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from flyscale.connectome import Connectome

ROOT = Path(__file__).resolve().parents[1]


def _get(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def merged_neuropil_count(c: Connectome) -> int:
    names = set()
    for nm in c.neuropils["name_to_code"]:
        names.add(re.sub(r"_[LR]$", "", nm))
    return len(names)


def derived(c: Connectome, threshold: int) -> dict:
    """Quantities the gate computes directly from the canonical graph (not from blocks)."""
    cc = c.thresholded(threshold) if threshold > 1 else c
    pre, post = cc.pre, cc.post
    m = pre != post
    n = c.n
    out_deg = np.bincount(pre[m], minlength=n)
    in_deg = np.bincount(post[m], minlength=n)
    active = (out_deg + in_deg) > 0
    r = float(np.corrcoef(out_deg[active], in_deg[active])[0, 1])

    # rich club as defined by the paper: total degree > 37 on the thresholded graph
    total = out_deg + in_deg
    rich = total > 37
    n_rich = int(rich.sum())
    e_in_rich = int((rich[pre] & rich[post]).sum())
    p_all = pre.size / (n * (n - 1))
    p_rich = e_in_rich / (n_rich * (n_rich - 1)) if n_rich > 1 else float("nan")

    return {
        "in_out_degree_pearson": round(r, 4),
        "mean_out_degree": round(float(pre.size / n), 4),
        "mean_total_degree": round(float(total[active].mean()), 4),
        "median_total_degree": float(np.median(total[active])),
        "n_neurons_with_any_edge": int(active.sum()),
        "merged_neuropils": merged_neuropil_count(c),
        "n_neuropil_labels_raw": len(c.neuropils["name_to_code"]),
        "rich_club_size_at_degree_37": n_rich,
        "rich_club_size_fraction": round(n_rich / n, 4),
        "edges_inside_rich_club": e_in_rich,
        "connection_probability": p_all,
        "rich_club_connection_probability": p_rich,
        "rich_club_density_ratio": round(p_rich / p_all, 3) if p_all else None,
        "threshold_used": int(threshold),
    }


def evaluate(checks: list[dict], baseline: dict, derived_vals: dict) -> list[dict]:
    rows = []
    for chk in checks:
        ours = _get(baseline, chk["ours_path"])
        if chk["ours_path"] == "merged_neuropils":
            ours = derived_vals["merged_neuropils"]
        if chk["ours_path"].startswith("derived."):
            ours = derived_vals[chk["ours_path"].split(".", 1)[1]]
        pub = chk["published"]
        tol_abs = chk.get("tolerance_abs")
        tol_rel = chk.get("tolerance_rel")
        if ours is None:
            rows.append({**chk, "ours": None, "status": "NOT MEASURED",
                         "delta": None, "tolerance": None})
            continue
        delta = float(ours) - float(pub)
        if tol_abs is not None:
            tol = float(tol_abs)
        elif tol_rel is not None:
            tol = abs(float(pub)) * float(tol_rel)
        else:
            tol = 0.0
        status = "PASS" if abs(delta) <= tol else "FAIL"
        rows.append({**chk, "ours": ours, "delta": round(delta, 6), "tolerance": tol,
                     "status": status})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ROOT / "results" / "phase0" / "flywire_baseline.json"))
    ap.add_argument("--canonical", default=str(ROOT / "data" / "processed" / "canonical_v783"))
    ap.add_argument("--reference", default=str(ROOT / "references" / "flywire_published_values.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase0" / "gate_result.json"))
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    ref = json.loads(Path(args.reference).read_text())
    c = Connectome(args.canonical)
    threshold = int(baseline.get("synapse_threshold", 5))
    d = derived(c, threshold)
    rows = evaluate(ref["checks"], baseline, d)

    hard = [r for r in rows if not r.get("advisory", False)]
    soft = [r for r in rows if r.get("advisory", False)]
    verdict = {
        "gate": "Phase 0 baseline vs published FlyWire values",
        "canonical_version": baseline.get("canonical_version"),
        "required_checks_passed": sum(r["status"] == "PASS" for r in hard),
        "required_checks_total": len(hard),
        "advisory_checks_passed": sum(r["status"] == "PASS" for r in soft),
        "advisory_checks_total": len(soft),
        "derived": d,
        "checks": rows,
    }
    verdict["passed"] = verdict["required_checks_passed"] == verdict["required_checks_total"]

    Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    hdr = f"{'check':32s} {'ours':>14s} {'published':>12s} {'delta':>11s} {'tol':>9s}  status"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ours = "-" if r["ours"] is None else f"{float(r['ours']):.6g}"
        delta = "-" if r["delta"] is None else f"{r['delta']:+.6g}"
        tol = "-" if r["tolerance"] is None else f"{r['tolerance']:.6g}"
        flag = "" if not r.get("advisory") else "  (advisory)"
        print(f"{r['name']:32s} {ours:>14s} {float(r['published']):>12.6g} "
              f"{delta:>11s} {tol:>9s}  {r['status']}{flag}")
    print()
    print(f"required: {verdict['required_checks_passed']}/{verdict['required_checks_total']} "
          f"| advisory: {verdict['advisory_checks_passed']}/{verdict['advisory_checks_total']} "
          f"| gate passed: {verdict['passed']}")
    print("derived:", json.dumps(d))
    print("wrote", args.out)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
