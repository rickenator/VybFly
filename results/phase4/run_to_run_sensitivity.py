"""Run-to-run sensitivity of the hyperbolic embedding (Phase 4 companion).

Two fits with the *same* configuration (dim=2, lr=0.05, batch=65536, 150 epochs, seed=0,
degree-ranked initialisation) were run on protocol variants that differ by 14 neurons
(0.01% of the connectome, the neurons without an annotation row, which were removed from
the analysis graph after the first run). That tiny change shifts the sampled non-edges and
therefore the SGD trajectory. This script evaluates both embeddings on the *identical*
held-out pair set of the delivered protocol so the spread is measured, not asserted.

It copies the second run's coordinates into the delivered artifacts and writes
results/phase4/run_to_run_sensitivity.json.

    python /tmp/phase4_run2.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from flyscale.connectome import Connectome                       # noqa: E402
from flyscale.geometry import build_protocol, evaluate_geometry   # noqa: E402

RES = ROOT / "results" / "phase4"
ART = RES / "artifacts"
doc = json.loads((RES / "geometry.json").read_text())
c = Connectome(doc["canonical"]["canonical_dir"])

run2_artifact = ART / "hyperbolic_2d_run2_coords.npy"
run2_src = run2_artifact if run2_artifact.exists() else Path("/tmp/proto6_D.npy")
if not run2_src.exists():
    raise SystemExit("second-run coordinates not found; nothing to compare")
run2 = np.load(run2_src)
if run2_src != run2_artifact:
    shutil.copyfile(run2_src, run2_artifact)

proto = build_protocol(c, threshold=doc["protocol"].get("threshold", doc["config"]["threshold_synapses"]),
                      test_frac=doc["protocol"]["test_frac"], seed=doc["protocol"]["seed"])
run1 = np.load(ART / "hyperbolic_2d_coords.npy")

out = {
    "why": "the two hyperbolic fits use the same configuration; the only difference is the "
           "protocol variant (14 neurons without an annotation row removed from the analysis "
           "graph after the first fit), which changes the sampled non-edges and hence the SGD "
           "trajectory. Both are scored on the identical held-out pairs of the delivered "
           "protocol, so the spread below is a measured run-to-run sensitivity.",
    "config": {"dim": 2, "lr": 0.05, "batch": 65536, "epochs": 150, "seed": 0,
               "init": "degree"},
    "runs": {},
}
for name, X, note in (("run1_delivered", run1, "this repository's script run (post-patch "
                                                "protocol: 132,181 analysis nodes)"),
                      ("run2_same_config", run2, "separate run of the same configuration on "
                                                "the pre-patch protocol (132,191 nodes)")):
    res = evaluate_geometry(proto, X, "hyperbolic", name)
    out["runs"][name] = {"note": note, "heldout": res["heldout"],
                         "connection_law": {k: res["connection_law"].get(k) for k in ("R", "T", "nll_mean")},
                         "artifact": f"results/phase4/artifacts/{'hyperbolic_2d_coords.npy' if name == 'run1_delivered' else 'hyperbolic_2d_run2_coords.npy'}"}

a1 = out["runs"]["run1_delivered"]["heldout"]["auc"]
a2 = out["runs"]["run2_same_config"]["heldout"]["auc"]
deg_auc = doc["degree_only_baseline"]["heldout"]["auc"]
anat_auc = doc["geometries"]["anatomical_xyz"]["heldout"]["auc"]
out["spread_auc"] = abs(a1 - a2)
out["interpretation"] = (
    f"two runs of the same hyperbolic configuration reach held-out AUC {a1:.4f} and {a2:.4f} "
    f"on identical held-out pairs (spread {abs(a1 - a2):.4f}), which is larger than both the "
    f"hyperbolic-vs-anatomical margin ({a1 - anat_auc:+.4f} for the delivered run) and the "
    f"hyperbolic-vs-degree-only margin ({a1 - deg_auc:+.4f}). Both runs beat the 3-D "
    f"anatomical coordinates ({anat_auc:.4f}), but the comparison against the degree-only "
    f"baseline ({deg_auc:.4f}) is not resolved by this protocol: it flips between runs.")
(RES / "run_to_run_sensitivity.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
print(json.dumps({k: out[k] for k in ("spread_auc", "interpretation")}, indent=2))
print("wrote", RES / "run_to_run_sensitivity.json")
