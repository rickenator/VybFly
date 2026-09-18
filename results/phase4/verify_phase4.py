"""Independent re-derivation + protocol replay of the Phase 4 results.

1. Replays build_protocol(c, threshold=5, test_frac=0.1, seed=0) from scratch and checks
   that the saved fit pairs, held-out pairs and negatives are reproduced exactly.
2. Re-fits the connection law (R, T) on the replayed fit pairs (positives + negatives)
   for every geometry and compares with the values in geometry.json.
3. Recomputes every held-out AUC / AP / log-likelihood from the saved coordinates and
   the replayed pair sets, and compares with geometry.json.

    python /tmp/verify_phase4.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flyscale.connectome import Connectome                                   # noqa: E402
from flyscale.geometry import (average_precision, bernoulli_log_likelihood,  # noqa: E402
                               build_protocol, connection_probability, euclidean_distance,
                               fit_connection_law, hyperbolic_distance_poincare, roc_auc)

RES = ROOT / "results" / "phase4"
doc = json.loads((RES / "geometry.json").read_text())
ART = RES / "artifacts"
TOL = 5e-4
problems = []

fit_pos = np.load(ART / "protocol_fit_pairs.npy").astype(np.int64)
test_pos = np.load(ART / "protocol_test_pairs.npy").astype(np.int64)
test_neg = np.load(ART / "protocol_eval_negatives.npy").astype(np.int64)
deg = np.load(ART / "fit_degrees.npy")

# ---------------------------------------------------------------- 1. protocol replay
c = Connectome(doc["canonical"]["canonical_dir"])
proto = build_protocol(c, threshold=doc["protocol"].get("threshold", doc["config"]["threshold_synapses"]),
                       test_frac=doc["protocol"]["test_frac"], seed=doc["protocol"]["seed"])
for label, saved, replayed in (("fit_pos", fit_pos, np.stack(proto["pairs"]["fit_pos"], 1)),
                              ("test_pos", test_pos, np.stack(proto["pairs"]["test_pos"], 1)),
                              ("test_neg", test_neg, np.stack(proto["pairs"]["test_neg"], 1)),
                              ("degrees", deg.astype(np.int64), proto["deg_total_fit"].astype(np.int64)),
                              ("node_mask", np.load(ART / "analysis_node_mask.npy"),
                               proto["eval_node_mask"])):
    same = saved.shape == replayed.shape and bool(np.array_equal(saved, replayed))
    print(f"replay {label:<10} shapes {saved.shape} vs {replayed.shape} -> identical: {same}")
    if not same:
        problems.append(f"replay mismatch for {label}")
fit_neg = np.stack(proto["pairs"]["fit_neg"], 1).astype(np.int64)

KNOWN = {
    "anatomical_xyz": "euclidean",
    "anatomical_xy": "euclidean",
    "anatomical_pca2": "euclidean",
    "anatomical_xyz_zscored": "euclidean",
    "spectral_16": "euclidean",
    "spectral_32": "euclidean",
    "hyperbolic_2d": "hyperbolic",
}


def coords_path(name):
    for cand in (f"{name}_coords.npy", f"{name}.npy"):
        if (ART / cand).exists():
            return ART / cand
    return None


FILES = {}
for n in doc["geometries"]:
    if n in KNOWN and coords_path(n):
        FILES[n] = (coords_path(n).name, KNOWN[n])
for n, kind in KNOWN.items():
    if n not in FILES and coords_path(n):       # geometry present as artifact but not scored
        FILES[n] = (coords_path(n).name, kind)


def dist(kind, X, i, j):
    return (hyperbolic_distance_poincare(X[i], X[j]) if kind == "hyperbolic"
            else euclidean_distance(X, i, j))


print(f"\n{'geometry':<16} {'AUC json':>9} {'AUC rechk':>9} {'AP json':>8} {'AP rechk':>8} "
      f"{'LL json':>9} {'LL rechk':>9} {'R json':>9} {'R rechk':>9} {'T rechk':>8}")
for name, (fname, kind) in FILES.items():
    if name not in doc["geometries"] or not (ART / fname).exists():
        print(f"{name:<16} (no artifact / not in json)")
        continue
    g = doc["geometries"][name]
    X = np.load(ART / fname)
    # ---- re-fit the law on the replayed fit pairs
    d_fp = dist(kind, X, fit_pos[:, 0], fit_pos[:, 1])
    d_fn = dist(kind, X, fit_neg[:, 0], fit_neg[:, 1])
    mfp, mfn = np.isfinite(d_fp), np.isfinite(d_fn)
    law2 = fit_connection_law(np.concatenate([d_fp[mfp], d_fn[mfn]]),
                              np.concatenate([np.ones(int(mfp.sum())), np.zeros(int(mfn.sum()))]))
    law = g["connection_law"]
    if abs(law2["R"] - law["R"]) > 1e-3 or abs(law2["T"] - law["T"]) > 1e-3:
        problems.append(f"{name}: refit law R={law2['R']} T={law2['T']} vs json "
                        f"R={law['R']} T={law['T']}")
    # ---- recompute held-out metrics
    d_tp, d_tn = dist(kind, X, test_pos[:, 0], test_pos[:, 1]), dist(kind, X, test_neg[:, 0], test_neg[:, 1])
    okp, okn = np.isfinite(d_tp), np.isfinite(d_tn)
    y = np.concatenate([np.ones(int(okp.sum())), np.zeros(int(okn.sum()))])
    d = np.concatenate([d_tp[okp], d_tn[okn]])
    auc, ap, = roc_auc(y, -d), average_precision(y, -d)
    ll = bernoulli_log_likelihood(y, connection_probability(d, law["R"], law["T"]))
    print(f"{name:<16} {g['heldout']['auc']:>9.6f} {auc:>9.6f} "
          f"{g['heldout']['average_precision']:>8.5f} {ap:>8.5f} "
          f"{g['heldout']['log_likelihood_mean']:>9.5f} {ll:>9.5f} "
          f"{law['R']:>9.4f} {law2['R']:>9.4f} {law2['T']:>8.4f}")
    for label, a, b in (("auc", g["heldout"]["auc"], auc),
                        ("ap", g["heldout"]["average_precision"], ap),
                        ("ll", g["heldout"]["log_likelihood_mean"], ll)):
        if abs(a - b) > TOL:
            problems.append(f"{name}.{label}: json={a!r} recomputed={b!r}")
    if name == "hyperbolic_2d":
        r = np.linalg.norm(X, axis=1)
        print(f"{'':<16} Poincare radii: max={r.max():.6f} (<1 required), median={np.median(r):.4f}")
        if r.max() >= 1.0:
            problems.append("hyperbolic coordinate outside the Poincare ball")

# ------------------------------------------------------- degree-only baseline
s = np.log(np.maximum(deg, 1.0))
sc = np.concatenate([s[test_pos[:, 0]] + s[test_pos[:, 1]], s[test_neg[:, 0]] + s[test_neg[:, 1]]])
y = np.concatenate([np.ones(test_pos.shape[0]), np.zeros(test_neg.shape[0])])
db = doc["degree_only_baseline"]
auc_d, ap_d = roc_auc(y, sc), average_precision(y, sc)
p_d = 1.0 / (1.0 + np.exp(-(db["a"] + db["b"] * sc)))
ll_d = bernoulli_log_likelihood(y, p_d)
print(f"\n{'degree_only':<16} {db['heldout']['auc']:>9.6f} {auc_d:>9.6f} "
      f"{db['heldout']['average_precision']:>8.5f} {ap_d:>8.5f} "
      f"{db['heldout']['log_likelihood_mean']:>9.5f} {ll_d:>9.5f}")
for label, a, b in (("auc", db["heldout"]["auc"], auc_d),
                    ("ap", db["heldout"]["average_precision"], ap_d),
                    ("ll", db["heldout"]["log_likelihood_mean"], ll_d)):
    if abs(a - b) > TOL:
        problems.append(f"degree_only.{label}: json={a!r} recomputed={b!r}")

p = doc["protocol"]
assert p["test_edges_inside_giant"] == test_pos.shape[0], (p["test_edges_inside_giant"], test_pos.shape)
assert p["fit_edges_inside_giant"] == fit_pos.shape[0], (p["fit_edges_inside_giant"], fit_pos.shape)
assert p["n_eval_negatives"] == test_neg.shape[0]
assert p["n_test_edges"] >= p["test_edges_inside_giant"] and p["n_fit_edges"] >= p["fit_edges_inside_giant"]
assert doc["canonical"]["canonical_version"] == "flywire-v783-canon-1"
assert doc["config"]["threshold_synapses"] == 5 and test_neg.shape[0] == test_pos.shape[0]
print(f"protocol consistency: full split {p['n_test_edges']} held-out / "
      f"{p['fit_edges_inside_giant']} fit pairs inside the analysis graph, "
      f"{p['n_eval_negatives']} negatives (1:1)")
print("json errors:", doc.get("errors"))
print("headline:", json.dumps(doc["headline"], indent=1))
print("PROBLEMS:", problems if problems else "none - protocol replays and every held-out number reproduces")
