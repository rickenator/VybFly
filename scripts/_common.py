"""Shared helpers for the scaling/closure phase scripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flyscale.connectome import Connectome
from flyscale.renorm import GeometryLaw, fit_connection_law
from flyscale.synthetic import GraphView, from_connectome

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "processed" / "canonical_v783"
PHASE4 = ROOT / "results" / "phase4"
PHASE5 = ROOT / "results" / "phase5"
PHASE6 = ROOT / "results" / "phase6"
PHASE7 = ROOT / "results" / "phase7"


def load_target(threshold: int = 5) -> GraphView:
    """The biological reference graph: canonical v783 with the published 5-synapse rule."""
    c = Connectome(CANON)
    c5 = c.thresholded(threshold)
    prov = {"kind": "canonical_v783_thr%d" % threshold, "n_source": int(c.n),
            "source_edges": int(c.pairs.shape[0]), "threshold": threshold}
    return from_connectome(c5, provenance=prov)


def anatomical_coords(g: GraphView, scale: bool = True) -> np.ndarray:
    """Anatomical neuron coordinates from the annotation table (xyz, nan -> mean imputed)."""
    import pandas as pd
    ann = pd.read_parquet(CANON / "neurons.parquet", columns=["idx", "pos_x", "pos_y", "pos_z"])
    ann = ann.sort_values("idx")
    xyz = np.array(ann[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64), copy=True)
    bad = ~np.isfinite(xyz).all(axis=1)
    if bad.any():
        xyz[bad] = np.nanmean(xyz[~bad], axis=0)
    if scale:
        xyz = (xyz - xyz.mean(axis=0)) / np.maximum(xyz.std(axis=0), 1e-9)
    return xyz


def _find_embedding_file(patterns: tuple[str, ...]) -> Path | None:
    d = PHASE4 / "artifacts"
    if not d.exists():
        return None
    for p in sorted(d.glob("*.npy")):
        low = p.name.lower()
        if any(pat in low for pat in patterns) and "labels" not in low:
            return p
    return None


def load_geometry(g: GraphView, kind: str = "auto", threshold: int = 5,
                  fit: bool = True, seed: int = 0) -> GeometryLaw:
    """Build the geometry law used by scaling.

    'anatomical' uses the neuron annotation coordinates; 'hyperbolic'/'spectral' load the
    Phase 4 embedding when it exists. `auto` prefers a Phase 4 hyperbolic embedding and falls
    back to anatomical coordinates, recording which was used.
    """
    g4 = PHASE4 / "geometry.json"
    fits = json.loads(g4.read_text()) if g4.exists() else {}

    if kind in ("auto", "hyperbolic", "poincare"):
        f = _find_embedding_file(("hyperbolic", "poincare", "h2", "ball"))
        if f is not None:
            coords = np.load(f)
            law = GeometryLaw(coords=coords, kind="hyperbolic", source=f"phase4:{f.name}")
            law.R, law.T = _fitted_or_fit(law, fits, "hyperbolic", g, fit, seed)
            return law
        if kind != "auto":
            raise FileNotFoundError(
                f"no hyperbolic embedding in {PHASE4/'artifacts'} - run scripts/phase4_geometry.py first")

    if kind in ("auto", "spectral"):
        f = _find_embedding_file(("spectral", "laplacian"))
        if f is not None:
            coords = np.load(f)
            law = GeometryLaw(coords=coords, kind="euclidean", source=f"phase4:{f.name}")
            law.R, law.T = _fitted_or_fit(law, fits, "spectral", g, fit, seed)
            return law
        if kind == "spectral":
            raise FileNotFoundError("no spectral embedding in results/phase4/artifacts")

    coords = anatomical_coords(g)
    law = GeometryLaw(coords=coords, kind="euclidean", source="annotation_xyz")
    law.R, law.T = _fitted_or_fit(law, fits, "anatomical", g, fit, seed)
    return law


def _fitted_or_fit(law: GeometryLaw, fits: dict, key: str, g: GraphView, do_fit: bool,
                   seed: int) -> tuple[float, float]:
    entry = None
    for k in (key, f"{key}_2d", f"{key}_law"):
        if isinstance(fits.get(k), dict) and "R" in fits[k]:
            entry = fits[k]
            break
    if entry:
        return float(entry["R"]), float(entry["T"])
    if not do_fit:
        return 1.0, 0.2
    f = fit_connection_law(law.coords, g.pre, g.post, kind=law.kind, seed=seed)
    law.source = f"{law.source}+law_fit"
    law.__dict__.setdefault("_fit", f)
    return float(f["R"]), float(f["T"])


def fit_summary(law: GeometryLaw) -> dict:
    return {"kind": law.kind, "dim": law.dim, "R": law.R, "T": law.T, "source": law.source,
            "fit": getattr(law, "_fit", None)}


def strip_arrays(obj):
    """Remove numpy/private keys so a measurement dict can be JSON-serialized."""
    if isinstance(obj, dict):
        return {k: strip_arrays(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [strip_arrays(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(strip_arrays(payload), indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")


CACHE = ROOT / "results" / "phase7" / "cache"


def get_signature(g: GraphView, tag: str, seed: int = 0, force: bool = False,
                  triad_max_edges: int = 4_000_000) -> dict:
    """closure.signature(g) with an on-disk cache.

    The reference-side signature is the expensive part of every closure comparison (the triad
    census alone is ~7 minutes on the biological graph), and it does not change between scales,
    so it is computed once and reused. The cache key includes the geometry tag because the
    latent-distance part depends on the coordinates.
    """
    from flyscale import closure
    CACHE.mkdir(parents=True, exist_ok=True)
    npz_path = CACHE / f"signature_{tag}.npz"
    json_path = CACHE / f"signature_{tag}.json"
    if npz_path.exists() and json_path.exists() and not force:
        z = np.load(npz_path, allow_pickle=False)
        meta = json.loads(json_path.read_text())
        return {
            "spectral": z["spectral"].tolist(),
            "leiden": z["leiden"],
            "triad": meta.get("triad"),
            "rich_club": {int(k): v for k, v in meta.get("rich_club", {}).items()},
            "celltype_matrix": (z["celltype_matrix"], meta.get("celltype_labels", [])),
            "latent_distances": z["latent_distances"],
            "two_node": meta.get("two_node"),
            "summary": meta.get("summary"),
            "_from_cache": True,
        }
    sig = closure.signature(g, triad_max_edges=triad_max_edges, seed=seed)
    cm = sig.get("celltype_matrix")
    np.savez_compressed(
        npz_path,
        spectral=np.asarray(sig.get("spectral", []), dtype=np.float64),
        leiden=np.asarray(sig.get("leiden") if sig.get("leiden") is not None else [], dtype=np.int64),
        celltype_matrix=(np.asarray(cm[0]) if cm else np.zeros((0, 0))),
        latent_distances=np.asarray(sig.get("latent_distances", []), dtype=np.float64),
    )
    json_path.write_text(json.dumps({
        "triad": sig.get("triad"),
        "rich_club": {str(k): v for k, v in (sig.get("rich_club") or {}).items()},
        "celltype_labels": (cm[1] if cm else []),
        "two_node": sig.get("two_node"),
        "summary": sig.get("summary"),
    }, indent=2, sort_keys=True) + "\n")
    print(f"cached signature -> {npz_path.name}")
    return sig
