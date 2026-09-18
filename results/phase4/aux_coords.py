"""Save the two auxiliary Phase 4 coordinate arrays that the main run does not persist,
and check they reproduce the geometry documented in geometry.json.

    python /tmp/phase4_aux_coords.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from flyscale.connectome import Connectome                     # noqa: E402
from flyscale.geometry import anatomical_xy, anatomical_xyz     # noqa: E402

RES = ROOT / "results" / "phase4"
ART = RES / "artifacts"
doc = json.loads((RES / "geometry.json").read_text())
c = Connectome(doc["canonical"]["canonical_dir"])

X3 = anatomical_xyz(c)
Xc = X3 - np.nanmean(X3, axis=0)
Xf = np.where(np.isfinite(Xc), Xc, 0.0)
u, s, vt = np.linalg.svd(Xf, full_matrices=False)
pca2 = Xf @ vt[:2].T
Xz = np.stack([(X3[:, k] - np.nanmean(X3[:, k])) / np.nanstd(X3[:, k]) for k in range(3)], axis=1)

np.save(ART / "anatomical_pca2_coords.npy", pca2)
np.save(ART / "anatomical_xyz_zscored_coords.npy", Xz)

evr = (s ** 2 / (s ** 2).sum())[:3]
doc_evr = doc["geometry_attributes"]["anatomical"]["pca_explained_variance_ratio"]
peak = float(np.abs(np.array(evr) - np.array(doc_evr)).max())
print("pca explained variance ratio recomputed:", np.round(evr, 6).tolist())
print("pca explained variance ratio in json   :", np.round(doc_evr, 6).tolist())
print("max abs difference:", peak, "-> geometry.json reproducible:", peak < 1e-9)
print("axis std recomputed:", np.round(np.nanstd(X3, axis=0), 3).tolist())
print("axis std in json   :", np.round(doc["geometry_attributes"]["anatomical"]["axis_std_nm"], 3).tolist())
# anatomical_xy must equal the first two columns of anatomical_xyz
xy_path = ART / "anatomical_xy_coords.npy"
if not xy_path.exists():
    xy_path = ART / "anatomical_xy.npy"
same = np.array_equal(np.load(xy_path), anatomical_xy(c), equal_nan=True)
print(f"{xy_path.name} == anatomical_xy(c):", same)
