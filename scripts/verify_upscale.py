"""Verification of the Vyb/CUDA 100x placement - checks the artifact, does not re-derive it.

Reads what the kernels wrote (results/upscale100/children.bin: 13,925,500 f64 xyz records) and the
canonical somata they came from, then reports:

  * the child count against 139,255 x 100 (an exact count, so a dropped or duplicated parent shows up)
  * the displacement statistics of each child from its parent, in nm: the source of the spread is the
    per-neuron sigma the kernel derived from the local occupancy, so these are the numbers to sanity
    check against the spacing of the biological somata
  * how far each family strays from its parent's cell (the kernel clamps sigma to the cell size, so a
    family must not leave a 10 um neighbourhood)
  * the fraction of children that land inside the brain's bounding box

    python scripts/verify_upscale.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_PARENTS = 139_255
C = 100
CELL_NM = 10_000.0


def main() -> int:
    kids = np.memmap(RES / "upscale100" / "children.bin", dtype="<f8", mode="r",
                     shape=(N_PARENTS * C, 3))
    coords = np.fromfile(ROOT / "data" / "processed" / "canonical_v783" / "bin" / "neurons.coords.f32",
                         dtype="<f4").reshape(-1, 6)[:, :3].astype(np.float64)
    parent = coords[np.repeat(np.arange(N_PARENTS), C)]
    d = np.asarray(kids) - parent
    r = np.sqrt((d * d).sum(axis=1))

    finite_parent = np.isfinite(coords).all(axis=1)
    print(f"children: {len(kids):,}  expected {N_PARENTS * C:,}  "
          f"{'MATCH' if len(kids) == N_PARENTS * C else 'MISMATCH'}")
    print(f"parents with a usable position: {int(finite_parent.sum()):,} of {N_PARENTS:,}")
    ok = np.isfinite(r)
    print(f"finite displacements: {int(ok.sum()):,}")
    print(f"|child - parent| per axis (nm): x mean {d[ok, 0].mean():+.1f} sd {d[ok, 0].std():.1f} | "
          f"y {d[ok, 1].mean():+.1f} sd {d[ok, 1].std():.1f} | z {d[ok, 2].mean():+.1f} sd {d[ok, 2].std():.1f}")
    q = np.percentile(r[ok], [50, 90, 99, 99.9])
    print(f"|child - parent| (nm): median {q[0]:.0f}  p90 {q[1]:.0f}  p99 {q[2]:.0f}  p99.9 {q[3]:.0f}  "
          f"max {r[ok].max():.0f}")
    print(f"families fully inside one {CELL_NM / 1000:.0f} um cell: "
          f"{float((r[ok] < CELL_NM).mean()) * 100:.2f}% of children")

    bbox = np.array([[np.nanmin(coords[:, i]), np.nanmax(coords[:, i])] for i in range(3)])
    pad = CELL_NM
    inside = ((kids[:, 0] >= bbox[0, 0] - pad) & (kids[:, 0] <= bbox[0, 1] + pad) &
              (kids[:, 1] >= bbox[1, 0] - pad) & (kids[:, 1] <= bbox[1, 1] + pad) &
              (kids[:, 2] >= bbox[2, 0] - pad) & (kids[:, 2] <= bbox[2, 1] + pad))
    print(f"children inside the parent bounding box +- {pad / 1000:.0f} um: "
          f"{float(inside.mean()) * 100:.4f}%")

    out = {
        "children": int(len(kids)),
        "expected_children": N_PARENTS * C,
        "parents_with_position": int(finite_parent.sum()),
        "displacement_nm": {
            "x_mean": float(d[ok, 0].mean()), "x_sd": float(d[ok, 0].std()),
            "y_mean": float(d[ok, 1].mean()), "y_sd": float(d[ok, 1].std()),
            "z_mean": float(d[ok, 2].mean()), "z_sd": float(d[ok, 2].std()),
            "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), "max": float(r[ok].max()),
        },
        "fraction_within_one_cell": float((r[ok] < CELL_NM).mean()),
        "fraction_inside_bbox_plus_cell": float(inside.mean()),
        "source": "results/upscale100/children.bin written by src/vyb_kernels/upscale100.vyb",
    }
    (RES / "upscale100" / "verify.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote results/upscale100/verify.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
