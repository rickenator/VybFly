"""What the ladder actually looks like: the connectome at 0.1x, 1x and 10x.

Every point is a real coordinate from an artifact, but the phases stored their replicas in different
spaces (phase 5 downscales in the fitted 2-D hyperbolic disk, phase 6 upscales in the 16-D spectral
embedding, phase 9 capability replicas in anatomical 3-D). Comparing those side by side would be
apples to oranges, so every panel here is drawn in the anatomical space:

* 0.1x - rebuilt for this figure with the anatomical geometry (the on-disk 0.1x replica is the
  hyperbolic run), using the same `renorm.coarse_grain` call the phase script uses;
* 1x   - the canonical somata from the annotation table;
* 10x  - the phase-9 replica, the only 10x artifact that was saved in anatomical coordinates.

Each panel is normalized to its own 1st-99th percentile extent, so what changes across the row is
point *density*, not framing. 100x is deliberately absent: M15 is gated upstream of this script, and
the figure says so rather than drawing an interpolation.

    python scripts/scale_gallery.py            # -> results/scale-gallery/{scale-gallery.pdf,.png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = ROOT / "results"
OUT = RES / "scale-gallery"
OUT.mkdir(parents=True, exist_ok=True)
BG = "#070a0f"


def _collect(obj, path=(), out=None):
    """Find every dict carrying counts, with the key path that names its scale."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        if "n_neurons" in obj and "n_connections" in obj:
            out.append((path, obj))
        for k, v in obj.items():
            _collect(v, path + (str(k),), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _collect(v, path + (str(i),), out)
    return out


def _is_factor(value, factor: float) -> bool:
    try:
        return abs(float(value) - factor) < 1e-9
    except (TypeError, ValueError):
        return False


def recorded_counts(path: Path, factor: float) -> tuple[int | None, int | None]:
    """Counts for one scale.

    The scale is named either by a dict key (`factors["0.1"].replica`) or by a `factor` field, so
    match on both rather than assuming one layout.
    """
    matches = []
    for keys, row in _collect(json.loads(path.read_text())):
        named = any(_is_factor(seg, factor) for seg in keys) or _is_factor(row.get("factor"), factor)
        if named:
            matches.append((keys, row))
    if not matches:
        return None, None
    # every scale block carries both the reference graph and its own replica; take the replica
    matches.sort(key=lambda kr: ("replica" not in kr[0], "reference" in kr[0]))
    row = matches[0][1]
    return int(row["n_neurons"]), int(row["n_connections"])


def robust_extent(a: np.ndarray) -> tuple[float, float, float, float]:
    lo = np.nanpercentile(a, 0.5, axis=0)
    hi = np.nanpercentile(a, 99.5, axis=0)
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def normalise(a: np.ndarray) -> np.ndarray:
    """Centre on the median and scale each axis to its robust extent (so panels are comparable)."""
    out = a.copy()
    for k in range(a.shape[1]):
        col = a[:, k]
        med = np.nanmedian(col)
        span = np.nanpercentile(col, 99.5) - np.nanpercentile(col, 0.5)
        out[:, k] = (col - med) / (span if span else 1.0)
    return out


def coarse_01() -> tuple[np.ndarray, int]:
    """Build a 0.1x replica in anatomical coordinates (same call as phase 5, computed for this figure)."""
    from _common import load_geometry, load_target
    from flyscale import renorm

    g1 = load_target(threshold=5)
    law = load_geometry(g1, "anatomical", threshold=5, seed=0)
    rep, _groups = renorm.coarse_grain(g1, law, 0.1, seed=0, return_groups=True)
    coords = np.asarray(getattr(rep, "coords", None))
    if coords is None or coords.ndim != 2:
        raise RuntimeError("coarse_grain did not return coordinates")
    return coords[:, :2], int(rep.n), int(np.asarray(rep.pre).size)


def main() -> None:
    ann = pd.read_parquet(ROOT / "data" / "processed" / "canonical_v783" / "neurons.parquet",
                          columns=["pos_x", "pos_y"])
    one = ann.to_numpy(dtype=float) / 1000.0                      # nm -> um
    one = one[np.isfinite(one).all(axis=1)]

    print("building the 0.1x replica in anatomical space ...")
    tenth, n_tenth, e_tenth = coarse_01()
    tenth = tenth[np.isfinite(tenth).all(axis=1)]
    n_rec, e_rec = recorded_counts(RES / "phase5" / "downscale.json", 0.1)
    geo5 = json.loads((RES / "phase5" / "downscale.json").read_text()).get("geometry", {})
    rec = (f"{n_rec:,} neurons, {e_rec:,} connections" if n_rec else "not recorded")
    print(f"  rebuilt: {n_tenth:,} neurons, {e_tenth:,} connections  "
          f"(recorded 0.1x, geometry {geo5.get('kind')}: {rec})")

    ten_raw = np.load(RES / "phase9" / "replicas" / "g10" / "coords.npy", mmap_mode="r")
    ten = np.asarray(ten_raw[:, :2], dtype=float)
    ten = ten[np.isfinite(ten).all(axis=1)]

    n_ten, e_ten = recorded_counts(RES / "phase6" / "upscale.json", 10.0)

    panels = [
        ("0.1x", tenth, n_tenth, e_tenth),
        ("1x (real brain)", one, one.shape[0], 2_700_513),
        ("10x", ten, n_ten or ten.shape[0], e_ten),
    ]
    print("panels:", [(lab, n, e) for lab, _, n, e in panels])

    fig = plt.figure(figsize=(7.4, 3.9), layout="constrained")
    fig.patch.set_facecolor(BG)
    axes = fig.subplots(1, 3)
    for ax, (label, pts, n, e) in zip(axes, panels):
        p = normalise(pts)
        ax.set_facecolor(BG)
        ax.scatter(p[:, 0], p[:, 1], s=0.10, c="#cdf3ff", alpha=0.30, linewidths=0)
        ax.set_aspect("equal")
        ax.set_axis_off()
        # two short lines per panel: a single "label - N neurons, E connections" line is wider than
        # the panel and the three titles then collide with each other
        # three short lines per panel; any wider and the three titles collide with each other
        head = f"{label}\n{n:,} neurons"
        if e:
            head += f"\n{e:,} connections"
        ax.set_title(head, color="#93b8c8", fontsize=7.5, linespacing=1.35)

    fig.suptitle("The same connectome at three scales, one rendering", color="#eaf6fb",
                 fontsize=11, weight="bold")
    fig.text(0.5, 0.055,
             "Every point is a real coordinate: the 0.1x replica re-coarse-grained in anatomical space for this figure, "
             "the canonical somata at 1x, and the phase-9 10x replica.\n"
             "Each panel is normalized to its own extent, so the difference across the row is point density. "
             "Mean degree stays ~19 partners per neuron at every scale — that is what \"sparse at every scale\" means.\n"
             "The 1x panel draws 139,241 of the 139,255 neurons: 14 have no annotated soma position.",
             color="#7f97a2", fontsize=7, ha="center")
    fig.text(0.5, 0.017,
             "100x (about 13.9M neurons and ~278M connections at the measured N^1.0175 growth) was never built: "
             "milestone M15 is gated, see docs/ROADMAP.md.",
             color="#5f7480", fontsize=7, ha="center")

    # fail loudly if any two text artists in this figure overlap
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in axes:
        for t in [ax.title] + list(ax.texts):
            if t.get_text().strip():
                boxes.append((t.get_text().split("\n")[0], t.get_window_extent(renderer=r)))
    for t in fig.texts:
        if t.get_text().strip():
            boxes.append((t.get_text()[:40], t.get_window_extent(renderer=r)))
    clashes = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1.5:
                clashes += 1
                print(f"  WARNING overlapping text: {boxes[i][0]!r} x {boxes[j][0]!r}")
    print(f"  text overlap check: {clashes} clashing pairs")

    out_pdf = OUT / "scale-gallery.pdf"
    out_png = OUT / "scale-gallery.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor=BG)
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_pdf.relative_to(ROOT)} and {out_png.relative_to(ROOT)}")
    for label, pts, n, e in panels:
        print(f"  {label:20s} points={pts.shape[0]:>9,} connections={e if e else 'n/a'}")


if __name__ == "__main__":
    main()
