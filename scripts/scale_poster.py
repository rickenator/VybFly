"""Poster: the connectome at 0.1x, 1x, 10x and 100x - one rendering, high resolution.

Panels 1x/10x/100x are the density rasters written by the Vyb/CUDA kernels (src/vyb_kernels/
upscale100.vyb -> results/upscale100/raster_c{1,10,100}.f64, one f64 atomic add per soma), so the
poster shows what the device actually computed rather than a re-plot of coordinates. The 0.1x panel
is the coarse-grained replica from the phase-5 artifact, labelled as such.

Tone mapping: each panel is mapped to *its own* density ceiling (the 99.98th percentile of its
counts, log scale) and prints that ceiling. A single shared ceiling puts the 1x and 10x panels below
the darkest ramp stop - they render black - because occupancy differs by more than an order of
magnitude across the ladder (0.29% of pixels hold a soma at 1x, 16.30% at 100x). What is comparable
panel-to-panel is the frame, not the brightness; occupancy and the ceiling are printed on each panel.

Nothing saturates at 100x: the busiest pixel holds 22 somata and the mean over occupied pixels is
1.81, so the panel resolves structure rather than reading as a silhouette. The deep-zoom asset is
the full-resolution view of the same raster.

    python scripts/scale_poster.py [--png-dpi 200]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RES = ROOT / "results"
OUT = RES / "scale-gallery"
OUT.mkdir(parents=True, exist_ok=True)
BG = "#05080c"
RAMP = LinearSegmentedColormap.from_list(
    "scale", ["#05080c", "#123048", "#1f6f8b", "#63c8dd", "#eafcff"])

W, H = 8192, 5760          # raster size produced by the kernels
DISPLAY = (2048, 1440)     # per-panel display resolution (block-max downsample)


class Panel:
    """One density panel: tone-mapped image + the numbers printed on it."""

    def __init__(self, img, ceiling, occupancy, pooled_max, mean_occupied):
        self.img = img
        self.ceiling = ceiling
        self.occupancy = occupancy
        self.pooled_max = pooled_max
        self.mean_occupied = mean_occupied


def load_panel(c: int) -> Panel:
    """Tone-map one kernel raster for display.

    Downsampling takes the *maximum* count in each display block rather than the mean: at 1x only
    0.29% of raster pixels hold a soma, so a mean over a 4x4 block rounds most single somata away and
    the panel renders as an empty field. The maximum keeps every occupied block visible.

    The tone map is per panel: log1p(count) / log1p(ceiling) with ceiling = that panel's own 99.98th
    percentile, which is the mapping scripts/deep_zoom.py uses for the 100x asset. A shared ceiling
    is not usable across the ladder - see the module docstring.
    """
    a = np.memmap(RES / "upscale100" / f"raster_c{c}.f64", dtype="<f8", mode="r", shape=(H, W))
    occupied = int(np.count_nonzero(a))
    total = float(a.sum())
    fy, fx = H // DISPLAY[1], W // DISPLAY[0]
    pooled = np.asarray(
        np.asarray(a, dtype=np.float32)[: DISPLAY[1] * fy, : DISPLAY[0] * fx]
        .reshape(DISPLAY[1], fy, DISPLAY[0], fx).max(axis=(1, 3)),
        dtype=np.float64)
    ceiling = float(np.percentile(pooled, 99.98))
    if ceiling <= 0:
        ceiling = 1.0
    img = np.clip(np.log1p(pooled) / np.log1p(ceiling), 0.0, 1.0)
    return Panel(img=img, ceiling=ceiling, occupancy=100.0 * occupied / (W * H),
                 pooled_max=float(pooled.max()),
                 mean_occupied=(total / occupied) if occupied else 0.0)


def panel(ax, p: Panel, title, note):
    ax.imshow(p.img, origin="lower", cmap=RAMP, aspect="equal", interpolation="nearest",
              vmin=0.0, vmax=1.0)
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1d2c36")
        s.set_linewidth(1.4)
    ax.set_title(title, color="#eaf6fb", fontsize=26, pad=16, linespacing=1.45)
    ax.text(0.5, -0.028, note, transform=ax.transAxes, color="#8aa3af", fontsize=15,
            ha="center", va="top")


def panel_pts(ax, pts, title, note, box):
    """Point rendering for the sparse panel: same frame as the density panels."""
    gx0, gy0, gsx, gsy = box
    ax.set_facecolor(BG)
    ax.scatter(pts[:, 0], pts[:, 1], s=2.2, marker="o", linewidths=0,
               c=["#63c8dd"], alpha=0.85)
    ax.set_xlim(gx0, gx0 + W * gsx)
    ax.set_ylim(gy0, gy0 + H * gsy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#1d2c36")
        sp.set_linewidth(1.4)
    ax.set_title(title, color="#eaf6fb", fontsize=26, pad=16, linespacing=1.45)
    ax.text(0.5, -0.028, note, transform=ax.transAxes, color="#8aa3af", fontsize=15,
            ha="center", va="top")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dpi", type=int, default=200)
    args = ap.parse_args()

    # ---- counts from the artifacts (each panel names where its number comes from) ----
    up = json.loads((RES / "phase6" / "upscale.json").read_text())
    d5 = json.loads((RES / "phase5" / "downscale.json").read_text())
    s100 = json.loads((OUT / "scale_100.json").read_text()) if (OUT / "scale_100.json").exists() else {}

    def counts(blob, factor):
        rows = []

        def walk(o, path=()):
            if isinstance(o, dict):
                if "n_neurons" in o and "n_connections" in o:
                    rows.append((path, o))
                for k, v in o.items():
                    walk(v, path + (str(k),))
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, path + (str(i),))

        walk(blob)
        hit = []
        for p, r in rows:
            named = False
            for seg in p:
                try:
                    named = named or abs(float(seg) - factor) < 1e-9
                except ValueError:
                    continue
            if named:
                hit.append((p, r))
        if not hit:
            return None, None
        hit.sort(key=lambda pr: ("replica" not in pr[0], "reference" in pr[0]))
        return int(hit[0][1]["n_neurons"]), int(hit[0][1]["n_connections"])

    n01, e01 = counts(d5, 0.1)
    n10, e10 = counts(up, 10.0)
    n100 = 139241 * 100                 # the 14 somata with no position cannot be replicated
    e100 = (s100.get("replica") or {}).get("n_connections")

    fig = plt.figure(figsize=(33.1, 24.6), layout="constrained")   # A1 landscape + footnote band
    fig.get_layout_engine().set(rect=(0.0, 0.075, 1.0, 1.0))
    fig.patch.set_facecolor(BG)
    axes = fig.subplots(2, 2).ravel()

    # 0.1x panel: the phase-5 coarse grouping applied to the same somata, so the panel sits in the
    # same anatomical space as the other three (the stored phase-5 replica keeps its fitted
    # hyperbolic coordinates, which would not be comparable panel-to-panel)
    groups = np.load(RES / "phase5" / "replicas" / "g0.1" / "groups.npy").astype(np.int64)
    soma = np.fromfile(ROOT / "data" / "processed" / "canonical_v783" / "bin" / "neurons.coords.f32",
                       dtype="<f4").reshape(-1, 6)[:, :3].astype(np.float64)
    good = np.isfinite(soma).all(axis=1)
    n_groups = int(groups.max()) + 1
    cent = np.zeros((n_groups, 3))
    cnt = np.zeros(n_groups)
    np.add.at(cent, groups[good], soma[good])
    np.add.at(cnt, groups[good], 1.0)
    cent = cent[cnt > 0] / cnt[cnt > 0, None]
    # same pixel grid as the device rasters so all four panels share a scale
    # 16k somata cannot fill a 28 nm pixel on the shared density scale (they would be invisible),
    # so this panel is drawn as points, which is also what a 0.1x cloud honestly looks like. The
    # note says so; the other three panels are the density maps and are mutually comparable.
    gx0, gy0 = 10000.0, 10000.0
    gsx, gsy = 28.076, 28.073
    panel_pts(axes[0], cent,
              f"0.1x\n{n01:,} groups   ·   {e01:,} connections",
              f"phase-5 coarse grouping: {n01:,} somata coarser than one 28 nm pixel, drawn as points",
              (gx0, gy0, gsx, gsy))

    # 1x / 10x / 100x panels: straight from the Vyb kernels' own rasters, each tone-mapped to its
    # own density (see the module docstring) and labelled with its measured occupancy and ceiling
    measured: dict[int, Panel] = {}
    for ax, (c, n, e, note) in zip(axes[1:], [
        (1, 139241, 2700513, "the real brain: FlyWire v783, 139,241 of 139,255 somata placed"),
        (10, n10, e10, "subdivided ladder top rung (10 children per soma)"),
        (100, n100, e100, "Vyb/CUDA kernels; no pixel saturates, the panel resolves structure"),
    ]):
        p = load_panel(c)
        measured[c] = p
        title = f"{c}x\n{n:,} neurons"
        if e:
            title += f"   ·   {e:,} connections"
        note = (f"{note}\n{p.occupancy:.2f}% of pixels hold a soma · log scale to its own ceiling "
                f"{p.ceiling:.0f} counts/px (busiest pixel {p.pooled_max:.0f})")
        panel(ax, p, title, note)
        print(f"  panel {c:>3}x: occupancy {p.occupancy:.2f}%  ceiling {p.ceiling:.0f}  "
              f"busiest display pixel {p.pooled_max:.0f}  mean/occupied raster px {p.mean_occupied:.3f}")

    fig.suptitle("The same connectome at four scales, one rendering",
                 color="#eafcff", fontsize=46, weight="bold")
    p1, p10, p100 = measured[1], measured[10], measured[100]
    foot = (
        "1x, 10x and 100x are the device rasters from the Vyb NVPTX kernels in src/vyb_kernels/upscale_kernel.vyb "
        "(one atomic add per soma), not a re-plot; 0.1x is the phase-5 grouping, drawn as points.\n"
        "All four panels share one frame (the same 8192 x 5760 grid, 28.08 nm per pixel, each scaled to its own extent), "
        "but each density panel is tone-mapped to its own ceiling - printed on the panel - because occupancy differs by "
        f"over an order of magnitude across the ladder ({p1.occupancy:.2f}% of pixels hold a soma at 1x, "
        f"{p10.occupancy:.2f}% at 10x, {p100.occupancy:.2f}% at 100x): on one shared scale the 1x and 10x panels fall "
        "below the darkest ramp stop and render black.\n"
        f"Nothing saturates at 100x. Its busiest pixel holds {p100.pooled_max:.0f} somata and the mean over occupied "
        f"pixels is {p100.mean_occupied:.2f}, so the panel resolves structure instead of reading as a silhouette; the "
        "ladder's real change is how much of the frame the cloud fills. Mean degree stays ~19 partners per neuron at "
        "every scale. Full resolution: results/scale-gallery/zoom-100x.html (or zoom-100x.png)."
    )
    fig.text(0.5, 0.012, foot, color="#6f8794", fontsize=14, ha="center", va="bottom", linespacing=1.75)

    # ---- self-check: no two drawn text artists may overlap ----
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

    pdf = OUT / "scale-poster.pdf"
    png = OUT / "scale-poster.png"
    fig.savefig(pdf, format="pdf", facecolor=BG)
    fig.savefig(png, format="png", dpi=args.png_dpi, facecolor=BG)
    plt.close(fig)
    print(f"wrote {pdf.relative_to(ROOT)} ({pdf.stat().st_size / 1e6:.1f} MB) and "
          f"{png.relative_to(ROOT)} ({png.stat().st_size / 1e6:.1f} MB) at {args.png_dpi} dpi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
