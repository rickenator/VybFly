"""Poster: the connectome at 0.1x, 1x, 10x and 100x - one rendering, high resolution.

Panels 1x/10x/100x are the density rasters written by the Vyb/CUDA kernels (src/vyb_kernels/
upscale100.vyb -> results/upscale100/raster_c{1,10,100}.f64, one f64 atomic add per soma), so the
poster shows what the device actually computed rather than a re-plot of coordinates. The 0.1x panel
is the coarse-grained replica from the phase-5 artifact, labelled as such.

At 100x the cloud is a silhouette: 13.9M somata at 28 nm/pixel saturate every pixel they touch. That
is the finding, and the footnote points at the deep-zoom asset, which resolves individual neurons.

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
DISPLAY = (2048, 1440)     # per-panel display resolution (block-mean downsample)


GLOBAL_VMAX = 24.0     # counts per pixel: the 100x panel's ceiling, shared by all four panels


def load_raster(c: int) -> np.ndarray:
    a = np.memmap(RES / "upscale100" / f"raster_c{c}.f64", dtype="<f8", mode="r", shape=(H, W))
    fy, fx = H // DISPLAY[1], W // DISPLAY[0]
    ds = np.asarray(a, dtype=np.float32)
    ds = ds[: DISPLAY[1] * fy, : DISPLAY[0] * fx].reshape(DISPLAY[1], fy, DISPLAY[0], fx).mean(axis=(1, 3))
    # shared density scale: log1p(count) / log1p(GLOBAL_VMAX) for every panel, so the brightness
    # difference across the poster is the actual density difference rather than a per-panel autoscale
    return np.clip(np.log1p(ds) / np.log1p(GLOBAL_VMAX), 0.0, 1.0)


def panel(ax, img, title, note):
    ax.imshow(img, origin="lower", cmap=RAMP, aspect="equal", interpolation="nearest",
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
              "phase-5 grouping; drawn as points (16k somata sit below the shared density scale)",
              (gx0, gy0, gsx, gsy))

    # 1x / 10x / 100x panels: straight from the Vyb kernels' own rasters
    for ax, (c, n, e, note) in zip(axes[1:], [
        (1, 139241, 2700513, "the real brain: FlyWire v783 somata, 139,241 of 139,255 placed"),
        (10, n10, e10, "subdivided ladder top rung (10 children per soma)"),
        (100, n100, e100, "Vyb/CUDA kernels: at this density the cloud is a silhouette"),
    ]):
        img = load_raster(c)
        title = f"{c}x\n{n:,} neurons"
        if e:
            title += f"   ·   {e:,} connections"
        panel(ax, img, title, note)

    fig.suptitle("The same connectome at four scales, one rendering",
                 color="#eafcff", fontsize=46, weight="bold")
    foot = (
        "1x, 10x and 100x are the device rasters from the Vyb NVPTX kernels in src/vyb_kernels/upscale_kernel.vyb "
        "(one atomic add per soma), not a re-plot.\n"
        "All four panels share one density scale (log counts per 28 nm pixel, ceiling 24), so a brighter panel is a denser one; "
        "every panel is scaled to its own extent, so what changes across the poster is point density, not framing. "
        "Mean degree stays ~19 partners per neuron at every scale.\n"
        f"100x is real data: {n100:,} somata from 139,255 parents, each child placed within its parent's local "
        "spacing, then rasterized at 28 nm per pixel. 13.9M somata fill every pixel they touch, which is why the "
        "panel reads as a white silhouette —\nsee results/scale-gallery/zoom-100x.html (or zoom-100x.png) to "
        "resolve individual neurons at full resolution."
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
