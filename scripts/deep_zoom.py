"""Build the zoomable 100x asset from the Vyb-generated density raster.

Input: results/upscale100/raster_c100.f64 - an 8192 x 5760 array of per-pixel soma counts, written
by src/vyb_kernels/upscale100.vyb on the GPU (one f64 atomic add per soma).

Why a Deep Zoom Image: at 8192 x 5760 with ~13.9M somata the interesting structure is both global
(the brain silhouette) and local (individual neurons, ~28 nm per pixel). A single PNG forces a
choice between the two; a DZI pyramid serves a tiled 256 px view at every level, so the viewer can
zoom from the whole brain to single cells without re-fetching more than a screenful of tiles.

Outputs (results/scale-gallery/):
    zoom-100x.png          full-resolution 8-bit render, viewable anywhere
    zoom-100x.dzi + _files/  the tile pyramid (Deep Zoom)
    zoom-100x.html         OpenSeadragon viewer for the pyramid

    python scripts/deep_zoom.py [--raster results/upscale100/raster_c100.f64] [--tiles]

The counts are also checked here: the raster must sum to exactly n_parents * c somata, which is an
independent confirmation that the placement loop covered every parent exactly once.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scale-gallery"
OUT.mkdir(parents=True, exist_ok=True)

RAMP = [(5, 8, 12), (18, 48, 72), (31, 111, 139), (99, 200, 221), (234, 252, 255)]


def ramp_lut(n: int = 256) -> np.ndarray:
    """Piecewise-linear dark->cyan->white ramp, the same one the poster uses."""
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    lut = np.zeros((n, 3), dtype=np.float64)
    for ch in range(3):
        lut[:, ch] = np.interp(np.linspace(0.0, 1.0, n), stops, [c[ch] for c in RAMP])
    return lut


def tiles_for(level_w: int, level_h: int, tile: int = 256) -> int:
    return math.ceil(level_w / tile) * math.ceil(level_h / tile)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=str(ROOT / "results" / "upscale100" / "raster_c100.f64"))
    ap.add_argument("--width", type=int, default=8192)
    ap.add_argument("--height", type=int, default=5760)
    ap.add_argument("--tiles", action="store_true", default=True)
    ap.add_argument("--no-tiles", dest="tiles", action="store_false")
    ap.add_argument("--n-parents", type=int, default=139241,
                    help="somata with a usable position: 139,241 of 139,255; the other 14 are "
                         "rejected by the raster's in-box test, so the count check uses this")
    ap.add_argument("--c", type=int, default=100)
    args = ap.parse_args()

    path = Path(args.raster)
    arr = np.memmap(path, dtype="<f8", mode="r", shape=(args.height, args.width))
    total = float(arr.sum())
    expected = args.n_parents * args.c
    nonzero = int(np.count_nonzero(arr))
    print(f"raster {args.width}x{args.height} from {path.name}")
    print(f"  somata in raster: {total:,.0f}  expected {expected:,} "
          f"({args.n_parents:,} placeable x {args.c})  "
          f"{'MATCH' if int(round(total)) == expected else 'MISMATCH'}")
    print(f"  occupied pixels: {nonzero:,} of {arr.size:,} ({100.0 * nonzero / arr.size:.2f}%)")
    print(f"  max count per pixel: {float(arr.max()):,.0f}")

    a = np.asarray(arr, dtype=np.float64).copy()
    vmax = float(np.percentile(a, 99.98))
    if vmax <= 0:
        vmax = 1.0
    norm = np.log1p(a) / math.log1p(vmax)
    np.clip(norm, 0.0, 1.0, out=norm)
    lut = ramp_lut(256)
    idx = (norm * 255.0).astype(np.uint8)
    rgb = lut[idx].astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)      # raster row 0 is the lowest y

    png = OUT / "zoom-100x.png"
    img.save(png, optimize=True)
    print(f"  wrote {png.relative_to(ROOT)} ({png.stat().st_size / 1e6:.1f} MB)")

    if args.tiles:
        tile_dir = OUT / "zoom-100x_files"
        tile_dir.mkdir(exist_ok=True)
        levels = []
        cur = img
        lvl = 0
        while True:
            w, h = cur.size
            n_tiles = tiles_for(w, h)
            for ty in range(math.ceil(h / 256)):
                for tx in range(math.ceil(w / 256)):
                    box = (tx * 256, ty * 256, min((tx + 1) * 256, w), min((ty + 1) * 256, h))
                    cur.crop(box).save(tile_dir / f"{lvl}_{tx}_{ty}.png", optimize=True)
            levels.append((lvl, w, h, n_tiles))
            if w <= 256 and h <= 256:
                break
            cur = cur.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
            lvl += 1
        size = sum(f.stat().st_size for f in tile_dir.glob("*.png"))
        print(f"  tiles: {sum(l[3] for l in levels):,} across {len(levels)} levels "
              f"({size / 1e6:.1f} MB)")
        (OUT / "zoom-100x.dzi").write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Image TileSize="256" Overlap="0" Format="png" '
            f'xmlns="http://schemas.microsoft.com/deepzoom/2008">\n'
            f'  <Size Width="{args.width}" Height="{args.height}"/>\n'
            "</Image>\n")
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FlyScale 100x connectome - deep zoom</title>
<style>html,body{{margin:0;height:100%;background:#05080c;color:#8fa9b5;
font:13px/1.5 ui-monospace,Menlo,monospace}}
#osd{{width:100%;height:100%}}
#hud{{position:fixed;left:12px;bottom:10px;background:#05080cd9;padding:8px 12px;border:1px solid #1d2c36}}
#hud b{{color:#eafcff}}</style></head>
<body>
<div id="osd"></div>
<div id="hud"><b>FlyScale 100x connectome</b> - {total:,.0f} somata from
{args.n_parents:,} parents, {args.width}x{args.height} px at ~28 nm/pixel.<br>
Scroll to zoom, drag to pan. The full-resolution render is <code>zoom-100x.png</code>.</div>
<script src="https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/openseadragon.min.js"></script>
<script>
OpenSeadragon({{ id: "osd", prefixUrl:
  "https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/images/",
  tileSources: "zoom-100x.dzi", showNavigator: true, navigatorPosition: "TOP_RIGHT",
  maxZoomPixelRatio: 8, visibilityRatio: 0.5, background: "#05080c" }});
</script>
</body></html>
"""
        (OUT / "zoom-100x.html").write_text(html)
        print(f"  wrote {OUT.relative_to(ROOT)}/zoom-100x.dzi, zoom-100x.html")
        (OUT / "zoom-100x.json").write_text(json.dumps({
            "raster": str(path.relative_to(ROOT)),
            "width": args.width, "height": args.height,
            "somata_in_raster": int(round(total)), "somata_expected": expected,
            "occupied_pixels": nonzero, "max_count_per_pixel": float(arr.max()),
            "nm_per_pixel_x": 28076 / 1000.0, "nm_per_pixel_y": 28073 / 1000.0,
            "levels": [{"level": l, "width": w, "height": h, "tiles": t} for l, w, h, t in levels],
            "note": "built by scripts/deep_zoom.py from the Vyb/CUDA raster; "
                    "counts come from the same placement pass that produced children.bin",
        }, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
