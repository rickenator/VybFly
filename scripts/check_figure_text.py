"""Find overlapping text inside every figure, objectively.

Renders each figure, then walks the artist tree and reports pairs of Text artists (titles, axis
labels, tick labels, legends, annotations) whose rendered bounding boxes intersect. Eyeballing
small figures misses these; this does not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import paper_make_figures as F

# the figure functions close their figure when they save it; keep them alive so we can inspect
plt.close = lambda *a, **k: None


def drawn_texts(fig) -> list[tuple[str, str, object]]:
    """Only text a reader can actually see.

    Walking fig.findobj(Text) also picks up tick labels that matplotlib has already replaced; those
    are no longer drawn but still report a bounding box, which invents overlaps that are not on the
    page. Collect from the live sources instead.
    """
    out: list[tuple[str, str, object]] = []
    for ax in fig.get_axes():
        if not ax.get_visible() or not ax.axison:   # axis-off panels draw no ticks
            continue
        cands: list[tuple[str, object]] = [("title", ax.title), ("xlabel", ax.xaxis.label),
                                          ("ylabel", ax.yaxis.label)]
        cands += [("xtick", t) for t in ax.get_xticklabels()]
        cands += [("ytick", t) for t in ax.get_yticklabels()]
        cands += [("text", t) for t in ax.texts]
        leg = ax.get_legend()
        if leg is not None:
            cands += [("legend", t) for t in leg.get_texts()]
        for kind, t in cands:
            s = t.get_text().strip()
            if s and t.get_visible():
                out.append((kind, s.replace("\n", " | ")[:52], t))
    for t in fig.texts:
        s = t.get_text().strip()
        if s and t.get_visible():
            out.append(("figtext", s.replace("\n", " | ")[:52], t))
    return out


def overlaps(fig) -> list[tuple[str, str, str, str, float]]:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for kind, s, t in drawn_texts(fig):
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            items.append((kind, s, bb))
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][2], items[j][2]
            dx = min(a.x1, b.x1) - max(a.x0, b.x0)
            dy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if dx > 1.5 and dy > 1.5:                # more than a hairline of overlap
                out.append((items[i][0], items[i][1], items[j][0], items[j][1], dx * dy))
    return sorted(out, key=lambda t: -t[4])


def main() -> None:
    figs = {
        "fig_cover": F.cover,
        "fig_geometry": F.geometry,
        "fig_scaling": F.scaling,
        "fig_downscale": F.downscale,
        "fig_closure": F.closure,
        "fig_dynamics": F.dynamics,
        "fig_capability": F.capability,
        "fig_energy": F.energy,
        "fig_gpu": F.gpu_equivalence,
    }
    total = 0
    for name, fn in figs.items():
        before = set(plt.get_fignums())
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            print(f"{name}: ERROR {exc}")
            continue
        new = [n for n in plt.get_fignums() if n not in before]
        if not new:
            print(f"{name}: no figure produced")
            continue
        fig = plt.figure(new[-1])
        hits = overlaps(fig)
        total += len(hits)
        print(f"{name:16s} {'OK' if not hits else f'{len(hits)} overlapping pairs'}")
        for ka, sa, kb, sb, area in hits[:6]:
            print(f"      {area:7.0f} px^2  {ka}:{sa!r}  x  {kb}:{sb!r}")
        plt.close(fig)
    print(f"\nTOTAL overlapping text pairs across the content figures: {total}")


if __name__ == "__main__":
    main()
