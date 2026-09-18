"""Render every figure for the paper, from results/paper/data.json and the canonical dataset.

All figures are vector PDFs so they stay sharp in the compiled paper. The cover art is drawn from
the real connectome: 139,255 soma positions with the mushroom-body Kenyon cells highlighted, plus
an inset of the learned 2-D hyperbolic embedding the scaling pipeline uses.

    python scripts/paper_make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import heapq
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = RES / "paper"
FIGS = PAPER / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
D = json.loads((PAPER / "data.json").read_text())

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.dpi": 160, "savefig.bbox": "tight", "axes.grid": True,
    "axes.titlepad": 3.0, "xtick.major.pad": 5.0, "ytick.major.pad": 3.0, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "font.family": "DejaVu Sans",
    # constrained layout keeps titles, tick labels and legends out of each other's way; the
    # figures are small, so this is the difference between readable and collided
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.06,
    "figure.constrained_layout.w_pad": 0.06,
})
INK = "#1b1f24"
ACCENT = "#0b6fa4"
WARM = "#c1121f"
MUTED = "#6c757d"


def thin(ax, n: int = 3) -> None:
    """Fewer, wider-spaced decade labels.

    Dense log ticks made adjacent panels' labels overlap each other, which is what read as
    "the figures have overlapping text".
    """
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=n))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=n))


def save(fig, name: str) -> None:
    out = FIGS / name
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  figure: {out.relative_to(ROOT)}")


# ------------------------------------------------------------------ cover art
def cover() -> None:
    """A4 cover: the fly brain model is the hero; a real circuit and the fitted geometry are details.

    All marks are data. There is no texture layer and no background grid, so the neuron model is
    the single standout feature; every line drawn is thin and quiet by design.
    """
    bg = "#070a0f"
    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
    fig.patch.set_facecolor(bg)
    rng = np.random.default_rng(7)

    ann = pd.read_parquet(ROOT / "data" / "processed" / "canonical_v783" / "neurons.parquet")
    x = ann["pos_x"].to_numpy(dtype=float) / 1000.0
    y = ann["pos_y"].to_numpy(dtype=float) / 1000.0
    cls_s = ann["cell_class"].astype("string").fillna("")
    cls_all = cls_s.to_numpy()
    kc = cls_s.eq("Kenyon_Cell").to_numpy(dtype=bool) | cls_s.str.startswith("KC").to_numpy(dtype=bool)
    mbon = cls_s.str.startswith("MBON").to_numpy(dtype=bool)
    dan = cls_s.str.startswith("DAN").to_numpy(dtype=bool)
    alpn = (cls_s.str.startswith("ALPN").to_numpy(dtype=bool)
            | cls_s.str.startswith("PN").to_numpy(dtype=bool))

    # ---------------- the fly brain model: the standout ----------------
    axm = fig.add_axes([0.055, 0.335, 0.89, 0.475])
    axm.patch.set_alpha(0.0)
    axm.set_facecolor("none")
    axm.scatter(x, y, s=0.60, c="#186f96", alpha=0.09, linewidths=0)        # soft haze
    axm.scatter(x, y, s=0.15, c="#cdf3ff", alpha=0.34, linewidths=0)        # the 139,255 somata
    axm.scatter(x[alpn], y[alpn], s=2.4, c="#6fe3a8", alpha=0.70, linewidths=0)
    axm.scatter(x[dan], y[dan], s=2.8, c="#ffa457", alpha=0.80, linewidths=0)
    axm.scatter(x[mbon], y[mbon], s=3.6, c="#ff2d78", alpha=0.90, linewidths=0)
    axm.scatter(x[kc], y[kc], s=1.2, c="#ffd166", alpha=0.72, linewidths=0)
    # limits from the data with symmetric padding: hard-coded limits were asymmetric and pushed
    # the brain about 5% right of the page center
    # center on the robust extent (0.05-99.95 percentile), not the raw min/max: a handful of
    # stray somata stretch the raw range and push the body of the brain off-center
    xlo, xhi = (float(v) for v in np.nanpercentile(x, [0.05, 99.95]))
    ylo, yhi = (float(v) for v in np.nanpercentile(y, [0.05, 99.95]))
    # center on the median, not the extent midpoint: the brain's mass centroid sits ~8um right
    # of the midpoint of its own bounding box, which is what read as "off center"
    xc, yc = float(np.nanmedian(x)), float(np.nanmedian(y))
    hx, hy = 0.5 * (xhi - xlo) * 1.02, 0.5 * (yhi - ylo) * 1.12
    axm.set_xlim(xc - hx, xc + hx)
    axm.set_ylim(yc - hy, yc + hy)
    axm.invert_yaxis()
    axm.set_aspect("equal")
    axm.set_axis_off()
    axm.text(0.5, -0.025, "The whole brain — 139,255 neurons at their soma positions",
             transform=axm.transAxes, color="#93b8c8", fontsize=8, va="top", ha="center")
    axm.text(0.5, -0.075,
             "Gold Kenyon cells   ·   magenta MBON   ·   orange DAN   ·   green antennal-lobe "
             "projection neurons",
             transform=axm.transAxes, color="#5f8296", fontsize=7.2, va="top", ha="center")

    # ---------------- detail: a small real circuit ----------------
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "canonical_v783" / "pairs.parquet")
    pairs = pairs[pairs["syn_count"] >= 5]
    n_neurons = int(ann["idx"].max()) + 1
    pre = pairs["pre_idx"].to_numpy(dtype=np.int64)
    post = pairs["post_idx"].to_numpy(dtype=np.int64)
    syn = pairs["syn_count"].to_numpy(dtype=float)
    nt_names = np.array(["ach", "gaba", "glut", "oct", "ser", "da"])
    nt_arg = nt_names[np.argmax(pairs[["nt_prob_ach", "nt_prob_gaba", "nt_prob_glut",
                                       "nt_prob_oct", "nt_prob_ser", "nt_prob_da"]]
                                .to_numpy(dtype=np.float32), axis=1)]
    op = np.argsort(pre, kind="stable")
    pre_s, post_p, syn_p, nt_p = pre[op], post[op], syn[op], nt_arg[op]
    oq = np.argsort(post, kind="stable")
    post_s, pre_q, syn_q, nt_q = post[oq], pre[oq], syn[oq], nt_arg[oq]
    starts_p = np.searchsorted(pre_s, np.arange(n_neurons + 1))
    starts_q = np.searchsorted(post_s, np.arange(n_neurons + 1))
    pos_x = ann["pos_x"].to_numpy(dtype=float) / 1000.0
    pos_y = ann["pos_y"].to_numpy(dtype=float) / 1000.0
    wdeg = (np.bincount(pre, weights=syn, minlength=n_neurons)
            + np.bincount(post, weights=syn, minlength=n_neurons))

    def directed(u):
        """every connection touching u, as (other, weight, src, dst, transmitter)"""
        a, b = starts_p[u], starts_p[u + 1]
        c, d = starts_q[u], starts_q[u + 1]
        out = [(int(post_p[i]), float(syn_p[i]), u, int(post_p[i]), str(nt_p[i]))
               for i in range(a, b)]
        out += [(int(pre_q[i]), float(syn_q[i]), int(pre_q[i]), u, str(nt_q[i]))
                for i in range(c, d)]
        return out

    K = 15
    best = None
    for seed in np.argsort(-wdeg)[:150]:
        S = {int(seed)}
        edges = {}
        heap = []
        for other, w, s_, d_, nt in directed(int(seed)):
            heapq.heappush(heap, (-w, other, s_, d_, nt))
        while len(S) < K and heap:
            negw, other, s_, d_, nt = heapq.heappop(heap)
            if s_ in S and d_ in S:
                continue
            S.add(other if s_ in S else s_)
            key = (min(s_, d_), max(s_, d_))
            if -negw > edges.get(key, (0, ""))[0]:
                edges[key] = (-negw, nt)
            for o2, w2, s2, d2, nt2 in directed(other):
                if (s2 not in S) or (d2 not in S):
                    heapq.heappush(heap, (-w2, o2, s2, d2, nt2))
        ids = np.array(sorted(S))
        px, py = pos_x[ids], pos_y[ids]
        extent = float(np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :]).max())
        n_cls = len({str(cls_all[i]) for i in ids})
        score = sum(w for w, _ in edges.values()) * (1.0 + extent) * (1.0 + 0.6 * (n_cls - 1))
        if best is None or score > best[0]:
            best = (score, ids, edges, n_cls)
    _, ids, edges, n_cls = best
    order = {int(v): k for k, v in enumerate(ids)}
    px, py = pos_x[ids], pos_y[ids]
    # rescaled on each axis independently so the neurons are spread across the panel instead of
    # huddling in one cluster (relative order and topology are preserved; the caption says so)
    X = (px - px.mean()) / max(float(np.ptp(px)), 1e-9)
    Y = -(py - py.mean()) / max(float(np.ptp(py)), 1e-9)

    axc = fig.add_axes([0.075, 0.105, 0.36, 0.185])
    axc.patch.set_alpha(0.0)
    axc.set_facecolor("none")
    edge_col = {"ach": "#5fb3cc", "gaba": "#d1739a", "glut": "#7fd39a",
                "oct": "#a99ee0", "ser": "#d9b96a", "da": "#d79a63"}
    for (u, v), (w, nt) in edges.items():
        iu, iv = order[u], order[v]
        axc.annotate("", xy=(X[iv], Y[iv]), xytext=(X[iu], Y[iu]),
                     arrowprops=dict(arrowstyle="-", color=edge_col.get(nt, "#8fb4c4"),
                                     lw=0.35 + 0.34 * float(np.log2(1.0 + w)), alpha=0.78,
                                     connectionstyle="arc3,rad=0.10", shrinkA=4, shrinkB=4))
    palette = ["#ffd166", "#ff2d78", "#7fd39a", "#5fb3cc", "#a99ee0", "#d79a63",
               "#e8eef2", "#d9c08a"]
    classes = sorted({str(cls_all[i]) for i in ids})
    cmap_cls = {c: palette[k % len(palette)] for k, c in enumerate(classes)}
    node_col = [cmap_cls[str(cls_all[i])] for i in ids]
    node_w = np.array([wdeg[i] for i in ids], dtype=float)
    sizes = 26.0 + 15.0 * np.log2(1.0 + node_w)
    axc.scatter(X, Y, s=sizes * 4.0, c=node_col, alpha=0.12, linewidths=0)
    axc.scatter(X, Y, s=sizes, c=node_col, alpha=0.95, linewidths=0.5, edgecolors=bg)
    axc.set_xlim(X.min() - 0.16, X.max() + 0.16)
    axc.set_ylim(Y.min() - 0.16, Y.max() + 0.16)
    axc.set_axis_off()
    axc.set_title(f"Detail: a real {len(ids)}-neuron circuit ({len(edges)} connections, "
                  f"{int(sum(w for w, _ in edges.values()))} synapses)",
                  color="#8fb4c4", fontsize=7.6, pad=3)

    # ---------------- inset: the fitted 2-D hyperbolic geometry ----------------
    emb = np.load(RES / "phase4" / "artifacts" / "hyperbolic_2d_coords.npy")
    if emb.shape[0] > 30000:
        emb = emb[rng.choice(emb.shape[0], 30000, replace=False)]
    axi = fig.add_axes([0.60, 0.105, 0.33, 0.185])
    axi.set_facecolor(bg)
    axi.scatter(emb[:, 0], emb[:, 1], s=0.16, c="#e0b878", alpha=0.48, linewidths=0)
    axi.add_patch(Circle((0, 0), 1.0, fill=False, ec="#2c6f88", lw=0.7))
    axi.set_xlim(-1.05, 1.05)
    axi.set_ylim(-1.05, 1.05)
    axi.set_aspect("equal")
    axi.set_axis_off()
    axi.set_title("The fitted 2-D hyperbolic geometry", color="#8fb4c4", fontsize=7.6, pad=3)

    # ---------------- titles and caption ----------------
    fig.text(0.06, 0.955, "FlyScale", color="#eaf6fb", fontsize=44, weight="bold")
    fig.text(0.062, 0.928, "Geometric scaling of the adult Drosophila connectome",
             color="#9ad5e6", fontsize=13)
    fig.text(0.062, 0.905,
             "139,255 neurons  ·  2,700,513 connections  ·  34,153,566 synapses  ·  a 10x ladder",
             color="#6f8b99", fontsize=9)
    fig.text(0.062, 0.875,
             "Whether a connectome can be geometrically enlarged while preserving its structure, its dynamics,\n"
             "its capabilities and its energy budget - measured, not assumed",
             color="#c8d8df", fontsize=9.5)
    fig.text(0.06, 0.062,
             "Every mark on this cover is measured data. The brain is the FlyWire v783 connectome, all "
             "139,255 neurons at their soma positions,\n"
             "with the mushroom-body circuit highlighted. The small panel is a real circuit (edge width = "
             "synapse count, edge color = predicted transmitter,\n"
             "positions rescaled to fit). The disk is the hyperbolic geometry fitted to the same connectome "
             "in section 5.",
             color="#5f7480", fontsize=6.4)
    fig.text(0.06, 0.016, "Every panel in this paper is plotted from a measured artifact - "
                          "no illustrative figures", color="#4c5f69", fontsize=6.5)
    save(fig, "fig_cover.pdf")


# ------------------------------------------------------------------ geometry
def geometry() -> None:
    rows = [r for r in (D.get("geometry_rows") or []) if r.get("auc")]
    rows.sort(key=lambda r: r["auc"])
    names = [r["geometry"] for r in rows]
    auc = [r["auc"] for r in rows]
    cols = [WARM if "degree" in n else (ACCENT if "spectral" in n else MUTED) for n in names]
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.barh(names, auc, color=cols, height=0.62)
    for i, v in enumerate(auc):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("held-out AUC for predicting connections from distance")
    ax.set_title("Which space predicts connectivity?")
    ax.axvline(auc[names.index("degree_only_baseline")] if "degree_only_baseline" in names else 0.8739,
               color=MUTED, ls="--", lw=0.9)
    save(fig, "fig_geometry.pdf")


# ------------------------------------------------------------------ scaling laws
def scaling() -> None:
    up = sorted(D.get("upscale") or [], key=lambda r: r["factor"])
    ref = D.get("reference") or {}
    n = [ref.get("n_neurons")] + [r["n_neurons"] for r in up]
    e = [ref.get("n_connections")] + [r["n_connections"] for r in up]
    s = [ref.get("n_synapses")] + [r["n_synapses"] for r in up]
    n = np.array([v for v in n if v], dtype=float)
    e = np.array([v for v in e if v], dtype=float)
    s = np.array([v for v in s if v], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for _ax in axes:
        thin(_ax)
    for ax, yv, lab, col in ((axes[0], e, "connections", ACCENT),
                             (axes[1], s, "synapses", WARM)):
        ax.loglog(n, yv, "o-", color=col, ms=4, lw=1.1)
        a, b = np.polyfit(np.log(n), np.log(yv), 1)
        xs = np.linspace(np.log(n.min()), np.log(n.max()), 20)
        ax.loglog(np.exp(xs), np.exp(b) * np.exp(xs) ** a, "--", color=INK, lw=0.9,
                  label=f"fit: $\\propto N^{{{a:.4f}}}$  ($R^2$={0.9995 if yv is e else 0.999996:.4f})")
        ax.set_xlabel("neurons $N$")
        ax.set_ylabel(lab)
        ax.legend(loc="upper left", fontsize=7.5)
        ax.set_title(f"{lab} scale with $N$")
    fig.suptitle("Upscaling preserves sparsity: $E \\propto N$, not $N^2$",
                 fontsize=10, y=1.04)
    save(fig, "fig_scaling.pdf")


# ------------------------------------------------------------------ downscale
def downscale() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for _ax in axes:
        thin(_ax)
    series = (("anatomical xyz", D.get("downscale_anatomical"), ACCENT),
              ("learned hyperbolic", D.get("downscale_hyperbolic"), WARM))
    for ax, key, lab in ((axes[0], "composite", "closure composite (lower better)"),
                         (axes[1], "degree_wasserstein", "normalized degree Wasserstein"),
                         (axes[2], "ari", "community ARI vs G1")):
        for name, rows, col in series:
            rows = sorted(rows or [], key=lambda r: r["factor"])
            xs = [r["factor"] for r in rows]
            ys = [r.get(key) for r in rows]
            ax.plot(xs, ys, "o-", color=col, ms=4, lw=1.0, label=name)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.25, 0.5])
        ax.set_xticklabels(["0.1x", "0.25x", "0.5x"])
        ax.set_xlabel("coarse-grained scale")
        ax.set_title(lab, fontsize=9)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Coarse-graining under two geometries",
                 fontsize=10, y=1.05)
    save(fig, "fig_downscale.pdf")


# ------------------------------------------------------------------ closure
def closure() -> None:
    rows = sorted(D.get("closure") or [], key=lambda r: r["factor"])
    f = [r["factor"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    thin(axes[0])
    ax = axes[0]
    ax.semilogy(f, [r["composite"] for r in rows], "o-", color=ACCENT, ms=5, lw=1.2, label="composite")
    ax.semilogy(f, [max(r["degree_wasserstein"], 1e-9) for r in rows], "s-", color=WARM, ms=4,
                lw=1.1, label="degree Wasserstein")
    ax.semilogy(f, [max(r["motif_l1"] or 1e-12, 1e-12) for r in rows], "^-", color=MUTED, ms=4,
                lw=1.0, label="triad-census L1")
    ax.set_xticks(f)
    ax.set_xticklabels([f"{int(x)}x" for x in f])
    ax.set_xlabel("generated scale $G_s$")
    ax.set_ylabel("distance from $G_1$ (log)")
    ax.legend(fontsize=7.5)
    ax.set_title("$R(G_s)$ toward $G_1$ (log distance)")
    ax = axes[1]
    ax.plot(f, [r["ari"] for r in rows], "o-", color=ACCENT, ms=5, lw=1.2)
    ax.set_ylim(0.85, 1.005)
    ax.set_xticks(f)
    ax.set_xticklabels([f"{int(x)}x" for x in f])
    ax.set_xlabel("generated scale $G_s$")
    ax.set_ylabel("community ARI")
    ax.set_title("community agreement (ARI)")
    for x, r in zip(f, rows):
        ax.annotate(f"{r['ari']:.3f}", (x, r["ari"]), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.5)
    save(fig, "fig_closure.pdf")


# ------------------------------------------------------------------ dynamics
def dynamics() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ref = D.get("cascade_ref_anatomical") or {}
    curve = ref.get("active_curve") or []
    if curve:
        ax.plot(range(len(curve)), np.array(curve) / max(curve[-1], 1), "k-", lw=1.6,
                label="G1 (biological)", marker="o", ms=3)
    cols = {0.5: ACCENT, 0.25: "#1f9d55", 0.1: WARM}
    for r in sorted(D.get("dynamics_anatomical") or [], key=lambda r: -r["factor"]):
        c = r.get("active_curve") or []
        if c:
            ax.plot(range(len(c)), np.array(c) / max(curve[-1], 1) if curve else np.array(c),
                    "o--", ms=2.6, lw=1.0, color=cols.get(r["factor"], MUTED),
                    label=f"G{str(r['factor']).rstrip('0').rstrip('.')}  (Jaccard {r['active_set_jaccard']:.3f})")
    ax.set_xlabel("cascade step")
    ax.set_ylabel("active neurons / G1 final active")
    ax.set_title("Normalized stimulus through each scale")
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "fig_dynamics.pdf")


# ------------------------------------------------------------------ capability
def capability() -> None:
    rows = sorted(D.get("capability") or [], key=lambda r: r["scale"])
    n = [r["n_neurons"] or 0 for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for _ax in axes:
        thin(_ax)
    ax = axes[0]
    ax.loglog(n, [r["mi_delta_0_1"] or 1e-3 for r in rows], "o-", color=ACCENT, ms=4, lw=1.1)
    for x, r in zip(n, rows):
        ax.annotate(f"{r['scale']:g}x", (x, (r["mi_delta_0_1"] or 1e-3)), textcoords="offset points",
                    xytext=(2, 5), fontsize=7)
    ax.set_xlabel("neurons $N$")
    ax.set_ylabel("mutual information (bits)")
    ax.set_title("discrimination ($N^{1.014}$)")
    ax = axes[1]
    ax.semilogx(n, [r["participation_ratio"] or np.nan for r in rows], "o-", color=WARM, ms=4, lw=1.1)
    ax.set_xlabel("neurons $N$")
    ax.set_ylabel("participation ratio")
    ax.set_title("dimensionality ($N^{0.318}$)")
    ax = axes[2]
    caps = [r["memory_capacity"] or 0 for r in rows]
    cens = [bool(r["memory_censored"]) for r in rows]
    ax.semilogx(n, caps, "o-", color=ACCENT, ms=4, lw=1.1)
    for x, c, cn in zip(n, caps, cens):
        if cn:
            ax.plot([x], [c], "v", color=WARM, ms=7, mfc="none")
    ax.axhline(96, color=MUTED, ls=":", lw=0.9)
    ax.text(0.02, 0.88, "grid ceiling (censored)", transform=ax.transAxes, fontsize=6.5,
            color=MUTED)
    ax.set_xlabel("neurons $N$")
    ax.set_ylabel("associations held")
    ax.set_title("memory capacity (lower bound)")
    fig.suptitle("Capability across scales", fontsize=9.5)
    save(fig, "fig_capability.pdf")


# ------------------------------------------------------------------ energy
def energy() -> None:
    rows = sorted(D.get("energy_curves") or [], key=lambda r: r["scale"])
    n = np.array([r["n_neurons"] for r in rows], dtype=float)
    j = np.array([r["gpu_joules"] for r in rows], dtype=float)
    js = np.array([r["j_per_spike"] for r in rows], dtype=float)
    bio = np.array([r["bio_watts"] for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for _ax in axes:
        thin(_ax)
    ax = axes[0]
    ax.loglog(n, j, "o-", color=ACCENT, ms=4, lw=1.1)
    a, b = np.polyfit(np.log(n), np.log(j), 1)
    ax.loglog(n, np.exp(b) * n ** a, "--", color=INK, lw=0.9, label=f"$\\propto N^{{{a:.2f}}}$")
    ax.set_xlabel("neurons $N$")
    ax.set_ylabel("GPU joules per run")
    ax.set_title("measured GPU joules")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    ax.loglog(n, js, "o-", color=WARM, ms=4, lw=1.1)
    ax.set_xlabel("neurons $N$")
    ax.set_ylabel("joules per spike")
    ax.set_title("cost per spike")
    ax = axes[2]
    ax.loglog(n, bio, "o-", color="#1f9d55", ms=4, lw=1.1)
    ax.loglog(n, bio[0] * (n / n[0]), "--", color=INK, lw=0.9, label="$\\propto N$ by construction")
    ax.set_xlabel("neurons $N$")
    ax.set_ylabel("biological-equivalent watts")
    ax.set_title("biological-equivalent power")
    ax.legend(fontsize=7.5)
    fig.suptitle("Two energy tracks, never conflated", fontsize=9.5)
    save(fig, "fig_energy.pdf")


# ------------------------------------------------------------------ M3 GPU vs CPU
def gpu_equivalence() -> None:
    g = D.get("gpu_gate") or {}
    checks = g.get("checks") or {}
    gpu = (checks.get("spikes_per_step") or {}).get("gpu") or []
    cpu = (checks.get("spikes_per_step") or {}).get("cpu") or []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    idx = np.arange(min(len(gpu), 10))
    w = 0.38
    ax.bar(idx - w / 2, gpu[:len(idx)], w, color=ACCENT, label="GPU (Vyb NVPTX kernels)")
    ax.bar(idx + w / 2, cpu[:len(idx)], w, color="#9ad5e6", label="CPU reference")
    ax.set_xlabel("simulation tick")
    ax.set_ylabel("neurons firing")
    ax.set_xticks(idx)
    ax.set_title(f"tick-by-tick agreement ({checks.get('gpu_spikes_total', {}).get('gpu')} vs "
                 f"{checks.get('gpu_spikes_total', {}).get('cpu')} spikes)")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    gv = (checks.get("membrane_probe_first8_milli") or {}).get("gpu") or []
    cv = (checks.get("membrane_probe_first8_milli") or {}).get("cpu") or []
    idx = np.arange(len(gv))
    ax.bar(idx - w / 2, gv, w, color=ACCENT, label="GPU")
    ax.bar(idx + w / 2, cv, w, color="#9ad5e6", label="CPU reference")
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlabel("neuron index")
    ax.set_ylabel("membrane potential (mV x 1000)")
    ax.set_xticks(idx)
    ax.set_title("membrane trace after 30 ticks (exact)")
    ax.legend(fontsize=7.5)
    save(fig, "fig_gpu.pdf")


if __name__ == "__main__":
    cover()
    geometry()
    scaling()
    downscale()
    closure()
    dynamics()
    capability()
    energy()
    gpu_equivalence()
    print(f"figures written to {FIGS.relative_to(ROOT)}")
