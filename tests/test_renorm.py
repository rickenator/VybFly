"""Self-test for the geometric renormalization machinery on a small synthetic graph.

Runs without pytest:  python tests/test_renorm.py

Checks the invariants PROJECT-VYBFLY.md §11-§13 depend on, before any of it is pointed at the
139,255-neuron dataset: subdivision must scale counts linearly (E proportional to N, never
N^2), keep mean degree and mean connection strength bounded, preserve the partner structure,
and coarse-graining back along the lineage must recover the source graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flyscale import closure  # noqa: E402
from flyscale.renorm import (GeometryLaw, coarse_grain, coarse_grain_by_lineage,  # noqa: E402
                             poincare_distance_block, poincare_exp_map,
                             typical_neighbour_distance, upscale)
from flyscale.synthetic import GraphView  # noqa: E402


def make_geometric_graph(n: int = 400, dim: int = 2, radius: float = 0.35,
                         seed: int = 0) -> GraphView:
    """A random geometric directed graph: nodes uniform in [0,1]^d, edges by distance."""
    rng = np.random.default_rng(seed)
    coords = rng.random((n, dim))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    A = (d < radius) & (d > 0)
    pre, post = np.nonzero(A)
    syn = rng.integers(1, 21, size=pre.size)
    nt = rng.integers(0, 6, size=pre.size).astype(np.int8)
    return GraphView(n=n, pre=pre, post=post, syn=syn, nt_code=nt,
                     root_ids=np.arange(10_000, 10_000 + n, dtype=np.int64),
                     cell_type=np.array(["ct%d" % (i % 7) for i in range(n)], dtype=object),
                     super_class=np.array(["central" if i % 3 else "optic" for i in range(n)],
                                          dtype=object),
                     coords=coords, provenance={"kind": "synthetic_geometric"})


def test_hyperbolic_geometry() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 2))
    x /= (np.linalg.norm(x, axis=1, keepdims=True) * 1.5)      # inside the unit ball
    d = poincare_distance_block(x, x)
    assert np.allclose(np.diag(d), 0.0, atol=1e-6), np.diag(d)
    assert np.allclose(d, d.T, atol=1e-9)
    # the exponential map moves exactly `length`
    u = rng.normal(size=(5, 2))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    length = 0.4
    y = poincare_exp_map(x, np.tanh(length / 2.0) * u)
    got = poincare_distance_block(x, y).diagonal()
    assert np.allclose(got, length, atol=2e-6), got
    assert np.all(np.linalg.norm(y, axis=1) < 1.0)


def test_upscale_scaling() -> None:
    g = make_geometric_graph(400)
    law = GeometryLaw(coords=g.coords, kind="euclidean", R=0.3, T=0.05)
    for factor in (2, 5):
        up = upscale(g, law, factor, seed=1)
        n_ratio = up.n / g.n
        e_ratio = up.pre.size / g.pre.size
        assert abs(n_ratio - factor) < 0.02, (factor, n_ratio)
        # §12: edges scale linearly with N, not quadratically
        assert abs(e_ratio - factor) / factor < 0.25, (factor, e_ratio)
        mean_deg_src = g.pre.size / g.n
        mean_deg_up = up.pre.size / up.n
        assert abs(mean_deg_up - mean_deg_src) / mean_deg_src < 0.35, (mean_deg_src, mean_deg_up)
        # mean connection strength (synapses per connection) is preserved by construction
        strength_ratio = up.syn.mean() / g.syn.mean()
        assert 0.5 < strength_ratio < 2.0, strength_ratio
        # lineage is recorded for the closure test
        assert up.parent_index is not None and up.parent_index.size == up.n
        assert np.array_equal(np.bincount(up.parent_index, minlength=g.n).sum(), up.n)
        print(f"  upscale x{factor}: n {g.n}->{up.n}  E {g.pre.size}->{up.pre.size}  "
              f"mean_deg {mean_deg_src:.1f}->{mean_deg_up:.1f}  "
              f"mean_strength {g.syn.mean():.1f}->{up.syn.mean():.1f}")


def test_lineage_closure() -> None:
    g = make_geometric_graph(400)
    law = GeometryLaw(coords=g.coords, kind="euclidean", R=0.3, T=0.05)
    up = upscale(g, law, 2, seed=2)
    back = coarse_grain_by_lineage(up)
    assert back.n == g.n, (back.n, g.n)
    corr = np.arange(g.n)                       # lineage groups are exactly the parents
    cmp = closure.compare(g, back, correspondence=corr, triad_max_edges=10**9)
    assert cmp["degree_wasserstein_normalised"] < 0.5, cmp["degree_wasserstein_normalised"]
    assert cmp["community_ari"] is not None and cmp["community_ari"] > 0.5, cmp["community_ari"]
    print(f"  closure 2x->back: E {g.pre.size} vs {back.pre.size}  "
          f"deg_wass_norm {cmp['degree_wasserstein_normalised']:.4f}  "
          f"ARI {cmp['community_ari']:.3f}  composite {cmp['composite']}")


def test_coarse_grain_geometry() -> None:
    g = make_geometric_graph(400)
    law = GeometryLaw(coords=g.coords, kind="euclidean", R=0.3, T=0.05)
    cg = coarse_grain(g, law, 0.5, seed=3)
    assert 150 <= cg.n <= 220, cg.n
    src_mean = g.pre.size / g.n
    cg_mean = cg.pre.size / cg.n
    assert abs(cg_mean - src_mean) / src_mean < 0.5, (src_mean, cg_mean)
    cmp = closure.compare(g, cg, correspondence=None, triad_max_edges=10**9)
    print(f"  coarse 0.5x: n {g.n}->{cg.n}  E {g.pre.size}->{cg.pre.size}  "
          f"mean_deg {src_mean:.1f}->{cg_mean:.1f}  "
          f"deg_wass_norm {cmp['degree_wasserstein_normalised']:.4f}  "
          f"triad_L1 {cmp['motif_divergence_L1']}")
    # correspondence-free comparison must still produce the distributional metrics
    for key in ("degree_wasserstein", "spectral_distance", "rich_club_distance"):
        assert cmp[key] is not None and cmp[key] == cmp[key], key
    # and the lineage-based partition of a coarse graining is recoverable for later closure
    assert cmp["community_ari"] is None


def run() -> None:
    print("hyperbolic geometry ...")
    test_hyperbolic_geometry()
    print("subdivision scaling ...")
    test_upscale_scaling()
    print("lineage closure ...")
    test_lineage_closure()
    print("geometric coarse graining ...")
    test_coarse_grain_geometry()
    print("renorm self-test PASSED")


def test_renorm() -> None:
    run()


if __name__ == "__main__":
    run()
