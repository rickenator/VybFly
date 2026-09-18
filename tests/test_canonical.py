"""Self-test for the canonical builder on synthetic data with the real release schema.

Runs without pytest:  python tests/test_canonical.py

It exercises the code paths that matter for the real 852 MB file (chunked streaming,
cross-chunk pair aggregation, neuropil keying, unmatched-root-id handling, autapses,
NT argmax, CSR construction) before the real data is trusted.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flyscale.connectome import Connectome, build_canonical  # noqa: E402
from flyscale import io as fio  # noqa: E402

ROOT_IDS = np.array([100, 200, 300, 400, 500], dtype=np.int64)
# pre, post, neuropil, syn_count, then the six NT probabilities (gaba, ach, glut, oct, ser, da)
ROWS = [
    (100, 200, "ME_L", 3, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0),
    (100, 200, "AL_R", 2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0),   # -> dominant gaba overall
    (200, 100, "AL_R", 1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0),
    (200, 300, "FB", 4, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0),
    (300, 300, "FB", 5, 0.0, 0.2, 0.0, 0.0, 0.6, 0.0),      # autapse, dominant ser
    (100, 999, "LO", 7, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0),      # unmatched post root id
    (200, 300, "FB", 1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0),      # same pair+neuropil as row 4
]


def make_raw(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    np.save(raw / "proofread_root_ids_783.npy", ROOT_IDS)
    cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count", *fio.NT_PROB_COLUMNS]
    df = pd.DataFrame(ROWS, columns=cols)
    df.to_feather(raw / "proofread_connections_783.feather")
    ann = pd.DataFrame({
        "root_id": ROOT_IDS,
        "super_class": ["central", "optic", "central", "sensory", None],
        "cell_type": ["CT1", "CT2", "CT1", "CT3", None],
        "top_nt": ["acetylcholine", "gaba", "serotonin", "acetylcholine", None],
        "side": ["left", "right", "left", "left", None],
    })
    ann.to_csv(raw / "Supplemental_file1_neuron_annotations.tsv", sep="\t", index=False)


def run() -> None:
    with tempfile.TemporaryDirectory() as td:
        raw, out = Path(td) / "raw", Path(td) / "canon"
        make_raw(raw)
        meta = build_canonical(raw, out, chunk_rows=2, force=True)   # chunk_rows=2 forces cross-chunk merges
        c = Connectome(out)
        cnt = meta["counts"]

        assert cnt["n_neurons"] == 5, cnt
        assert cnt["n_raw_rows_streamed"] == 7, cnt
        assert cnt["n_raw_rows_with_unmatched_root_id"] == 1, cnt
        assert cnt["n_pairs"] == 4, cnt                      # 100->200, 200->100, 200->300, 300->300
        assert cnt["n_synapses_pairs"] == 3 + 2 + 1 + 4 + 1 + 5, cnt
        assert cnt["n_autapse_pairs"] == 1, cnt
        assert cnt["n_neuropils"] == 3, cnt                  # ME_L, AL_R, FB (LO row dropped)
        assert cnt["n_reciprocal_pairs"] == 1, cnt          # (100,200)/(200,100); autapse excluded
        assert cnt["n_edges_in_reciprocal_pairs"] == 2, cnt

        pairs = c.pairs.set_index(["pre_idx", "post_idx"])
        assert pairs.loc[(0, 1), "syn_count"] == 5           # 3 (ME_L) + 2 (AL_R)
        assert pairs.loc[(2, 2), "syn_count"] == 5           # autapse kept
        assert pairs.loc[(1, 2), "syn_count"] == 5           # 4 + 1 across two published rows

        # 100->200 is 3 synapses gaba-heavy in ME_L and 2 acetylcholine-clean in AL_R:
        # weighted mean gaba = (3*0.1 + 2*0.9)/5 = 0.42, ach = (3*0.2 + 2*0.0)/5 = 0.12
        assert abs(pairs.loc[(0, 1), "nt_prob_gaba"] - 0.42) < 1e-6
        assert pairs.loc[(0, 1), "nt_code"] == fio.NT_TYPES.index("gaba")
        assert pairs.loc[(2, 2), "nt_code"] == fio.NT_TYPES.index("ser")

        # CSR must be consistent with the pair list
        assert np.array_equal(c.out_degree(), np.bincount(c.pre, minlength=5))
        assert np.array_equal(c.in_degree(), np.bincount(c.post, minlength=5))
        assert np.array_equal(c.weighted_out_degree(), np.bincount(c.pre, weights=c.syn, minlength=5))
        assert np.array_equal(c.weighted_in_degree(), np.bincount(c.post, weights=c.syn, minlength=5))
        A = c.adjacency(include_autapses=True)
        assert A.shape == (5, 5) and A.nnz == 4, A.nnz
        assert A[0, 1] == 1 and A[2, 2] == 1
        Aw = c.adjacency(include_autapses=False, weighted=True)
        assert Aw.nnz == 3 and Aw[0, 1] == 5.0

        # neuropil stratification preserved: 100->200 exists in two neuropils
        e = c.edges
        assert len(e) == 5, len(e)                        # 4 pairs, one of them split over 2 neuropils
        row = e[(e.pre_idx == 0) & (e.post_idx == 1)]
        assert len(row) == 2 and int(row.syn_count.sum()) == 5

        # annotation join: row 4 has no annotation -> NaN, not zero
        assert c.neurons.loc[4, "cell_type"] != c.neurons.loc[4, "cell_type"]  # NaN
        assert c.neurons.loc[0, "cell_type"] == "CT1"

        # restricted view: pairs with both endpoints inside the allowed set survive
        allow = np.array([True, True, True, False, False])
        sub = c.restricted_to(allow)
        assert sub.pre.size == 4, sub.pre.size
        assert np.array_equal(sub.out_degree(), np.bincount(sub.pre, minlength=5))
        print("meta counts:", cnt)
        print("neuropils:", c.neuropils["name_to_code"])
        print("canonical self-test PASSED")


def test_canonical() -> None:
    run()


if __name__ == "__main__":
    run()
