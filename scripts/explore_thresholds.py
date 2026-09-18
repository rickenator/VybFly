"""Check the published connectivity convention: connections are thresholded at 5 synapses.

Run:  python scripts/explore_thresholds.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAIRS = "data/processed/canonical_v783/pairs.parquet"
N = 139255

p = pd.read_parquet(PAIRS, columns=["pre_idx", "post_idx", "syn_count"])
pre_all = p["pre_idx"].to_numpy()
post_all = p["post_idx"].to_numpy()
syn_all = p["syn_count"].to_numpy()

print(f"pairs (no threshold): {len(p):,d}   synapses: {int(syn_all.sum()):,d}   "
      f"mean syn/pair: {syn_all.mean():.3f}")
key_all = pre_all.astype(np.int64) * N + post_all
print()
print(f"{'thr':>4s} {'connections':>13s} {'synapses':>13s} {'mean syn':>9s} "
      f"{'mean deg':>9s} {'reciprocity':>12s}")
for k in (1, 2, 3, 4, 5, 6, 8, 10, 20):
    m = syn_all >= k
    pre, post, syn = pre_all[m], post_all[m], syn_all[m]
    if pre.size == 0:
        continue
    keys = pre.astype(np.int64) * N + post
    rev = post.astype(np.int64) * N + pre
    rec = np.isin(rev, keys)
    no_aut = pre != post
    recip = rec[no_aut].sum() / max(1, no_aut.sum())
    deg = np.bincount(pre, minlength=N) + np.bincount(post, minlength=N)
    print(f"{k:>4d} {pre.size:>13,d} {int(syn.sum()):>13,d} {syn.mean():>9.3f} "
          f"{deg.mean():>9.2f} {recip:>12.4f}")

# published anchors: v783 codex reports 3,732,460 connections; the Nature network paper
# (v630, 5-synapse threshold) reports 2,613,129 connections and reciprocity 0.138
print()
print("published: v783 codex connections = 3,732,460 ; v630 (thr 5) = 2,613,129, "
      "reciprocity 0.138, mean in/out degree of an intrinsic neuron 20.5, "
      "mean connection strength 12.6 synapses")
