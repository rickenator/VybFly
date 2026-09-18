"""Phase 0b: canonical graph metrics + reproducible baseline JSON.

    python scripts/phase0_metrics.py [--heavy] [--sources 1000]

Writes results/phase0/flywire_baseline.json (checkpointed block by block) plus binary
artifacts (matrices, community labels) in results/phase0/artifacts/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyscale.connectome import Connectome
from flyscale.metrics import compute_baseline

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default=str(ROOT / "data" / "processed" / "canonical_v783"))
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase0" / "flywire_baseline.json"))
    ap.add_argument("--sources", type=int, default=1000, help="BFS source samples per variant")
    ap.add_argument("--heavy", action="store_true", help="also triad census + rich-club null")
    ap.add_argument("--blocks", nargs="*", default=None,
                    help="compute only these blocks, merging into an existing results file")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    c = Connectome(args.canonical)
    res = compute_baseline(c, args.raw, Path(args.out), n_bfs_sources=args.sources,
                           heavy=args.heavy, seed=args.seed, only_blocks=args.blocks)
    print(json.dumps(res["timings"], indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
