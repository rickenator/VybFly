"""Phase 0a: build the canonical FlyWire v783 dataset from the raw release files.

    python scripts/phase0_build.py [--force]

Writes data/processed/canonical_v783/ (see flyscale.connectome docstring) and prints the
headline counts that the Phase 0 gate checks against the published release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyscale.connectome import Connectome, build_canonical

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "canonical_v783"))
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    meta = build_canonical(args.raw, args.out, chunk_rows=args.chunk_rows, force=args.force)
    print(json.dumps(meta["counts"], indent=2))

    c = Connectome(args.out)
    print("\ncanonical dataset:", args.out)
    print(json.dumps(c.summary(), indent=2))
    print("\nneuropils:", c.neuropils["name_to_code"])
    nt = c.pairs["nt_code"].value_counts().sort_index()
    print("\nconnections by dominant neurotransmitter:", dict(zip(range(len(nt)), nt.tolist())))
    print("autapses:", int((~c.mask_autapses()).sum()))
    print("annotated cell types:", int(c.neurons['cell_type'].nunique()),
          "| super classes:", int(c.neurons['super_class'].nunique()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
