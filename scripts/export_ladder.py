"""Export one ladder scale's harness inputs from a saved replica graph.

    python scripts/export_ladder.py --scale 100                       # from results/phase6/replicas/g100
    python scripts/export_ladder.py --scale 100 --src <dir> --out <dir> --stim <path>

The ladder engines (src/vyb/ladder_*.vyb) read a directory containing

    bin/csr.out.indptr.i64     n + 1 int64
    bin/csr.out.indices.i32    one int32 target per edge, grouped by source
    bin/csr.out.syn.i32        one int32 synapse count per edge, same order

plus a stimulus file produced by scripts/make_stim.cpp for the same n. This script writes
the CSR half from a replica saved by scripts/phase6_upscale.py --save, and calls the built
stimulus generator for the other half so a scale's inputs are never hand-assembled.

Scale 1 is the canonical FlyWire graph itself (threshold 5), which is what the published
ladder's s1 is.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_target  # noqa: E402
from flyscale.synthetic import load_graph  # noqa: E402

FLYWIRE_V783_NEURONS = 139255
PHASE6 = ROOT / "results" / "phase6"


def export_csr(g, dest: Path) -> dict:
    """Write the three CSR files the ladder harness reads (out-CSR, i64 indptr)."""
    bin_dir = dest / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    indptr = np.zeros(g.n + 1, dtype=np.int64)
    np.add.at(indptr, g.pre + 1, 1)
    np.cumsum(indptr, out=indptr)
    order = np.argsort(g.pre, kind="stable")
    indices = g.post[order].astype(np.int32)
    syn = np.rint(g.syn[order]).astype(np.int32)
    indptr.tofile(bin_dir / "csr.out.indptr.i64")
    indices.tofile(bin_dir / "csr.out.indices.i32")
    syn.tofile(bin_dir / "csr.out.syn.i32")
    return {
        "n_neurons": int(g.n),
        "n_edges": int(g.pre.size),
        "n_synapses": int(syn.astype(np.int64).sum()),
        "mean_out_degree": round(float(g.pre.size) / g.n, 4),
        "indptr_bytes": int(indptr.nbytes),
        "indices_bytes": int(indices.nbytes),
        "syn_bytes": int(syn.nbytes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True,
                    help="ladder scale (1 = canonical, else the replica's factor, e.g. 100)")
    ap.add_argument("--src", default=None, help="replica dir (default results/phase6/replicas/g<scale>)")
    ap.add_argument("--out", default=None, help="ladder dir (default results/ladder/s<scale>)")
    ap.add_argument("--stim", default=None, help="stimulus path (default <out>/stim_s<scale>.bin)")
    ap.add_argument("--stim-bin", default=None, help="path to the built make_stim binary")
    args = ap.parse_args()

    scale = args.scale
    tag = f"s{scale:g}"
    out = Path(args.out) if args.out else ROOT / "results" / "ladder" / tag
    stim = Path(args.stim) if args.stim else out / f"stim_{tag}.bin"

    if scale == 1.0:
        g = load_target(threshold=5)
        source = "canonical_v783 (threshold 5)"
    else:
        src = Path(args.src) if args.src else PHASE6 / "replicas" / f"g{scale:g}"
        if not (src / "graph.npz").exists():
            raise SystemExit(f"no saved replica at {src}; run scripts/phase6_upscale.py "
                             f"--factors {scale:g} --save first")
        g = load_graph(src)
        source = str(src)

    summary = {"scale": scale, "source": source, "ladder_dir": str(out), "stim": str(stim)}
    summary["csr"] = export_csr(g, out)

    stim_bin = Path(args.stim_bin) if args.stim_bin else Path("/tmp/make_stim")
    if not stim_bin.exists():
        cc = subprocess.run(["g++", "-O2", "-o", str(stim_bin),
                             str(ROOT / "scripts" / "make_stim.cpp")], check=False,
                            capture_output=True, text=True)
        if cc.returncode != 0:
            raise SystemExit(f"building make_stim failed:\n{cc.stderr}")
    res = subprocess.run([str(stim_bin), str(int(g.n)), str(stim)], check=True,
                         capture_output=True, text=True)
    summary["stim_stdout"] = res.stdout.strip()

    (out / "export.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
