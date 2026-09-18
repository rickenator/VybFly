"""Phase 0c: export the canonical dataset as flat little-endian binary for Vyb.

    python scripts/export_bin.py

Python is used here only as a packaging step (it has the feather/parquet readers). The
output is the interface the Vyb-native loader consumes: header-less arrays described by
bin/meta.json, readable with io::read_at + byte assembly. See docs/VYB-PORT.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

NEURON_TEXT_COLUMNS = ("root_id", "super_class", "cell_class", "cell_type", "supertype",
                       "top_nt", "side")
NEURON_FLOAT_COLUMNS = ("pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default=str(ROOT / "data" / "processed" / "canonical_v783"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "canonical_v783" / "bin"))
    args = ap.parse_args()

    src, out = Path(args.canonical), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    neurons = pd.read_parquet(src / "neurons.parquet")
    pairs = pd.read_parquet(src / "pairs.parquet",
                            columns=["pre_idx", "post_idx", "syn_count", "nt_code"])
    z = np.load(src / "csr.npz")
    neuropils = json.loads((src / "neuropils.json").read_text())
    n, m = len(neurons), len(pairs)
    arrays: dict[str, dict] = {}

    def emit(name: str, arr: np.ndarray, dtype: str) -> None:
        a = np.ascontiguousarray(arr, dtype=np.dtype(dtype))
        (out / name).write_bytes(a.tobytes())
        arrays[name] = {"dtype": str(a.dtype), "count": int(a.size), "bytes": int(a.nbytes)}

    emit("neurons.root_id.i64", neurons["root_id"].to_numpy(), "<i8")
    for col in NEURON_TEXT_COLUMNS:
        if col == "root_id":
            continue
        vals = neurons[col].fillna("").astype(str).to_numpy()
        (out / f"neurons.{col}.txt").write_text("\n".join(vals.tolist()) + "\n")
        arrays[f"neurons.{col}.txt"] = {"dtype": "text", "count": int(vals.size),
                                        "bytes": int((out / f'neurons.{col}.txt').stat().st_size)}
    floats = np.stack([neurons[c].fillna(np.nan).to_numpy(np.float32)
                       for c in NEURON_FLOAT_COLUMNS], axis=1)
    emit("neurons.coords.f32", floats.reshape(-1), "<f4")

    emit("pairs.pre.i32", pairs["pre_idx"].to_numpy(), "<i4")
    emit("pairs.post.i32", pairs["post_idx"].to_numpy(), "<i4")
    emit("pairs.syn.i32", pairs["syn_count"].to_numpy(), "<i4")
    emit("pairs.nt.i8", pairs["nt_code"].to_numpy(), "i1")

    for key, name in (("out_indptr", "csr.out.indptr.i64"), ("out_indices", "csr.out.indices.i32"),
                      ("out_syn", "csr.out.syn.i32"), ("in_indptr", "csr.in.indptr.i64"),
                      ("in_indices", "csr.in.indices.i32"), ("in_syn", "csr.in.syn.i32")):
        emit(name, z[key], "<i8" if "indptr" in name else "<i4")

    names = neuropils["code_to_name"]
    (out / "neuropils.txt").write_text("\n".join(names) + "\n")
    nt_order = json.loads((src / "meta.json").read_text())["conventions"]["nt_order"]

    meta = {
        "format": "flyscale-flat-binary-1",
        "endianness": "little",
        "source_canonical": str(src),
        "source_canonical_version": json.loads((src / "meta.json").read_text())["canonical_version"],
        "n_neurons": n,
        "n_pairs": m,
        "n_neuropils": len(names),
        "nt_order": nt_order,
        "neuron_text_columns": [c for c in NEURON_TEXT_COLUMNS if c != "root_id"],
        "neuron_float_columns": list(NEURON_FLOAT_COLUMNS),
        "files": arrays,
        "notes": [
            "arrays are header-less; element type and count are in this file",
            "neuron index i corresponds to the i-th line of every neurons.*.txt file",
            "the threshold-5 connectivity view is a filter on pairs.syn.i32",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    total = sum(v["bytes"] for v in arrays.values())
    print(f"wrote {len(arrays)} arrays to {out} ({total / 1e6:.1f} MB)")
    print(json.dumps({k: v["count"] for k, v in arrays.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
