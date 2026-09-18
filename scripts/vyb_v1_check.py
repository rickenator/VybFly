"""Cross-check the Vyb-native loader output against an independent Python computation.

    python scripts/vyb_v1_check.py --vyb-output results/phase0/vyb_counts.txt

The Vyb program (src/vyb/phase0_counts.vyb) decodes the flat-binary canonical dataset with
its own byte-level reader; this script recomputes every invariant it prints, straight from
the same bytes, and fails loudly on any mismatch. It is the V1 gate of docs/VYB-PORT.md:
Vyb's numbers must equal the Python reference's numbers exactly (these are integer
invariants, so no tolerance is warranted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_vyb_output(path: Path) -> dict:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            try:
                values[k.strip()] = int(v.strip())
            except ValueError:
                values[k.strip()] = v.strip()
    return values


def oracle(bin_dir: Path) -> dict:
    meta = json.loads((bin_dir / "meta.json").read_text())
    n = meta["n_neurons"]
    m = meta["n_pairs"]

    indptr = np.fromfile(bin_dir / "csr.out.indptr.i64", dtype="<i8", count=n + 1)
    deg = np.diff(indptr)
    syn = np.fromfile(bin_dir / "pairs.syn.i32", dtype="<i4", count=m)
    pre = np.fromfile(bin_dir / "pairs.pre.i32", dtype="<i4", count=m)
    post = np.fromfile(bin_dir / "pairs.post.i32", dtype="<i4", count=m)
    ids = np.fromfile(bin_dir / "neurons.root_id.i64", dtype="<i8", count=n)

    thr = syn >= 5
    return {
        "n_neurons": n,
        "n_indptr_entries": int(indptr.size),
        "indptr_head0": int(indptr[0]),
        "indptr_tail": int(indptr[n]),
        "out_degree_sum": int(deg.sum()),
        "out_degree_max": int(deg.max()),
        "out_degree_zero_neurons": int((deg == 0).sum()),
        "n_connections": m,
        "synapse_total": int(syn.sum()),
        "connections_thr5": int(thr.sum()),
        "synapses_thr5": int(syn[thr].sum()),
        "max_connection_strength": int(syn.max()),
        "min_connection_strength": int(syn.min()),
        "syn_elements_seen": int(syn.size),
        "pre_elements_seen": int(pre.size),
        "pre_sum": int(pre.sum()),
        "pre_max": int(pre.max()),
        "pre_min": int(pre.min()),
        "post_elements_seen": int(post.size),
        "post_sum": int(post.sum()),
        "post_max": int(post.max()),
        "root_ids": int(ids.size),
        "root_id_seen": int(ids.size),
        "root_id_max": int(ids.max()),
        "root_id_min": int(ids.min()),
        # the sum of 139k root ids overflows int64; both sides wrap identically, which makes
        # it a useful equality check on the whole decoded array
        "root_id_sum": int(np.sum(ids, dtype=np.int64)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vyb-output", default=str(ROOT / "results" / "phase0" / "vyb_counts.txt"))
    ap.add_argument("--bin", default=str(ROOT / "data" / "processed" / "canonical_v783" / "bin"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase0" / "vyb_v1_check.json"))
    args = ap.parse_args()

    vyb = read_vyb_output(Path(args.vyb_output))
    ref = oracle(Path(args.bin))

    rows, failures = [], []
    for key, expected in ref.items():
        got = vyb.get(key)
        ok = (got == expected)
        if not ok:
            failures.append(key)
        rows.append({"key": key, "vyb": got, "python_reference": expected,
                     "match": ok, "kind": "integer invariant"})

    width = max(len(r["key"]) for r in rows)
    print(f"{'invariant'.ljust(width)}  {'vyb (native)':>16s}  {'python reference':>16s}  ok")
    for r in rows:
        print(f"{r['key'].ljust(width)}  {str(r['vyb']):>16s}  {str(r['python_reference']):>16s}  "
              f"{'yes' if r['match'] else 'NO'}")

    verdict = {
        "gate": "Vyb-native loader (V1) vs Python reference",
        "vyb_output": str(args.vyb_output),
        "bin_dir": str(args.bin),
        "checks": rows,
        "matched": len(rows) - len(failures),
        "total": len(rows),
        "passed": not failures,
        "failures": failures,
        "vyb_reported_ok": "vyb-phase0-counts-ok" in Path(args.vyb_output).read_text(),
    }
    Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(f"\n{verdict['matched']}/{verdict['total']} invariants match | "
          f"vyb ok marker: {verdict['vyb_reported_ok']} | passed: {verdict['passed']}")
    print("wrote", args.out)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
