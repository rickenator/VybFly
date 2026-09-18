#!/usr/bin/env python3
"""Fetch the canonical FlyWire (FAFB v783) source data for FlyScale Phase 0.

Sources (all public, no credentials required):
  * Zenodo record 10676866 (DOI 10.5281/zenodo.10676866, CC-BY-4.0)
      - proofread_connections_783.feather      edge list (pre/post root id, synapse count, transmitter)
      - proofread_root_ids_783.npy             the 139,255 proofread neuron root ids
      - per_neuron_neuropil_count_{pre,post}   per-neuron synapse counts stratified by neuropil
  * GitHub flyconnectome/flywire_annotations (Schlegel et al. 2024 supplemental files)
      - Supplemental_file1_neuron_annotations.tsv   cell type / transmitter / region annotations
      - Supplemental_file3_summary_with_ngl_links.csv
  * Zenodo record 12572930 (DOI 10.5281/zenodo.12572930, MIT) - flywire-network-analysis
      published analysis scripts for the network-statistics paper (Phase 0 gate reference)

The 9.5 GB flywire_synapses_783.feather (individual synapse point cloud) is deliberately NOT
fetched here; it is only needed for morphology/geometry work in a later phase.

Every file is size- and md5-verified where Zenodo publishes a checksum, and the retrieval is
recorded in data/raw/MANIFEST.json for reproducibility (see PROJECT-VYBFLY.md section 24).

Usage:  python scripts/fetch_data.py [--include-synapses] [--only NAME ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ZENODO_783 = "https://zenodo.org/api/records/10676866/files/{key}/content"
GH_ANNOT = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/{key}"

# name -> (url, md5 or None, size or None, required, description)
SOURCES: dict[str, dict] = {
    "proofread_root_ids_783.npy": dict(
        url=ZENODO_783.format(key="proofread_root_ids_783.npy"),
        md5="e0e6c19732fd8c7a4e39a2d170105421", size=1114168, required=True,
        desc="proofread neuron root ids, n=139,255 (v783)",
    ),
    "proofread_connections_783.feather": dict(
        url=ZENODO_783.format(key="proofread_connections_783.feather"),
        md5="f48f972d262323a102aed49af1396b8a", size=852022274, required=True,
        desc="canonical directed edge list (pre/post root id, synapse count, transmitter)",
    ),
    "per_neuron_neuropil_count_post_783.feather": dict(
        url=ZENODO_783.format(key="per_neuron_neuropil_count_post_783.feather"),
        md5="bb5999f10920ade803d9f37097a43a56", size=233843050, required=False,
        desc="per-neuron postsynaptic synapse counts by neuropil",
    ),
    "per_neuron_neuropil_count_pre_783.feather": dict(
        url=ZENODO_783.format(key="per_neuron_neuropil_count_pre_783.feather"),
        md5="90fcdb42c1ba05ed92820840fa1e6ba0", size=16853770, required=False,
        desc="per-neuron presynaptic synapse counts by neuropil",
    ),
    "flywire_synapses_783.feather": dict(
        url=ZENODO_783.format(key="flywire_synapses_783.feather"),
        md5="f8f1b97c9d4b0ea9b4c8b287f6b99091", size=9492998242, required=False,
        desc="individual synapse point cloud (~9.5 GB) - later phase only", only_with="synapses",
    ),
    "Supplemental_file1_neuron_annotations.tsv": dict(
        url=GH_ANNOT.format(key="Supplemental_file1_neuron_annotations.tsv"),
        md5=None, size=31718505, required=True,
        desc="Schlegel et al. 2024 whole-brain annotations (cell type, transmitter, ...)",
    ),
    "Supplemental_file3_summary_with_ngl_links.csv": dict(
        url=GH_ANNOT.format(key="Supplemental_file3_summary_with_ngl_links.csv"),
        md5=None, size=758670, required=False,
        desc="annotation summary rows with Neuroglancer links",
    ),
    "flywire-network-analysis.zip": dict(
        url="https://zenodo.org/api/records/12572930/files/flywire-network-analysis.zip/content",
        md5="9a71bc3d0bb820cd5bbe4f6fd20235c3", size=36893964, required=False,
        desc="published network-statistics analysis code (reference for the Phase 0 gate)",
    ),
}


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(dest: Path, spec: dict) -> tuple[bool, str | None, str | None]:
    """Verify a downloaded file. Returns (ok, md5, note).

    The md5 published in the Zenodo API is authoritative; its `size` field is stale for
    records that were re-uploaded in place (record 12572930 advertises 36,893,964 bytes for a
    36,875,651-byte zip whose md5 matches). So when a checksum exists it decides, and a size
    disagreement is reported as a stale-metadata note rather than a failure.
    """
    if not dest.exists():
        return False, None, None
    size = dest.stat().st_size
    if spec["md5"]:
        got = md5_of(dest)
        if got != spec["md5"]:
            return False, got, f"md5 mismatch (got {got}, expected {spec['md5']})"
        note = None
        if spec["size"] is not None and size != spec["size"]:
            note = (f"checksum matches but published size is stale "
                    f"({size} vs advertised {spec['size']})")
        return True, got, note
    if spec["size"] is not None and size != spec["size"]:
        return False, None, f"size mismatch (got {size}, expected {spec['size']})"
    return True, None, None


def fetch(name: str, spec: dict, force: bool = False) -> dict:
    dest = RAW / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "url": spec["url"], "dest": str(dest.relative_to(ROOT)),
           "expected_size": spec["size"], "expected_md5": spec["md5"], "desc": spec["desc"]}

    if not force:
        ok, got_md5, note = verify(dest, spec)
        if ok:
            rec.update(status="cached", bytes=dest.stat().st_size, md5=got_md5)
            if note:
                rec["note"] = note
            print(f"[cached ] {name}" + (f" - {note}" if note else ""))
            return rec
        if dest.exists() and note:
            print(f"[partial] {name}: {note} - refetching/resuming")

    t0 = time.time()
    cmd = ["curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "5",
           "--connect-timeout", "30", "-C", "-", "-o", str(dest), spec["url"]]
    subprocess.run(cmd, check=True)
    dt = time.time() - t0

    ok, got_md5, note = verify(dest, spec)
    if not ok:
        raise SystemExit(f"VERIFY FAILED {name}: {note}")
    size = dest.stat().st_size
    rec.update(status="downloaded", bytes=size, md5=got_md5,
               seconds=round(dt, 1), mbps=round(size / 1e6 / max(dt, 1e-9), 1))
    if note:
        rec["note"] = note
    print(f"[fetched] {name}  {size/1e6:.1f} MB in {dt:.0f}s ({rec['mbps']} MB/s)"
          + (f" - {note}" if note else ""))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-synapses", action="store_true",
                    help="also fetch the 9.5 GB synapse point cloud")
    ap.add_argument("--only", nargs="*", default=None, help="fetch only these file names")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for name, spec in SOURCES.items():
        if args.only and name not in args.only:
            continue
        if spec.get("only_with") == "synapses" and not args.include_synapses:
            print(f"[skipped] {name} (pass --include-synapses to fetch)")
            continue
        try:
            records.append(fetch(name, spec, force=args.force))
        except SystemExit as exc:
            print(f"[FAILED ] {name}: {exc}")
            failures.append(name)

    manifest = RAW / "MANIFEST.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    old.update({
        "dataset": "FlyWire FAFB v783 whole-brain connectome",
        "zenodo_doi": "10.5281/zenodo.10676866",
        "zenodo_version": "783.0",
        "license": "CC-BY-4.0",
        "annotations_source": "github.com/flyconnectome/flywire_annotations (Schlegel et al. 2024)",
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {**old.get("files", {}), **{r["name"]: r for r in records}},
    })
    manifest.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")
    print(f"\nmanifest: {manifest}")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
