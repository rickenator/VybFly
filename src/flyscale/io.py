"""Raw-source readers for the FlyWire FAFB v783 release.

Everything here is a thin, streaming, schema-checked wrapper over the files fetched by
scripts/fetch_data.py. No analysis logic lives in this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.feather as feather

# Eckstein, Bates et al. 2024 neurotransmitter prediction columns, in the order published.
NT_PROB_COLUMNS = ("gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg")
NT_TYPES = ("gaba", "ach", "glut", "oct", "ser", "da")

CONNECTIONS_COLUMNS = (
    "pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count", *NT_PROB_COLUMNS,
)


def load_root_ids(path: str | Path) -> np.ndarray:
    """The published array of proofread neuron root ids (int64, unsorted in the file)."""
    ids = np.load(path)
    if ids.dtype.kind not in "iu":
        raise ValueError(f"{path}: expected an integer root-id array, got {ids.dtype}")
    return np.sort(ids.astype(np.int64))


def iter_feather_chunks(path: str | Path, chunk_rows: int = 2_000_000,
                        columns: tuple[str, ...] | None = None) -> Iterator[pd.DataFrame]:
    """Yield a feather file in row chunks so an 850 MB / 100M-row table never has to be
    materialised at once (the release notes recommend this pattern)."""
    table = feather.read_table(path, columns=list(columns) if columns else None)
    for start in range(0, table.num_rows, chunk_rows):
        yield table.slice(start, min(chunk_rows, table.num_rows - start)).to_pandas()


def check_connections_schema(df: pd.DataFrame) -> None:
    missing = [c for c in CONNECTIONS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"connections table missing columns: {missing}")


def load_annotations(path: str | Path) -> pd.DataFrame:
    """Schlegel et al. 2024 whole-brain annotation table (Supplemental file 1)."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "root_id" not in df.columns:
        raise ValueError(f"{path}: expected a 'root_id' column, got {list(df.columns)}")
    df["root_id"] = df["root_id"].astype(np.int64)
    return df


def load_synapses(path: str | Path, chunk_rows: int = 5_000_000) -> Iterator[pd.DataFrame]:
    """The ~130M-row individual-synapse point cloud (only fetched with --include-synapses)."""
    yield from iter_feather_chunks(
        path, chunk_rows,
        columns=("pre_pt_root_id", "post_pt_root_id", "neuropil",
                 "pre_pt_position_x", "pre_pt_position_y", "pre_pt_position_z",
                 "post_pt_position_x", "post_pt_position_y", "post_pt_position_z"),
    )
