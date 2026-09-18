# FlyScale dataset: sources, conventions, and the Phase 0 gate

Technical reference for the canonical dataset. The narrative version is in the
[README](../README.md); the scope document is [`PROJECT-VYBFLY.md`](../PROJECT-VYBFLY.md).

## Data sources

Source: **FlyWire FAFB v783** whole-brain connectome (139,255 neurons, 54.5M synapses).

| what | where |
|---|---|
| Connectivity | Zenodo [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866) (CC-BY-4.0) — `proofread_connections_783.feather` (one row per neuron pair per neuropil), `proofread_root_ids_783.npy`, per-neuron neuropil counts |
| Annotations | `github.com/flyconnectome/flywire_annotations` Supplemental file 1 (Schlegel et al. 2024) — cell type, superclass, neurotransmitter, lineage, side |
| Reference code | Zenodo [10.5281/zenodo.12572930](https://doi.org/10.5281/zenodo.12572930) — the published network-statistics analysis scripts (Lin et al. 2024), used for definition alignment. Not committed; the values extracted from it are in `references/flywire_published_values.json` |

All files are size- and md5-verified against the published checksums; the retrieval is recorded in
`data/raw/MANIFEST.json`. The 9.5 GB individual-synapse point cloud is deliberately not fetched
(the geometry phase does not need it).

## Connectivity convention (important)

The published network analyses threshold connections at **5 synapses per connection**
(Lin et al. 2024; Dorkenwald et al. 2024). The raw v783 release is *not* thresholded: it contains
15,091,983 unique pairs at threshold 1. The convention was adopted because the data confirms it:

| threshold | connections | mean synapses/connection | reciprocity |
|---|---|---|---|
| 1 | 15,091,983 | 3.611 | 0.2660 |
| 5 | 2,700,513 | **12.647** | **0.1398** |
| published (v630, thr 5) | 2,613,129 | **12.6** | **0.138** |

The agreement at threshold 5 on mean connection strength and reciprocity is the evidence that the
canonical pair aggregation (summing synapses across neuropils) is correct. All metrics are computed
on the threshold-5 view, with a threshold-robustness block recorded alongside.

Known open discrepancy: the codex v783 dataset card reports 3,732,460 connections, which no synapse
threshold reproduces exactly from the release (threshold 4 gives 3,537,227). It is recorded as an
unresolved convention in `references/flywire_published_values.json` rather than silently dropped.

## Canonical dataset

`data/processed/canonical_v783/` (git-ignored; rebuild with `scripts/phase0_build.py`; see
`flyscale.connectome` for the full schema):

| file | contents |
|---|---|
| `neurons.parquet` | 139,255 rows, index order = ascending root id, joined annotations |
| `pairs.parquet` | unique directed pairs with synapse count and dominant neurotransmitter |
| `edges.parquet` | one row per (pre, post, neuropil) exactly as published |
| `csr.npz` | outgoing/incoming CSR over the pair graph |
| `neuropils.json`, `meta.json` | label codes and provenance/counts/conventions |
| `bin/` | flat little-endian binary for the Vyb loader (`docs/VYB-PORT.md`) |

## Phase 0 gate result (v783, threshold 5 unless noted)

**11/11 required checks pass**, 7/9 advisory (`results/phase0/gate_result.json`). Where the
reference implementation and the published paper agree:

| quantity | ours (v783) | published (v630) | Δ |
|---|---|---|---|
| neurons | 139,255 | 139,255 (v783) | 0 |
| mean synapses / connection | 12.6471 | 12.6 | +0.4% |
| connection reciprocity | 0.13977 | 0.138 | +1.3% |
| mean directed path, giant SCC | 4.4558 | 4.42 | +0.8% |
| giant SCC size | 119,756 | 119,404 | +0.3% |
| rich-club connection probability | 0.0008681 | 0.000870 | -0.2% |
| neurons with total degree > 37 | 40,412 (29.0%) | 40,218 (~30%) | +0.5% |
| mean undirected path, giant WCC | 3.9972 | 3.91 | +2.2% |
| global clustering (undirected) | 0.044762 | 0.0477 | -6.2% |
| in/out degree correlation | 0.7415 | 0.76 | -2.4% |
| neuropil labels | 79 | 78 | +1 |

Deliberately *not* matching, and recorded rather than hidden: the codex v783 "3,732,460
connections" (no synapse threshold reproduces it — threshold 4 gives 3,537,227), and the SCC/WCC
*fractions* (0.860 / 0.951 vs 0.933 / 0.988), which fall because v783 has ~11k extra sparsely
connected neurons while the giant components themselves are the same absolute size.

Also computed: triad census (450.06 trillion triads, 177,055 fully connected), Leiden communities
(giant WCC: 20 communities, modularity 0.693), threshold-robustness sweep (1/2/3/4/5/10 synapses),
neuropil-to-neuropil and cell-type mixing matrices, normalised Laplacian spectrum, degree
distributions with Gini and tail exponents.

## Reproducing the dataset

```sh
cd ~/Projects/VybFly
uv venv .venv --python 3.12
.venv/bin/python scripts/fetch_data.py            # ~1.2 GB, md5-verified
.venv/bin/python scripts/phase0_build.py          # -> data/processed/canonical_v783/
.venv/bin/python scripts/phase0_metrics.py        # add --heavy for triad census + rich-club null
.venv/bin/python scripts/phase0_gate.py           # -> results/phase0/gate_result.json
.venv/bin/python scripts/export_bin.py            # flat binary for the Vyb loader
```

The Python package is the *reference oracle*; production execution moves to Vyb
(see `docs/VYB-PORT.md`).
