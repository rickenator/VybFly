"""Phase 8 / M11 — mushroom-body associative learning on the FlyWire v783 connectome.

PROJECT-VYBFLY.md §14 asks for biologically relevant learning mechanisms, prioritizing the
mushroom body (MB), with Kenyon cells (KC) coding stimuli sparsely and dopaminergic circuits
carrying reward/valence.  This module implements exactly that, on the *real* MB subgraph of
the canonical v783 dataset:

  ALPN (antennal-lobe projection neurons, 685)       -- real, sparse, syn-count weighted
      |
      v   sparse Kenyon-cell code  (k-winner-take-all, fixed sparsity)
  KC   (Kenyon cells, 5177)                          -- the plastic pathway
      |
      v   reward-modulated Hebbian (three-factor) plasticity, dopamine-gated
  MBON (mushroom-body output neurons, 96)
      ^
      |   dopamine-like teaching signal, routed through the real DAN wiring
  DAN  (PAM = reward, PPL1/PPL2 = punishment, 331)

Everything that shapes the computation is taken from the connectome: which KCs a given MBON
listens to, the relative synaptic weights, which MBONs a reward DAN can gate, and the
ALPN->KC fan-in that turns an odor into a sparse KC code.  The only synthetic element is
the odor -> glomerulus input pattern (no odor data exists in FlyWire), which is drawn as a
sparse random pattern over the 56 glomeruli defined by the uniglomerular projection neurons.

Design constraints from the scope document:

  * "Do not permit learning rules to destroy the source architecture without explicit
    experimental reason."  -> plasticity is *masked* to synapses that exist in the
    connectome at the published 5-synapse threshold.  No synapse is created, none is
    deleted, and the sign/type of every synapse is untouched.  Every run reports the
    measured structural drift so the constraint is auditable.
  * homeostatic / synaptic normalization -> the per-KC weight budget (row sum, i.e. the
    total output synaptic weight of each Kenyon cell onto the MBON layer) is held at its
    anatomical value.  This is the anti-blow-up term.
  * controls that make the result interpretable -> degree-preserving target shuffles of the
    KC->MBON pathway and of the ALPN->KC projection, no-plasticity runs, and a random-weight
    readout, so "training dominated the architecture" (§27) can be quantified rather than
    asserted.

The readout is fixed across every condition (the MBONs the measured reward DANs innervate),
so ablating the dopamine gate or the odor code never moves the goalposts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

# --------------------------------------------------------------------------- labels ----
#: exact annotation vocabulary used to define each population (Schlegel et al. 2024
#: annotations as distributed in the canonical_v783 dataset).
MB_LABELS = {
    "kc": {"column": "cell_class", "value": "Kenyon_Cell"},
    "mbon": {"column": "cell_class", "value": "MBON"},
    "dan": {"column": "cell_class", "value": "DAN"},
    "alpn": {"column": "cell_class", "value": "ALPN"},
    "upn": {"column": "cell_sub_class", "value": "uniglomerular"},
}

#: neuropil codes that belong to the mushroom body proper (calyx, medial lobe, vertical
#: lobe, pedunculus).  CRE/SIP are the MB output tracts and are reported separately.
MB_NEUROPILS = ("MB_CA", "MB_ML", "MB_VL", "MB_PED")

#: nt_code ordering in pairs.parquet / edges.parquet (Eckstein, Bates et al. 2024 argmax)
NT_CODES = ("gaba", "acetylcholine", "glutamate", "octopamine", "serotonin", "dopamine")

#: gaba, ach, glut, oct, ser, da


# ------------------------------------------------------------------ subgraph extraction --
@dataclass
class MBSubgraph:
    """Mushroom-body populations, as global neuron indices into the Connectome."""

    kc: np.ndarray
    mbon: np.ndarray
    dan: np.ndarray
    pam: np.ndarray          # reward dopaminergic (PAM*) subset of dan
    ppl: np.ndarray          # punishment dopaminergic (PPL1/PPL2) subset of dan
    alpn: np.ndarray
    upn: np.ndarray          # uniglomerular projection neurons (define the glomeruli)
    channel_of_alpn: np.ndarray   # glomerulus id per ALPN, -1 if the ALPN has none
    channel_names: list[str]
    labels: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    @property
    def n_kc(self) -> int:
        return int(self.kc.size)


def extract_mb_subgraph(c, threshold: int = 5, alpn_channel_source: str = "uniglomerular"
                        ) -> MBSubgraph:
    """Pull the mushroom-body populations out of the canonical connectome annotations.

    `c` is a flyscale.connectome.Connectome.  Returns global indices (the `idx` column),
    which are the indices used by pairs.parquet / edges.parquet.  `threshold` is only used
    to document which published convention the caller intends; the populations themselves
    are annotation-only and do not depend on it.
    """
    n = c.neurons
    ann = n.set_index("idx")

    def pop(column: str, value: str) -> np.ndarray:
        m = ann[column].astype(str).to_numpy() == value
        return np.flatnonzero(m).astype(np.int64)

    kc = pop(*MB_LABELS["kc"].values())
    mbon = pop(*MB_LABELS["mbon"].values())
    dan = pop(*MB_LABELS["dan"].values())
    alpn = pop(*MB_LABELS["alpn"].values())
    upn = np.intersect1d(pop(*MB_LABELS["upn"].values()), alpn)

    ct = ann["cell_type"].astype(str)
    pam = np.intersect1d(dan, np.flatnonzero(ct.str.startswith("PAM").to_numpy()))
    ppl = np.intersect1d(dan, np.flatnonzero(ct.str.startswith("PPL").to_numpy()))

    # glomerulus channel per ALPN.  A uniglomerular PN's cell_type is "<glomerulus>_<class>"
    # (e.g. DA1_lPN, DL2d_adPN, VP1l+_lvPN).  Multiglomerular cells are only assigned to a
    # channel when their prefix is already in the glomerulus vocabulary (e.g. DM3, DA4m);
    # the CB*/M_* cells are genuinely multi-glomerular and stay unassigned.
    names = sorted({str(x).split("_")[0] for x in ct.to_numpy()[upn]})
    index_of = {name: i for i, name in enumerate(names)}
    channel_of_alpn = np.full(alpn.size, -1, dtype=np.int64)
    pos = {int(a): i for i, a in enumerate(alpn)}
    for a in alpn:
        prefix = str(ann["cell_type"].iat[a]).split("_")[0]
        if prefix in index_of:
            channel_of_alpn[pos[int(a)]] = index_of[prefix]
    if alpn_channel_source == "all":
        # every ALPN becomes its own channel (fallback, keeps multiglomerular cells as
        # separate input units).  Not used by default; kept so the choice is auditable.
        extra = sorted({str(ann["cell_type"].iat[int(a)]).split("_")[0] for a in alpn})
        index_of = {name: i for i, name in enumerate(extra)}
        names = extra
        channel_of_alpn = np.array(
            [index_of[str(ann["cell_type"].iat[int(a)]).split("_")[0]] for a in alpn],
            dtype=np.int64)

    labels = {
        "kc": {"cell_class": "Kenyon_Cell", "cell_types": sorted(set(str(x) for x in ct.to_numpy()[kc])),
               "cell_sub_classes": sorted(set(str(x) for x in ann["cell_sub_class"].astype(str).to_numpy()[kc]))},
        "mbon": {"cell_class": "MBON", "cell_types": sorted(set(str(x) for x in ct.to_numpy()[mbon]))},
        "dan": {"cell_class": "DAN",
                "cell_types": sorted(set(str(x) for x in ct.to_numpy()[dan]))},
        "pam": {"cell_type prefix": "PAM"},
        "ppl": {"cell_type prefix": "PPL"},
        "alpn": {"cell_class": "ALPN",
                 "cell_sub_classes": sorted(set(
                     str(x) for x in ann["cell_sub_class"].astype(str).to_numpy()[alpn]))},
        "upn": {"cell_sub_class": "uniglomerular", "n_glomeruli": len(names)},
    }
    counts = {k: int(v.size) for k, v in
              dict(kc=kc, mbon=mbon, dan=dan, pam=pam, ppl=ppl, alpn=alpn, upn=upn).items()}
    return MBSubgraph(kc=kc, mbon=mbon, dan=dan, pam=pam, ppl=ppl, alpn=alpn, upn=upn,
                      channel_of_alpn=channel_of_alpn, channel_names=names,
                      labels=labels, counts=counts)


def mb_pair_stats(c, sub: MBSubgraph, threshold: int = 5,
                  neuropils: dict | None = None) -> dict:
    """Measured synapse statistics of the extracted MB subgraph at `threshold` synapses."""
    p = c.pairs
    pre = p["pre_idx"].to_numpy()
    post = p["post_idx"].to_numpy()
    syn = p["syn_count"].to_numpy()
    nt = p["nt_code"].to_numpy()
    keep = syn >= int(threshold)
    pre, post, syn, nt = pre[keep], post[keep], syn[keep], nt[keep]

    def block(a: np.ndarray, b: np.ndarray, name: str) -> dict:
        m = np.isin(pre, a) & np.isin(post, b)
        return {"pathway": name, "pairs": int(m.sum()), "synapses": int(syn[m].sum()),
                "nt_codes": {NT_CODES[k]: int(v) for k, v in
                             zip(*np.unique(nt[m], return_counts=True))} if m.any() else {}}

    out = {
        "threshold_synapses": int(threshold),
        "KC->MBON": block(sub.kc, sub.mbon, "KC->MBON"),
        "ALPN->KC": block(sub.alpn, sub.kc, "ALPN->KC"),
        "DAN->MBON": block(sub.dan, sub.mbon, "DAN->MBON"),
        "PAM->MBON": block(sub.pam, sub.mbon, "PAM->MBON"),
        "PPL->MBON": block(sub.ppl, sub.mbon, "PPL->MBON"),
        "DAN->KC": block(sub.dan, sub.kc, "DAN->KC"),
        "KC->KC": block(sub.kc, sub.kc, "KC->KC"),
    }
    m = np.isin(pre, sub.kc) & np.isin(post, sub.mbon)
    if m.any():
        posts = post[m]
        _, cnt = np.unique(posts, return_counts=True)
        pres = pre[m]
        _, kcnt = np.unique(pres, return_counts=True)
        out["KC->MBON"]["MBONs_with_input"] = int(cnt.size)
        out["KC->MBON"]["KCs_with_output"] = int(kcnt.size)
        out["KC->MBON"]["KCs_per_MBON_mean"] = float(cnt.mean())
        out["KC->MBON"]["MBONs_per_KC_mean"] = float(kcnt.mean())

    if neuropils is not None:
        e = c.edges
        epre = e["pre_idx"].to_numpy()
        epost = e["post_idx"].to_numpy()
        ec = e["neuropil_code"].to_numpy()
        es = e["syn_count"].to_numpy()
        me = (es >= int(threshold)) & np.isin(epre, sub.kc) & np.isin(epost, sub.mbon)
        name = neuropils["code_to_name"]
        vv = {name[k]: int(v) for k, v in zip(*np.unique(ec[me], return_counts=True))}
        out["KC->MBON"]["by_neuropil_rows"] = vv
        out["KC->MBON"]["mb_neuropil_rows"] = {
            k: v for k, v in vv.items()
            if any(k.startswith(nm) for nm in MB_NEUROPILS)}
    return out


# --------------------------------------------------------------- graph construction ----
def shuffle_bipartite_targets(pre: np.ndarray, post: np.ndarray, syn: np.ndarray,
                              n_swaps: int, seed: int = 0,
                              n_columns: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Degree-preserving target randomisation of a bipartite graph, by edge swaps.

    Repeatedly pick two edges and exchange their targets; a swap is accepted only when it
    creates no duplicate (pre, post) pair.  The out-degree of every `pre` node and the
    in-degree of every `post` node are therefore *exactly* preserved, the number of edges is
    unchanged, and the synapses attached to an edge follow the edge.  This is the standard
    configuration-model randomisation used for connectome null models, and it is the
    shuffled-connectivity control the Phase 8 protocol asks for.
    """
    rng = np.random.default_rng(seed)
    pre = np.asarray(pre, dtype=np.int64).copy()
    post = np.asarray(post, dtype=np.int64).copy()
    syn = np.asarray(syn).copy()
    m = pre.size
    if m < 2 or n_swaps <= 0:
        return pre, post, syn, {"swaps_attempted": 0, "swaps_accepted": 0, "duplicates": 0}
    n_col = int(n_columns) if n_columns is not None else int(post.max()) + 1
    key = pre * n_col + post
    occupied = set(key.tolist())
    a = rng.integers(0, m, size=n_swaps)
    b = rng.integers(0, m, size=n_swaps)
    accepted = 0
    for i, j in zip(a.tolist(), b.tolist()):
        if i == j:
            continue
        pi, pj = int(pre[i]), int(pre[j])
        qi, qj = int(post[i]), int(post[j])
        if pi == pj or qi == qj:
            continue
        ka, kb = pi * n_col + qj, pj * n_col + qi
        if ka in occupied or kb in occupied:
            continue
        occupied.discard(key[i])
        occupied.discard(key[j])
        post[i], post[j] = qj, qi
        key[i], key[j] = ka, kb
        occupied.add(ka)
        occupied.add(kb)
        accepted += 1
    dup = int(m - np.unique(key).size)
    return pre, post, syn, {"swaps_attempted": int(n_swaps), "swaps_accepted": int(accepted),
                            "duplicate_pairs": dup, "edges": int(m)}


@dataclass
class MBGraph:
    """The model mushroom body: real fan-in, real synapses, optionally randomised wiring."""

    W0: np.ndarray                 # (n_kc, n_mbon) anatomical synapses, dense (496k cells)
    mask: np.ndarray               # W0 > 0 : exactly the synapses plasticity may touch
    A: sparse.csr_matrix           # (n_alpn, n_kc) row-normalized ALPN->KC drive
    channel_of_alpn: np.ndarray
    channel_names: list[str]
    pam_gate: np.ndarray           # (n_mbon,) MBONs reachable by a reward (PAM) DAN
    ppl_gate: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_kc(self) -> int:
        return int(self.W0.shape[0])

    @property
    def n_mbon(self) -> int:
        return int(self.W0.shape[1])


def _block_matrix(c, threshold: int, pre_ids: np.ndarray, post_ids: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synapse-count matrix restricted to (pre in pre_ids) x (post in post_ids)."""
    p = c.pairs
    pm = p["syn_count"].to_numpy() >= int(threshold)
    pre = p["pre_idx"].to_numpy()[pm]
    post = p["post_idx"].to_numpy()[pm]
    syn = p["syn_count"].to_numpy()[pm]
    row_of = pd.Series(np.arange(pre_ids.size), index=pre_ids)
    col_of = pd.Series(np.arange(post_ids.size), index=post_ids)
    keep = np.isin(pre, pre_ids) & np.isin(post, post_ids)
    r = row_of.reindex(pre[keep]).to_numpy()
    cc = col_of.reindex(post[keep]).to_numpy()
    return r.astype(np.int64), cc.astype(np.int64), syn[keep].astype(np.float64)


def build_mb_graph(c, sub: MBSubgraph, threshold: int = 5, *, seed: int = 0,
                   shuffle_kc_mbon: bool = False, shuffle_alpn_kc: bool = False,
                   shuffle_dan_gate: bool = False, permute_weights: bool = False,
                   n_swap_factor: int = 10) -> MBGraph:
    """Assemble the model MB, optionally under one of the control manipulations."""
    rk, ck, sk = _block_matrix(c, threshold, sub.kc, sub.mbon)
    ra, ca, sa = _block_matrix(c, threshold, sub.alpn, sub.kc)
    rp, cp, sp = _block_matrix(c, threshold, sub.pam, sub.mbon)
    rpp, cpp, spp = _block_matrix(c, threshold, sub.ppl, sub.mbon)
    meta: dict = {"threshold": int(threshold), "shuffle_kc_mbon": shuffle_kc_mbon,
                  "shuffle_alpn_kc": shuffle_alpn_kc, "shuffle_dan_gate": shuffle_dan_gate,
                  "permute_weights": permute_weights, "seed": int(seed)}

    if shuffle_kc_mbon:
        rk2, ck2, sk2, info = shuffle_bipartite_targets(
            rk, ck, sk, n_swap_factor * rk.size, seed=seed + 11, n_columns=sub.mbon.size)
        meta["kc_mbon_shuffle"] = info
        rk, ck, sk = rk2, ck2, sk2

    if permute_weights:
        rng = np.random.default_rng(seed + 23)
        sk = rng.permutation(sk)
        meta["weights_permuted"] = True

    W0 = np.zeros((sub.kc.size, sub.mbon.size), dtype=np.float64)
    np.add.at(W0, (rk, ck), sk)

    if shuffle_alpn_kc:
        ra2, ca2, sa2, info = shuffle_bipartite_targets(
            ra, ca, sa, n_swap_factor * ra.size, seed=seed + 37, n_columns=sub.kc.size)
        meta["alpn_kc_shuffle"] = info
        ra, ca, sa = ra2, ca2, sa2

    A = sparse.csr_matrix((sa, (ra, ca)), shape=(sub.alpn.size, sub.kc.size))
    rs = np.asarray(A.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    A = sparse.diags(1.0 / rs) @ A

    pam_gate = np.zeros(sub.mbon.size, dtype=bool)
    pam_gate[cp] = True
    ppl_gate = np.zeros(sub.mbon.size, dtype=bool)
    ppl_gate[cpp] = True
    if shuffle_dan_gate:
        _, cp2, _, info = shuffle_bipartite_targets(
            rp, cp, sp, n_swap_factor * rp.size, seed=seed + 53, n_columns=sub.mbon.size)
        # edge-swap randomisation of the PAM->MBON bipartite graph; the gate is rebuilt from
        # the randomised in-degree (how many reward-DAN edges each MBON receives), so the
        # number of reward DAN synapses and the DAN out-degrees are exactly preserved while
        # *which* MBONs the reward teaching signal can reach is randomised.
        gate_counts = np.bincount(cp2, minlength=sub.mbon.size)
        pam_gate = gate_counts > 0
        meta["dan_gate_shuffle"] = info
        meta["dan_gate_shuffled_mbons"] = int(pam_gate.sum())
    meta["pam_gate_mbons"] = int(pam_gate.sum())
    meta["ppl_gate_mbons"] = int(ppl_gate.sum())
    meta["kc_mbon_edges"] = int(rk.size)
    meta["kc_mbon_synapses"] = float(sk.sum())
    meta["alpn_kc_edges"] = int(ra.size)
    meta["pam_mbon_edges"] = int(rp.size)
    return MBGraph(W0=W0, mask=W0 > 0, A=A, channel_of_alpn=sub.channel_of_alpn,
                   channel_names=sub.channel_names, pam_gate=pam_gate, ppl_gate=ppl_gate,
                   meta=meta)


# ------------------------------------------------------------------------ odor coding ----
@dataclass
class OdorCodes:
    X: np.ndarray                  # (n_odors, n_kc) binary sparse KC codes
    channels: np.ndarray           # (n_odors, channels_per_odor) active glomeruli
    active_per_odor: np.ndarray
    info: dict = field(default_factory=dict)


def odor_codes(g: MBGraph, n_odors: int, *, sparsity: float = 0.05,
               channels_per_odor: int = 5, seed: int = 0, gain_range=(0.5, 1.0),
               lognormal_sigma: float = 0.25) -> OdorCodes:
    """Sparse Kenyon-cell codes for synthetic odorants.

    An odor is a sparse pattern over glomeruli (the real antennal-lobe channels: each
    glomerulus is the set of uniglomerular projection neurons carrying that glomerulus
    name).  That pattern is pushed through the *measured* ALPN->KC synaptic fan-in (row
    normalized, so each active projection neuron contributes unit drive), scaled by a
    per-Kenyon-cell log-normal gain, and then passed through a k-winner-take-all with a
    fixed active fraction.  The k-WTA is what makes the code sparse in the same way real
    Kenyon cells are: a fixed number of cells fire per stimulus, chosen by relative drive.
    """
    rng = np.random.default_rng(seed)
    n_kc = g.n_kc
    k = max(1, int(round(sparsity * n_kc)))
    n_ch = len(g.channel_names)
    cpk = int(min(channels_per_odor, n_ch))
    X = np.zeros((n_odors, n_kc), dtype=np.float64)
    channels = np.zeros((n_odors, cpk), dtype=np.int64)
    odour_gain = np.zeros((n_odors, cpk), dtype=np.float64)
    for o in range(n_odors):
        ch = rng.choice(n_ch, size=cpk, replace=False)
        channels[o] = ch
        gains = rng.uniform(gain_range[0], gain_range[1], size=cpk)
        odour_gain[o] = gains
        u = np.zeros(g.A.shape[0], dtype=np.float64)
        for c_i, gain in zip(ch.tolist(), gains.tolist()):
            members = np.flatnonzero(g.channel_of_alpn == c_i)
            if members.size:
                u[members] = gain
        drive = np.asarray(g.A.T @ u).ravel()
        drive = np.maximum(drive, 0.0) * rng.lognormal(0.0, lognormal_sigma, size=n_kc)
        idx = np.argpartition(-drive, k - 1)[:k]
        X[o, idx] = 1.0
    info = {"sparsity": float(sparsity), "active_per_odor": int(k), "n_kc": n_kc,
            "channels_per_odor": cpk, "n_channels": n_ch,
            "gain_range": list(gain_range), "lognormal_sigma": float(lognormal_sigma),
            "seed": int(seed)}
    return OdorCodes(X=X, channels=channels, active_per_odor=np.full(n_odors, k),
                     info=info)


# --------------------------------------------------------------------- plasticity rule ----
@dataclass
class PlasticityConfig:
    """Reward-modulated Hebbian (three-factor) plasticity with a dopamine gate.

        eligibility  e_ij = x_i * y_j
        update       dw_ij = eta * da * g_j * e_ij * (w_ij / y_ref if multiplicative else 1)
        applied      w_ij <- w_ij + dw_ij           (only where the synapse exists)
        bounded      w_ij <- clip(w_ij, lo * w0_ij, hi * w0_ij)
        homeostatic  row sums <- anatomical row sums

    `da` is the scalar dopamine-like teaching signal, `g_j` the anatomical dopamine gate
    (1 where a DAN actually connects to MBON j, else 0), which routes the teaching signal to
    a specific set of MBON dendrites instead of broadcasting it.  `gate` and `pool` are
    deliberately separate: `gate` says which synapses the teaching signal can reach, `pool`
    says which MBONs the behavioral readout is taken from, so the gate ablation leaves the
    readout untouched.
    """

    eta: float = 0.5
    plasticity: bool = True
    da_mode: str = "reward_only"    # 'reward_only' | 'rpe' | 'signed'
    gate: str = "pam"               # 'pam' | 'ppl' | 'both' | 'all'
    pool: str = "pam"               # READOUT pool: 'pam' (anatomical PAM-innervated MBONs)
    #                                 | 'ppl' | 'all'; independent of `gate`, which only
    #                                 sets which synapses the dopamine signal can reach
    normalize: bool = True
    max_factor: float | None = None
    min_factor: float | None = None
    punish_gain: float = 1.0
    code_jitter: float = 0.0
    """Fraction of the active Kenyon-cell set re-drawn on each *presentation* of an odor.

    Kenyon-cell responses to the same odor vary from trial to trial (the PN->KC drive is
    noisy), so learning has to average over presentations rather than being complete after
    one.  0.0 makes the codes deterministic and acquisition effectively single-trial."""

    rule: str = "multiplicative"
    """'multiplicative' (default) or 'additive'.

    multiplicative:  dw_ij = eta * da * g_j * x_i * (y_j / y_ref) * w_ij
    additive:        dw_ij = eta * da * g_j * x_i * y_j

    Both are three-factor (pre x post x dopamine) Hebbian rules; the multiplicative form is
    the soft-bound / synaptic-scaling variant, in which eta is the *fraction* of the current
    weight deposited per rewarded presentation and the acquisition time constant is ~1/eta
    presentations.  The additive form saturates within a handful of presentations at any eta
    because the raw MBON activity (order 100) exceeds the per-synapse weight (order 10)."""


class MBLearner:
    """Weight state + the associative-learning protocol on top of an MBGraph."""

    def __init__(self, g: MBGraph, cfg: PlasticityConfig, code: OdorCodes, seed: int = 0):
        self.g = g
        self.cfg = cfg
        self.code = code
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.W = g.W0.copy()
        self.W_start = g.W0.copy()
        self.row_budget = g.W0.sum(axis=1)          # anatomical per-KC output weight
        self.row_active = self.row_budget > 0
        self.n_updates = 0
        if cfg.gate == "pam":
            self.da_gate = g.pam_gate
        elif cfg.gate == "ppl":
            self.da_gate = g.ppl_gate
        elif cfg.gate == "both":
            self.da_gate = g.pam_gate | g.ppl_gate
        else:
            self.da_gate = np.ones(g.n_mbon, dtype=bool)
        # The readout is a property of the connectome, not of the plasticity manipulation:
        # it is the anatomically PAM-innervated MBON pool for every condition.  Only the
        # `all` setting (a readout-sensitivity check) or an explicit alternative changes it,
        # so ablating the dopamine gate does not simultaneously move the goalposts.
        if cfg.pool == "all" or not g.pam_gate.any():
            self.readout_pool = np.ones(g.n_mbon, dtype=bool)
        elif cfg.pool == "ppl":
            self.readout_pool = g.ppl_gate.copy()
        else:
            self.readout_pool = g.pam_gate.copy()
        self.da_sign = (g.pam_gate.astype(float) - cfg.punish_gain * g.ppl_gate.astype(float)
                        if cfg.da_mode == "signed" else np.ones(g.n_mbon))
        # RPE mode: the dopamine signal must be in the same units as the reward, so the
        # readout is normalized by its own untrained maximum response over the odor set.
        self.value_scale = 1.0
        if cfg.da_mode == "rpe":
            y = self.response()
            v = float(np.max(y[:, self._pool("readout")].mean(axis=1)))
            self.value_scale = v if v > 1e-9 else 1.0
        # multiplicative rule: the update is expressed as a fraction of the current weight,
        # so the post-synaptic activity is normalized by its own untrained mean level.
        self.value_ref = 1.0
        if cfg.rule == "multiplicative":
            y = self.response()
            v = float(y[:, self._pool("readout")].mean())
            self.value_ref = v if v > 1e-9 else 1.0
        # the untrained readout, used to report the *learned* component of any performance
        # change separately from the innate bias of the naive connectome readout
        self.value_initial = self.value(pool="readout")

    # -- forward pass -------------------------------------------------------------------
    def response(self, X: np.ndarray | None = None) -> np.ndarray:
        """MBON activity y = W^T x for every odor (n_odors, n_mbon)."""
        X = self.code.X if X is None else X
        return X @ self.W

    def value(self, X: np.ndarray | None = None, pool: str = "readout") -> np.ndarray:
        """Scalar learned value per odor: mean MBON activity over the readout pool."""
        y = self.response(X)
        m = self._pool(pool)
        return y[:, m].mean(axis=1)

    def _pool(self, pool: str = "readout") -> np.ndarray:
        if pool in ("readout", "readout_pool"):
            return self.readout_pool
        if pool == "all":
            return np.ones(self.g.n_mbon, dtype=bool)
        if pool == "pam":                       # anatomical PAM gate, whatever the cfg says
            return (self.g.pam_gate.copy() if self.g.pam_gate.any()
                    else np.ones(self.g.n_mbon, dtype=bool))
        raise ValueError(f"unknown readout pool {pool!r}")

    def jitter(self, x: np.ndarray) -> np.ndarray:
        """One noisy presentation of a code: redraw `code_jitter` of the active KC set.

        The active count is preserved, so sparsity is unchanged; only *which* Kenyon cells
        carry the stimulus varies between presentations, which is the trial-to-trial
        variability a biological KC population shows.
        """
        active = np.flatnonzero(x)
        n_j = int(round(self.cfg.code_jitter * active.size))
        if n_j <= 0 or active.size == 0:
            return x
        x = x.copy()
        drop = self.rng.choice(active, size=n_j, replace=False)
        add = self.rng.choice(self.g.n_kc, size=n_j, replace=False)
        x[drop] = 0.0
        x[add] = 1.0
        return x

    # -- learning -----------------------------------------------------------------------
    def reward_update(self, o: int, da_scale: float = 1.0) -> float:
        """One three-factor update on odor `o` with teaching signal `da`.

        Only Kenyon cells that are active for this odor are touched (x is sparse), so the
        update is O(active KCs x MBONs) rather than O(n_kc x n_mbon).
        """
        cfg = self.cfg
        if not cfg.plasticity:
            return 0.0
        x = self.code.X[o]
        if cfg.code_jitter > 0.0:
            x = self.jitter(x)
        active = np.flatnonzero(x)
        if active.size == 0:
            self.n_updates += 1
            return 0.0
        xa = x[active]
        sub = self.W[active]                            # (n_active, n_mbon) copy
        y = sub.T @ xa                                  # pre-update MBON activity
        da = float(da_scale)
        if cfg.da_mode == "rpe":
            pool = self._pool("readout")
            da = da - float(y[pool].mean()) / self.value_scale
        self.n_updates += 1
        if da == 0.0:
            return 0.0
        mask = self.g.mask[active]
        if cfg.rule == "multiplicative":
            yhat = y / self.value_ref                       # normalized post-synaptic activity
            fac = 1.0 + cfg.eta * da * self.da_gate[None, :] * self.da_sign[None, :] * yhat[None, :]
            fac = np.clip(fac, 0.0, 10.0)
            sub = np.where(mask, sub * fac, 0.0)
        else:
            g = cfg.eta * da * self.da_gate[None, :] * self.da_sign[None, :] * mask
            sub = np.maximum(sub + y[None, :] * g, 0.0)
        if cfg.max_factor is not None:
            sub = np.minimum(sub, cfg.max_factor * self.g.W0[active])
        if cfg.min_factor is not None:
            sub = np.maximum(sub, cfg.min_factor * self.g.W0[active])
        if cfg.normalize:
            budget = self.row_budget[active]
            rs = sub.sum(axis=1)
            scale = np.ones_like(rs)
            nz = budget > 0
            scale[nz] = budget[nz] / np.maximum(rs[nz], 1e-12)
            sub = sub * scale[:, None]
        self.W[active] = sub
        return da

    def drift(self) -> dict:
        """Measured structural drift: how much of the source architecture survived."""
        m = self.g.mask
        w0 = self.g.W0[m]
        w1 = self.W[m]
        ratio = w1 / np.maximum(w0, 1e-12)
        row0 = self.g.W0.sum(axis=1)[self.row_active]
        row1 = self.W.sum(axis=1)[self.row_active]
        from scipy.stats import spearmanr
        rho = float(spearmanr(w0, w1).statistic) if w0.size > 2 else float("nan")
        return {
            "edges_preserved": bool(np.array_equal(self.W > 0, self.g.W0 > 0)),
            "synapses_created": int(np.sum((self.W > 0) & ~m)),
            "synapses_lost": int(np.sum((self.W <= 0) & m)),
            "mean_abs_relative_change": float(np.mean(np.abs(w1 - w0) / np.maximum(w0, 1e-12))),
            "median_weight_ratio": float(np.median(ratio)),
            "fraction_beyond_2x": float(np.mean(ratio > 2.0)),
            "fraction_below_half": float(np.mean(ratio < 0.5)),
            "weight_spearman_initial_vs_final": rho,
            "row_budget_max_rel_error": float(np.max(np.abs(row1 - row0) / np.maximum(row0, 1e-12))),
        }


# ------------------------------------------------------------------------- protocols ----
def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-aware rank separation: fraction of (pos, neg) pairs correctly ordered, ties 0.5."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(np.mean((d > 0).astype(float) + 0.5 * (d == 0).astype(float)))


def zero_below_tolerance(d: np.ndarray, ref: np.ndarray, rel: float = 1e-9) -> np.ndarray:
    """Zero changes that are floating-point noise (below `rel` of the reference scale).

    Without this, a condition whose readout is invariant by construction (e.g. a readout
    over the whole MBON layer under per-KC weight normalization) would report an arbitrary
    'learned' rank separation computed from 1e-14 rounding differences instead of the
    correct answer, a tie.
    """
    tol = max(float(np.max(np.abs(ref))) * rel, 1e-12)
    out = np.array(d, dtype=float, copy=True)
    out[np.abs(out) <= tol] = 0.0
    return out


def acquisition(learner: MBLearner, n_odors: int = 8, rewarded=(0,), trials: int = 300,
                probe_every: int = 10, mode: str = "reward", pool: str = "readout",
                seed: int = 0) -> dict:
    """Classic conditioning: one target odor among N, associatively rewarded or punished.

    `mode='reward'` delivers the teaching signal only on target-odor trials (+1), which is
    how reward dopaminergic (PAM) neurons behave; `mode='punish'` delivers -1 on target
    trials, the PPL1 convention.  Returns the acquisition curve measured as (a) the value of
    the target odor, (b) the rank separation between the target and every non-target odor
    (AUC over all target/non-target odor pairs), and (c) the fraction of probes at which the
    target odor is the arg-max of the readout.
    """
    rng = np.random.default_rng(seed + 101)
    rewarded = np.asarray(rewarded, dtype=np.int64)
    X = learner.code.X[:n_odors]
    is_target = np.zeros(n_odors, dtype=bool)
    is_target[rewarded] = True
    target_da = 1.0 if mode == "reward" else -1.0
    curve = {"trials": [], "value_target": [], "value_nontarget": [], "auc": [],
             "auc_learned": [], "argmax_hit": [], "mean_mbon_response_target": [],
             "mean_mbon_response_other": []}
    v0 = learner.value_initial[:n_odors]
    order = rng.integers(0, n_odors, size=trials)
    for t in range(1, trials + 1):
        o = int(order[t - 1])
        learner.reward_update(o, target_da if is_target[o] else 0.0)
        if t % probe_every == 0 or t == trials:
            v = learner.value(X, pool=pool)
            vr = v[rewarded]
            vu = v[~is_target]
            curve["trials"].append(t)
            curve["value_target"].append(float(vr.mean()))
            curve["value_nontarget"].append(float(vu.mean()))
            dv = zero_below_tolerance(v - v0, v)
            curve["auc"].append(rank_auc(vr, vu))
            curve["auc_learned"].append(rank_auc(dv[rewarded], dv[~is_target]))
            curve["argmax_hit"].append(float(np.argmax(v) in set(rewarded.tolist())))
            y = learner.response(X)
            curve["mean_mbon_response_target"].append(float(y[rewarded].mean()))
            curve["mean_mbon_response_other"].append(float(y[~is_target].mean()))
    return curve


def capacity_protocol(learner: MBLearner, n_odors: int = 64, n_rewarded: int = 1,
                      trials: int | None = None, pool: str = "readout", seed: int = 0,
                      probe_every: int | None = 50, da_value: float = 1.0) -> dict:
    """How many odor->reward associations can the same plastic pathway hold at once?

    `n_rewarded` odors out of `n_odors` are rewarded; the network is trained on a random
    interleaved stream and then scored on its ability to rank *every* rewarded odor above
    *every* unrewarded one.  Capacity is read off a sweep of `n_rewarded`.  `da_value` is the
    sign of the teaching signal, so the same protocol also runs as punishment learning (in
    which case `pair_auc_inverted` is the separation in the learned direction).
    """
    if trials is None:
        trials = int(max(200, 60 * n_rewarded))
    if probe_every is None:
        probe_every = int(max(50, trials // 40))
    rng = np.random.default_rng(seed + 313)
    X = learner.code.X[:n_odors]
    v0 = learner.value_initial[:n_odors]
    idx = rng.permutation(n_odors)
    rewarded = np.sort(idx[:n_rewarded])
    is_rewarded = np.zeros(n_odors, dtype=bool)
    is_rewarded[rewarded] = True
    order = rng.integers(0, n_odors, size=trials)
    curve = {"trials": [], "auc": [], "auc_learned": []}
    for t in range(1, trials + 1):
        o = int(order[t - 1])
        learner.reward_update(o, da_value if is_rewarded[o] else 0.0)
        if t % probe_every == 0 or t == trials:
            v = learner.value(X, pool=pool)
            curve["trials"].append(t)
            dv_c = zero_below_tolerance(v - v0, v)
            curve["auc"].append(rank_auc(v[rewarded], v[~is_rewarded]))
            curve["auc_learned"].append(rank_auc(dv_c[rewarded], dv_c[~is_rewarded]))
    v = learner.value(X, pool=pool)
    dv = zero_below_tolerance(v - v0, v)
    pair_auc = rank_auc(v[rewarded], v[~is_rewarded])
    pair_auc_learned = rank_auc(dv[rewarded], dv[~is_rewarded])
    min_rewarded = float(v[rewarded].min())
    max_rewarded = float(v[rewarded].max())
    frac_above = float(np.mean(v[~is_rewarded] < min_rewarded))
    frac_below = float(np.mean(v[~is_rewarded] > max_rewarded))
    return {"n_odors": n_odors, "n_rewarded": n_rewarded, "trials": trials,
            "da_value": float(da_value),
            "rewarded_idx": rewarded.tolist(),
            "pair_auc": pair_auc, "pair_auc_inverted": 1.0 - pair_auc,
            "pair_auc_learned": pair_auc_learned,
            "separation_frac": max(frac_above, frac_below),
            "value_rewarded_mean": float(v[rewarded].mean()),
            "value_unrewarded_mean": float(v[~is_rewarded].mean()),
            "curve": curve}


def subsample_kcs(g: MBGraph, frac: float, seed: int = 0) -> MBGraph:
    """A smaller model mushroom body: a random `frac` of the real Kenyon cells.

    Used for the KC-population-size sweep (learned capacity as a function of the number of
    Kenyon cells).  The MBON layer, the surviving KCs' real synapses and the dopamine gate
    are unchanged; only the KC population is subsampled, and sparsity is held fixed as a
    fraction of the remaining KCs.
    """
    n_keep = int(round(frac * g.n_kc))
    rng = np.random.default_rng(seed + 907)
    idx = np.sort(rng.choice(g.n_kc, size=n_keep, replace=False))
    A = g.A[:, idx]
    rs = np.asarray(A.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    A = sparse.diags(1.0 / rs) @ A
    meta = dict(g.meta)
    meta.update({"kc_fraction": float(frac), "n_kc_kept": int(n_keep),
                 "kc_subsample_seed": int(seed)})
    return MBGraph(W0=g.W0[idx].copy(), mask=g.mask[idx].copy(), A=A.tocsr(),
                   channel_of_alpn=g.channel_of_alpn, channel_names=g.channel_names,
                   pam_gate=g.pam_gate.copy(), ppl_gate=g.ppl_gate.copy(), meta=meta)


def code_overlap(code: OdorCodes) -> dict:
    """Measured KC-code statistics: sparsity, pairwise overlap, and the chance level."""
    X = code.X
    n_kc = X.shape[1]
    k = float(X.sum(axis=1).mean())
    Xb = X > 0
    inter = Xb.astype(np.float64) @ Xb.T.astype(np.float64)
    n = X.shape[0]
    iu = np.triu_indices(n, k=1)
    ov = inter[iu]
    return {"n_odors": int(n), "n_kc": int(n_kc), "active_per_odour": k,
            "sparsity_measured": k / n_kc,
            "mean_pairwise_overlap_cells": float(ov.mean()),
            "mean_pairwise_overlap_frac_of_active": float(ov.mean() / max(k, 1)),
            "expected_overlap_if_independent": float(k * k / n_kc)}
