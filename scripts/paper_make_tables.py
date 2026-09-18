"""Generate the paper's LaTeX tables from results/paper/data.json.

Numbers are formatted here, once, so the prose and the tables cannot drift apart; every table
names its source artifact in the caption.

    python scripts/paper_make_tables.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "paper"
TAB = PAPER / "tables"
TAB.mkdir(parents=True, exist_ok=True)
D = json.loads((PAPER / "data.json").read_text())


def esc(s) -> str:
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace("$", r"\$"))


def cnt(v, fallback="--"):
    """Integer counts: never scientific notation, thousands grouped with a thin space."""
    if v is None:
        return fallback
    try:
        i = int(round(float(v)))
    except (TypeError, ValueError):
        return fallback
    return f"{i:,}".replace(",", r"\,")


def num(v, digits=4, fallback="--", scale=1.0):
    if v is None:
        return fallback
    try:
        f = float(v) * scale
    except (TypeError, ValueError):
        return fallback
    if f == 0:
        return "0"
    if abs(f) >= 1e5 or abs(f) < 1e-3:
        return f"{f:.3g}".replace("e-0", "e-").replace("e+0", "e")
    return f"{f:.{digits}g}" if abs(f) >= 1000 else f"{f:.{digits}f}".rstrip("0").rstrip(".")


def th(*cols) -> str:
    return " & ".join(cols) + r" \\"


def write(name: str, body: str, resize: bool = False) -> None:
    """Emit one LaTeX table, normalised and bounded.

    Wide numeric tables are scaled to the text width so they cannot overflow the margin; a table
    that needs scaling is one whose column count or identifiers grew, which is worth knowing.
    """
    # A non-raw source string silently eats the backslash of any LaTeX command whose first
    # letter is a Python escape (\approx -> BEL+"pprox", \times -> TAB+"imes", \rho -> CR+"ho").
    # Repair exactly those, then normalise any other stray control character.
    for mangled, fixed in (("\x07pprox", r"\approx"), ("\x07lpha", r"\alpha"), ("\x07st", r"\ast"),
                           ("\rho", r"\rho"), ("\times", r"\times"), ("\beta", r"\beta"),
                           ("\nu", r"\nu"), ("\x0ctorall", r"\forall"), ("\x0c", r"\f"),
                           ("\x0b", r"\v"), ("\x08", r"\b")):
        body = body.replace(mangled, fixed)
    body = body.replace("\t", " ")

    bs = chr(92)
    # some cells were authored with doubled backslashes (a raw source string); collapse those
    for cmd in ("rho", "approx", "pm", "alpha", "to", "times", "dagger", "mu", "ge", "le",
                "propto", "ast", "S", "checkmark", "ding"):
        body = body.replace(bs + bs + cmd, bs + cmd)
    for punct in (",", ";", ":", "!"):
        body = body.replace(bs + bs + punct, bs + punct)
    body = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", body)
    if resize:
        body = r"\resizebox{0.96\textwidth}{!}{%" + "\n" + body + "\n}"
    # text-mode specials are fatal in LaTeX (a bare _ is a missing-$ error, % eats the cell):
    # escape them, but leave anything inside $...$ math alone
    out = []
    in_math = False
    for k, ch in enumerate(body):
        if ch == "$" and (k == 0 or body[k - 1] != bs):
            in_math = not in_math
            out.append(ch)
            continue
        if not in_math and ch in "_^%" and (k == 0 or body[k - 1] != bs):
            out.append(bs + ch)
            continue
        out.append(ch)
    body = "".join(out)
    for line in body.split(chr(10)):
        if (line.replace(chr(92) + "$", "").count("$") % 2):
            print(f"  WARNING unbalanced math in {name}: {line[:70]}")
    p = TAB / name
    p.write_text("{\\small\\setlength{\\tabcolsep}{3pt}\n" + body + "\n}\n")
    print(f"  table: {p.relative_to(ROOT)}")


# ------------------------------------------------------------------ dataset
def dataset() -> None:
    ds = D.get("dataset") or {}
    ref = D.get("reference") or {}
    rows = [
        ("neurons (annotated)", cnt(ds.get("n_annotated_neurons")), "canonical counts"),
        ("neurons (in graph)", cnt(ds.get("n_neurons")), "canonical counts"),
        ("edge rows released", cnt(ds.get("n_edges_rows")), "unthresholded"),
        ("connections (all)", cnt(ds.get("n_pairs")), "threshold $\geq 1$ synapse"),
        ("connections (scaling graph)", cnt(ref.get("n_connections")), "threshold $\geq 5$ synapses"),
        ("synapses (all)", cnt(ds.get("n_synapses_pairs")), "threshold $\geq 1$"),
        ("synapses (scaling graph)", cnt(ref.get("n_synapses")), "threshold $\geq 5$"),
        ("autapse pairs", cnt(ds.get("n_autapse_pairs")), "none in v783"),
        ("reciprocal pairs", cnt(ds.get("n_reciprocal_pairs")), "threshold $\geq 1$"),
        ("reciprocity", num(ds.get("reciprocity"), 4), "threshold $\geq 1$"),
        ("mean out-degree", num(ref.get("mean_out_degree"), 4), "scaling graph"),
        ("mean synapses per connection", num(ref.get("mean_synapses_per_connection"), 4),
         "published: 12.6"),
        ("neuropils", cnt(ds.get("n_neuropils")), "annotated regions"),
    ]
    body = [r"\begin{tabular}{l r p{8.0cm}}", r"\toprule",
            r"quantity & value & note \\", r"\midrule"]
    body += [f"{esc(a)} & {b} & {c} \\\\" for a, b, c in rows]
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_dataset.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ gate checks
def gate() -> None:
    checks = D.get("p0_gate_checks") or []
    body = [r"\begin{tabular}{l c p{7.4cm}}", r"\toprule",
            r"check & class & measured vs published \\", r"\midrule"]
    for c in checks:
        cls = "advisory" if c.get("advisory") else "required"
        mark = r"\checkmark" if c.get("passed") else r"\ding{55}"
        body.append(f"{esc(c.get('name'))} & {cls} & {mark}~{esc(c.get('comment'))[:150]} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_gate.tex", "\n".join(body) + "\n", resize=True)


# ------------------------------------------------------------------ geometry
def geometry() -> None:
    rows = D.get("geometry_rows") or []
    body = [r"\begin{tabular}{l r r r r r}", r"\toprule",
            r"geometry & AUC & AP & mean $\log L$ & $R$ & $T$ \\", r"\midrule"]
    for r in rows:
        body.append(" & ".join([
            esc(r["geometry"]).replace(r"\_", " "), num(r.get("auc"), 4), num(r.get("ap"), 4),
            num(r.get("loglik"), 4), num(r.get("R"), 3), num(r.get("T"), 3)]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_geometry.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ downscale
def downscale() -> None:
    body = [r"\begin{tabular}{l l r r r r r r}", r"\toprule",
            r"geometry & scale & neurons & connections & mean deg. & composite & deg.\ Wasserstein & ARI \\",
            r"\midrule"]
    for label, key in (("anatomical xyz", "downscale_anatomical"),
                       ("learned hyperbolic", "downscale_hyperbolic")):
        for r in sorted(D.get(key) or [], key=lambda x: -x["factor"]):
            body.append(" & ".join([
                esc(label), f"{r['factor']:g}$\times$", cnt(r.get("n_neurons")),
                cnt(r.get("n_connections")), num(r.get("mean_out_degree"), 4),
                num(r.get("composite"), 4), num(r.get("degree_wasserstein"), 4),
                num(r.get("ari"), 4)]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_downscale.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ dynamics
def dynamics() -> None:
    ref = D.get("cascade_ref_anatomical") or {}
    body = [r"\begin{tabular}{l r r r r r}", r"\toprule",
            r"scale & final active fraction & ratio to $G_1$ & active-set Jaccard & precision & latency $\Delta$ \\",
            r"\midrule"]
    body.append(" & ".join([r"$G_1$ (biological)", num(ref.get("final_active_fraction"), 4), "--",
                            "--", "--", "--"]) + r" \\")
    for r in sorted(D.get("dynamics_anatomical") or [], key=lambda x: -x["factor"]):
        body.append(" & ".join([
            f"$G_{{{r['factor']:g}}}$", num(r.get("final_active_fraction"), 4),
            num(r.get("final_fraction_ratio"), 4), num(r.get("active_set_jaccard"), 4),
            num(r.get("active_set_precision"), 4), num(r.get("latency_delta_steps"), 0)]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_dynamics.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ upscale
def upscale() -> None:
    ref = D.get("reference") or {}
    body = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
            r"scale & neurons & connections & synapses & mean deg. & mean strength & composite \\",
            r"\midrule"]
    body.append(" & ".join([r"$G_1$", cnt(ref.get("n_neurons")), cnt(ref.get("n_connections")),
                            cnt(ref.get("n_synapses")), num(ref.get("mean_out_degree"), 4),
                            num(ref.get("mean_synapses_per_connection"), 4), "--"]) + r" \\")
    for r in sorted(D.get("upscale") or [], key=lambda x: x["factor"]):
        body.append(" & ".join([
            f"$G_{{{r['factor']:g}}}$", cnt(r.get("n_neurons")), cnt(r.get("n_connections")),
            cnt(r.get("n_synapses")), num(r.get("mean_out_degree"), 4),
            num(r.get("mean_strength"), 4), num(r.get("composite"), 4)]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_upscale.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ closure
def closure() -> None:
    body = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
            r"scale & neurons in $R(G_s)$ & connections in $R(G_s)$ & composite & degree Wasserstein & triad $L_1$ & ARI \\",
            r"\midrule"]
    for r in sorted(D.get("closure") or [], key=lambda x: x["factor"]):
        body.append(" & ".join([
            f"$G_{{{r['factor']:g}}}$", cnt(r.get("n_neurons")), cnt(r.get("n_connections")),
            num(r.get("composite"), 6), num(r.get("degree_wasserstein"), 6),
            num(r.get("motif_l1"), 3), num(r.get("ari"), 6)]) + r" \\")
    body += [r"\midrule",
             r"\multicolumn{2}{l}{biological $G_1$ for reference} & " +
             f"{num((D.get('reference') or {}).get('n_connections'), 0)} & -- & -- & -- & -- \\\\",
             r"\bottomrule", r"\end{tabular}"]
    write("tab_closure.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ capability
def capability() -> None:
    body = [r"\begin{tabular}{l r r r r r r r r}", r"\toprule",
            r"scale & neurons & min sep. $\Delta$ & MI (bits, overlap 0.9) & memory capacity & temporal depth & sequence depth & participation ratio & gen. gap \\",
            r"\midrule"]
    for r in sorted(D.get("capability") or [], key=lambda x: x["scale"]):
        cap = num(r.get("memory_capacity"), 0)
        if r.get("memory_censored"):
            cap += r"$^{\ast}$"
        body.append(" & ".join([
            f"{r['scale']:g}$\times$", cnt(r.get("n_neurons")), num(r.get("min_delta"), 3),
            num(r.get("mi_delta_0_1"), 4), cap, num(r.get("temporal_depth"), 0),
            num(r.get("sequence_depth"), 0), num(r.get("participation_ratio"), 4),
            num(r.get("gen_gap"), 4)]) + r" \\")
    body += [r"\midrule",
             r"\multicolumn{9}{l}{\footnotesize $^{\ast}$ at the 96-class grid ceiling: a lower bound, not a measured capacity} \\",
             r"\bottomrule", r"\end{tabular}"]
    write("tab_capability.tex", "\n".join(body) + "\n", resize=True)


# ------------------------------------------------------------------ energy
def energy() -> None:
    body = [r"\begin{tabular}{l r r r r r r r}", r"\toprule",
            r"scale & neurons & GPU joules & J per spike & J per synaptic event & wall seconds & biological W & capacity per watt \\",
            r"\midrule"]
    for r in sorted(D.get("energy_curves") or [], key=lambda x: x["scale"]):
        body.append(" & ".join([
            f"{r['scale']:g}$\times$", cnt(r.get("n_neurons")), num(r.get("gpu_joules"), 4),
            num(r.get("j_per_spike"), 3), num(r.get("j_per_event"), 3), num(r.get("wall_s"), 4),
            num(r.get("bio_watts"), 3), num(r.get("cap_per_watt"), 3)]) + r" \\")
    body += [r"\midrule",
             r"\multicolumn{8}{l}{\footnotesize hardware column measured on an RTX 3090 (GPU device only; CPU/DRAM power unavailable);} \\",
             r"\multicolumn{8}{l}{\footnotesize biological column is $P_0 N/N_0$ from the cited calorimetry anchor - a property of $N$, never summed with the hardware column} \\",
             r"\bottomrule", r"\end{tabular}"]
    write("tab_energy.tex", "\n".join(body) + "\n", resize=True)


# ------------------------------------------------------------------ milestones
def milestones() -> None:
    rows = [
        ("M0", "reproduce FlyWire graph statistics", "done",
         "11/11 required checks vs published values; 7/9 advisory"),
        ("M1", "CPU neural baseline (LIF, whole connectome)", "done",
         r"dense $=$ sparse spike-for-spike (0 delta, $\rho=1.0$); 1.0 s in 120 s wall"),
        ("M2", "Vyb discrete-event runtime", "PASS",
         "spike-for-spike against an independent reference, 4/4 replications"),
        ("M3", "event-driven GPU backend", "PASS",
         r"CPU $\approx$ GPU on the fixture; 0 launch errors (required upstream fix \#270)"),
        ("M4", "energy instrumentation", "done",
         "GPU power, counters, parity; RAPL unavailable as non-root"),
        ("M5", "latent geometry", "done",
         "spectral 32-D wins (0.952); hyperbolic beats anatomical 3-D"),
        ("M6", "downscaling 0.5/0.25/0.1x", "done", "composites 0.179 / 0.267 / 0.276"),
        ("M7", "dynamic validation of downscaled scales", "done",
         "active-set Jaccard 0.909 / 0.950 / 0.853"),
        ("M8", "2x inverse scaling", "done", "sparsity preserved, strengths bounded"),
        ("M9", "closure R(G_2) toward G_1 (lineage)", "done$^{\dagger}$",
         "composite 0.0184, degree Wasserstein 0.0074, ARI 0.912"),
        ("M10", "10x graph (1.39M neurons)", "done", "closure composite 0.00098, ARI 1.0"),
        ("M11", "plasticity / associative learning", "done",
         "acquisition AUC 0.51 to 1.00; geometry adds nothing over shuffled controls"),
        ("M12", "capability scaling benchmarks", "done",
         "discrimination and dimensionality grow; temporal/sequence depth flat"),
        ("M13", "energy curves vs scale", "done", "438 / 500 / 946 / 2022 GPU joules"),
        ("M14", "capability per watt", "done (bound)", "0.07 / 0.62 / 0.61 / 0.53; censored"),
        ("M15", "100x graph (~14M neurons)", "gated",
         "needs the sibling-edge placement vectorised (1.4e9 distances at $c=100$)"),
    ]
    body = [r"\begin{tabular}{l p{5.0cm} l p{6.0cm}}", r"\toprule",
            r"id & milestone & state & evidence \\", r"\midrule"]
    body += [f"{a} & {esc(b)} & {c} & {d} \\\\" for a, b, c, d in rows]
    body += [r"\midrule",
             r"\multicolumn{4}{l}{\footnotesize $^{\dagger}$ closure demonstrates invertibility of the generator, not self-similarity (see \S 9).} \\",
             r"\bottomrule", r"\end{tabular}"]
    write("tab_milestones.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ defects
def defects() -> None:
    rows = [
        ("st\\_* stored the operand's LLVM type, not the intrinsic element type",
         "`st\\_i32(ptr, i64)` emitted an 8-byte store whose high word landed in the neighbouring 4-byte slot",
         "kernel-mode data corruption in the GPU pipeline", "fixed upstream (PR \\#273)"),
        ("a parenthesised operand after `*` inside an addition",
         "the multiplication's left operand is typed as a pointer; `p + s * (q)` fails to compile outside kernel mode",
         "blocks ordinary arithmetic expressions", "open (probe saved)"),
        ("`Vec<Vec<T>>` is unusable", "four failure modes incl. silent garbage from `.get(i).get(0)` and `.push` into a temporary",
         "forced the DES scheduler into a structure-of-arrays design", "open"),
        ("re-borrowing an existing borrow segfaults", "`f(borrow(v))` where `v` is already borrowed (exit 139)",
         "API shape constraint in the DES runtime", "open"),
        ("raw io byte buffers corrupt across a module boundary",
         "the same 16 bytes read inline give 0, 62 and via a module function give a 64-bit value",
         "forced decoding inside the module", "open"),
        ("optional `Vec` returns are rejected", "a function declared `Vec<UInt8>?` cannot `return` a `Vec<UInt8>`",
         "forced a rewrite of the loader", "open"),
        ("`Vec` parameters are not mutated for the caller", "out-parameters come back empty",
         "design constraint for the loader", "open"),
        ("device intrinsics return `CInt`", "`tid\\_x`/`blk\\_x`/`dim\\_x`/`ld\\_i32` need an explicit `as Int` in assignments",
         "documented; cost a compile cycle", "open (by design?)"),
        ("`ptr` is a reserved struct-field name", "rejected with a message that never names the token", "minor ergonomics", "open"),
        ("`String.to_int`/`to_float` do not exist", "documented in the reference manual but absent in the compiler",
         "the DES app ships its own integer parser", "open"),
        ("byte-at-a-time decode throughput", "$\approx 29\\,\mu$s per element ($\approx$34k elements/s); a 15M-element pass takes $\approx$7 min",
         "project-level blocker for a whole-connectome Vyb simulation; a bulk byte$\to$int read is required", "open"),
    ]
    body = [r"\begin{tabular}{p{4.4cm} p{5.6cm} p{3.6cm} l}", r"\toprule",
            r"defect found while building this project & reproduction / observed behaviour & consequence here & status \\",
            r"\midrule"]
    body += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_defects.tex", "\n".join(body) + "\n", resize=True)


# ------------------------------------------------------------------ criteria
def criteria() -> None:
    crit = D.get("criteria") or []
    body = [r"\begin{tabular}{p{10.6cm} l}", r"\toprule", r"criterion (\S 25) & verdict \\", r"\midrule"]
    for c in crit:
        body.append(f"{esc(c.get('criterion'))} & \\textbf{{{esc(c.get('verdict'))}}} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_criteria.tex", "\n".join(body) + "\n")


if __name__ == "__main__":
    for fn in (dataset, gate, geometry, downscale, dynamics, upscale, closure, capability,
               energy, criteria, milestones, defects):
        fn()
    print(f"tables written to {TAB.relative_to(ROOT)}")
