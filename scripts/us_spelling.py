"""Convert British spellings to US spellings across the repository's published text.

Three targeted passes, so nothing programmatic breaks:

* prose files (``*.md``, ``*.tex``, ``*.txt``) are converted wholesale;
* Python files are converted only inside comments and string literals, via ``tokenize`` — identifiers
  and dict keys are left alone, because renaming those would silently change an artifact's schema;
* JSON files are converted only in string *values*, never in keys.

Run from the repository root:  python scripts/us_spelling.py [--check]
"""
from __future__ import annotations

import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# British -> US, longest first so 'coloured' is handled before 'colour'
PAIRS: list[tuple[str, str]] = [
    ("colour", "color"), ("colours", "colors"), ("coloured", "colored"), ("colourful", "colorful"),
    ("centre", "center"), ("centres", "centers"), ("centred", "centered"), ("centring", "centering"),
    ("behaviour", "behavior"), ("behaviours", "behaviors"),
    ("neighbouring", "neighboring"), ("neighbourhood", "neighborhood"),
    ("neighbour", "neighbor"), ("neighbours", "neighbors"),
    ("normalisation", "normalization"), ("normalised", "normalized"), ("normalises", "normalizes"),
    ("normalising", "normalizing"), ("normalise", "normalize"),
    ("initialisation", "initialization"), ("initialised", "initialized"), ("initialising", "initializing"),
    ("initialise", "initialize"),
    ("summarised", "summarized"), ("summarise", "summarize"),
    ("optimised", "optimized"), ("optimise", "optimize"),
    ("utilised", "utilized"), ("utilise", "utilize"),
    ("serialised", "serialized"), ("serialise", "serialize"),
    ("vectorising", "vectorizing"), ("vectorised", "vectorized"), ("vectorise", "vectorize"),
    ("generalised", "generalized"), ("generalise", "generalize"),
    ("characterised", "characterized"), ("characterise", "characterize"),
    ("minimised", "minimized"), ("minimise", "minimize"),
    ("maximised", "maximized"), ("maximise", "maximize"),
    ("standardised", "standardized"), ("standardise", "standardize"),
    ("visualised", "visualized"), ("visualise", "visualize"),
    ("categorised", "categorized"), ("categorise", "categorize"),
    ("prioritised", "prioritized"), ("prioritise", "prioritize"),
    ("customised", "customized"), ("customise", "customize"),
    ("hypothesised", "hypothesized"), ("hypothesise", "hypothesize"),
    ("parallelised", "parallelized"), ("parallelise", "parallelize"),
    ("recognised", "recognized"), ("recognise", "recognize"),
    ("analysed", "analyzed"), ("analysing", "analyzing"), ("analyse", "analyze"),
    ("labelled", "labeled"), ("labelling", "labeling"),
    ("modelled", "modeled"), ("modelling", "modeling"),
    ("odours", "odors"), ("odour", "odor"),
    ("greyscale", "grayscale"), ("grey-scale", "gray-scale"), ("greyed", "grayed"),
    ("greying", "graying"), ("grey", "gray"),
    ("metres", "meters"), ("metre", "meter"),
    ("artefacts", "artifacts"), ("artefact", "artifact"),
    ("programmes", "programs"), ("programme", "program"),
    ("favourite", "favorite"), ("favoured", "favored"), ("favours", "favors"), ("favour", "favor"),
    ("flavours", "flavors"), ("flavour", "flavor"),
    ("honours", "honors"), ("honour", "honor"),
    ("licence", "license"), ("defence", "defense"), ("practise", "practice"),
    ("towards", "toward"), ("whilst", "while"), ("amongst", "among"), ("learnt", "learned"),
    ("labour", "labor"), ("harbour", "harbor"), ("rumour", "rumor"),
    ("vapours", "vapors"), ("vapour", "vapor"), ("sulphur", "sulfur"),
    # derived forms and -isation nouns: missed by the first pass and found by a wider sweep
    ("behavioural", "behavioral"), ("behaviourally", "behaviorally"), ("colouring", "coloring"),
    ("organised", "organized"), ("organise", "organize"), ("organising", "organizing"),
    ("organisation", "organization"), ("organisations", "organizations"), ("organiser", "organizer"),
    ("realised", "realized"), ("realise", "realize"), ("realising", "realizing"),
    ("realisation", "realization"),
    ("specialised", "specialized"), ("specialise", "specialize"),
    ("specialisation", "specialization"), ("specialising", "specializing"),
    ("emphasised", "emphasized"), ("emphasise", "emphasize"), ("emphasising", "emphasizing"),
    ("optimisation", "optimization"), ("optimisations", "optimizations"),
    ("optimising", "optimizing"), ("optimiser", "optimizer"),
    ("utilisation", "utilization"), ("utilisations", "utilizations"),
    ("serialisation", "serialization"),
    ("generalisation", "generalization"), ("generalisations", "generalisations".replace("sations", "zations")),
    ("categorisation", "categorization"), ("visualisation", "visualization"),
    ("visualisations", "visualizations"), ("standardisation", "standardization"),
    ("minimisation", "minimization"), ("minimising", "minimizing"),
    ("maximisation", "maximization"), ("maximising", "maximizing"),
    ("characterisation", "characterization"), ("summarisation", "summarization"),
    ("prioritisation", "prioritization"), ("prioritising", "prioritizing"),
    ("customisation", "customization"), ("parallelisation", "parallelization"),
    ("analyser", "analyzer"),
    ("favourable", "favorable"), ("favourably", "favorably"), ("favourites", "favorites"),
    ("unfavourable", "unfavorable"),
    ("honourable", "honorable"), ("honoured", "honored"), ("honouring", "honoring"),
    ("ageing", "aging"), ("judgement", "judgment"),
]

PATTERNS = [(re.compile(rf"\b{re.escape(b)}\b", re.IGNORECASE), b, u) for b, u in PAIRS]
PERCENT = re.compile(r"\bper cent\b", re.IGNORECASE)


def _cased(src: str, dst: str, found: str) -> str:
    if found.isupper():
        return dst.upper()
    if found[:1].isupper():
        return dst.capitalize()
    return dst


PY_KEY = re.compile(r"""(['\"])([A-Za-z_][A-Za-z0-9_]*)\1(\s*:)""")


def convert_python_text(text: str) -> str:
    """Convert comments, docstrings and printed text, but never a dict key.

    Keys such as {"optimiser": ...} are written into artifacts; renaming one here would desync the
    code from the JSON already on disk.
    """
    keys: list[str] = []

    def stash(m: re.Match) -> str:
        keys.append(m.group(0))
        return f"\x00{len(keys) - 1}\x00"

    stashed = PY_KEY.sub(stash, text)
    converted = convert(stashed)
    return re.sub(r"\x00(\d+)\x00", lambda m: keys[int(m.group(1))], converted)


def convert(text: str) -> str:
    def repl(m: re.Match) -> str:
        return _cased(m.group(0), m.group(0), m.group(0))
    out = text
    for pat, brit, us in PATTERNS:
        out = pat.sub(lambda m, b=brit, u=us: _cased(m.group(0), u, m.group(0)), out)
    out = PERCENT.sub(lambda m: _cased(m.group(0), "percent", m.group(0)), out)
    return out


def convert_python(text: str) -> str:
    """Rewrite comments and string literals only, leaving identifiers untouched."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError):
        return text
    out = []
    for tok in tokens:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            out.append((tok.type, convert(tok.string)))
        else:
            out.append((tok.type, tok.string))
    try:
        return tokenize.untokenize(out)
    except Exception:
        return text


JSON_KEY = re.compile(r'"(?:[^"\\]|\\.)*"(\s*:)')


def convert_json_text(text: str) -> str:
    """Convert string values, never keys, without reformatting the artifact.

    Keys are stashed, the remaining text is converted, then the keys are restored verbatim.
    """
    keys: list[str] = []

    def stash(m: re.Match) -> str:
        keys.append(m.group(0))
        return f"\x00{len(keys) - 1}\x00"

    stashed = JSON_KEY.sub(stash, text)
    converted = convert(stashed)
    return re.sub(r"\x00(\d+)\x00", lambda m: keys[int(m.group(1))], converted)


def main() -> None:
    check = "--check" in sys.argv
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and ".git" not in p.parts and ".venv" not in p.parts
             and p.name != "us_spelling.py"          # its tables list the British spellings
             and p.suffix in {".md", ".tex", ".txt", ".py", ".json", ".vyb", ".cu", ".cuh"}]
    changed = []
    for p in files:
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if p.suffix == ".py":
            new = convert_python_text(text)
        elif p.suffix == ".json":
            new = convert_json_text(text)
        else:
            new = convert(text)
        if new != text:
            changed.append(p.relative_to(ROOT))
            if not check:
                p.write_text(new)
    print(f"{'would change' if check else 'changed'} {len(changed)} files")
    for c in changed[:40]:
        print("  ", c)


if __name__ == "__main__":
    main()
