"""Refresh and verify the domain-stripped review corpus (cluster E).

The corpus is a standalone copy of the phase-congruency implementation, its tests, its
spec and its plan, with every trace of the application domain removed, so a reviewer can
judge the mathematics without learning what the software is for. It also carries the
reference implementations, the papers and the golden fixture.

**The corpus is durable and additive.** Do not rebuild it. When a source file in the
repository changes, run::

    python math_review_corpus.py refresh --sandbox <root> --repo <root>

which re-derives only the files that come from the repo, applies the rename table below,
and leaves ``refs/``, ``refimpl/``, ``papers/``, ``tests/fixtures/``, ``kernels/_data.py``
and ``kernels/_standalone.py`` untouched -- those are hand-authored or third-party and are
the whole reason the corpus is worth keeping.

Then::

    python math_review_corpus.py verify --sandbox <root> --repo <root>

runs the seven gates and exits non-zero if any fails.

Gate 7 is the one that matters. ``refresh`` cannot invent prose: if a repo file gains a new
sentence containing a banned word, gate 1 fails and prints the offending lines for a human
to rewrite. That is deliberate -- a strip pass that silently guesses is worse than one that
stops.

stdlib only. Never imports the host package. Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys
from collections import Counter

# --------------------------------------------------------------------------------------
# The ban-list. `refs/` is NOT excluded from gate 1: it is provably clean (0 hits across
# all 11 files), so including it catches contamination of the reference copies. Only
# `papers/` is excluded, and only because it is third-party text we may not alter -- its
# two hits are the word "Biolog", from the journal *Biological Cybernetics* in the
# reference lists of felsberg2004 and shi2019.
# --------------------------------------------------------------------------------------
BAN_WORDS = [
    "colony", "colonies", "agar", "plate", "yeast", "fungi", "fungal", "hypha", "hyphal",
    "mycel", "septa", "microbe", "microbio", "phenotyp", "petri", "culture", "biolog",
    "organism",
]
BAN_RE = re.compile("|".join(BAN_WORDS), re.IGNORECASE)

EXCLUDE_DIRS = {"papers", "__pycache__", ".pytest_cache"}

# --------------------------------------------------------------------------------------
# Rename table, longest-first. Applied in order, so `FocusEdgeMonogenicPhase` must precede
# `FocusEdgePhase`, and `plates` must precede `plate`.
# --------------------------------------------------------------------------------------
RENAMES: list[tuple[str, str]] = [
    ("phenotypic._core._image.Image", "Image"),
    ("FocusEdgeMonogenicPhase", "MonogenicPhaseCongruencyEnhancer"),
    ("FocusEdgePhase", "OrientedPhaseCongruencyEnhancer"),
    ("load_synth_yeast_plate", "load_sample_image_a"),
    ("load_yeast_plate", "load_sample_image_b"),
    ("load_fungi_plate", "load_sample_image_c"),
    ("shipped plates", "shipped sample images"),
    ("real plates", "natural images"),
    ("real plate", "natural image"),
    ("the plate", "the sample image"),
    ("plates", "sample images"),
    ("plate", "sample image"),
    ("colony boundary", "step edge"),
    ("colony boundaries", "step edges"),
    ("hyphal ridge", "line feature"),
    ("hyphal ridges", "line features"),
    ("colonies", "objects"),
    ("colony", "object"),
]

# Per-file import rewiring. The stripped tree is a flat package: `kernels/` and `tests/`.
# These run AFTER `RENAMES`, so the right-hand sides already carry the new class names.
IMPORT_REWRITES: dict[str, list[tuple[str, str]]] = {
    "kernels/_monogenic_kernels.py": [],
    "kernels/_focus_edge_phase.py": [
        ("from ..abc_ import EdgeResponseEnhancer", "from ._standalone import EdgeResponseEnhancer"),
        ("from ..sdk_.typing_ import TuneSpec", "from .typing_ import TuneSpec"),
        ("    from phenotypic._core._image import Image", "    from ._standalone import Image"),
    ],
    "kernels/_focus_edge_monogenic_phase.py": [
        ("from ..abc_ import EdgeResponseEnhancer", "from ._standalone import EdgeResponseEnhancer"),
        ("from ..sdk_.typing_ import MonogenicOutput, TuneSpec", "from .typing_ import MonogenicOutput, TuneSpec"),
        ("    from phenotypic._core._image import Image", "    from ._standalone import Image"),
    ],
    "tests/test_monogenic_kernels.py": [
        ("from phenotypic.data import load_sample_image_c, load_sample_image_a, load_sample_image_b",
         "from kernels._data import load_sample_image_a, load_sample_image_b, load_sample_image_c"),
        ("from phenotypic.enhance._monogenic_kernels import", "from kernels._monogenic_kernels import"),
        ('_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures"',
         '_FIXTURE = Path(__file__).resolve().parent / "fixtures"'),
    ],
    "tests/test_focus_edge_monogenic_phase.py": [
        ("from phenotypic import Image, ImagePipeline", "from kernels._standalone import Image, ImagePipeline"),
        ("from phenotypic.data import load_sample_image_a", "from kernels._data import load_sample_image_a"),
        ("from phenotypic.enhance import MonogenicPhaseCongruencyEnhancer, OrientedPhaseCongruencyEnhancer",
         "from kernels import MonogenicPhaseCongruencyEnhancer, OrientedPhaseCongruencyEnhancer"),
        ("from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency",
         "from kernels._monogenic_kernels import monogenic_phase_congruency"),
    ],
    "tests/test_phase_congruency.py": [
        ("from phenotypic import Image", "from kernels._standalone import Image"),
        ("from phenotypic.data import load_sample_image_a", "from kernels._data import load_sample_image_a"),
        ("from phenotypic.enhance import OrientedPhaseCongruencyEnhancer",
         "from kernels import OrientedPhaseCongruencyEnhancer\nfrom kernels.typing_ import TuneSpec"),
        ("import phenotypic.enhance._focus_edge_phase as fep", "import kernels._focus_edge_phase as fep"),
        ("from phenotypic.enhance._monogenic_kernels import EPSILON_MONOGENIC",
         "from kernels._monogenic_kernels import EPSILON_MONOGENIC"),
        ("from phenotypic.sdk_.typing_ import TuneSpec", "from kernels.typing_ import TuneSpec"),
    ],
    "verify_claims.py": [
        ("**not** import ``phenotypic``", "**not** import the host package"),
    ],
}

# Prose the rename table cannot derive: sentences whose *framing*, not merely whose nouns,
# names the application domain. Encoded once, so a refresh reproduces the same judgement
# instead of asking a human to re-make it. `refresh` refuses to overwrite a live file while
# any banned word survives, so a new domain sentence in the repo stops the pipeline rather
# than leaking. Keys are the repo text, verbatim.
PROSE_PATCHES: dict[str, list[tuple[str, str]]] = {
    "kernels/_focus_edge_phase.py": [
        (
            "    Best For:\n"
            "        - Colony boundaries that vary in opacity or contrast across the sample image\n"
            "          due to pigmentation differences, agar depth variation, or object age.\n"
            "        - Plates with scanner vignetting or uneven illumination where",
            "    Best For:\n"
            "        - Step edges whose contrast varies across the field of view, so that a\n"
            "          single gradient threshold cannot hold everywhere.\n"
            "        - Images with vignetting or uneven illumination where",
        ),
    ],
}

# repo path -> sandbox path. Only these are re-derived by `refresh`.
DERIVED: dict[str, str] = {
    "src/phenotypic/enhance/_monogenic_kernels.py": "kernels/_monogenic_kernels.py",
    "src/phenotypic/enhance/_focus_edge_phase.py": "kernels/_focus_edge_phase.py",
    "src/phenotypic/enhance/_focus_edge_monogenic_phase.py": "kernels/_focus_edge_monogenic_phase.py",
    "tests/unit/enhance/_kovesi_synthetic.py": "tests/_kovesi_synthetic.py",
    "tests/unit/enhance/test_monogenic_kernels.py": "tests/test_monogenic_kernels.py",
    "tests/unit/enhance/test_focus_edge_monogenic_phase.py": "tests/test_focus_edge_monogenic_phase.py",
    "tests/unit/enhance/test_phase_congruency.py": "tests/test_phase_congruency.py",
    "docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py": "verify_claims.py",
}

# Hand-authored or third-party. `refresh` must never touch these.
PRESERVED = [
    "kernels/_standalone.py", "kernels/_data.py", "kernels/typing_.py", "kernels/__init__.py",
    "tests/__init__.py", "conftest.py", "tests/fixtures/phasecongmono_golden.npz",
    "refs", "refimpl", "papers", "spec", "plan",
]

MANIFEST = [
    "kernels/_monogenic_kernels.py", "kernels/_focus_edge_phase.py",
    "kernels/_focus_edge_monogenic_phase.py", "kernels/typing_.py",
    "tests/test_monogenic_kernels.py", "tests/_kovesi_synthetic.py",
    "tests/test_focus_edge_monogenic_phase.py", "tests/test_phase_congruency.py",
    "tests/fixtures/phasecongmono_golden.npz", "verify_claims.py",
    "plan/plan.md", "plan/reviews", "spec/README.md", "spec/references.md",
    "spec/drift-register.md", "spec/monogenic-phase-congruency.md",
    "spec/color-phase-congruency.md", "spec/conformal-lift.md",
    "refs", "refimpl", "papers",
]

# The only file whose structure must be *identical* to the repo original. It has no host
# imports, so nothing legitimate can change. The rest legitimately diverge at imports and
# data loaders; for those we compare numeric-constant and operator multisets instead.
MUST_BE_AST_IDENTICAL = [
    ("src/phenotypic/enhance/_monogenic_kernels.py", "kernels/_monogenic_kernels.py"),
    ("docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py", "verify_claims.py"),
]

# `parents[2]` in the repo test becomes `parent` in the flat sandbox: a directory depth,
# not logic. Recorded so gate 7 does not flag it forever.
KNOWN_CONSTANT_DELTAS = {"tests/test_monogenic_kernels.py": {"2": -1}}


def _iter_files(root: pathlib.Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if set(rel.parts) & EXCLUDE_DIRS:
            continue
        if p.name.endswith(".candidate"):  # a blocked refresh's work-in-progress
            continue
        yield p


def _numeric_profile(path: pathlib.Path) -> tuple[Counter, Counter] | None:
    """Numeric constants and binary operators. ``None`` if the file is missing or unparseable.

    Returning ``None`` rather than raising keeps a broken sandbox file reportable as a gate
    failure instead of a traceback -- a gate that dies does not tell you which gate died.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    consts = Counter(
        repr(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float, complex))
        and not isinstance(n.value, bool)
    )
    ops = Counter(type(n.op).__name__ for n in ast.walk(tree) if isinstance(n, ast.BinOp))
    return consts, ops


def _strip(text: str, sand_rel: str) -> str:
    """Rename identifiers, rewire imports, then apply the recorded prose judgements."""
    for old, new in PROSE_PATCHES.get(sand_rel, []):
        text = text.replace(old, new)
    for old, new in RENAMES:
        text = text.replace(old, new)
    for old, new in IMPORT_REWRITES.get(sand_rel, []):
        text = text.replace(old, new)
    return text


def refresh(sandbox: pathlib.Path, repo: pathlib.Path, only: list[str] | None) -> int:
    """Re-derive the repo-sourced files. **Never overwrites a live file with dirty text.**

    A candidate that still contains a banned word means the repo grew a sentence the tables
    do not cover. Overwriting would leak the domain into a corpus whose entire purpose is
    not to have one, and the leak would be invisible until the reviewer read it. So the
    candidate is kept beside the target for a human to finish, the live file is left exactly
    as it was, and the exit code is non-zero.
    """
    targets = {k: v for k, v in DERIVED.items() if not only or v in only or k in only}
    if not targets:
        print(f"no derived file matches {only!r}", file=sys.stderr)
        return 2

    promoted, blocked = [], []
    for repo_rel, sand_rel in targets.items():
        src = repo / repo_rel
        if not src.is_file():
            print(f"MISSING repo source: {repo_rel}", file=sys.stderr)
            return 1

        text = _strip(src.read_text(encoding="utf-8"), sand_rel)
        residual = [
            (i, line.strip()[:88])
            for i, line in enumerate(text.splitlines(), 1)
            if BAN_RE.search(line)
        ]

        dst = sandbox / sand_rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if residual:
            cand = dst.with_suffix(dst.suffix + ".candidate")
            cand.write_text(text, encoding="utf-8")
            blocked.append((sand_rel, residual, cand))
            continue

        dst.write_text(text, encoding="utf-8")
        promoted.append(sand_rel)

    for f in promoted:
        print(f"  refreshed {f}")

    if blocked:
        print("\nBLOCKED -- these still name the domain. The live files were NOT touched.")
        for sand_rel, residual, cand in blocked:
            print(f"\n  {sand_rel}  ({len(residual)} lines)  candidate: {cand.name}")
            for i, line in residual[:12]:
                print(f"    {i:>4}: {line}")
        print("\nStrip those lines in the candidate, then either move it over the target or")
        print("record the rewrite in PROSE_PATCHES so the next refresh reproduces it.")
        print("Never let refresh guess at prose.")
        return 1

    print("\nrefresh clean. Run `verify`.")
    return 0


def verify(sandbox: pathlib.Path, repo: pathlib.Path, python: list[str]) -> int:
    failures: list[str] = []

    # ---- gates 1-3: the corpus reveals nothing about the domain ------------------------
    for gate, pattern, label in (
        (1, BAN_RE, "ban-list"),
        (2, re.compile(r"import phenotypic"), "'import phenotypic'"),
        (3, re.compile(r"phenotypic", re.IGNORECASE), "'phenotypic'"),
    ):
        hits = []
        for p in _iter_files(sandbox):
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if pattern.search(line):
                        hits.append(f"{p.relative_to(sandbox)}:{i}: {line.strip()[:90]}")
            except (UnicodeDecodeError, OSError):
                continue
        status = "PASS" if not hits else "FAIL"
        print(f"GATE {gate}  {label:24s} hits={len(hits):3d}  {status}")
        for h in hits[:10]:
            print(f"          {h}")
        if hits:
            failures.append(f"gate {gate}")

    # ---- gate 4: manifest completeness -------------------------------------------------
    missing = [m for m in MANIFEST if not (sandbox / m).exists()]
    print(f"GATE 4  manifest                 missing={len(missing)}  {'PASS' if not missing else 'FAIL'}")
    for m in missing:
        print(f"          MISSING {m}")
    if missing:
        failures.append("gate 4")

    # ---- gate 5: the stripped checks run standalone -------------------------------------
    proc = subprocess.run(python + ["verify_claims.py"], cwd=sandbox, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    last = out.strip().splitlines()[-1] if out.strip() else "<no output>"
    ok5 = (
        proc.returncode == 0
        and "21/21 checks passed" in out
        and "max|dpc|" in out
        and "SKIPPED" not in out
        and "FIXTURE MISSING" not in out
    )
    print(f"GATE 5  verify_claims standalone exit={proc.returncode}  '{last[:40]}'  {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        failures.append("gate 5")

    # ---- gate 6: stripped kernels reproduce the golden fixture --------------------------
    snippet = (
        "import sys,numpy as np;sys.path.insert(0,'.')\n"
        "from kernels._monogenic_kernels import monogenic_phase_congruency as f\n"
        "from tests._kovesi_synthetic import step2line,starsine,circsine,noiseonf,unit_variance as u\n"
        "g=np.load('tests/fixtures/phasecongmono_golden.npz');n=64\n"
        "s=np.zeros((n,n));s[:,n//2:]=1.0\n"
        "c={'step':s,'step2line':u(step2line(n)),'starsine':u(starsine(n,ncycles=8)),"
        "'circsine':u(circsine(n,wavelength=16.0)),'noiseonf':u(noiseonf(n,1.5,seed=1))}\n"
        "w=0.0;ok=True\n"
        "for k,i in c.items():\n"
        "    r=f(i,periodic=True)\n"
        "    w=max(w,float(np.abs(g[k+'__pc']-r.pc).max()))\n"
        "    ok&=bool(np.allclose(g[k+'__pc'],r.pc,rtol=1e-6,atol=1e-9))\n"
        "print(f'{w:.4e} {ok}')\n"
    )
    proc = subprocess.run(python + ["-c", snippet], cwd=sandbox, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    ok6 = proc.returncode == 0 and tail.endswith("True")
    print(f"GATE 6  golden fixture           {tail or proc.stderr.strip()[:50]}  {'PASS' if ok6 else 'FAIL'}")
    if not ok6:
        failures.append("gate 6")

    # ---- gate 7: the transform touched no logic -----------------------------------------
    tool = pathlib.Path(__file__).with_name("ast_structural_equivalence.py")
    ok7 = True
    for repo_rel, sand_rel in MUST_BE_AST_IDENTICAL:
        proc = subprocess.run(
            [sys.executable, str(tool), str(repo / repo_rel), str(sandbox / sand_rel)],
            capture_output=True, text=True,
        )
        identical = proc.returncode == 0
        ok7 &= identical
        print(f"GATE 7  {sand_rel:28s} {'IDENTICAL' if identical else 'DIVERGES'}")

    for repo_rel, sand_rel in DERIVED.items():
        if (repo_rel, sand_rel) in MUST_BE_AST_IDENTICAL:
            continue
        a = _numeric_profile(repo / repo_rel)
        b = _numeric_profile(sandbox / sand_rel)
        if a is None or b is None:
            ok7 = False
            which = repo_rel if a is None else sand_rel
            print(f"GATE 7  {sand_rel:28s} UNPARSEABLE OR MISSING ({which})")
            continue
        co, oo = a
        cs, os_ = b
        allowed = KNOWN_CONSTANT_DELTAS.get(sand_rel, {})
        dc = {k: (co[k], cs[k]) for k in set(co) | set(cs) if cs[k] - co[k] != allowed.get(k, 0)}
        do = {k: (oo[k], os_[k]) for k in set(oo) | set(os_) if oo[k] != os_[k]}
        if dc or do:
            ok7 = False
            print(f"GATE 7  {sand_rel:28s} CONSTANT/OPERATOR DRIFT")
            if dc:
                print(f"          constants (repo,sandbox): {dc}")
            if do:
                print(f"          operators (repo,sandbox): {do}")
        else:
            print(f"GATE 7  {sand_rel:28s} constants+operators match")
    if not ok7:
        failures.append("gate 7")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all 7 gates pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh and verify the domain-stripped review corpus.")
    ap.add_argument("mode", choices=["verify", "refresh"])
    ap.add_argument("--sandbox", required=True, type=pathlib.Path)
    ap.add_argument("--repo", required=True, type=pathlib.Path)
    ap.add_argument("--only", nargs="*", help="refresh: limit to these sandbox paths")
    ap.add_argument(
        "--python", default=None,
        help="verify: interpreter for the sandbox, e.g. \"uv run --project <repo> python\"",
    )
    a = ap.parse_args()

    if not a.sandbox.is_dir():
        print(f"sandbox not found: {a.sandbox}", file=sys.stderr)
        return 2

    if a.mode == "refresh":
        return refresh(a.sandbox, a.repo, a.only)

    python = a.python.split() if a.python else ["uv", "run", "--project", str(a.repo), "python"]
    return verify(a.sandbox, a.repo, python)


if __name__ == "__main__":
    raise SystemExit(main())
