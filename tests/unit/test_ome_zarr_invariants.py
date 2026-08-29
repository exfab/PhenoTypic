"""Source-tree invariants for the OME-Zarr store. Each guards a silent failure.

Grep-style gates, in the same spirit as
``tests/unit/schema/test_no_metadata_literals.py``. Every one here guards an
invariant a future edit could violate with **no other test noticing** -- either
because the wrong behaviour still produces the right answer (a hand-joined
suffix resolves to the same path until an rgb-less store appears) or because
what changes is cost rather than result (a recursive glob that walks into every
store returns a byte-identical file list).

**Comments and docstrings are excluded from every scan**, by parsing rather
than by prefix-matching a line. The plan's draft skipped only lines beginning
with ``#``, which left ``test_objmap_path_is_never_hard_coded`` red on day one
against five *docstrings* that exist precisely to tell readers not to hard-code
that path. A gate that fires on its own documentation gets relaxed, not obeyed.

**Every allow-list entry is checked for being live**
(:func:`test_every_allow_list_entry_still_earns_its_exemption`). An exemption
outlives the code it was written for, and a stale one widens the gate forever
with nothing to say so.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "phenotypic"
PY = sorted(SRC.rglob("*.py"))


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Every physical line occupied by a module/class/function docstring."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


@lru_cache(maxsize=None)
def _scannable_lines(path: Path) -> tuple[tuple[int, str], ...]:
    """``(line number, line)`` for every line that is neither comment nor docstring.

    Cached: the tree is parsed once per file rather than once per gate, which
    turns twelve full-tree walks into one.
    """
    text = path.read_text(encoding="utf-8")
    skip = _docstring_lines(ast.parse(text))
    return tuple(
        (number, line)
        for number, line in enumerate(text.splitlines(), 1)
        if number not in skip and not line.lstrip().startswith("#")
    )


def _hits(pattern: str, *, allow: frozenset[str] = frozenset()) -> list[str]:
    """Every non-comment, non-docstring source line matching *pattern*.

    Args:
        pattern: A regular expression searched against each line.
        allow: Basenames exempted from the scan. Justify each one where it is
            passed, and keep it live -- see
            :func:`test_every_allow_list_entry_still_earns_its_exemption`.

    Returns:
        ``"<relative path>:<line number>: <stripped line>"`` for each hit.
    """
    rx = re.compile(pattern)
    out: list[str] = []
    for path in PY:
        if path.name in allow:
            continue
        for number, line in _scannable_lines(path):
            if rx.search(line):
                out.append(f"{path.relative_to(SRC)}:{number}: {line.strip()}")
    return out


#: ``(pattern, allow)`` for each gate below, so the exemptions can be checked
#: for being live in one place. Keyed by the gate it belongs to.
GATES: dict[str, tuple[str, frozenset[str]]] = {
    "store_suffix": (
        r'\.ome\.zarr"',
        frozenset({"ngff_.py", "_preview_cache.py", "_image_pipeline_core.py"}),
    ),
    "objmap_path": (r"rgb/labels/objmap", frozenset()),
    "hdf_write": (
        r'save2hdf5|save_intermediate_layers|h5py\.File\([^)]*["\'](w|a|r\+)',
        frozenset({"hdf_.py", "_metadata_migration.py"}),
    ),
    "metadata_prefix": (
        r'startswith\(\s*["\']Metadata_',
        frozenset({"_metadata_migration.py"}),
    ),
    "store_rglob": (r'rglob\(\s*f?["\'][^"\']*\.ome\.zarr', frozenset()),
    "store_suffix_rglob": (
        r'rglob\(\s*f["\'][^"\']*\{STORE_SUFFIX\}',
        frozenset(),
    ),
}


def test_store_suffix_is_joined_in_exactly_one_place() -> None:
    """Per-image store paths come from ``zarr_store_path``, never a hand-join.

    ``.ome.zarr`` is a *double* suffix, so a hand-joined path and a
    ``Path.stem`` taken back off it disagree silently -- ``img.ome`` is a
    plausible-looking wrong name that propagates into parquet filenames and
    completion markers rather than raising.

    Two exemptions, both verified rather than assumed:

    * ``ngff_.py`` defines ``STORE_SUFFIX``.
    * ``_preview_cache.py`` (``BASE_STORE_NAME``, and the per-node
      ``f"{i:02d}_{op_key}.ome.zarr"``) and ``_image_pipeline_core.py`` (five
      sites around ``_run_operations``' node capture) name **builder-preview**
      node stores: a temp-cache tree keyed by node index, not by dataset and
      image stem. Neither file imports ``zarr_store_path`` -- asserted below --
      so neither is in the results-store business at all.

    Everything that *is* -- all of ``_cli/``, the GUI viewers, and
    ``_image_io_handler.py`` -- is covered with no exemption. So is
    ``_io_constants.py``, which the plan's draft exempted unnecessarily: it
    composes the path from ``STORE_SUFFIX`` and names the literal only in
    prose.
    """
    pattern, allow = GATES["store_suffix"]
    assert _hits(pattern, allow=allow) == []


def test_the_builder_preview_exemption_is_not_a_results_store_exemption() -> None:
    """The two exempted files must stay out of the per-image store namespace."""
    for name in ("gui/builder/_preview_cache.py", "_core/_pipeline_parts/_image_pipeline_core.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert "zarr_store_path" not in text, (
            f"{name} is exempt from the suffix gate but now builds results-store "
            f"paths; the exemption must be narrowed or the call routed properly"
        )


def test_objmap_path_is_never_hard_coded() -> None:
    """An rgb-less store puts the label under ``gray``, not ``rgb``.

    The label attaches to the *primary* series, which is ``rgb`` when present
    and ``gray`` otherwise. A reader must resolve the path from
    ``phenotypic.labels.objmap``. Hard-coding it works on every store that has
    an ``rgb`` series and fails on exactly the ones that do not -- and ``gray``
    is the primary series in every rgb-less store, so the failure is a
    ``FileNotFoundError`` on real data that no fixture with an ``rgb`` layer
    ever reproduces.
    """
    pattern, allow = GATES["objmap_path"]
    assert _hits(pattern, allow=allow) == []


def test_no_module_still_writes_hdf() -> None:
    """Phase 6 keeps h5py READERS for migration; only WRITE paths must be gone.

    The scan is **wider than the plan's**, which matched only ``"w``: an HDF
    writer reintroduced as ``mode="a"`` or ``"r+"`` would have passed it. Both
    are matched here, which is why the two exemptions exist and why they are
    both live:

    * ``hdf_.py`` keeps ``safe_writer`` / ``swmr_writer`` / ``strict_writer``,
      the append-mode plumbing the migration read path opens against.
    * ``_metadata_migration.py`` keeps ``_migrate_hdf_copy`` (:1490) and
      ``_rollback_hdf`` (:2493), both ``h5py.File(temp, "r+")`` against a
      ``shutil.copy2`` **temp copy**, never the source. Pass 1 excludes ``.h5``
      from ``NON_IMAGE_KINDS`` outright, so neither is reached by ``--mode
      migrate`` today; the code is retained deliberately, documented at
      ``_metadata_migration.py:88-101``.

    Every other module is scanned unexempted, including ``_hdf_to_zarr.py`` and
    ``_image_io_handler.py`` -- the plan exempted both, but their h5py calls are
    all ``"r"``, so exempting them only blinded the gate to a writer added
    there later.
    """
    pattern, allow = GATES["hdf_write"]
    assert _hits(pattern, allow=allow) == []


def test_metadata_ownership_is_never_a_prefix_check() -> None:
    """CLAUDE.md: use ``metadata_owner_for_header``, never string parsing.

    ``_metadata_migration.py:269`` is the one sanctioned carve-out -- the
    centralized canonicalization helper, which is exactly where string handling
    is allowed to live. (The plan cited :210; the line is 269.)
    """
    pattern, allow = GATES["metadata_prefix"]
    assert _hits(pattern, allow=allow) == []


def test_no_recursive_glob_for_stores() -> None:
    """``rglob`` walks INTO every store: ~400k stat calls at 10k images.

    This is the gate class that matters most and is hardest to get from a
    behavioural test, because **an assertion about results cannot see cost**. A
    recursive walk filtered back to the same file list returns a byte-identical
    answer while doing all the work it was meant to avoid; the results-viewer
    inventory's own guard has to count ``os.scandir`` calls for exactly this
    reason.

    The f-string form is matched too, so ``sweep_orphan_parts`` -- which lives
    in ``ngff_.py`` and once used this very pattern -- cannot exempt itself.
    """
    for key in ("store_rglob", "store_suffix_rglob"):
        pattern, allow = GATES[key]
        assert _hits(pattern, allow=allow) == [], key


#: Identifiers whose ``.stem`` would be a store directory's, not an image's.
_STORE_NAMES = frozenset({"store", "store_path", "target_store", "zarr_store"})

#: Calls whose result is a store path, so ``f(...).stem`` is the same defect.
_STORE_CALLS = frozenset({"zarr_store_path", "new_part_path", "promote_store"})


def _store_stem_sites() -> list[str]:
    """Every ``<store>.stem`` in the source tree, resolved by AST not by grep."""
    out: list[str] = []
    for path in PY:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and node.attr == "stem"):
                continue
            value = node.value
            named = isinstance(value, ast.Name) and (
                value.id in _STORE_NAMES or value.id.endswith(("_store", "_store_path"))
            )
            called = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in _STORE_CALLS
            )
            if named or called:
                out.append(
                    f"{path.relative_to(SRC)}:{node.lineno}: {ast.unparse(node)}"
                )
    return out


def test_path_stem_is_never_taken_of_a_store_directory() -> None:
    """``.ome.zarr`` is a DOUBLE suffix, so ``Path.stem`` yields ``img.ome``.

    That is a plausible-looking wrong name rather than an error: it propagates
    into parquet filenames and completion-marker keys, and
    ``zarr_store_path(out, ds, "img.ome")`` then resolves to a store that does
    not exist -- so every image reprocesses forever, silently, on every run.
    ``store_stem`` (``_io_constants.py:1528``) is the correct call and raises
    rather than falling back.

    This replaces the phase's `grep -rn "\.stem" src/phenotypic/_cli/`
    exit criterion, which returns ~100 hits and asks a reader to confirm each
    is a *source image* path. Every one of them is, today -- ``img.stem``,
    ``image.stem``, ``item.stem``, ``Path(image_name).stem`` -- so a manual
    review passes and teaches nothing. Resolving the receiver by AST turns a
    hundred-line eyeball pass into a check that fails when a store path
    actually acquires a ``.stem``.
    """
    assert _store_stem_sites() == []


@pytest.mark.parametrize("gate", sorted(GATES))
def test_every_allow_list_entry_still_earns_its_exemption(gate: str) -> None:
    """A stale exemption widens its gate forever with nothing to say so.

    Each allowed basename must still produce at least one hit when it is *not*
    exempted. When the code an exemption was written for is deleted, this
    fails and the exemption goes with it, rather than silently covering
    whatever is added to that file next.
    """
    pattern, allow = GATES[gate]
    for name in sorted(allow):
        others = frozenset(allow - {name})
        assert any(hit.split(":")[0].endswith(name) for hit in _hits(pattern, allow=others)), (
            f"{gate}: '{name}' is exempted but no longer matches; drop it"
        )


# ---------------------------------------------------------------------------
# Four candidate gates were considered and dropped as UNABLE TO FAIL. Recorded
# so they are not "helpfully" re-added:
#
#   test_no_pid_in_a_part_directory_name
#       The regex needs `getpid` and `part` on one PHYSICAL line; the only real
#       instance was wrapped across lines, and Phase 6 deleted it anyway. The
#       real guard is test_part_name_carries_no_pid in test_ngff_promote.py.
#   test_scale_vectors_are_never_powers_of_two
#       r'"scale":\s*\[?\s*2\s*\*\*' matches a JSON-literal-with-Python-exponent
#       form no implementation emits. Phase 1's
#       test_scale_vector_comes_from_actual_shapes_not_powers_of_two is the
#       real guard.
#   test_resume_state_never_lives_in_ngff_metadata
#       r'labels.*stage2|stage2.*ome\.labels' matches nothing plausible. Phase
#       3 Task 3.4's differential parity test is what catches that defect.
#   test_zarr_errors_are_caught_not_propagated
#       BaseZarrError subclasses ValueError, so the assertion held with or
#       without it in the tuple.
# ---------------------------------------------------------------------------
