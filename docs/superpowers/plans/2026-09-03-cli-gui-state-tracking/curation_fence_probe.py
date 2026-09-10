"""Does curating a run in the results viewer make it undiscoverable?

Real writers only: `build_complete_viewer_run` publishes the tree (including
`publish_aggregate_snapshot`), and the curation goes through
`CurationLabels.mark`, the call the radial menu makes.

`core_readable` is `not state_requires_success_markers(...) or
aggregate_proof_is_current(...)`. The first probe assumed the fixture left the
flag unset and split the arms on forcing it; the fixture sets it to True, so
that split carved nothing. The arms below split on the axis that actually
decides the predicate:

  FENCED    success_markers_required = True   (as the fixture publishes it)
  UNFENCED  success_markers_required = False  (the legacy tree)

Each arm ASSERTS the flag is what its name claims before doing anything, so an
arm cannot silently run under the other condition.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# Derived from __file__, never hard-coded: this file lives at
# <repo>/docs/superpowers/plans/<topic>/, and a committed reproducer that names
# one worktree absolutely runs in exactly one checkout. The same mistake in a
# test's grep root is register entry 45's neighbour.
REPO = Path(__file__).resolve().parents[4]
if not (REPO / "src" / "phenotypic").is_dir():
    raise SystemExit(f"not a phenotypic checkout: {REPO}")
sys.path.insert(0, str(REPO))

import polars as pl  # noqa: E402

from phenotypic._cli._cli_completion import (  # noqa: E402
    state_requires_success_markers,
)
from phenotypic._cli._cli_state_management import (  # noqa: E402
    load_processing_state,
    save_processing_state,
)
from phenotypic.gui.results_viewer._curation_labels import (  # noqa: E402
    CurationLabels,
)
from phenotypic.gui.results_viewer._filtered_state import (  # noqa: E402
    KEY_IMAGE_FILE,
    KEY_OBJECT_LABEL,
)
from phenotypic.gui.results_viewer._output_root import (  # noqa: E402
    OutputRoot,
    core_readable,
)
from phenotypic.sdk_ import (  # noqa: E402
    BundleLayout,
    aggregate_proof_is_current,
)
from tests._output_layout import build_complete_viewer_run  # noqa: E402

PROOF = Path(".phenotypic") / "aggregate_publication.json"

# Scalar columns only. A nested column (Centroid as List(Float64)) makes the
# helper's write_measurements_mirror raise ComputeError: CSV format does not
# support nested data.
FRAME = pl.DataFrame(
    {
        "Metadata_Dataset": ["plate", "plate"],
        KEY_IMAGE_FILE: ["a", "b"],
        KEY_OBJECT_LABEL: [1, 2],
        "Size_Area": [10.0, 20.0],
    }
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fence(root: Path, when: str) -> dict[str, tuple[int, str]]:
    """Print, and return, the live (size, sha) of every fenced artifact."""
    print(f"    [{when}] aggregate_proof_is_current = "
          f"{aggregate_proof_is_current(root)}")
    live: dict[str, tuple[int, str]] = {}
    proof_path = root / PROOF
    if not proof_path.is_file():
        print("      (no aggregate proof on disk)")
        return live
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    for name, desc in (proof.get("required_outputs") or {}).items():
        path = root / desc["path"]
        if not path.is_file():
            print(f"      {name:22s} MISSING at {desc['path']}")
            continue
        size, digest = path.stat().st_size, _sha(path)
        live[name] = (size, digest)
        match = size == desc["size"] and digest == desc["sha256"]
        print(f"      {name:22s} match={match!s:5s} "
              f"size {desc['size']} -> {size}")
    return live


def _discover(root: Path, cache_root: Path) -> str:
    try:
        OutputRoot.discover(root, cache_root=cache_root)
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        return f"{type(exc).__name__}: {exc}"
    return "OK"


def _arm(name: str, *, want_flag: bool) -> None:
    print(f"\n{'=' * 72}\nARM {name}  "
          f"(success_markers_required == {want_flag})\n{'=' * 72}")
    tmp = Path(tempfile.mkdtemp(prefix=f"curation-fence-{name}-"))
    try:
        root = build_complete_viewer_run(tmp / "run", frame=FRAME,
                                         stems=("a", "b"))
        state = load_processing_state(root)
        published = (
            state.config.get("success_markers_required", False)
            if state is not None
            else None
        )
        print(f"    as published: success_markers_required = {published!r}")
        if published is not want_flag:
            if state is None:
                print("    CANNOT set: no processing state"); return
            state.config["success_markers_required"] = want_flag
            save_processing_state(state, root)
            print(f"    set to {want_flag!r}")

        # The arm asserts its own condition rather than assuming it took.
        actual = state_requires_success_markers(root)
        print(f"    ASSERT state_requires_success_markers == {want_flag}: "
              f"actual={actual}")
        assert actual is want_flag, (
            f"arm {name} claims {want_flag} but the tree reports {actual}"
        )

        layout = BundleLayout.detect(root)
        print(f"    core_readable (before) = {core_readable(layout)}")
        before = _fence(root, "before")
        print(f"    discover      (before) = "
              f"{_discover(root, tmp / 'cache-before')}")

        bound = OutputRoot.discover(root, cache_root=tmp / "cache-bind")
        frame = bound.clean_master_df
        row = frame.row(0, named=True)
        image, obj = str(row[KEY_IMAGE_FILE]), int(row[KEY_OBJECT_LABEL])
        labels = CurationLabels.load(bound.layout, frame)
        category = labels.categories()[0]
        print(f"    curating {image!r} object {obj} as {category!r}")
        try:
            labels.mark(image, obj, category)
            print("    curation WRITTEN")
        except Exception as exc:  # noqa: BLE001
            print(f"    curation REFUSED: {type(exc).__name__}: {exc}")
            print("    -> outcome 3: a CAS guard prevents the rewrite")
            return

        print(f"    core_readable (after)  = {core_readable(layout)}")
        after = _fence(root, "after")
        changed = sorted(k for k in before if before.get(k) != after.get(k))
        print(f"    fenced artifacts whose bytes CHANGED: {changed or 'none'}")
        print(f"    discover      (after)  = "
              f"{_discover(root, tmp / 'cache-after')}")
    except Exception:  # noqa: BLE001 - the arm's own failure is a result
        print("    ARM RAISED:")
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _arm("FENCED", want_flag=True)
    _arm("UNFENCED", want_flag=False)
    print("\ndone")
