"""Every registered callback must write to an id the layout actually mounts.

A viewer-wide invariant, not a detail of any one surface. Dash's
``suppress_callback_exceptions`` relaxes SERVER-side validation only: a
callback whose Output names a component absent from the layout still
registers, still runs, and still returns — and then the browser's
dash-renderer throws ``ReferenceError: A nonexistent object was used in
an Output`` and discards the WHOLE response. For a multi-output callback
that means the outputs which *do* exist are never applied either, so a
working control silently stops updating.

That is exactly what unmounting the QC tab did to the Colony spotlight
readout: ``colony_view`` registered one callback writing to both
``colony-dim-readout`` and ``qc-review-dim-readout``, and losing the
second froze the first. No Python-level test could see it — the failure
lives in the renderer — and the e2e that caught it was a *filter* test
asserting a clean console. This test makes the same class of defect a
unit failure.

**Pattern-matching ids are excluded on purpose.** A dict id
(``{"type": ..., "index": ALL}``) legitimately matches zero components
at layout time — the components it targets are created by other
callbacks. Only string ids name a component the layout is expected to
carry, so only string ids are checked; asserting on dict ids would
false-positive on every wildcard callback in the viewer.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import IMAGE

from tests._output_layout import (
    write_complete_manifest,
    write_master,
    write_measurements_mirror,
)


def _walk(node):
    """Yield every component in a built layout tree."""
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif children is not None:
        yield from _walk(children)


@pytest.fixture()
def output_root(tmp_path: Path) -> OutputRoot:
    """A discoverable two-image output root with overlays."""
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 2,
            str(IMAGE.IMAGE_NAME): ["img-A", "img-B"],
            "Object_Label": [1, 1],
            "Bbox_CenterRR": [10.0, 10.0],
            "Bbox_CenterCC": [10.0, 10.0],
            "Size_Area": [100.0, 100.0],
        }
    )
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    for stem in ("img-A", "img-B"):
        PILImage.new("RGB", (64, 64), (128, 128, 128)).save(overlays / f"{stem}.png")
    write_complete_manifest(tmp_path, total_images=2)

    return OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )


def test_no_registered_callback_writes_to_an_unmounted_id(
    output_root: OutputRoot,
) -> None:
    """No callback Output names a string id the built layout does not carry."""
    app = create_app(output_root)
    layout = app.layout() if callable(app.layout) else app.layout
    mounted = {
        node.id for node in _walk(layout) if isinstance(getattr(node, "id", None), str)
    }

    dangling: set[tuple[str, str]] = set()
    for key in app.callback_map:
        # The callback_map key encodes its outputs as
        # ``..<id>.<prop>...<id>.<prop>..`` with an optional ``@<hash>``
        # suffix on ``allow_duplicate`` outputs.
        for segment in key.strip(".").split("..."):
            segment = segment.strip(".").split("@", 1)[0]
            if "." not in segment or segment.startswith("{"):
                continue  # no property, or a pattern-matching dict id
            component_id, prop = segment.rsplit(".", 1)
            if component_id.startswith("{"):
                continue  # pattern-matching id -- may match zero components
            if component_id not in mounted:
                dangling.add((component_id, prop))

    assert not dangling, (
        "these callback Outputs name components the layout does not mount, so "
        "dash-renderer will throw and DISCARD the whole callback response "
        f"(taking any co-outputs with it): {sorted(dangling)}"
    )
