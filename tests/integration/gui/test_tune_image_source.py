"""Integration tests for the Curate Image Source picker (Task B-IMG).

The Curate view carries a sandbox-bounded directory picker that sets the plate
**Image Source** used to load plates for overlay rendering. The picker reuses
the builder's ``directory_tree`` (folder-only) inside a ``dbc.Modal``, bounded
by the sandbox root, and pre-fills from the bound run's ``run.json``
``images_dir``.

Three behaviours are pinned here:

* :func:`resolve_image_source` accepts an in-sandbox directory and rejects an
  out-of-sandbox path (the modal can't escape the sandbox boundary on a shared
  SSH tunnel).
* :func:`plate_image_path` resolves ``<Image Source>/<plate_name>``.
* the loaded Curate view exposes the picker ids and pre-fills the store from
  ``root.images_dir``; an unset Image Source surfaces a prompt instead of an
  overlay attempt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic.gui.shell import SandboxRoot


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


def _run_with_images(tmp_path: Path, images_dir: Path) -> object:
    """Write a run.json marker carrying ``images_dir`` and discover the root."""
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tools_ import tune_cache_run_marker_path

    marker = tune_cache_run_marker_path(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "storage_url": None,
                "study_name": "tune",
                "is_multi_objective": False,
                "images_dir": str(images_dir),
            }
        )
    )
    return TuneRunRoot.discover(tmp_path)


def test_resolve_image_source_accepts_in_sandbox(sandbox: SandboxRoot) -> None:
    from phenotypic.gui.tune._image_source import resolve_image_source

    plates = sandbox.root / "plates"
    plates.mkdir()
    resolved = resolve_image_source(sandbox, str(plates))
    assert resolved == plates.resolve()


def test_resolve_image_source_rejects_escape(sandbox: SandboxRoot) -> None:
    from phenotypic.gui.tune._image_source import resolve_image_source

    # A path outside the sandbox root is refused (None, not an exception).
    assert resolve_image_source(sandbox, "/etc") is None
    assert resolve_image_source(sandbox, "../../escape") is None


def test_resolve_image_source_rejects_non_directory(sandbox: SandboxRoot) -> None:
    from phenotypic.gui.tune._image_source import resolve_image_source

    a_file = sandbox.root / "plate.tif"
    a_file.write_bytes(b"")
    assert resolve_image_source(sandbox, str(a_file)) is None


def test_plate_image_path_joins_source_and_plate() -> None:
    from phenotypic.gui.tune._image_source import plate_image_path

    src = Path("/data/plates")
    assert plate_image_path(str(src), "plate_01.tif") == src / "plate_01.tif"


def test_load_plate_grid_reads_a_real_plate(sandbox: SandboxRoot) -> None:
    """A successful overlay load returns a ``GridImage``.

    Regression: the loader must call ``GridImage.imread(path)``, not
    ``GridImage(str(path))`` — the constructor rejects a path string
    (``ValueError: Input must be a NumPy array``), which silently broke every
    Curate A/B overlay render. The prior B-IMG tests only exercised the
    sandbox-rejection path, never a successful load, so this slipped through.
    """
    import numpy as np
    from skimage.io import imsave

    from phenotypic import GridImage
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.gui.tune._curate_overlays import load_plate_grid

    plates = sandbox.root / "plates"
    plates.mkdir()
    rgb = np.asarray(load_synth_yeast_plate().rgb[:])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb if rgb.max() > 1 else rgb * 255, 0, 255).astype(np.uint8)
    imsave(str(plates / "plate.tif"), rgb)

    grid = load_plate_grid(str(plates), "plate.tif", sandbox=sandbox)
    assert isinstance(grid, GridImage)


def test_curate_view_exposes_picker_ids(tmp_path: Path) -> None:
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tools_ import trials_parquet_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    images = tmp_path / "calibration"
    images.mkdir()
    parquet = trials_parquet_path(tmp_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[Trial(number=0, params={"0.sigma": 1.0}, score=0.5, terms={}, n_images=2)]
    ).to_parquet(parquet)
    root = TuneRunRoot.discover(tmp_path)
    assert isinstance(root, TuneRunRoot)

    sandbox = SandboxRoot.from_path(tmp_path)
    app = create_app(root=root, url_prefix="/tune/", sandbox=sandbox)
    layout = str(app.layout)
    for component_id in (
        "tune-image-source-store",
        "tune-image-source-modal",
        "tune-btn-pick-image-source",
        "tune-btn-image-source-confirm",
    ):
        assert component_id in layout


def test_curate_store_prefilled_from_run_images_dir(tmp_path: Path) -> None:
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune import _ids as ids

    images = tmp_path / "calibration"
    images.mkdir()
    root = _run_with_images(tmp_path, images)
    sandbox = SandboxRoot.from_path(tmp_path)
    app = create_app(root=root, url_prefix="/tune/", sandbox=sandbox)

    def _find_store(node: object) -> object | None:
        if getattr(node, "id", None) == ids.TUNE_IMAGE_SOURCE_STORE:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                found = _find_store(child)
                if found is not None:
                    return found
        elif children is not None:
            return _find_store(children)
        return None

    store = _find_store(app.layout)
    assert store is not None
    assert store.data == str(images)


def test_curate_store_does_not_prefill_out_of_sandbox_marker_path(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune import _ids as ids

    outside = tmp_path.parent / "outside-plates"
    outside.mkdir()
    root = _run_with_images(tmp_path, outside)
    sandbox = SandboxRoot.from_path(tmp_path)
    app = create_app(root=root, url_prefix="/tune/", sandbox=sandbox)

    def _find_store(node: object) -> object | None:
        if getattr(node, "id", None) == ids.TUNE_IMAGE_SOURCE_STORE:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                found = _find_store(child)
                if found is not None:
                    return found
        elif children is not None:
            return _find_store(children)
        return None

    store = _find_store(app.layout)
    assert store is not None
    assert store.data is None


def test_plate_listing_rejects_out_of_sandbox_source(tmp_path: Path) -> None:
    from phenotypic.gui.tune._callbacks import _list_plate_names

    outside = tmp_path.parent / "outside-list"
    outside.mkdir()
    (outside / "secret.tif").write_bytes(b"not an image")
    sandbox = SandboxRoot.from_path(tmp_path)

    assert _list_plate_names(str(outside), sandbox=sandbox) == []


def test_curate_prompt_when_image_source_unset(tmp_path: Path) -> None:
    """A run with no images_dir shows a 'point me at the plate images' prompt."""
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tools_ import trials_parquet_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    parquet = trials_parquet_path(tmp_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[Trial(number=0, params={}, score=0.5, terms={}, n_images=1)]
    ).to_parquet(parquet)
    # Legacy root: discovered with images_dir is None.
    root = TuneRunRoot.discover(tmp_path)
    assert root.images_dir is None

    sandbox = SandboxRoot.from_path(tmp_path)
    app = create_app(root=root, url_prefix="/tune/", sandbox=sandbox)
    layout = str(app.layout)
    assert "tune-curate-prompt" in layout


def test_tune_image_source_builds_shared_source_payload(tmp_path: Path) -> None:
    from phenotypic.gui.shell._source_context import source_payload_from_path
    from phenotypic.gui.tune._callbacks import (
        _source_payload_for_tune_image_source,
    )

    plates = tmp_path / "plates"
    plates.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)

    payload = _source_payload_for_tune_image_source(sandbox, str(plates), None)
    expected = source_payload_from_path(sandbox, plates, source="tune")

    assert payload == expected
    assert payload is not None
    assert payload["source"] == "tune"


def test_shared_source_initializes_tune_when_no_current_source(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import source_payload_from_path
    from phenotypic.gui.tune._callbacks import _tune_image_source_from_shared

    plates = tmp_path / "plates"
    plates.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = source_payload_from_path(sandbox, plates, source="manual")

    assert _tune_image_source_from_shared(sandbox, payload, None) == str(
        plates.resolve()
    )


def test_shared_source_does_not_override_bound_tune_source(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._source_context import source_payload_from_path
    from phenotypic.gui.tune._callbacks import _tune_image_source_from_shared

    plates = tmp_path / "plates"
    bound = tmp_path / "bound"
    plates.mkdir()
    bound.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = source_payload_from_path(sandbox, plates, source="manual")

    assert _tune_image_source_from_shared(sandbox, payload, str(bound)) is None
