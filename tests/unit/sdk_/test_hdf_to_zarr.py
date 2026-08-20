"""Legacy HDF -> store conversion must equal a freshly written store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.sdk_._hdf_to_zarr import (
    migrate_hdf_to_zarr,
    migrate_run_hdf_to_zarr,
)
from phenotypic.sdk_.ngff_ import (
    PhenotypicAttr,
    read_phenotypic_attributes,
    valid_staged_store,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_hdf"


@pytest.mark.parametrize(
    "layout",
    ["v1_flat", "v2_grouped", "v2_enh_gray", "v2_grid", "v2_image_type", "v2_work_id"],
)
def test_conversion_produces_a_valid_store(layout: str, tmp_path: Path) -> None:
    store = migrate_hdf_to_zarr(FIXTURES / layout / "img.h5", tmp_path / "img.ome.zarr")
    assert valid_staged_store(store) is True


@pytest.mark.parametrize("layout", ["v1_flat", "v2_grouped", "v2_enh_gray"])
def test_converted_store_conforms(layout: str, tmp_path: Path) -> None:
    from tests._ngff_conformance import assert_store_conforms

    assert_store_conforms(
        migrate_hdf_to_zarr(FIXTURES / layout / "img.h5", tmp_path / "img.ome.zarr")
    )


def test_enh_gray_maps_to_detect_mat(tmp_path: Path) -> None:
    """The pre-rename layer name; dropping it silently loses the detection matrix."""
    store = migrate_hdf_to_zarr(
        FIXTURES / "v2_enh_gray" / "img.h5", tmp_path / "img.ome.zarr"
    )
    block = read_phenotypic_attributes(store)
    assert "detect_mat" in block[PhenotypicAttr.SERIES]
    assert "enh_gray" not in block[PhenotypicAttr.SERIES]
    assert Image.load_layer_zarr(store, "detect_mat").any()


def test_converted_store_matches_the_fixture_s_AUTHORED_metadata(
    tmp_path: Path,
) -> None:
    """Assert against what the fixture was BUILT with, not against another load.

    A comparison of ``migrate_hdf_to_zarr(...)`` against
    ``Image.load_hdf5(...).save2zarr(...)`` runs both sides through the SAME
    ``_load_v2_grouped``, so any loader-level fidelity loss compares equal and
    the test certifies the bug (ledger MIG-2).
    """
    from tests.fixtures.legacy_hdf._generate import V2_IMAGE_TYPE_AUTHORED

    store = migrate_hdf_to_zarr(
        FIXTURES / "v2_image_type" / "img.h5", tmp_path / "converted.ome.zarr"
    )
    protected = read_phenotypic_attributes(store)[PhenotypicAttr.METADATA][
        "protected"
    ]
    assert protected["Metadata_ImageType"] == V2_IMAGE_TYPE_AUTHORED


def test_grid_state_survives_conversion(tmp_path: Path) -> None:
    """``grid_finder is not None`` is NOT the assertion -- it never fails.

    ``GridImage.__init__`` mints a ``CenteredAutoGridFinder`` whenever none is
    supplied, so a conversion that dropped the stored finder entirely still
    produces a non-``None`` one with the right ``nrows``/``ncols``. The fixture
    therefore carries a *different* class with a *non-default* parameter, and
    this asserts on both. Verified: dropping the stored ``grid_finder_json``
    survived the weaker assertion.
    """
    from phenotypic import GridImage

    from tests.fixtures.legacy_hdf._generate import (
        V2_GRID_FINDER_CLASS,
        V2_GRID_FINDER_RESIDUAL_FRACTION,
        V2_GRID_SHAPE,
    )

    store = migrate_hdf_to_zarr(
        FIXTURES / "v2_grid" / "img.h5", tmp_path / "grid.ome.zarr"
    )
    back = GridImage.load_zarr(store)
    assert (back.nrows, back.ncols) == V2_GRID_SHAPE
    assert type(back.grid_finder).__name__ == V2_GRID_FINDER_CLASS
    assert back.grid_finder.residual_fraction == V2_GRID_FINDER_RESIDUAL_FRACTION

    block = read_phenotypic_attributes(store)[PhenotypicAttr.GRID]
    assert block["nrows"] == V2_GRID_SHAPE[0]
    assert block["ncols"] == V2_GRID_SHAPE[1]
    assert block["grid_finder"]["class"] == V2_GRID_FINDER_CLASS


def test_work_id_survives_conversion(tmp_path: Path) -> None:
    """FLOW-1: without this, every migrated image reclassifies "stage1"."""
    from phenotypic._cli._cli_staged_resume import staged_store_matches_work_id

    from tests.fixtures.legacy_hdf._generate import V2_WORK_ID_AUTHORED

    store = migrate_hdf_to_zarr(
        FIXTURES / "v2_work_id" / "img.h5", tmp_path / "wid.ome.zarr"
    )
    assert staged_store_matches_work_id(store, V2_WORK_ID_AUTHORED) is True


def test_converted_equals_a_freshly_written_store(tmp_path: Path) -> None:
    """Structural equivalence only -- see the test above for content fidelity."""
    converted = migrate_hdf_to_zarr(
        FIXTURES / "v2_grouped" / "img.h5", tmp_path / "converted.ome.zarr"
    )
    fresh = Image.load_hdf5(FIXTURES / "v2_grouped" / "img.h5").save2zarr(
        tmp_path / "fresh.ome.zarr"
    )
    for layer in ("gray", "detect_mat", "objmap"):
        np.testing.assert_array_equal(
            Image.load_layer_zarr(converted, layer),
            Image.load_layer_zarr(fresh, layer),
        )
    a = read_phenotypic_attributes(converted)
    b = read_phenotypic_attributes(fresh)
    for key in (PhenotypicAttr.SERIES, PhenotypicAttr.LABELS, PhenotypicAttr.METADATA):
        assert a[key] == b[key]


def test_legacy_headers_are_canonicalized_in_the_same_pass(tmp_path: Path) -> None:
    store = migrate_hdf_to_zarr(
        FIXTURES / "v1_flat" / "img.h5", tmp_path / "img.ome.zarr"
    )
    block = read_phenotypic_attributes(store)
    for section in ("protected", "public"):
        for key in block[PhenotypicAttr.METADATA][section]:
            assert key.startswith("Metadata_"), key


def test_a_store_written_from_legacy_headers_comes_out_canonical(
    tmp_path: Path,
) -> None:
    """The invariant that makes header-only store migration unnecessary.

    Task 5.5 was cut because a legacy per-topic header cannot survive into a
    store: both paths that reach ``_metadata`` canonicalize first. This pins
    the accessor half -- ``MetadataAccessor._resolve_key`` maps a known legacy
    spelling to its schema member before storing. The ingest half is pinned by
    ``test_legacy_headers_are_canonicalized_in_the_same_pass`` above, which
    converts a v1-flat file whose stored headers are legacy.

    If either stops canonicalizing, this fails and the cut has to be
    revisited -- rather than silently shipping stores with legacy headers and
    no migration path.

    Note:
        Assigning through the private ``image._metadata.public`` dict is NOT
        this invariant and is deliberately not tested as one. ``save2zarr``
        writes the sections verbatim, so a direct poke at that dict does reach
        the store unchanged -- but nothing in PhenoTypic writes metadata that
        way, and the HDF writer's normalize-on-write was never a documented
        contract.
    """
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    from tests.fixtures.legacy_hdf._generate import (
        V1_CANONICAL_PUBLIC_HEADER,
        V1_LEGACY_PUBLIC_HEADER,
        V1_PUBLIC_VALUE,
    )

    image = Image(load_synth_yeast_plate())
    image.metadata[V1_LEGACY_PUBLIC_HEADER] = V1_PUBLIC_VALUE
    store = image.save2zarr(tmp_path / "legacy.ome.zarr")

    public = read_phenotypic_attributes(store)[PhenotypicAttr.METADATA]["public"]
    assert V1_LEGACY_PUBLIC_HEADER not in public
    assert public[V1_CANONICAL_PUBLIC_HEADER] == V1_PUBLIC_VALUE


def test_source_is_retained_by_default(tmp_path: Path) -> None:
    src = tmp_path / "img.h5"
    src.write_bytes((FIXTURES / "v2_grouped" / "img.h5").read_bytes())
    migrate_hdf_to_zarr(src, tmp_path / "img.ome.zarr")
    assert src.is_file()


def test_source_deletion_is_opt_in(tmp_path: Path) -> None:
    src = tmp_path / "img.h5"
    src.write_bytes((FIXTURES / "v2_grouped" / "img.h5").read_bytes())
    migrate_hdf_to_zarr(src, tmp_path / "img.ome.zarr", keep_source=False)
    assert not src.exists()


def test_run_migration_is_idempotent(legacy_run: Path) -> None:
    """Re-running after an interruption is the recovery procedure."""
    first = migrate_run_hdf_to_zarr(legacy_run)
    second = migrate_run_hdf_to_zarr(legacy_run)
    assert first.converted > 0
    assert second.converted == 0
    assert second.skipped == first.converted


def test_dry_run_writes_nothing(legacy_run: Path) -> None:
    from phenotypic.sdk_ import dataset_zarr_dir

    report = migrate_run_hdf_to_zarr(legacy_run, dry_run=True)
    assert report.converted > 0
    assert not dataset_zarr_dir(legacy_run, "ds").exists()


def test_a_failed_conversion_is_reported_not_raised(legacy_run: Path) -> None:
    corrupt = next((legacy_run / "results").rglob("*.h5"))
    corrupt.write_bytes(b"not an hdf")
    report = migrate_run_hdf_to_zarr(legacy_run)
    assert len(report.failed) == 1
    assert report.failed[0][0] == corrupt


# ---------------------------------------------------------------------------
# Generator fidelity -- this window closes at Phase 6
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(Image, "save2hdf5"),
    reason=(
        "Image.save2hdf5 is gone (Phase 6). The legacy_hdf goldens can no "
        "longer be checked against production; see "
        "tests/fixtures/legacy_hdf/_generate.py."
    ),
)
@pytest.mark.parametrize("subject", ["image", "grid"])
def test_the_generator_matches_the_real_writer(subject: str, tmp_path: Path) -> None:
    """One-time fidelity check, only possible before Phase 6 deletes save2hdf5.

    After that, the goldens are unfalsifiable against production forever. The
    comparison covers every group's and dataset's shape, dtype **and attrs**,
    plus the root attrs -- so ``detect_mode`` and the encoded metadata are in
    scope, not just the array skeleton.
    """
    import h5py

    from tests.fixtures.legacy_hdf._generate import (
        build_fixture_grid_image,
        build_fixture_image,
        write_v2_grid,
        write_v2_grouped,
    )

    if subject == "image":
        image = build_fixture_image()
        writer = write_v2_grouped
    else:
        image = build_fixture_grid_image()
        writer = write_v2_grid

    real, generated = tmp_path / "real.h5", tmp_path / "generated.h5"
    image.save2hdf5(real)
    writer(generated, image)

    def _shape(path: Path):
        out: dict = {}
        with h5py.File(path, "r") as fh:
            fh.visititems(
                lambda name, obj: out.__setitem__(
                    name,
                    (
                        getattr(obj, "shape", None),
                        str(getattr(obj, "dtype", "")),
                        dict(obj.attrs),
                    ),
                )
            )
            return out, dict(fh.attrs)

    assert _shape(generated) == _shape(real)


# ---------------------------------------------------------------------------
# Data loss -- compared against the SOURCE, not against another store
# ---------------------------------------------------------------------------


def _source_layers(src: Path) -> dict[str, np.ndarray]:
    """Read every image layer straight out of a legacy ``.h5`` with h5py.

    An independent oracle. Every other fidelity test here compares a converted
    store against a *freshly written* one -- both through ``save2zarr``, so a
    writer that drops a whole series drops it on both sides and the comparison
    passes. Verified: that mutation survived the suite until this existed.
    """
    import h5py

    with h5py.File(src, "r") as handle:
        group = (
            handle["layers"]
            if int(handle.attrs.get("schema_version", 1)) >= 2
            and "layers" in handle
            else handle
        )
        return {
            name: group[name][()]
            for name in ("rgb", "gray", "detect_mat", "enh_gray", "objmap")
            if name in group
        }


@pytest.mark.parametrize("layout", ["v1_flat", "v2_grouped", "v2_enh_gray"])
def test_every_source_layer_reaches_the_store(layout: str, tmp_path: Path) -> None:
    """No layer the ``.h5`` held may be missing or altered in the store.

    ``--delete-sources`` makes conversion irreversible, so "the store carries
    everything the file did" has to be checked against the file itself.
    """
    src = FIXTURES / layout / "img.h5"
    store = migrate_hdf_to_zarr(src, tmp_path / "img.ome.zarr")

    series = read_phenotypic_attributes(store)[PhenotypicAttr.SERIES]
    labels = read_phenotypic_attributes(store)[PhenotypicAttr.LABELS]
    assert set(series) == {"rgb", "gray", "detect_mat"}, series
    assert "objmap" in labels, labels

    for name, expected in _source_layers(src).items():
        # The pre-rename layer is the SAME data under the current name.
        target = "detect_mat" if name == "enh_gray" else name
        # ``load_layer_zarr`` already returns rgb as (H, W, C), which is the
        # layout the HDF held, so no axis handling belongs here.
        actual = Image.load_layer_zarr(store, target)
        np.testing.assert_array_equal(
            actual, expected, err_msg=f"{layout}: layer {name!r} did not survive"
        )


def _source_metadata(src: Path) -> dict[str, dict[str, object]]:
    """Read the metadata sections straight out of a legacy ``.h5``.

    The second half of the independent oracle. Keys are canonicalized with the
    **public** ``ensure_metadata_prefix``, deliberately not with the private
    ``_normalize_stored_metadata_items`` the loaders use -- an oracle sharing
    the machinery under test proves nothing about it.
    """
    import json

    import h5py

    from phenotypic.sdk_ import ensure_metadata_prefix

    def _value(raw: object) -> object:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return int(raw) if raw.isdigit() else raw
        return raw

    sections: dict[str, dict[str, object]] = {
        "protected": {},
        "public": {},
        "imported": {},
    }
    with h5py.File(src, "r") as handle:
        if int(handle.attrs.get("schema_version", 1)) >= 2 and "metadata" in handle:
            for name in sections:
                if name in handle["metadata"]:
                    attrs = handle["metadata"][name].attrs
                    sections[name] = {
                        ensure_metadata_prefix(str(key)): _value(attrs[key])
                        for key in attrs
                    }
        else:
            for name, group in (
                ("protected", "protected_metadata"),
                ("public", "public_metadata"),
            ):
                if group in handle:
                    attrs = handle[group].attrs
                    sections[name] = {
                        ensure_metadata_prefix(str(key)): _value(attrs[key])
                        for key in attrs
                    }
    return sections


@pytest.mark.parametrize("layout", ["v1_flat", "v2_grouped", "v2_image_type"])
def test_every_source_metadata_key_reaches_the_store(
    layout: str, tmp_path: Path
) -> None:
    """No metadata the ``.h5`` held may be missing or altered in the store.

    Compared against the FILE, not against a second store: a writer that drops
    a whole section drops it on both sides of a store-to-store comparison.
    """
    src = FIXTURES / layout / "img.h5"
    store = migrate_hdf_to_zarr(src, tmp_path / "img.ome.zarr")
    stored = read_phenotypic_attributes(store)[PhenotypicAttr.METADATA]

    for section, expected in _source_metadata(src).items():
        for key, value in expected.items():
            assert key in stored[section], (
                f"{layout}: {section} lost {key!r}"
            )
            assert stored[section][key] == value, (
                f"{layout}: {section}[{key!r}] changed "
                f"{value!r} -> {stored[section][key]!r}"
            )
