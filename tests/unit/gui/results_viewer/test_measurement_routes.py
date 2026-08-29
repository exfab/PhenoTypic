"""The per-image measurement JSON route: the allow-list, and the error contract.

The property that matters most here is negative. ``column`` is user-supplied
and the viewer is unauthenticated, so the test that earns its place is
:func:`test_an_undeclared_column_never_opens_the_parquet` -- it deletes the
payload and still expects a 400, which no implementation that reads first and
validates after can pass.

The rest pins the contract ``/zarr/`` already established, because Plate and
Colony must agree about one store: absent store or absent ``tables``
descriptor -> 404, undecodable store -> 422, bad column -> 400.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import polars as pl
import pytest

from phenotypic import Image
from phenotypic.gui.results_viewer._measurement_routes import (
    register_measurement_routes,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import IMAGE, OBJECT
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    PreparedEmbeddedMeasurementTable,
    zarr_store_path,
)
from phenotypic.sdk_.ngff_ import PhenotypicAttr, STORE_ROOT_JSON

from tests._output_layout import write_master

DATASET = "d1"
MEASURED_STEM = "img-measured"
UNMEASURED_STEM = "img-unmeasured"

#: Values the fixture store's embedded table carries, keyed by
#: ``Object_Label``. Spelled once so the route assertions and the join
#: assertion cannot drift apart.
AREAS: dict[int, float] = {1: 12.0, 2: 512.0, 3: 128.0}


def _table() -> PreparedEmbeddedMeasurementTable:
    """An embedded table carrying one numeric and one string column."""
    labels = sorted(AREAS)
    return PreparedEmbeddedMeasurementTable(
        frame=pd.DataFrame(
            {
                str(OBJECT.LABEL): labels,
                "Shape_Area": [AREAS[label] for label in labels],
                "ColorLab_MedoidColorHex": ["#a08866"] * len(labels),
            }
        ),
        measurement_columns=(
            str(OBJECT.LABEL),
            "Shape_Area",
            "ColorLab_MedoidColorHex",
        ),
        join_status="not_requested",
        join_keys=(),
        metadata_snapshot_sha256="",
    )


@pytest.fixture(scope="module")
def run_template(tmp_path_factory) -> Path:
    """One run with a measured store and an unmeasured one.

    The unmeasured store is not a defect: a ``--mode process`` run never
    writes a ``tables`` descriptor, and the route must answer that with 404
    rather than pretending measurement is pending.
    """
    root = tmp_path_factory.mktemp("measurement-route-run")
    write_master(
        root,
        pl.DataFrame(
            {
                "Metadata_Dataset": [DATASET, DATASET],
                str(IMAGE.IMAGE_NAME): [MEASURED_STEM, UNMEASURED_STEM],
                "Object_Label": [1, 1],
                "Bbox_CenterRR": [16.0, 16.0],
                "Bbox_CenterCC": [16.0, 16.0],
            }
        ),
    )
    (root / "results" / DATASET / "measurements").mkdir(parents=True)
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    measured = zarr_store_path(root, DATASET, MEASURED_STEM)
    measured.parent.mkdir(parents=True, exist_ok=True)
    Image(arr=pixels).save2zarr(measured, measurement_table=_table())
    Image(arr=pixels).save2zarr(zarr_store_path(root, DATASET, UNMEASURED_STEM))
    return root


class RouteFixture:
    """A registered measurement route over a private copy of a run."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_root = OutputRoot.discover(
            root, cache_root=root.parent / ".test-phenotypic-viewer-cache"
        )
        app = dash.Dash(f"measurement-routes-{id(self)}")
        # Dash 4.x validates the layout in a before_request hook; a trivial
        # layout keeps that from 500-ing before the blueprint is reached.
        app.layout = dash.html.Div()
        register_measurement_routes(app, self.output_root)
        self.client = app.server.test_client()

    def store(self, stem: str) -> Path:
        return zarr_store_path(self.root, DATASET, stem)

    def get(self, stem: str = MEASURED_STEM, **params):
        query = "&".join(f"{key}={value}" for key, value in params.items())
        return self.client.get(f"/measurements/{DATASET}/{stem}?{query}")


@pytest.fixture
def route(run_template: Path, tmp_path: Path) -> RouteFixture:
    root = tmp_path / "run"
    shutil.copytree(run_template, root, symlinks=True)
    return RouteFixture(root)


# ---------------------------------------------------------------------------
# The happy path, and the join it rests on
# ---------------------------------------------------------------------------


def test_serves_one_column_keyed_by_object_label(route: RouteFixture) -> None:
    """Every declared label maps to the value written for THAT label."""
    resp = route.get(column="Shape_Area")
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    assert payload["column"] == "Shape_Area"
    assert payload["n"] == len(AREAS)
    assert payload["min"] == min(AREAS.values())
    assert payload["max"] == max(AREAS.values())
    # JSON object keys are strings; the value must still reach the label it
    # was measured for.
    assert payload["values"] == {
        str(label): value for label, value in AREAS.items()
    }


def test_serves_only_the_requested_column(route: RouteFixture) -> None:
    """The other columns of the table do not ride along."""
    payload = json.loads(route.get(column="Shape_Area").data)
    assert set(payload) == {"column", "values", "min", "max", "n"}
    assert "ColorLab_MedoidColorHex" not in json.dumps(payload)


def test_non_finite_values_are_valid_json_nulls(
    route: RouteFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NaN and infinities never leak as non-standard JSON tokens."""
    from phenotypic.gui.results_viewer import _measurement_routes as routes

    monkeypatch.setattr(
        routes,
        "read_embedded_measurement_column",
        lambda *_args: {1: float("nan"), 2: float("inf"), 3: float("-inf")},
    )
    response = route.get(column="Shape_Area")
    assert response.status_code == 200
    assert b"NaN" not in response.data
    assert b"Infinity" not in response.data
    assert json.loads(response.data)["values"] == {
        "1": None,
        "2": None,
        "3": None,
    }


# ---------------------------------------------------------------------------
# The allow-list
# ---------------------------------------------------------------------------


def test_an_undeclared_column_is_400(route: RouteFixture) -> None:
    resp = route.get(column="Shape_NoSuchThing")
    assert resp.status_code == 400


def test_an_undeclared_column_never_opens_the_parquet(
    route: RouteFixture,
) -> None:
    """The check runs against the descriptor, before any file read.

    Deleting the payload makes the two orderings distinguishable: validate
    first and this is a 400; read first and it is a 500 (or a 404 from the
    ``OSError`` handler). Without this test a route that reads and then
    filters passes every other assertion in this module.
    """
    (route.store(MEASURED_STEM) / MEASUREMENT_TABLE_RELATIVE_PATH).unlink()
    assert route.get(column="Shape_NoSuchThing").status_code == 400


def test_a_path_traversal_column_is_400(route: RouteFixture) -> None:
    """A column name is a closed value set, never a path."""
    assert route.get(column="../../zarr.json").status_code == 400


def test_a_non_numeric_column_is_400(route: RouteFixture) -> None:
    """``ColorLab_MedoidColorHex`` is declared, and still has no scale."""
    assert route.get(column="ColorLab_MedoidColorHex").status_code == 400


def test_a_missing_column_parameter_is_400(route: RouteFixture) -> None:
    assert route.client.get(
        f"/measurements/{DATASET}/{MEASURED_STEM}"
    ).status_code == 400


def test_an_unsafe_path_component_is_400(route: RouteFixture) -> None:
    assert route.client.get(
        f"/measurements/{DATASET}/..?column=Shape_Area"
    ).status_code in (400, 404)


# ---------------------------------------------------------------------------
# The error contract, matching ``/zarr/``
# ---------------------------------------------------------------------------


def test_an_unknown_image_is_404(route: RouteFixture) -> None:
    assert route.get(stem="img-nope", column="Shape_Area").status_code == 404


def test_a_store_with_no_tables_descriptor_is_404(
    route: RouteFixture,
) -> None:
    """A ``--mode process`` run never measures. That is 404, not 'pending'."""
    resp = route.get(stem=UNMEASURED_STEM, column="Shape_Area")
    assert resp.status_code == 404


def test_a_store_this_build_cannot_decode_is_422(
    route: RouteFixture,
) -> None:
    """422, not 404: a decode failure is run-wide and actionable.

    404 would tell the user "no such image", which is false, and would make
    this route disagree with ``/zarr/`` and ``crop_colony`` about one store.
    """
    root_json = route.store(MEASURED_STEM) / STORE_ROOT_JSON
    document = json.loads(root_json.read_text(encoding="utf-8"))
    block = document["attributes"][PhenotypicAttr.ROOT]
    block[PhenotypicAttr.STORE_SCHEMA_VERSION] = 999_999
    root_json.write_text(json.dumps(document), encoding="utf-8")
    assert route.get(column="Shape_Area").status_code == 422


def test_a_store_with_no_phenotypic_block_is_404(
    route: RouteFixture,
) -> None:
    """``require_readable_store`` raises ``KeyError``, which is no ``OSError``.

    Unnamed, that reaches Flask as a 500 -- and with ``--debug`` plus the
    documented ``--host 0.0.0.0`` a 500 is the Werkzeug interactive debugger.
    """
    root_json = route.store(MEASURED_STEM) / STORE_ROOT_JSON
    document = json.loads(root_json.read_text(encoding="utf-8"))
    document["attributes"].pop(PhenotypicAttr.ROOT)
    root_json.write_text(json.dumps(document), encoding="utf-8")
    assert route.get(column="Shape_Area").status_code == 404
