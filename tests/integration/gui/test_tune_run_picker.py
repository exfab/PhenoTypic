"""Integration tests for the runtime tune run-picker / bind (Chunk C).

The hub mounts ``/tune/`` in its empty (run-unbound) state; the user binds a
tune output directory from the page itself. These tests close the gap the
headless unit tests miss: the bind callback can be correctly *implemented* (its
pure :func:`~phenotypic.gui.tune._run_picker.discover_run_payload` helper passes)
yet *mis-wired* — a wrong-arity confirm closure, a store-write that 500s, or a
body-swap that fails to render the loaded views. Those only surface on the real
``/_dash-update-component`` round trip, so each test POSTs the confirm trigger
through the Flask test client (the "Flask client, no browser" pattern from
``test_tune_callback_wiring.py``) and asserts the response shape.

Covered:

* **Empty-state surface** — ``create_app(root=None, sandbox=...)`` renders the
  Bind-run button, the run-root store, the swappable page body, and the picker
  modal, but NO loaded-view component yet.
* **Bind success** — confirming a real tune output writes ``{"path": ...}`` into
  ``tune-run-root-store`` AND swaps the page body to a layout carrying the
  loaded views (the Monitor objective figure id appears).
* **Bind rejection** — confirming a non-tune directory returns a 200 with a
  clear ``tune-run-picker-note`` and leaves the store / body untouched
  (``no_update``), never a 500.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from dash.development.base_component import Component

from phenotypic.gui.shell import SandboxRoot


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_tune_run(run_dir: Path) -> None:
    """Write a discoverable, optuna-free tune output under ``run_dir``.

    A ``.pht-tune-cache/run.json`` marker (the run-START sidecar) with a null
    ``storage_url`` plus a 2-trial ``trials.parquet`` journal and a
    ``tuning_spec.json`` — enough for ``TuneRunRoot.discover`` to bind and for
    the loaded views (Monitor / Space) to render against a real spec + journal.
    """
    from phenotypic import ImagePipeline
    from phenotypic.analysis import ExpectedVsDetectedCount
    from phenotypic.detect import OtsuDetector
    from phenotypic.enhance import GaussianBlur
    from phenotypic.sdk_ import (
        trials_parquet_path,
        tune_cache_run_marker_path,
        tuning_spec_path,
    )
    from phenotypic.tune import (
        Budget,
        Categorical,
        Evaluator,
        Knob,
        QCScorer,
        RandomConfig,
        SearchSpace,
        TuningSpec,
    )
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    run_dir.mkdir(parents=True, exist_ok=True)

    marker = tune_cache_run_marker_path(run_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "storage_url": None,
                "study_name": "tune",
                "is_multi_objective": False,
                "images_dir": None,
            }
        )
    )

    csv = run_dir / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["plate1"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),)
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(csv), groupby=["Metadata_ImageName"]
            )
        ),
        evaluator=Evaluator(),
        strategy=RandomConfig(n_trials=5),
        budget=Budget(n_trials=5),
    )
    spec_path = tuning_spec_path(run_dir)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(spec.model_dump_json(indent=2))

    parquet = trials_parquet_path(run_dir)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[
            Trial(number=0, params={"0.sigma": 3.0}, score=0.30, terms={}, n_images=2),
            Trial(number=1, params={"0.sigma": 2.0}, score=0.60, terms={}, n_images=2),
        ]
    ).to_parquet(parquet)


def _empty_state_app(sandbox_root: Path):  # type: ignore[no-untyped-def]
    """A tune app mounted empty-state (``root=None``) with a bound sandbox."""
    from phenotypic.gui.tune import create_app

    sandbox = SandboxRoot.from_path(sandbox_root)
    return create_app(root=None, url_prefix="/tune/", sandbox=sandbox)


def _walk(component: Component):
    yield component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            if isinstance(child, Component):
                yield from _walk(child)
    elif isinstance(children, Component):
        yield from _walk(children)


# ---------------------------------------------------------------------------
# Dash dispatch helpers (mirrors test_tune_callback_wiring)
# ---------------------------------------------------------------------------

def _find_output_key(app, *id_property_substrings: str) -> str:
    for key in app.callback_map:
        if all(sub in key for sub in id_property_substrings):
            return key
    raise KeyError(id_property_substrings)


def _outputs_from_key(output_key: str) -> list[dict[str, str]]:
    import re

    body = output_key.strip(".")
    outputs: list[dict[str, str]] = []
    for seg in re.split(r"\.\.\.", body):
        seg = seg.strip(".").split("@", 1)[0]
        component_id, prop = seg.rsplit(".", 1)
        outputs.append({"id": component_id, "property": prop})
    return outputs


def _post_confirm(app, browse_dir: str):  # type: ignore[no-untyped-def]
    """POST the run-picker confirm trigger; return the parsed Dash response."""
    client = app.server.test_client()
    out_key = _find_output_key(
        app, "tune-run-root-store.data", "tune-page-body.children"
    )
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": _outputs_from_key(out_key),
            "inputs": [
                {
                    "id": "tune-btn-run-picker-confirm",
                    "property": "n_clicks",
                    "value": 1,
                }
            ],
            "state": [
                {
                    "id": "tune-run-picker-browse-dir",
                    "property": "data",
                    "value": browse_dir,
                }
            ],
            "changedPropIds": ["tune-btn-run-picker-confirm.n_clicks"],
        },
    )
    return resp


# ---------------------------------------------------------------------------
# Empty-state surface
# ---------------------------------------------------------------------------

def test_empty_state_renders_picker_store_body_modal(tmp_path: Path) -> None:
    """The empty-state mount carries picker, store, body, and poll placeholders."""
    app = _empty_state_app(tmp_path)
    layout = str(app.layout)
    for needed in (
        "tune-btn-pick-run",
        "tune-run-root-store",
        "tune-page-body",
        "tune-run-picker-modal",
        "tune-subtab-monitor",
    ):
        assert needed in layout, f"empty-state layout missing {needed!r}"
    # The empty Monitor destination is pollable for registry/log updates, but
    # loaded-only Launch/Curate/Space components wait until a run is bound.
    component_ids = {getattr(component, "id", None) for component in _walk(app.layout)}
    assert "tune-study-poll" in component_ids
    assert "tune-objective-figure" in component_ids
    assert "tune-launch-command" not in component_ids


# ---------------------------------------------------------------------------
# Bind success
# ---------------------------------------------------------------------------

def test_bind_run_populates_store_and_swaps_in_loaded_views(tmp_path: Path) -> None:
    """Confirming a real tune output writes the store AND renders the loaded body."""
    run_dir = tmp_path / "tune_run"
    _make_tune_run(run_dir)
    sandbox = SandboxRoot.from_path(tmp_path)
    resolved_run = str(sandbox.resolve("tune_run"))

    app = _empty_state_app(tmp_path)
    resp = _post_confirm(app, resolved_run)
    assert resp.status_code == 200
    response = resp.get_json()["response"]

    # The run-root store now carries the discovered run path.
    assert response["tune-run-root-store"]["data"] == {"path": resolved_run}
    # The picker label + cleared note + closed modal resolved over HTTP.
    assert response["tune-run-picker-label"]["children"] == resolved_run
    assert response["tune-run-picker-note"]["children"] == ""
    assert response["tune-run-picker-modal"]["is_open"] is False
    assert response["tune-active-destination-store"]["data"] == "monitor"
    assert "tune-view-hidden" not in response["tune-destview-monitor"]["className"]
    # The page body was swapped to the loaded four-view layout: its serialised
    # children carry the Monitor objective figure + the Launch command ids.
    body_str = json.dumps(response["tune-page-body"]["children"])
    assert "tune-objective-figure" in body_str
    assert "tune-view-monitor" in body_str
    assert "tune-launch-command" in body_str


# ---------------------------------------------------------------------------
# Bind rejection (note, not a 500)
# ---------------------------------------------------------------------------

def test_bind_non_tune_dir_returns_note_not_500(tmp_path: Path) -> None:
    """Confirming a non-tune directory yields a clear note and leaves state alone."""
    plain = tmp_path / "just_a_folder"
    plain.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    resolved_plain = str(sandbox.resolve("just_a_folder"))

    app = _empty_state_app(tmp_path)
    resp = _post_confirm(app, resolved_plain)
    # Crucially a 200 (a clear note), never a 500.
    assert resp.status_code == 200
    response = resp.get_json()["response"]
    note = response["tune-run-picker-note"]["children"]
    assert "Not a tune output" in note
    # The store + body + label were left untouched (no_update → absent from the
    # response, or unchanged), so nothing bound.
    assert "tune-run-root-store" not in response
    assert "tune-page-body" not in response


def test_bind_out_of_sandbox_dir_is_refused(tmp_path: Path) -> None:
    """An out-of-sandbox path is refused with a note, never a 500 or a bind."""
    app = _empty_state_app(tmp_path)
    # An absolute path outside the sandbox root.
    resp = _post_confirm(app, "/etc")
    assert resp.status_code == 200
    response = resp.get_json()["response"]
    note = response["tune-run-picker-note"]["children"]
    assert "escapes the sandbox" in note
    assert "tune-run-root-store" not in response
