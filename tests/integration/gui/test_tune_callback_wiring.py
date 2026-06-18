"""Flask-test-client callback-wiring tests for the tune sub-app (no browser).

These close the gap the headless unit tests miss: a Dash server callback can be
correctly *implemented* (its pure helper passes) yet *mis-wired* — wrong output
arity, a store-write that 500s, or a ``State`` that isn't reachable from the
active sub-view. Those only surface on the real ``/_dash-update-component``
round trip, so each test POSTs a representative trigger and asserts a 200 with
the expected output shape (the "Flask client, no browser" pattern from
``tests/integration/gui/test_lifecycle.py``).

Covered callbacks:

* **Monitor poll** (``tune-study-poll`` → objective/importance/badge/table/note)
  — confirms the poll reads ``tune-run-root-store`` as ``State`` from the page
  root after the store was hoisted out of the Monitor sub-view.
* **Shortlist pin** (pattern-matching card click → ``tune-ab-store`` +
  per-card ``className``) — the arity/store-write contract: a wrong-arity pin
  closure 500s here even though the pure ``pinned_pair`` helper passes.
* **Set as winner** (``tune-btn-set-winner`` → winner note + toast) — drives the
  success path end-to-end (reads the spec's base pipeline, writes
  ``deliverables/best_pipeline.json``), confirming the multi-output
  ``allow_duplicate`` toast wiring resolves over HTTP.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from phenotypic.gui.shell import SandboxRoot


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _runnable_spec(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A round-trippable ``TuningSpec`` whose first op is a tunable GaussianBlur.

    Mirrors ``test_tune_space._runnable_spec`` so ``read_base_pipeline`` yields a
    base whose ``0.sigma`` the winner override lands on.
    """
    from phenotypic import ImagePipeline
    from phenotypic.analysis import ExpectedVsDetectedCount
    from phenotypic.detect import OtsuDetector
    from phenotypic.enhance import GaussianBlur
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

    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["plate1"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
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


def _wiring_app(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A loaded tune app over a 3-trial journal + a tuning_spec.json + sandbox."""
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.sdk_ import trials_parquet_path, tuning_spec_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    (tmp_path / "calibration").mkdir()

    spec = _runnable_spec(tmp_path)
    spec_path = tuning_spec_path(tmp_path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(spec.model_dump_json(indent=2))

    parquet = trials_parquet_path(tmp_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[
            # trial 0 overrides 0.sigma -> 3.0 (the winner check asserts it lands).
            Trial(number=0, params={"0.sigma": 3.0}, score=0.30, terms={}, n_images=2),
            Trial(number=1, params={"0.sigma": 2.0}, score=0.60, terms={}, n_images=2),
            Trial(number=2, params={"0.sigma": 1.0}, score=0.45, terms={}, n_images=2),
        ]
    ).to_parquet(parquet)

    root = TuneRunRoot.discover(tmp_path)
    sandbox = SandboxRoot.from_path(tmp_path)
    return create_app(root=root, url_prefix="/tune/", sandbox=sandbox)


# ---------------------------------------------------------------------------
# Dash dispatch helpers (build the /_dash-update-component payload from the
# app's own callback_map so allow_duplicate output hashes never have to be
# hand-encoded).
# ---------------------------------------------------------------------------

def _find_output_key(app, *id_property_substrings: str) -> str:
    """Return the callback_map key (the exact ``output`` string) containing all
    ``id_property_substrings`` — e.g. ``("tune-winner-note.children",)``."""
    for key in app.callback_map:
        if all(sub in key for sub in id_property_substrings):
            return key
    raise KeyError(id_property_substrings)


def _outputs_from_key(output_key: str) -> list[dict[str, str]]:
    """Parse a (possibly multi-output, possibly allow_duplicate) output key into
    the ``outputs`` list Dash's response formatter needs.

    The key shape is ``..a.prop...b.prop@hash...c.prop@hash..``; the ``@hash``
    suffix is the allow_duplicate disambiguator and is dropped (the response is
    keyed only by id + property).
    """
    body = output_key.strip(".")
    outputs: list[dict[str, str]] = []
    for seg in re.split(r"\.\.\.", body):
        seg = seg.strip(".").split("@", 1)[0]
        component_id, prop = seg.rsplit(".", 1)
        outputs.append({"id": component_id, "property": prop})
    return outputs


# ---------------------------------------------------------------------------
# Monitor poll
# ---------------------------------------------------------------------------

def test_monitor_poll_callback_returns_figures_over_http(tmp_path: Path) -> None:
    """The Monitor poll POST returns 200 with all six outputs filled.

    Also a hoist regression: the poll reads ``tune-run-root-store`` as ``State``;
    after the store moved to the page root it must still be reachable, so the
    badge resolves to a real label (not the empty placeholder).
    """
    app = _wiring_app(tmp_path)
    client = app.server.test_client()

    out_key = _find_output_key(
        app, "tune-objective-figure.figure", "tune-monitor-note.children"
    )
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": _outputs_from_key(out_key),
            "inputs": [
                {"id": "tune-study-poll", "property": "n_intervals", "value": 1}
            ],
            "state": [
                {
                    "id": "tune-run-root-store",
                    "property": "data",
                    "value": {"path": str(tmp_path)},
                }
            ],
            "changedPropIds": ["tune-study-poll.n_intervals"],
        },
    )
    assert resp.status_code == 200
    response = resp.get_json()["response"]
    # All six outputs present (the figures, badge, table, note).
    for component_id in (
        "tune-objective-figure",
        "tune-importance-figure",
        "tune-gap-badge",
        "tune-trials-table",
        "tune-monitor-note",
    ):
        assert component_id in response
    # The objective figure carries the running-best line + scatter (2 trials → a
    # real figure, not an exception).
    assert "data" in response["tune-objective-figure"]["figure"]
    # The badge resolved from the journal (store reachable from the page root).
    assert isinstance(response["tune-gap-badge"]["children"], str)


# ---------------------------------------------------------------------------
# Shortlist pin (pattern-matching → tune-ab-store)
# ---------------------------------------------------------------------------

def test_shortlist_pin_callback_writes_ab_store_over_http(tmp_path: Path) -> None:
    """Clicking a shortlist card POSTs 200 and writes the A/B store.

    The arity/store-write gap: ``_pin_card`` returns ``(store, [classes...])``
    against ``Output(tune-ab-store.data)`` + a pattern-matching ``className``
    list. A wrong-arity closure 500s here; the pure ``pinned_pair`` unit test
    would not catch it.
    """
    app = _wiring_app(tmp_path)
    client = app.server.test_client()

    def _card(trial: int) -> dict[str, object]:
        return {"trial": trial, "type": "tune-shortlist-card"}

    out_key = _find_output_key(app, "tune-ab-store.data", "tune-shortlist-card")
    # The multi (ALL) output resolves to one entry per shortlist card.
    outputs = [
        {"id": "tune-ab-store", "property": "data"},
        [{"id": _card(t), "property": "className"} for t in (0, 1, 2)],
    ]
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": outputs,
            # Card 1 was the one clicked (its n_clicks ticked to 1).
            "inputs": [
                [
                    {
                        "id": _card(t),
                        "property": "n_clicks",
                        "value": 1 if t == 1 else 0,
                    }
                    for t in (0, 1, 2)
                ]
            ],
            "state": [
                {
                    "id": "tune-ab-store",
                    "property": "data",
                    "value": {"a": None, "b": None},
                },
                [{"id": _card(t), "property": "id", "value": _card(t)} for t in (0, 1, 2)],
            ],
            "changedPropIds": ['{"trial":1,"type":"tune-shortlist-card"}.n_clicks'],
        },
    )
    assert resp.status_code == 200
    response = resp.get_json()["response"]
    # The store took trial 1 into the (empty) slot A.
    assert response["tune-ab-store"]["data"] == {"a": 1, "b": None}
    # The clicked card carries the slot-A highlight class.
    clicked = response['{"trial":1,"type":"tune-shortlist-card"}']["className"]
    assert "tune-shortlist-card-a" in clicked


# ---------------------------------------------------------------------------
# Set as winner (allow_duplicate multi-output → note + toast)
# ---------------------------------------------------------------------------

def test_set_winner_callback_writes_best_pipeline_over_http(tmp_path: Path) -> None:
    """Clicking "Set as winner" POSTs 200, writes best_pipeline.json, notes it.

    Drives the full success path over HTTP (read base from the spec → build →
    atomic write), exercising the multi-output ``allow_duplicate`` toast wiring.
    """
    from phenotypic import ImagePipeline
    from phenotypic.sdk_ import best_pipeline_path

    app = _wiring_app(tmp_path)
    client = app.server.test_client()

    out_key = _find_output_key(
        app, "tune-winner-note.children", "tune-curate-toast"
    )
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": _outputs_from_key(out_key),
            "inputs": [
                {"id": "tune-btn-set-winner", "property": "n_clicks", "value": 1}
            ],
            "state": [
                # Slot A = trial 0 (the winner is slot A).
                {"id": "tune-ab-store", "property": "data", "value": {"a": 0, "b": None}},
                {
                    "id": "tune-run-root-store",
                    "property": "data",
                    "value": {"path": str(tmp_path)},
                },
            ],
            "changedPropIds": ["tune-btn-set-winner.n_clicks"],
        },
    )
    assert resp.status_code == 200
    note = resp.get_json()["response"]["tune-winner-note"]["children"]
    assert "trial 0" in note

    # The winner was written and the trial-0 override (0.sigma -> 3.0) landed.
    written = best_pipeline_path(tmp_path)
    assert written.exists()
    restored = ImagePipeline.from_json(written.read_text())
    assert list(restored.get_ops().values())[0].sigma == 3.0


def test_set_winner_without_pin_reports_toast_over_http(tmp_path: Path) -> None:
    """No slot-A pin → a guidance toast (still a 200, correct output arity)."""
    app = _wiring_app(tmp_path)
    client = app.server.test_client()

    out_key = _find_output_key(
        app, "tune-winner-note.children", "tune-curate-toast"
    )
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": out_key,
            "outputs": _outputs_from_key(out_key),
            "inputs": [
                {"id": "tune-btn-set-winner", "property": "n_clicks", "value": 1}
            ],
            "state": [
                {"id": "tune-ab-store", "property": "data", "value": {"a": None, "b": None}},
                {
                    "id": "tune-run-root-store",
                    "property": "data",
                    "value": {"path": str(tmp_path)},
                },
            ],
            "changedPropIds": ["tune-btn-set-winner.n_clicks"],
        },
    )
    assert resp.status_code == 200
    response = resp.get_json()["response"]
    # The toast opened with the "pin slot A first" guidance.
    assert response["tune-curate-toast"]["is_open"] is True
    assert "slot A" in response["tune-curate-toast"]["children"]
