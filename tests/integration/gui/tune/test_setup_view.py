import json
from pathlib import Path

from dash.development.base_component import Component

from phenotypic.gui.shell._ids import TUNE_PIPELINE_PATH_STORE
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.tune import _ids as ids
from phenotypic.gui.tune import create_app
from phenotypic.gui.tune._callbacks import (
    active_authored_spec_path,
    authored_spec_descriptor,
    setup_gate_state,
    setup_pipeline_path_from_sources,
)
from phenotypic.gui.tune._nav import destination_view_id
from phenotypic.gui.tune._run_root import TuneRunRoot
from phenotypic.sdk_ import trials_parquet_path
from phenotypic.tune._study_store import JournalStudyStore, Trial


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                yield from _walk(child)
        elif children is not None:
            yield from _walk(children)


def _component_by_id(layout, component_id: str):
    for component in _walk(layout):
        if getattr(component, "id", None) == component_id:
            return component
    raise AssertionError(f"component {component_id!r} not found")


def _journal_run(path: Path) -> TuneRunRoot:
    store = JournalStudyStore(
        trials=[Trial(number=0, params={}, score=0.4, terms={}, n_images=1)]
    )
    trials_parquet_path(path).parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(trials_parquet_path(path))
    return TuneRunRoot.discover(path)


def test_setup_is_the_landing_destination():
    app = create_app(root=None, url_prefix="/tune/")
    setup = _component_by_id(app.layout, destination_view_id("setup"))
    run = _component_by_id(app.layout, destination_view_id("run"))
    monitor = _component_by_id(app.layout, destination_view_id("monitor"))

    assert "tune-view-hidden" not in setup.className
    assert "tune-view-hidden" in run.className
    assert "tune-view-hidden" in monitor.className


def test_run_bound_app_lands_on_monitor_destination(tmp_path: Path):
    app = create_app(root=_journal_run(tmp_path), url_prefix="/tune/")
    setup = _component_by_id(app.layout, destination_view_id("setup"))
    monitor = _component_by_id(app.layout, destination_view_id("monitor"))

    assert "tune-view-hidden" in setup.className
    assert "tune-view-hidden" not in monitor.className


def test_search_space_locked_until_pipeline_chosen():
    app = create_app(root=None, url_prefix="/tune/")
    search_space = _component_by_id(app.layout, ids.TUNE_SETUP_SEARCH_SPACE)
    continue_button = _component_by_id(app.layout, ids.TUNE_SETUP_CONTINUE)

    assert "tune-setup-locked" in search_space.className
    assert continue_button.disabled is True

    search_class, scorer_class, disabled, note = setup_gate_state(
        "yeast_plate_pipeline.json.pht-pipe",
        "layout.csv",
    )
    assert "tune-setup-locked" not in search_class
    assert "tune-setup-locked" not in scorer_class
    assert disabled is False
    assert "yeast_plate_pipeline.json.pht-pipe" in note


def test_setup_continue_requires_metadata_after_pipeline_chosen():
    search_class, scorer_class, disabled, note = setup_gate_state(
        "yeast_plate_pipeline.json.pht-pipe"
    )

    assert "tune-setup-locked" not in search_class
    assert "tune-setup-locked" not in scorer_class
    assert disabled is True
    assert "metadata" in note.lower()


def test_setup_pipeline_input_and_shell_store_feed_pipeline_store():
    app = create_app(root=None, url_prefix="/tune/")
    assert _component_by_id(app.layout, ids.TUNE_SETUP_PIPELINE_INPUT) is not None
    assert _component_by_id(app.layout, ids.TUNE_SETUP_METADATA_INPUT) is not None

    setup_store_callbacks = [
        meta
        for callback_id, meta in app.callback_map.items()
        if f"{ids.TUNE_SETUP_PIPELINE_STORE}.data" in callback_id
    ]
    assert setup_store_callbacks
    callback_inputs = json.dumps(setup_store_callbacks[0]["inputs"])
    assert ids.TUNE_SETUP_PIPELINE_INPUT in callback_inputs
    assert TUNE_PIPELINE_PATH_STORE in callback_inputs


def test_setup_pipeline_source_prefers_explicit_typed_path():
    assert setup_pipeline_path_from_sources(
        "typed.json.pht-pipe",
        "handoff.json.pht-pipe",
    ) == "typed.json.pht-pipe"
    assert setup_pipeline_path_from_sources("typed.json.pht-pipe", None) == (
        "typed.json.pht-pipe"
    )
    assert setup_pipeline_path_from_sources("notes.txt", None) is None


def test_setup_gate_callback_enables_run_destination():
    app = create_app(root=None, url_prefix="/tune/")
    callback_id = next(
        callback_id
        for callback_id in app.callback_map
        if f"{ids.TUNE_SETUP_CONTINUE}.disabled" in callback_id
        and "tune-dest-run.disabled" in callback_id
    )
    assert "tune-dest-run.disabled" in callback_id


def test_run_callbacks_consume_authored_spec_store():
    app = create_app(root=None, url_prefix="/tune/")
    callbacks = [
        meta
        for callback_id, meta in app.callback_map.items()
        if ids.TUNE_RUN_COMMAND in callback_id or ids.TUNE_RUN_STATUS in callback_id
    ]
    assert callbacks
    encoded = json.dumps(
        [
            {
                "inputs": meta["inputs"],
                "state": meta["state"],
            }
            for meta in callbacks
        ]
    )
    assert ids.TUNE_SETUP_AUTHORED_SPEC_STORE in encoded


def test_authored_spec_descriptor_invalidates_when_setup_inputs_change():
    descriptor = authored_spec_descriptor(
        path="/root/.phenotypic-gui/presets/tune/a.json.pht-tune",
        pipeline_path="/root/a.json.pht-pipe",
        metadata_path="/root/layout.csv",
        setup_signature="current",
    )

    assert active_authored_spec_path(
        descriptor,
        pipeline_path="/root/a.json.pht-pipe",
        metadata_path="/root/layout.csv",
        setup_signature="current",
    ) == "/root/.phenotypic-gui/presets/tune/a.json.pht-tune"
    assert active_authored_spec_path(
        descriptor,
        pipeline_path="/root/b.json.pht-pipe",
        metadata_path="/root/layout.csv",
        setup_signature="current",
    ) is None
    assert active_authored_spec_path(
        descriptor,
        pipeline_path="/root/a.json.pht-pipe",
        metadata_path="/root/other-layout.csv",
        setup_signature="current",
    ) is None
    assert active_authored_spec_path(
        descriptor,
        pipeline_path="/root/a.json.pht-pipe",
        metadata_path="/root/layout.csv",
        setup_signature="edited",
    ) is None


def test_run_destination_exposes_form_and_deploy_callbacks():
    app = create_app(root=None, url_prefix="/tune/")
    assert _component_by_id(app.layout, ids.TUNE_RUN_OUTPUT_DIR) is not None
    assert _component_by_id(app.layout, ids.TUNE_RUN_STRATEGY) is not None
    assert _component_by_id(app.layout, ids.TUNE_RUN_STORAGE_MODE) is not None
    assert _component_by_id(app.layout, ids.TUNE_RUN_STORAGE_ENV) is not None
    assert _component_by_id(app.layout, ids.TUNE_RUN_PORTABLE_COMMAND) is not None
    assert _component_by_id(app.layout, ids.TUNE_RUN_COPY) is not None
    assert _component_by_id(app.layout, ids.TUNE_RUN_DEPLOY).disabled is True

    callbacks = "\n".join(app.callback_map)
    assert ids.TUNE_RUN_COMMAND in callbacks
    assert ids.TUNE_RUN_STATUS in callbacks


def test_setup_exposes_pipeline_and_metadata_pickers(tmp_path: Path):
    app = create_app(
        root=None,
        sandbox=SandboxRoot.from_path(tmp_path),
        url_prefix="/tune/",
    )

    assert _component_by_id(app.layout, ids.TUNE_SETUP_PICK_PIPELINE) is not None
    assert _component_by_id(app.layout, ids.TUNE_SETUP_PICK_METADATA) is not None
    assert _component_by_id(app.layout, ids.TUNE_SETUP_PIPELINE_MODAL) is not None
    assert _component_by_id(app.layout, ids.TUNE_SETUP_METADATA_MODAL) is not None
