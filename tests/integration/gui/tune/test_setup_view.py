import json
from pathlib import Path

from dash.development.base_component import Component

from phenotypic.gui.shell._ids import TUNE_PIPELINE_PATH_STORE
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.tune import _ids as ids
from phenotypic.gui.tune import create_app
from phenotypic.gui.tune._callbacks import (
    _build_command_from_controls,
    active_authored_spec_path,
    authored_spec_launch_defaults,
    authored_spec_descriptor,
    setup_authoring_signature,
    setup_gate_state,
    setup_pipeline_path_from_sources,
)
from phenotypic.gui.tune._setup_authoring import (
    SetupPathResolution,
    build_setup_draft,
    write_setup_draft_receipt,
)
from phenotypic.gui.tune._nav import destination_view_id
from phenotypic.gui.tune._run_root import TuneRunRoot
from phenotypic.sdk_ import trials_parquet_path
from phenotypic.tune._study_store import JournalStudyStore, Trial
from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tune import (
    Budget,
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
    TuningSpec,
    infer_search_space,
)
from phenotypic.tune.score import QCScorer
from phenotypic.tune.strategy import GridConfig, OptunaConfig, RandomConfig


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
    inputs = app.callback_map[callback_id]["inputs"]
    assert [entry["id"] for entry in inputs] == [
        ids.TUNE_SETUP_DRAFT_STORE,
        ids.TUNE_SETUP_AUTHORED_SPEC_STORE,
    ]


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
    assert ids.TUNE_SETUP_DRAFT_STORE in encoded
    assert ids.TUNE_SETUP_PIPELINE_STORE not in encoded
    assert ids.TUNE_SETUP_METADATA_STORE not in encoded
    assert ids.TUNE_SETUP_SIGNATURE_STORE not in encoded


def test_continue_consumes_only_the_revisioned_setup_draft():
    app = create_app(root=None, url_prefix="/tune/")
    callback = next(
        meta
        for callback_id, meta in app.callback_map.items()
        if f"{ids.TUNE_SETUP_AUTHORED_SPEC_STORE}.data" in callback_id
        and f"{ids.TUNE_SETUP_GATE}.children" in callback_id
    )

    state_ids = [entry["id"] for entry in callback["state"]]
    assert state_ids == [ids.TUNE_SETUP_DRAFT_STORE]
    encoded = json.dumps(
        {"inputs": callback["inputs"], "state": callback["state"]}
    )
    assert ids.TUNE_SETUP_PIPELINE_INPUT not in encoded
    assert ids.TUNE_SETUP_METADATA_INPUT not in encoded


def test_draft_callback_serializes_only_opaque_receipt_for_credential_spec(
    tmp_path: Path,
) -> None:
    secret = "setup-spec-browser-secret"
    pipeline = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    metadata = tmp_path / "layout.csv"
    metadata.write_text(
        "MetadataImage_ImageName,Object_Label\nplate.tif,1\n",
        encoding="utf-8",
    )
    existing = TuningSpec(
        pipeline=pipeline,
        search_space=infer_search_space(pipeline).to_search_space(),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(metadata),
                groupby=["MetadataImage_ImageName"],
            )
        ),
        evaluator=Evaluator(),
        strategy=OptunaConfig(
            sampler="tpe",
            n_trials=5,
            storage_url=f"postgresql+psycopg://tuner:{secret}@db/tune",
        ),
        budget=Budget(n_trials=5),
    )
    spec_path = tmp_path / "credentialed.json.pht-tune"
    spec_path.write_text(existing.model_dump_json(), encoding="utf-8")
    app = create_app(
        root=None,
        url_prefix="/tune/",
        sandbox=SandboxRoot.from_path(tmp_path),
    )
    output_key = next(
        key
        for key in app.callback_map
        if key.split("@", 1)[0] == f"{ids.TUNE_SETUP_DRAFT_STORE}.data"
    )
    callback = app.callback_map[output_key]
    inputs = []
    for entry in callback["inputs"]:
        component_id = entry["id"]
        if component_id == ids.TUNE_SETUP_PIPELINE_STORE:
            value = {
                "path": str(spec_path),
                "source": "typed",
                "issues": [],
            }
        elif component_id == ids.TUNE_SETUP_METADATA_STORE:
            value = {"path": None, "source": "unset", "issues": []}
        else:
            value = []
        inputs.append({**entry, "value": value})

    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": {
                "id": ids.TUNE_SETUP_DRAFT_STORE,
                "property": "data",
            },
            "inputs": inputs,
            "state": [],
            "changedPropIds": [f"{ids.TUNE_SETUP_PIPELINE_STORE}.data"],
        },
    )

    assert response.status_code == 200
    receipt = response.get_json()["response"][ids.TUNE_SETUP_DRAFT_STORE]["data"]
    serialized = json.dumps(receipt)
    assert set(receipt) == {"version", "handle", "revision"}
    assert secret not in serialized
    assert "storage_url" not in serialized
    assert "spec_json" not in serialized


def test_write_receipt_does_not_bless_source_mutation_before_descriptor(
    tmp_path: Path,
) -> None:
    pipeline_path = tmp_path / "pipeline.json.pht-pipe"
    metadata_path = tmp_path / "layout.csv"
    pipeline_path.write_text(
        ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]).to_json(),
        encoding="utf-8",
    )
    metadata_path.write_text(
        "MetadataImage_ImageName,Object_Label\nplate.tif,1\n",
        encoding="utf-8",
    )
    draft = build_setup_draft(
        pipeline=SetupPathResolution(pipeline_path, "typed"),
        metadata=SetupPathResolution(metadata_path, "typed"),
    )
    receipt = write_setup_draft_receipt(
        sandbox_root=tmp_path,
        draft=draft,
    )
    pipeline_path.write_text(
        ImagePipeline(ops=[GaussianBlur(sigma=4.0), OtsuDetector()]).to_json(),
        encoding="utf-8",
    )
    descriptor = authored_spec_descriptor(
        path=str(receipt.path),
        pipeline_path=str(pipeline_path),
        metadata_path=str(metadata_path),
        setup_signature=draft.revision,
        write_receipt=receipt,
    )

    assert descriptor["source_fingerprint"] == draft.source_fingerprint
    assert active_authored_spec_path(
        descriptor,
        pipeline_path=str(pipeline_path),
        metadata_path=str(metadata_path),
        setup_signature=draft.revision,
    ) is None


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


def test_existing_spec_controls_preserve_seed_budget_and_storage(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "layout.csv"
    metadata.write_text(
        "MetadataImage_ImageName,Object_Label\nplate.tif,1\n",
        encoding="utf-8",
    )
    pipeline = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    original = TuningSpec(
        pipeline=pipeline,
        search_space=infer_search_space(pipeline).to_search_space(),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(metadata),
                groupby=["MetadataImage_ImageName"],
            )
        ),
        evaluator=Evaluator(),
        strategy=RandomConfig(seed=71, n_trials=13),
        budget=Budget(n_trials=11, max_failures=2),
    )
    authored = tmp_path / "existing.json.pht-tune"
    authored.write_text(original.model_dump_json(), encoding="utf-8")
    signature = setup_authoring_signature(
        pipeline_path=str(authored),
        metadata_path=None,
        replace_scorer=False,
        edits={},
    )
    defaults = authored_spec_launch_defaults(authored)
    descriptor = authored_spec_descriptor(
        path=str(authored),
        pipeline_path=str(authored),
        metadata_path=None,
        setup_signature=signature,
        launch_defaults=defaults,
    )
    images = tmp_path / "plate images"
    images.mkdir()
    output = tmp_path / "tune output"
    command = _build_command_from_controls(
        sandbox=SandboxRoot.from_path(tmp_path),
        authored_descriptor=descriptor,
        pipeline_store={"path": str(authored), "issues": []},
        metadata_store={"path": None, "issues": []},
        replace_values=[],
        setup_signature=signature,
        shared_source=None,
        images_override=str(images),
        output_dir=str(output),
        strategy="random",
        n_trials=13,
        storage_mode="spec",
        storage_local_path=None,
        storage_environment_name=None,
        n_workers=None,
        slurm_partition=None,
        slurm_mem=None,
        slurm_time=None,
        held_out_fraction=None,
        cv_group=None,
        mode="local",
        screen_values=[],
    )

    assert command.issues == ()
    assert "--strategy" not in command.semantic_tail
    assert "--n-trials" not in command.semantic_tail
    assert "--storage-url" not in command.semantic_tail
    reloaded = TuningSpec.model_validate_json(authored.read_text(encoding="utf-8"))
    assert reloaded.strategy == original.strategy
    assert reloaded.budget == original.budget


def test_existing_spec_trial_override_preserves_authored_strategy(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "layout.csv"
    metadata.write_text(
        "MetadataImage_ImageName,Object_Label\nplate.tif,1\n",
        encoding="utf-8",
    )
    pipeline = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    original = TuningSpec(
        pipeline=pipeline,
        search_space=infer_search_space(pipeline).to_search_space(),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(metadata),
                groupby=["MetadataImage_ImageName"],
            )
        ),
        evaluator=Evaluator(),
        strategy=RandomConfig(seed=71, n_trials=13),
        budget=Budget(n_trials=13),
    )
    authored = tmp_path / "existing.json.pht-tune"
    authored.write_text(original.model_dump_json(), encoding="utf-8")
    signature = setup_authoring_signature(
        pipeline_path=str(authored),
        metadata_path=None,
        replace_scorer=False,
        edits={},
    )
    descriptor = authored_spec_descriptor(
        path=str(authored),
        pipeline_path=str(authored),
        metadata_path=None,
        setup_signature=signature,
        launch_defaults=authored_spec_launch_defaults(authored),
    )
    images = tmp_path / "images"
    images.mkdir()
    command = _build_command_from_controls(
        sandbox=SandboxRoot.from_path(tmp_path),
        authored_descriptor=descriptor,
        pipeline_store={"path": str(authored), "issues": []},
        metadata_store={"path": None, "issues": []},
        replace_values=[],
        setup_signature=signature,
        shared_source=None,
        images_override=str(images),
        output_dir=str(tmp_path / "output"),
        strategy="random",
        n_trials=21,
        storage_mode="spec",
        storage_local_path=None,
        storage_environment_name=None,
        n_workers=None,
        slurm_partition=None,
        slurm_mem=None,
        slurm_time=None,
        held_out_fraction=None,
        cv_group=None,
        mode="local",
        screen_values=[],
    )

    assert command.issues == ()
    assert "--strategy" not in command.semantic_tail
    assert command.semantic_tail[-2:] == ("--n-trials", "21")
    reloaded = TuningSpec.model_validate_json(authored.read_text(encoding="utf-8"))
    assert reloaded.strategy.seed == 71


def test_existing_grid_spec_never_emits_trial_override(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "layout.csv"
    metadata.write_text(
        "MetadataImage_ImageName,Object_Label\nplate.tif,1\n",
        encoding="utf-8",
    )
    pipeline = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    original = TuningSpec(
        pipeline=pipeline,
        search_space=SearchSpace(
            knobs=(
                Knob(
                    key="1.ignore_zeros",
                    domain=Categorical(choices=(True, False)),
                ),
            )
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(metadata),
                groupby=["MetadataImage_ImageName"],
            )
        ),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    authored = tmp_path / "grid.json.pht-tune"
    authored.write_text(original.model_dump_json(), encoding="utf-8")
    signature = setup_authoring_signature(
        pipeline_path=str(authored),
        metadata_path=None,
        replace_scorer=False,
        edits={},
    )
    descriptor = authored_spec_descriptor(
        path=str(authored),
        pipeline_path=str(authored),
        metadata_path=None,
        setup_signature=signature,
        launch_defaults=authored_spec_launch_defaults(authored),
    )
    images = tmp_path / "images"
    images.mkdir()
    command = _build_command_from_controls(
        sandbox=SandboxRoot.from_path(tmp_path),
        authored_descriptor=descriptor,
        pipeline_store={"path": str(authored), "issues": []},
        metadata_store={"path": None, "issues": []},
        replace_values=[],
        setup_signature=signature,
        shared_source=None,
        images_override=str(images),
        output_dir=str(tmp_path / "output"),
        strategy="grid",
        n_trials=50,
        storage_mode="spec",
        storage_local_path=None,
        storage_environment_name=None,
        n_workers=None,
        slurm_partition=None,
        slurm_mem=None,
        slurm_time=None,
        held_out_fraction=None,
        cv_group=None,
        mode="local",
        screen_values=[],
    )

    assert command.issues == ()
    assert "--strategy" not in command.semantic_tail
    assert "--n-trials" not in command.semantic_tail


def test_descriptor_invalidates_when_authored_content_changes(tmp_path: Path):
    source = tmp_path / "pipeline.json.pht-pipe"
    source.write_text(ImagePipeline(ops=[]).to_json(), encoding="utf-8")
    authored = tmp_path / "authored.json.pht-tune"
    authored.write_text("first", encoding="utf-8")
    signature = setup_authoring_signature(
        pipeline_path=str(source),
        metadata_path=None,
        replace_scorer=False,
        edits={},
    )
    descriptor = authored_spec_descriptor(
        path=str(authored),
        pipeline_path=str(source),
        metadata_path=None,
        setup_signature=signature,
    )
    authored.write_text("second", encoding="utf-8")

    assert active_authored_spec_path(
        descriptor,
        pipeline_path=str(source),
        metadata_path=None,
        setup_signature=signature,
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
