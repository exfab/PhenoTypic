from phenotypic.gui.tune._validation import (
    Issue,
    can_deploy,
    preflight_issues,
    spec_path_issue,
    validate_setup,
)
from phenotypic.gui.tune._callbacks import _load_spec_preflight_issues
from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.enhance import GaussianBlur
from phenotypic.tune import Evaluator, GridConfig, QCScorer
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune._search_space import FloatRange, IntRange, Knob, SearchSpace
from phenotypic.tune._search_space._targets import Param


def _space(*domains):
    knobs = tuple(
        Knob(target=Param(op=i, field=f"f{i}"), domain=d)
        for i, d in enumerate(domains)
    )
    return SearchSpace(knobs=knobs)


def test_no_active_knobs_is_an_issue():
    issues = validate_setup(_space(), scorer_kind="qc", metadata_present=True)
    assert any("no active knobs" in i.message.lower() for i in issues)
    assert all(i.blocks == "both" for i in issues)


def test_low_equal_high_is_an_issue():
    issues = validate_setup(
        _space(IntRange(low=20, high=20)), scorer_kind="qc", metadata_present=True
    )
    assert any("low" in i.message.lower() for i in issues)


def test_qc_scorer_needs_metadata():
    issues = validate_setup(
        _space(FloatRange(low=1.0, high=6.0, step=0.5)),
        scorer_kind="qc",
        metadata_present=False,
    )
    assert any(
        "metadata" in i.message.lower() and i.section == "scorer"
        for i in issues
    )


def test_clean_spec_has_no_issues():
    issues = validate_setup(
        _space(FloatRange(low=1.0, high=6.0, step=0.5)),
        scorer_kind="qc",
        metadata_present=True,
    )
    assert issues == []


def test_grid_with_continuous_float_is_a_run_issue():
    issues = preflight_issues(
        _space(FloatRange(low=1.0, high=6.0)),
        strategy="grid",
    )
    assert len(issues) == 1
    assert issues[0].section == "strategy"
    assert issues[0].blocks == "deploy"


def test_grid_with_stepped_float_is_clean():
    assert preflight_issues(
        _space(FloatRange(low=1.0, high=6.0, step=0.5)),
        strategy="grid",
    ) == []


def test_optuna_with_continuous_float_is_clean():
    assert preflight_issues(
        _space(FloatRange(low=1.0, high=6.0)),
        strategy="tpe",
    ) == []


def test_can_deploy_only_when_no_blocking_issues():
    assert can_deploy([], []) is True
    assert can_deploy([Issue("scorer", "x")], []) is False
    assert can_deploy([], [Issue("strategy", "x", blocks="deploy")]) is False


def test_pipeline_path_requires_authored_tuning_spec_before_deploy():
    issue = spec_path_issue("pipeline.json.pht-pipe")
    assert issue is not None
    assert issue.blocks == "deploy"


def test_tuning_spec_path_is_deployable():
    assert spec_path_issue("tuning_spec.json.pht-tune") is None


def test_run_preflight_reads_authored_spec_and_blocks_grid_continuous_float(
    tmp_path,
):
    metadata = tmp_path / "layout.csv"
    metadata.write_text(
        "Metadata_ImageName,Object_Label\n"
        + "\n".join(f"plate,{i}" for i in range(96)),
        encoding="utf-8",
    )
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0)]),
        search_space=SearchSpace(
            knobs=(
                Knob(
                    target=Param(op=0, field="sigma"),
                    domain=FloatRange(low=1.0, high=6.0),
                ),
            )
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(metadata),
                groupby=["Metadata_ImageName"],
            )
        ),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    spec_path = tmp_path / "spec.json.pht-tune"
    spec_path.write_text(spec.model_dump_json(), encoding="utf-8")

    issues = _load_spec_preflight_issues(str(spec_path), "grid")

    assert issues
    assert "continuous float" in issues[0]
