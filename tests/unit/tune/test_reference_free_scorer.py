from __future__ import annotations

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape, MeasureSize
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    ReferenceFreeScorer,
    SearchSpace,
)
from phenotypic.tune._scoring._reference_free_scorer import (
    ReferenceFreeScorer as _RFS,
    _bounded_inverse,
    _clamp01,
)
from phenotypic.tune._spec import Budget, TuningSpec


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _measured_plate() -> tuple[object, pd.DataFrame]:
    """A processed synth plate + its Shape/Size measurement frame."""
    image = load_synth_yeast_plate()
    pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape(), MeasureSize()])
    measurements = pipe.measure(image, apply_post=False)
    return image, measurements


# --------------------------------------------------------------------------- #
# fixed-normalization helpers
# --------------------------------------------------------------------------- #
def test_clamp01_clamps_out_of_range():
    assert _clamp01(-0.5) == 0.0
    assert _clamp01(0.0) == 0.0
    assert _clamp01(0.5) == pytest.approx(0.5)
    assert _clamp01(1.0) == 1.0
    assert _clamp01(9.4) == 1.0  # the synth-plate Shape_Solidity > 1 quirk
    assert _clamp01(float("nan")) == 0.0


def test_bounded_inverse_folds_dispersion_to_unit_interval():
    # 1/(1+x): x=0 -> 1 (no dispersion = best); grows -> 0; monotone decreasing.
    assert _bounded_inverse(0.0) == pytest.approx(1.0)
    assert _bounded_inverse(1.0) == pytest.approx(0.5)
    assert _bounded_inverse(0.10) > _bounded_inverse(0.50)
    assert 0.0 < _bounded_inverse(1000.0) < 0.01
    assert _bounded_inverse(float("inf")) == 0.0


# --------------------------------------------------------------------------- #
# Task 1 — term shape, fixed-normalization, column reuse
# --------------------------------------------------------------------------- #
def test_score_image_returns_stable_proxy_terms():
    scorer = ReferenceFreeScorer()
    image, measurements = _measured_plate()
    terms = scorer.score_image(image, measurements)
    assert set(terms) == {"ShapeRegularity", "Contrast", "SizeCV"}
    for value in terms.values():
        assert 0.0 <= value <= 1.0


def test_score_image_includes_count_term_when_count_check_configured(tmp_path):
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {
            "Metadata_ImageName": ["Synthetic96PlateWithObjects"] * 96,
            "Object_Label": list(range(96)),
        }
    ).to_csv(csv, index=False)
    scorer = ReferenceFreeScorer(
        count_check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        )
    )
    image, measurements = _measured_plate()
    terms = scorer.score_image(image, measurements)
    assert set(terms) == {"Count", "ShapeRegularity", "Contrast", "SizeCV"}
    # perfect 96-vs-96 grid count → Count term is 1.0
    assert terms["Count"] == pytest.approx(1.0)


def test_keys_are_stable_across_images():
    scorer = ReferenceFreeScorer()
    image_a, meas_a = _measured_plate()
    image_b, meas_b = _measured_plate()
    assert set(scorer.score_image(image_a, meas_a)) == set(
        scorer.score_image(image_b, meas_b)
    )


def test_shape_regularity_reuses_schema_columns_no_recompute(monkeypatch):
    # The shape term must read Shape_* columns from the frame, never recompute
    # geometry from the image. Drop the shape columns -> the term must degrade
    # to the neutral floor rather than recomputing from the mask/objmap.
    scorer = ReferenceFreeScorer()
    image, measurements = _measured_plate()
    stripped = measurements.drop(
        columns=[c for c in measurements.columns if str(c).startswith("Shape_")]
    )
    terms = scorer.score_image(image, stripped)
    # ShapeRegularity is still a key (stable), at the neutral floor when absent.
    assert terms["ShapeRegularity"] == pytest.approx(0.0)


def test_shape_regularity_clamps_solidity_quirk_into_unit_interval():
    # The synth plate's Shape_Solidity mean is > 1; the term must stay in [0, 1].
    scorer = ReferenceFreeScorer()
    image, measurements = _measured_plate()
    assert 0.0 <= scorer.score_image(image, measurements)["ShapeRegularity"] <= 1.0


def test_contrast_term_reads_image_foreground_background():
    # Otsu between-class separation η ∈ [0, 1]; the well-separated synth plate
    # scores high (fixed-normalized, NOT min-max over a grid).
    scorer = ReferenceFreeScorer()
    image, measurements = _measured_plate()
    assert scorer.score_image(image, measurements)["Contrast"] > 0.5


def test_size_cv_term_is_high_for_uniform_sizes():
    # A frame whose Size_Area is constant has zero CV → SizeCV term == 1.0.
    scorer = ReferenceFreeScorer()
    image, _ = _measured_plate()
    uniform = pd.DataFrame(
        {
            "Metadata_ImageName": ["p"] * 24,
            "Size_Area": [500.0] * 24,
            "Shape_Solidity": [0.95] * 24,
            "Shape_Circularity": [0.9] * 24,
            "Shape_Eccentricity": [0.1] * 24,
        }
    )
    assert scorer.score_image(image, uniform)["SizeCV"] == pytest.approx(1.0)


def test_size_cv_uses_replicate_groups_when_configured():
    # Two strains with *within-group* uniform sizes but very different means
    # → within-replicate CV is 0 (perfect) even though the pooled CV is large.
    scorer = ReferenceFreeScorer(replicate_groupby=["Metadata_Strain"])
    image, _ = _measured_plate()
    frame = pd.DataFrame(
        {
            "Metadata_Strain": ["A"] * 12 + ["B"] * 12,
            "Size_Area": [100.0] * 12 + [900.0] * 12,
        }
    )
    grouped = scorer.score_image(image, frame)["SizeCV"]
    # Pooled (no grouping) would be dragged down by the across-strain spread.
    pooled = ReferenceFreeScorer().score_image(image, frame)["SizeCV"]
    assert grouped == pytest.approx(1.0)
    assert grouped > pooled


def test_empty_measurements_floor_to_zero():
    scorer = ReferenceFreeScorer()
    image, _ = _measured_plate()
    terms = scorer.score_image(image, pd.DataFrame())
    assert set(terms) == {"ShapeRegularity", "Contrast", "SizeCV"}
    assert terms["ShapeRegularity"] == 0.0
    assert terms["SizeCV"] == 0.0


# --------------------------------------------------------------------------- #
# Task 1 — construction + registry round-trip
# --------------------------------------------------------------------------- #
def test_keyword_only_construction():
    with pytest.raises(TypeError):
        ReferenceFreeScorer(None)  # type: ignore[misc]


def test_round_trips_through_registry(tmp_path):
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["p"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    gt_dir = tmp_path / "gt_masks"
    gt_dir.mkdir()
    scorer = ReferenceFreeScorer(
        count_check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        ),
        replicate_groupby=["Metadata_Strain"],
        gt_masks_source=gt_dir,
    )
    reloaded = ReferenceFreeScorer.model_validate_json(scorer.model_dump_json())
    assert isinstance(reloaded, ReferenceFreeScorer)
    assert reloaded.replicate_groupby == ["Metadata_Strain"]
    assert reloaded.gt_masks_source == gt_dir
    # the path-configured count check rehydrates from disk and still scores
    assert reloaded.count_check is not None
    assert reloaded.count_check.metadata_source == str(csv)


def test_round_trips_inside_tuning_spec(tmp_path):
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(
                Knob(
                    key="0.ignore_zeros",
                    domain=Categorical(choices=(True, False)),
                ),
            )
        ),
        scorer=ReferenceFreeScorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    back = TuningSpec.model_validate_json(spec.model_dump_json())
    assert isinstance(back.scorer, ReferenceFreeScorer)


# --------------------------------------------------------------------------- #
# Task 2 — availability() + meta_validate() gate (structure / abstain logic)
# --------------------------------------------------------------------------- #
def test_availability_is_false_before_meta_validate_runs():
    # Fail-safe: until the gate runs and passes, the scorer is unavailable so the
    # engine degrades to QCScorer.
    assert ReferenceFreeScorer().availability() is False


def test_availability_reads_cached_flag_cheaply(monkeypatch):
    scorer = ReferenceFreeScorer()
    # availability() is a cheap cached-boolean read — it must NOT load GT or
    # recompute the correlation.
    monkeypatch.setattr(
        scorer,
        "_load_gt_masks",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("availability() must not load GT")
        ),
    )
    assert scorer.availability() is False
    object.__setattr__(scorer, "_meta_validated", True)
    assert scorer.availability() is True


def test_meta_validate_abstains_without_gt():
    # No gt_masks_source configured → cannot validate → flag stays False
    # (degrade to QCScorer).
    scorer = ReferenceFreeScorer()
    passed = scorer.meta_validate([load_synth_yeast_plate()], grid={})
    assert passed is False
    assert scorer.availability() is False


def test_meta_validate_abstains_on_weak_correlation(monkeypatch):
    scorer = ReferenceFreeScorer(gt_masks_source="/some/gt")
    # Stub the proxy-vs-GT correlation to a weak rho (< 0.7 enable threshold).
    monkeypatch.setattr(scorer, "_load_gt_masks", lambda images: {"x": object()})
    monkeypatch.setattr(scorer, "_proxy_gt_spearman", lambda *a, **k: 0.42)
    passed = scorer.meta_validate([load_synth_yeast_plate()], grid={})
    assert passed is False
    assert scorer.availability() is False


def test_meta_validate_enables_on_strong_correlation(monkeypatch):
    scorer = ReferenceFreeScorer(gt_masks_source="/some/gt")
    # Stub a strong rho (>= 0.7) → the gate flips the cached flag to enabled.
    monkeypatch.setattr(scorer, "_load_gt_masks", lambda images: {"x": object()})
    monkeypatch.setattr(scorer, "_proxy_gt_spearman", lambda *a, **k: 0.85)
    passed = scorer.meta_validate([load_synth_yeast_plate()], grid={})
    assert passed is True
    assert scorer.availability() is True


def test_meta_validate_caches_the_flag(monkeypatch):
    scorer = ReferenceFreeScorer(gt_masks_source="/some/gt")
    monkeypatch.setattr(scorer, "_load_gt_masks", lambda images: {"x": object()})
    monkeypatch.setattr(scorer, "_proxy_gt_spearman", lambda *a, **k: 0.9)
    scorer.meta_validate([load_synth_yeast_plate()], grid={})
    # After a passing run, availability() reads the cached flag — no re-correlation.
    monkeypatch.setattr(
        scorer,
        "_proxy_gt_spearman",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("availability() must not re-correlate")
        ),
    )
    assert scorer.availability() is True


def test_meta_validation_flag_does_not_serialize():
    # The cached gate flag is run-local state, not part of the recipe — it must
    # not round-trip (a reloaded scorer re-validates, fail-safe to unavailable).
    scorer = ReferenceFreeScorer(gt_masks_source="/some/gt")
    object.__setattr__(scorer, "_meta_validated", True)
    reloaded = ReferenceFreeScorer.model_validate_json(scorer.model_dump_json())
    assert reloaded.availability() is False


# --------------------------------------------------------------------------- #
# coefficient of variation uses the SAMPLE std (ddof=1), not the population std
# --------------------------------------------------------------------------- #
def test_coefficient_of_variation_uses_sample_std_ddof1():
    # [10, 20, 30]: sample std (ddof=1) = 10, mean = 20 → CV = 0.5 exactly.
    # The old population std (ddof=0) would give sqrt(200/3)/20 ≈ 0.4082, so this
    # pin distinguishes the two estimators.
    cv = _RFS._coefficient_of_variation(pd.Series([10.0, 20.0, 30.0]))
    assert cv == pytest.approx(0.5)


def test_coefficient_of_variation_two_value_group_ddof1():
    # [100, 200]: ddof=1 std = sqrt(5000) ≈ 70.7107, mean = 150 → CV ≈ 0.4714.
    cv = _RFS._coefficient_of_variation(pd.Series([100.0, 200.0]))
    assert cv == pytest.approx(0.4714045207910317)
    # Sanity: strictly above the ddof=0 value (0.3333), i.e. genuinely Bessel-corrected.
    assert cv > 1.0 / 3.0


def test_size_cv_term_reflects_ddof1_fold():
    # End-to-end: a single [10,20,30] group folds via 1/(1+CV) with the ddof=1 CV.
    scorer = ReferenceFreeScorer()
    image, _ = _measured_plate()
    frame = pd.DataFrame({"Size_Area": [10.0, 20.0, 30.0]})
    assert scorer.score_image(image, frame)["SizeCV"] == pytest.approx(
        _bounded_inverse(0.5)
    )


# --------------------------------------------------------------------------- #
# is_unattended_safe checks the RAW rho against the 0.8 bar (not the 0.7 verdict)
# --------------------------------------------------------------------------- #
def test_is_unattended_safe_false_before_meta_validate():
    # No gate run → raw rho is -inf → not unattended-safe (and not available).
    scorer = ReferenceFreeScorer()
    assert scorer.is_unattended_safe() is False
    assert scorer.availability() is False


def test_enable_bar_does_not_imply_unattended_bar(monkeypatch):
    # rho in [0.7, 0.8): the proxy is ENABLED (availability True) but NOT
    # unattended-safe — the two thresholds must not be conflated.
    scorer = ReferenceFreeScorer(gt_masks_source="/some/gt")
    monkeypatch.setattr(scorer, "_load_gt_masks", lambda images: {"x": object()})
    monkeypatch.setattr(scorer, "_proxy_gt_spearman", lambda *a, **k: 0.75)
    scorer.meta_validate([load_synth_yeast_plate()], grid={})
    assert scorer.availability() is True  # cleared the 0.7 enable bar
    assert scorer.is_unattended_safe() is False  # but NOT the 0.8 unattended bar


def test_unattended_bar_met_when_rho_at_or_above_threshold(monkeypatch):
    scorer = ReferenceFreeScorer(gt_masks_source="/some/gt")
    monkeypatch.setattr(scorer, "_load_gt_masks", lambda images: {"x": object()})
    monkeypatch.setattr(scorer, "_proxy_gt_spearman", lambda *a, **k: 0.8)
    scorer.meta_validate([load_synth_yeast_plate()], grid={})
    assert scorer.availability() is True
    assert scorer.is_unattended_safe() is True


def test_unattended_safe_resets_when_gate_abstains(monkeypatch):
    # A passing run, then a re-run that abstains (no GT) must clear BOTH verdicts
    # — the stored raw rho is reset to -inf so a stale 0.8 cannot linger.
    scorer = ReferenceFreeScorer(gt_masks_source="/some/gt")
    monkeypatch.setattr(scorer, "_load_gt_masks", lambda images: {"x": object()})
    monkeypatch.setattr(scorer, "_proxy_gt_spearman", lambda *a, **k: 0.95)
    scorer.meta_validate([load_synth_yeast_plate()], grid={})
    assert scorer.is_unattended_safe() is True
    # Now the GT disappears → abstain.
    monkeypatch.setattr(scorer, "_load_gt_masks", lambda images: {})
    scorer.meta_validate([load_synth_yeast_plate()], grid={})
    assert scorer.availability() is False
    assert scorer.is_unattended_safe() is False
