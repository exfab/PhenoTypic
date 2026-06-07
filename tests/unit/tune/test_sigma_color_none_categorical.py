"""Phase 3: ``LocalEdgeDenoise.sigma_color`` exposes its ``None`` auto mode.

``sigma_color=None`` selects the bilateral filter's noise-std auto-estimate. The
old ``TuneSpec(0.02, 0.5, log=True)`` float window could never sample ``None``,
so the optimizer never tried the auto mode. The field now carries a categorical
``TuneSpec(categories=[None, 0.02, 0.05, 0.1, 0.2, 0.5])`` whose inference yields
a ``Categorical`` containing ``None``; Optuna can suggest it, and the build path
applies it through the op constructor (whose validator accepts ``None``).
"""
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.detect import CannyDetector
from phenotypic.enhance import LocalEdgeDenoise
from phenotypic.tune import Categorical, infer_search_space
from phenotypic.tune._evaluation import build_pipeline


def _sigma_color_domain():
    pipe = ImagePipeline(ops=[LocalEdgeDenoise(), CannyDetector()])
    space = infer_search_space(pipe)
    return next(k for k in space.knobs if k.key == "0.sigma_color").domain


def test_sigma_color_infers_categorical_with_none():
    domain = _sigma_color_domain()
    assert isinstance(domain, Categorical)
    assert None in domain.choices
    # Representative explicit sigmas are still searchable alongside None.
    assert set(domain.choices) == {None, 0.02, 0.05, 0.1, 0.2, 0.5}


def test_sigma_color_knob_source_is_tune_spec():
    pipe = ImagePipeline(ops=[LocalEdgeDenoise(), CannyDetector()])
    space = infer_search_space(pipe)
    knob = next(k for k in space.knobs if k.key == "0.sigma_color")
    # A categories override resolves via Tier-1 (no numeric ⊆ check) and is not
    # flagged for review.
    assert knob.source == "tune_spec"
    assert knob.needs_review is False


def test_optuna_can_suggest_none_for_sigma_color():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    choices = list(_sigma_color_domain().choices)

    seen: set = set()

    def objective(trial: "optuna.Trial") -> float:
        seen.add(trial.suggest_categorical("0.sigma_color", choices))
        return 0.0

    # GridSampler enumerates every choice, so None is guaranteed to be drawn.
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler({"0.sigma_color": choices})
    )
    study.optimize(objective, n_trials=len(choices))
    assert None in seen


def test_none_constructs_and_build_path_applies_it():
    # The op itself accepts None (auto-estimate mode).
    assert LocalEdgeDenoise(sigma_color=None).sigma_color is None

    # build_pipeline overlays the sampled None onto the op via fresh
    # reconstruction (the op's field_validator accepts None).
    base = ImagePipeline(ops=[LocalEdgeDenoise(sigma_color=0.1), CannyDetector()])
    built = build_pipeline(base, {"0.sigma_color": None})
    denoise = list(built.get_ops().values())[0]
    assert isinstance(denoise, LocalEdgeDenoise)
    assert denoise.sigma_color is None


def test_build_path_applies_explicit_sigma_color():
    base = ImagePipeline(ops=[LocalEdgeDenoise(), CannyDetector()])
    built = build_pipeline(base, {"0.sigma_color": 0.2})
    denoise = list(built.get_ops().values())[0]
    assert denoise.sigma_color == 0.2
