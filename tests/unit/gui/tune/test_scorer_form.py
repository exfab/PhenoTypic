from phenotypic.gui._param_forms import param_form
from phenotypic.gui.tune._scorer_form import scorer_operation_info
from phenotypic.tune.score import QCScorer


def test_scorer_operation_info_exposes_pydantic_fields():
    info = scorer_operation_info(QCScorer)
    assert info.parameters
    assert set(info.parameters) <= set(QCScorer.model_fields)


def test_param_form_renders_a_scorer_without_registry():
    info = scorer_operation_info(QCScorer)
    form = param_form(info, current_values={}, form_id_prefix="tune-scorer")
    assert form is not None
