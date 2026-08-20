"""The agent-facing projection of an operation's contract.

The MCP catalog hands an agent JSON, not Python objects, so everything it
needs to author a valid operation has to survive that projection —
including the two facts ``model_json_schema()`` structurally cannot state:
that a parameter takes another *operation*, and that a parameter is a raw
NumPy array rather than the bare ``{"type": "array"}`` it reports as.
"""

from __future__ import annotations

import pytest

from phenotypic._services.catalog import describe_operation, list_operations


def test_constraints_are_json_schema_keywords():
    """Field(200.0, gt=0.0) reports exclusiveMinimum, not gt."""
    desc = describe_operation("FlattenIllumination")
    sigma = next(p for p in desc["params"] if p["name"] == "sigma")
    assert sigma["constraints"] == {"exclusiveMinimum": 0.0}
    assert "gt" not in sigma["constraints"]


def test_constraints_exclude_annotation_and_structure_keys():
    """``default``/``description``/``title``/``type`` are not constraints."""
    sigma = next(
        p for p in describe_operation("BlurGauss")["params"] if p["name"] == "sigma"
    )
    assert set(sigma["constraints"]) & {"default", "description", "title", "type"} == set()


def test_description_defaults_to_first_sentence():
    terse = describe_operation("BlurGauss")
    verbose = describe_operation("BlurGauss", verbose=True)
    t = next(p for p in terse["params"] if p["name"] == "sigma")["description"]
    v = next(p for p in verbose["params"] if p["name"] == "sigma")["description"]
    assert t.count(".") == 1
    assert len(v) > len(t)
    assert v.startswith(t)


def test_first_sentence_does_not_split_on_a_decimal_point():
    """A decimal inside the first sentence must not end it.

    Exercised directly rather than through a shipped docstring: every
    docstring that happens to carry a decimal carries it *after* the first
    full stop, so a naive ``(?<=[.!?])`` split would pass on all of them.
    """
    from phenotypic._services.catalog import _first_sentence

    assert (
        _first_sentence("Scales by 0.5 before thresholding. Then more text.")
        == "Scales by 0.5 before thresholding."
    )
    assert _first_sentence("Typical range: 0.5--5.0.") == "Typical range: 0.5--5.0."
    assert _first_sentence("") is None
    assert _first_sentence(None) is None


def test_terse_description_keeps_a_whole_sentence():
    """The shipped fixture, kept as a smoke check on the real data."""
    verbose = describe_operation("BlurGauss", verbose=True)
    v = next(p for p in verbose["params"] if p["name"] == "sigma")["description"]
    assert "0.5--5.0" in v, "fixture drifted; pick another decimal-bearing param"
    terse = describe_operation("BlurGauss")
    t = next(p for p in terse["params"] if p["name"] == "sigma")["description"]
    assert t.endswith("pixels.")


def test_no_tunable_field_on_a_class():
    """``ParamInfo`` carries no suggested domain; tunability needs a position."""
    for param in describe_operation("BlurGauss")["params"]:
        assert "tunable" not in param
        assert "suggested_domain" not in param


def test_operation_valued_params_are_flagged():
    """OperationField erases to Any; the schema alone cannot say this."""
    desc = describe_operation("FilamentousFungiDetector")
    nested = [p for p in desc["params"] if p["is_operation"]]
    assert nested, "nested operation params must be discoverable"
    assert {p["name"] for p in nested} == {"inoculum_detector"}


def test_the_schema_alone_could_not_have_flagged_it():
    """Pin the gap this field fills: the raw schema branch is literally ``{}``."""
    from phenotypic.detect import FilamentousFungiDetector

    prop = FilamentousFungiDetector.model_json_schema()["properties"][
        "inoculum_detector"
    ]
    assert prop["anyOf"] == [{}, {"type": "null"}]


def test_raw_array_params_report_type_ndarray():
    """``NdArrayField``'s schema is a bare array with no shape or dtype."""
    shape = next(
        p for p in describe_operation("MaskDilation")["params"] if p["name"] == "shape"
    )
    assert "ndarray" in shape["type"]
    assert "array" not in shape["type"].replace("ndarray", "")


def test_closed_value_sets_surface_as_choices():
    shape = next(
        p for p in describe_operation("MaskDilation")["params"] if p["name"] == "shape"
    )
    assert shape["choices"] == ["auto", "square", "diamond", "disk"]
    sigma = next(
        p for p in describe_operation("BlurGauss")["params"] if p["name"] == "sigma"
    )
    assert sigma["choices"] is None


def test_json_schema_is_verbatim():
    from phenotypic.enhance import BlurGauss

    assert describe_operation("BlurGauss")["json_schema"] == BlurGauss.model_json_schema()


def test_the_whole_descriptor_is_json_serializable():
    """It travels over MCP as JSON; a stray ndarray default would break it."""
    import json

    for name in ("BlurGauss", "MaskDilation", "FilamentousFungiDetector", "QCScorer"):
        json.dumps(describe_operation(name))


def test_layers_modified_follows_the_pipeline_helper():
    assert describe_operation("BlurGauss")["layers_modified"] == ["detect_mat"]
    assert describe_operation("OtsuDetector")["layers_modified"] == ["objmap"]
    # ``_layers_modified_by`` returns None for a measurer — it populates the
    # table, not a layer.
    assert describe_operation("MeasureSize")["layers_modified"] == []


def test_layers_modified_agrees_with_the_live_helper():
    """Anti-drift: the class-level twin must match the instance-level original.

    ``_layers_modified_by`` dispatches on ``isinstance`` and so needs an
    instance the catalog does not have. This asserts the two agree for every
    registered operation that constructs with defaults.
    """
    from phenotypic._core._pipeline_parts._image_pipeline_core import (
        _layers_modified_by,
    )
    from phenotypic._services.registry import get_registry
    from phenotypic.abc_._base_operation import BaseOperation

    checked = 0
    measurers_checked = 0
    post_checked = 0
    for name, info in get_registry().get_all().items():
        if not (isinstance(info.cls, type) and issubclass(info.cls, BaseOperation)):
            continue
        try:
            instance = info.cls()
        except Exception:  # noqa: BLE001 — required params, heavy deps
            continue
        expected = list(_layers_modified_by(instance) or ())
        assert describe_operation(name)["layers_modified"] == expected, name
        checked += 1
        if info.category == "Measure":
            measurers_checked += 1
        if info.category == "Post":
            post_checked += 1
    assert checked > 20, f"only {checked} operations exercised — coverage too thin"
    # ``MeasureFeatures``, ``PostMeasurement`` and ``PrefabPipeline`` are
    # ``BaseOperation`` but NOT ``ImageOperation``; a guard on the narrower
    # base reports ``[]`` for all of them and this loop would never notice.
    assert measurers_checked > 3, "no measurers exercised"
    assert post_checked > 0, "no post transforms exercised"


def test_scorers_are_describable():
    """§3.1: `catalog_operation_detail{name:"QCScorer"}` like any operation."""
    desc = describe_operation("QCScorer")
    assert desc["category"] == "Scorer"
    assert desc["params"]


def test_unknown_name_raises_keyerror():
    with pytest.raises(KeyError, match="NotAnOperation"):
        describe_operation("NotAnOperation")


def test_list_operations_rows_are_compact():
    """Token discipline: list rows carry no JSON schema."""
    result = list_operations(category="Detector", query=None, limit=100)
    rows = result["operations"]
    assert rows
    assert all(set(r) == {
        "name", "category", "summary", "n_params", "has_nested_operations"
    } for r in rows)
    assert all(r["category"] == "Detector" for r in rows)
    fungi = next(r for r in rows if r["name"] == "FilamentousFungiDetector")
    assert fungi["n_params"] == 20
    assert fungi["has_nested_operations"] is True


def test_list_operations_truncates_and_reports_the_total():
    result = list_operations(category=None, query=None, limit=3)
    assert len(result["operations"]) == 3
    assert result["truncated"] is True
    assert result["total"] > 3

    everything = list_operations(category=None, query=None, limit=10_000)
    assert everything["truncated"] is False
    assert everything["total"] == len(everything["operations"])


def test_list_operations_query_matches_name_and_summary():
    by_name = list_operations(category=None, query="otsu", limit=100)
    assert "OtsuDetector" in {r["name"] for r in by_name["operations"]}

    by_summary = list_operations(category=None, query="otsu", limit=100)
    assert all(
        "otsu" in r["name"].lower() or "otsu" in (r["summary"] or "").lower()
        for r in by_summary["operations"]
    )


def test_an_aliased_param_is_not_projected_empty():
    """The schema publishes an aliased field under its alias, not its name.

    ``RemoveGridOutliers.cutoff_multiplier`` carries
    ``AliasChoices("stddev_multiplier", "cutoff_multiplier")`` and appears in
    ``model_json_schema()`` as ``stddev_multiplier``. A plain name lookup
    finds nothing and reports the parameter with no type, no default and no
    constraints — an agent would then read its default as ``None``.
    """
    from phenotypic.refine import RemoveGridOutliers

    assert "cutoff_multiplier" not in RemoveGridOutliers.model_json_schema()["properties"]

    param = next(
        p
        for p in describe_operation("RemoveGridOutliers")["params"]
        if p["name"] == "cutoff_multiplier"
    )
    assert param["default"] == 1.5
    assert param["type"] == "number"
    assert param["required"] is False


def test_every_param_with_a_default_reports_it():
    """Sweep: no registered parameter may report a default it does not have."""
    from phenotypic._services.registry import get_registry

    offenders = []
    for name, info in get_registry().get_all().items():
        if not hasattr(info.cls, "model_fields"):
            continue
        described = {p["name"]: p for p in describe_operation(name)["params"]}
        for pname, param in info.parameters.items():
            if pname not in info.cls.model_fields:
                continue  # not a pydantic field; nothing for the schema to say
            if param.has_default and param.default is not None:
                if described[pname]["default"] is None:
                    offenders.append(f"{name}.{pname}")
    assert not offenders, f"defaults lost in projection: {offenders}"


def test_an_optional_bounded_param_keeps_its_bound():
    """``float | None = Field(None, gt=0)`` puts the bound on an ``anyOf`` branch.

    The top level of such a property carries only ``anyOf``/``default``/
    ``title``, so a projection that reads the top level alone reports
    ``constraints: {}`` and tells an agent the value is unconstrained.
    """
    from phenotypic.enhance import BayesShrinkEnhancer

    prop = BayesShrinkEnhancer.model_json_schema()["properties"]["gat_scale_factor"]
    assert "exclusiveMinimum" not in prop, "fixture drifted: bound is now top-level"

    param = next(
        p
        for p in describe_operation("BayesShrinkEnhancer")["params"]
        if p["name"] == "gat_scale_factor"
    )
    assert param["constraints"] == {"exclusiveMinimum": 0}


def test_no_param_hides_a_bound_declared_on_a_branch():
    """Sweep: an empty ``constraints`` must mean the schema declares none.

    The single-parameter test above proves the spelling for one field; this
    proves *where the projection looks*, for every registered class. Without
    it the hole reopens one optional-with-bound field at a time — 13 of them
    were reporting ``{}`` when this sweep was written.
    """
    from phenotypic._services.catalog import _NON_CONSTRAINT_KEYS, _property_for
    from phenotypic._services.registry import get_registry

    offenders: list[str] = []
    branch_sourced = 0
    for name, info in get_registry().get_all().items():
        if not hasattr(info.cls, "model_fields"):
            continue
        properties = info.cls.model_json_schema().get("properties", {})
        for param in describe_operation(name)["params"]:
            prop = _property_for(info.cls, param["name"], properties)
            top_level = set(prop) - _NON_CONSTRAINT_KEYS
            on_branches: set[str] = set()
            for key in ("anyOf", "oneOf"):
                for branch in prop.get(key, []) or []:
                    if not isinstance(branch, dict) or branch.get("type") == "null":
                        continue
                    on_branches |= set(branch) - _NON_CONSTRAINT_KEYS
            if not param["constraints"] and on_branches:
                offenders.append(f"{name}.{param['name']} lost {sorted(on_branches)}")
            if param["constraints"] and not top_level and on_branches:
                branch_sourced += 1

    assert not offenders, f"constraints dropped in projection: {offenders}"
    assert branch_sourced >= 13, (
        f"only {branch_sourced} branch-declared constraints seen — the sweep "
        "no longer exercises the fallback it exists to guard"
    )


def test_a_zero_field_operation_advertises_no_parameters():
    """``MeasureSize`` takes nothing; the descriptor must not invent ``data``.

    A pydantic model with no fields has a falsy ``model_fields``, and a
    truthiness test on it sends the class down the ``inspect.signature``
    path — where pydantic's own ``(self, /, **data)`` turns up as a
    *required* parameter named ``data`` that the class rejects.
    """
    import pydantic
    import pytest as _pytest

    from phenotypic.measure import MeasureSize

    assert MeasureSize.model_fields == {}
    with _pytest.raises(pydantic.ValidationError, match="Extra inputs"):
        MeasureSize(data=None)  # type: ignore[call-arg]

    assert describe_operation("MeasureSize")["params"] == []


def test_no_descriptor_reports_a_param_that_is_not_a_field():
    """Sweep: for a pydantic class, every projected param is a real field.

    ``test_every_param_with_a_default_reports_it`` skips non-fields with a
    ``continue``, so it steps around exactly this — a parameter that does
    not exist cannot lose a default it never had.
    """
    from phenotypic._services.registry import get_registry

    offenders = [
        f"{name}.{param['name']}"
        for name, info in get_registry().get_all().items()
        if hasattr(info.cls, "model_fields")
        for param in describe_operation(name)["params"]
        if param["name"] not in info.cls.model_fields
    ]
    assert not offenders, f"parameters advertised but not declared: {offenders}"
