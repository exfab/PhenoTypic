"""Unit tests for ``phenotypic.tools_._column_ref``.

The marker is purely informational at runtime — these tests pin the
runtime equivalence (``ColumnRef`` is still a ``str``) plus the
``Annotated`` metadata round-trip the GUI registry depends on.
"""
from __future__ import annotations

from typing import get_args

import pytest

from phenotypic.tools_ import ColumnRef, ColumnRefList
from phenotypic.tools_._column_ref import _ColumnRefMarker


class TestColumnRefMarker:
    def test_marker_default_source(self):
        m = _ColumnRefMarker()
        assert m.source == "measurements"

    def test_marker_custom_source(self):
        m = _ColumnRefMarker("master_measurements")
        assert m.source == "master_measurements"

    def test_marker_equality(self):
        a = _ColumnRefMarker("measurements")
        b = _ColumnRefMarker("measurements")
        c = _ColumnRefMarker("master_measurements")
        assert a == b
        assert a != c
        assert hash(a) == hash(b)

    def test_marker_repr(self):
        assert "measurements" in repr(_ColumnRefMarker())


class TestColumnRefAnnotation:
    def test_columnref_carries_marker(self):
        args = get_args(ColumnRef)
        assert args[0] is str
        assert isinstance(args[1], _ColumnRefMarker)
        assert args[1].source == "measurements"

    def test_columnreflist_carries_marker(self):
        args = get_args(ColumnRefList)
        # Args[0] is List[str] — the carrier type.
        assert get_args(args[0]) == (str,)
        assert isinstance(args[1], _ColumnRefMarker)


class TestColumnRefRuntimeEquivalence:
    """ColumnRef must be a plain str / list[str] at runtime.

    Otherwise existing analyzer code that does ``isinstance(self.on, str)``
    or ``self.groupby + [...]`` would break.
    """

    def test_columnref_value_is_a_str(self):
        from phenotypic.analysis import EdgeCorrector

        ec = EdgeCorrector(on="Shape_Area", groupby=["Metadata_Strain"])
        assert isinstance(ec.on, str)
        assert ec.on == "Shape_Area"

    def test_columnreflist_value_is_a_list(self):
        from phenotypic.analysis import EdgeCorrector

        ec = EdgeCorrector(on="Shape_Area", groupby=["Metadata_Strain"])
        assert isinstance(ec.groupby, list)
        assert ec.groupby == ["Metadata_Strain"]


class TestAnalyzerSignatureMarkers:
    """Each user-facing analyzer subclass must carry the marker."""

    @pytest.mark.parametrize(
        "cls_name,expected_params",
        [
            ("EdgeCorrector", {"on", "groupby", "time_label"}),
            ("TukeyOutlierRemover", {"on", "groupby"}),
            (
                "LogGrowthModel",
                {"on", "groupby", "time_label", "Kmax_label"},
            ),
            ("LinearSoftplus", {"on", "groupby", "time_label"}),
            ("DoubleSoftplus", {"on", "groupby", "time_label"}),
        ],
    )
    def test_subclass_marker_coverage(self, cls_name, expected_params):
        import phenotypic.analysis as analysis_module

        cls = getattr(analysis_module, cls_name)
        # The analyzers are pydantic models: the ``_ColumnRefMarker`` is
        # carried on each field's ``FieldInfo`` rather than on a
        # hand-written ``__init__`` signature. Pydantic lifts the marker
        # into ``FieldInfo.metadata``; for union-typed fields (e.g.
        # ``Kmax_label: ColumnRef | None``) it stays on a union branch's
        # ``__metadata__``.
        marked = set()
        for name, field in cls.model_fields.items():
            if any(
                isinstance(m, _ColumnRefMarker) for m in field.metadata
            ):
                marked.add(name)
                continue
            for branch in get_args(field.annotation):
                if any(
                    isinstance(m, _ColumnRefMarker)
                    for m in getattr(branch, "__metadata__", ())
                ):
                    marked.add(name)
                    break
        assert expected_params.issubset(marked)
