"""Unit tests for OperationRegistry and ParamInfo."""

from __future__ import annotations

from typing import List, Optional

import pytest

from phenotypic.abc_ import ObjectDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.gui._operation_registry import (
    OperationRegistry,
    ParamInfo,
    OperationInfo,
    get_registry,
)


class TestParamInfo:
    """Test ParamInfo dataclass."""

    def test_param_info_creation(self):
        """Test creating ParamInfo."""
        info = ParamInfo(
                name="sigma",
                type_hint=float,
                default=1.0,
                has_default=True,
                is_operation=False,
                is_pipeline=False,
                is_optional=False,
        )
        assert info.name == "sigma"
        assert info.type_hint is float
        assert info.default == 1.0
        assert info.has_default is True


class TestOperationInfo:
    """Test OperationInfo dataclass."""

    def test_operation_info_creation(self):
        """Test creating OperationInfo."""
        info = OperationInfo(
                cls=GaussianBlur,
                name="GaussianBlur",
                category="Enhancer",
                module="phenotypic.enhance",
                docstring="Test docstring",
                parameters={},
        )
        assert info.cls is GaussianBlur
        assert info.name == "GaussianBlur"
        assert info.category == "Enhancer"


class TestOperationRegistry:
    """Test OperationRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry for each test."""
        reg = OperationRegistry()
        reg.discover()
        return reg

    def test_discover_operations(self, registry):
        """Test that operations are discovered."""
        assert len(registry._operations) > 0
        assert "GaussianBlur" in registry._operations
        assert "OtsuDetector" in registry._operations

    def test_get_categories(self, registry):
        """Test getting operation categories."""
        categories = registry.get_categories()
        assert "Enhancer" in categories
        assert "Detector" in categories
        assert isinstance(categories, list)
        # Should be sorted
        assert categories == sorted(categories)

    def test_get_by_category(self, registry):
        """Test getting operations by category."""
        enhancers = registry.get_by_category("Enhancer")
        assert len(enhancers) > 0
        assert all(info.category == "Enhancer" for info in enhancers)

        # Check specific operations
        enhancer_names = [info.name for info in enhancers]
        assert "GaussianBlur" in enhancer_names
        assert "EnhanceLocalContrast" in enhancer_names

    def test_legacy_bm3d_alias_not_registered(self, registry):
        """Deserializer aliases must not appear as duplicate palette entries."""
        assert registry.get("EnhanceBlockMatch") is not None
        assert registry.get("BM3DDenoiser") is None

    def test_get_operation(self, registry):
        """Test getting specific operation by name."""
        info = registry.get("GaussianBlur")
        assert info is not None
        assert info.cls is GaussianBlur
        assert info.name == "GaussianBlur"
        assert info.category == "Enhancer"

    def test_get_nonexistent_operation(self, registry):
        """Test getting operation that doesn't exist."""
        info = registry.get("NonexistentOperation")
        assert info is None

    def test_get_all_operations(self, registry):
        """Test getting all operations."""
        all_ops = registry.get_all()
        assert isinstance(all_ops, dict)
        assert len(all_ops) > 0
        assert "GaussianBlur" in all_ops

    def test_create_instance(self, registry):
        """Test creating operation instance."""
        blur = registry.create_instance("GaussianBlur")
        assert isinstance(blur, GaussianBlur)

        # With parameters
        blur2 = registry.create_instance("GaussianBlur", sigma=2.5)
        assert blur2.sigma == 2.5

    def test_create_instance_nonexistent(self, registry):
        """Test creating instance of nonexistent operation."""
        with pytest.raises(KeyError):
            registry.create_instance("NonexistentOperation")

    def test_extract_parameters(self, registry):
        """Test parameter extraction."""
        info = registry.get("GaussianBlur")
        assert "sigma" in info.parameters

        param_info = info.parameters["sigma"]
        assert param_info.name == "sigma"
        assert param_info.has_default is True

    def test_extract_parameters_with_operations(self, registry):
        """Test parameter extraction for operations with nested operation params."""
        # FilamentousFungiDetector has operation parameters
        info = registry.get("FilamentousFungiDetector")
        if info:  # Only test if this detector exists
            assert "inoculum_detector" in info.parameters
            param = info.parameters["inoculum_detector"]
            assert param.is_operation or param.is_pipeline
            assert param.is_optional  # It's Union[..., None]

    def test_global_registry(self):
        """Test global registry singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2  # Should be same instance

    def test_operation_has_docstring(self, registry):
        """Test that operations include docstrings."""
        info = registry.get("GaussianBlur")
        assert info.docstring is not None
        assert len(info.docstring) > 0

    def test_operation_module_path(self, registry):
        """Test that module paths are captured."""
        info = registry.get("GaussianBlur")
        assert "phenotypic" in info.module
        assert "enhance" in info.module


class TestPointPickerMarker:
    """Operations that mix in ``PointPickerMixin`` are flagged in the registry.

    The Dash builder uses these flags to swap a free-form text input for an
    interactive point picker on the matching parameter.
    """

    @pytest.fixture
    def registry(self):
        reg = OperationRegistry()
        reg.discover()
        return reg

    def test_point_picker_marker_propagates(self, registry):
        """ManualPointDetector and ManualRefine advertise the mixin marker."""
        det = registry.get("ManualPointDetector")
        assert det is not None
        assert det.is_point_pickable is True
        assert det.point_picker_param == "centers"

        sel = registry.get("ManualRefine")
        assert sel is not None
        assert sel.is_point_pickable is True
        assert sel.point_picker_param == "centers"

    def test_non_pickable_ops_have_falsy_marker(self, registry):
        """Operations without the mixin do not gain a stray pickable flag."""
        otsu = registry.get("OtsuDetector")
        assert otsu is not None
        assert otsu.is_point_pickable is False
        assert otsu.point_picker_param is None

        blur = registry.get("GaussianBlur")
        assert blur is not None
        assert blur.is_point_pickable is False
        assert blur.point_picker_param is None

    def test_threshold_based_manual_detector_is_not_pickable(self, registry):
        """UserThreshold takes a scalar threshold, not points — not flagged."""
        man = registry.get("UserThreshold")
        assert man is not None
        assert man.is_point_pickable is False
        assert man.point_picker_param is None


class TestColumnRefDetection:
    """`_extract_parameters` populates `ParamInfo.column_ref` from `Annotated`."""

    @pytest.fixture(scope="class")
    def registry(self):
        reg = OperationRegistry()
        reg.discover()
        return reg

    @pytest.mark.parametrize(
            "cls_name,param_name,expected_multi",
            [
                ("EdgeCorrector", "on", False),
                ("EdgeCorrector", "groupby", True),
                ("EdgeCorrector", "time_label", False),
                ("TukeyOutlierRemover", "on", False),
                ("TukeyOutlierRemover", "groupby", True),
                ("LogGrowthModel", "on", False),
                ("LogGrowthModel", "groupby", True),
                ("LogGrowthModel", "time_label", False),
                ("LinearLagModel", "on", False),
                ("LinearLagModel", "groupby", True),
                ("LinearLagModel", "time_label", False),
                ("LinearCapAndLagModel", "on", False),
                ("LinearCapAndLagModel", "groupby", True),
                ("LinearCapAndLagModel", "time_label", False),
            ],
    )
    def test_column_ref_populated(
            self, registry, cls_name, param_name, expected_multi
    ):
        info = registry.get(cls_name)
        assert info is not None, f"{cls_name} not registered"
        p = info.parameters.get(param_name)
        assert p is not None, f"{cls_name}.{param_name} missing"
        assert p.column_ref is not None, (
            f"{cls_name}.{param_name} has no column_ref"
        )
        assert p.column_ref.source == "measurements"
        assert p.column_ref.multi is expected_multi
        assert p.column_ref.with_alt is False

    def test_kmax_label_is_column_ref_with_alt(self, registry):
        """`Kmax_label: ColumnRef | None` — the alt branch flips with_alt."""
        info = registry.get("LogGrowthModel")
        assert info is not None
        p = info.parameters.get("Kmax_label")
        assert p is not None
        assert p.column_ref is not None
        assert p.column_ref.source == "measurements"
        assert p.column_ref.multi is False
        assert p.column_ref.with_alt is True

    def test_non_column_params_have_no_column_ref(self, registry):
        info = registry.get("EdgeCorrector")
        assert info is not None
        for name in ("nrows", "ncols", "top_n", "pvalue"):
            p = info.parameters.get(name)
            assert p is not None
            assert p.column_ref is None, f"{name} should not be a column ref"

    def test_non_analyzer_op_has_no_column_ref(self, registry):
        """Builder-side operations don't carry the marker."""
        blur = registry.get("GaussianBlur")
        assert blur is not None
        for p in blur.parameters.values():
            assert p.column_ref is None


class TestIsListDetection:
    """`_extract_parameters` populates ``ParamInfo.is_list`` for list-typed params.

    The flag distinguishes list-typed aux ports (e.g.
    ``CompositeDetector.detectors: List[Union[ObjectDetector, ImagePipeline]]``)
    from scalar variants (e.g.
    ``FilamentousFungiDetector.inoculum_detector: Union[ObjectDetector,
    ImagePipeline, None]``) so the GUI builder can render multi-port
    ``+``/``×`` controls only on list slots.
    """

    @pytest.fixture(scope="class")
    def registry(self):
        reg = OperationRegistry()
        reg.discover()
        return reg

    def test_composite_detector_detectors_is_list(self, registry):
        """``CompositeDetector.detectors: List[Union[Op, Pipeline]]``."""
        info = registry.get("CompositeDetector")
        assert info is not None
        p = info.parameters.get("detectors")
        assert p is not None
        assert p.is_list is True
        assert p.is_operation is True
        assert p.is_pipeline is True
        assert p.is_optional is False

    def test_filamentous_inoculum_is_scalar_optional(self, registry):
        """``inoculum_detector: Union[Op, Pipeline, None]`` is scalar+optional."""
        info = registry.get("FilamentousFungiDetector")
        assert info is not None
        p = info.parameters.get("inoculum_detector")
        assert p is not None
        assert p.is_list is False
        assert p.is_operation is True
        assert p.is_pipeline is True
        assert p.is_optional is True

    def test_optional_list_of_operations(self):
        """``Optional[List[ObjectDetector]]`` peels both wrappers."""

        class _SyntheticOptionalListOp:
            def __init__(self, param: Optional[List[ObjectDetector]] = None):
                self.param = param

        reg = OperationRegistry()
        params = reg._extract_parameters(_SyntheticOptionalListOp)
        p = params["param"]
        assert p.is_list is True
        assert p.is_operation is True
        assert p.is_optional is True

    def test_bare_list_no_args(self):
        """Bare ``list`` annotation flags ``is_list`` without op/pipeline."""

        class _SyntheticBareListOp:
            def __init__(self, param: list = []):
                self.param = param

        reg = OperationRegistry()
        params = reg._extract_parameters(_SyntheticBareListOp)
        p = params["param"]
        assert p.is_list is True
        assert p.is_operation is False
        assert p.is_pipeline is False

    def test_scalar_param_is_not_list(self, registry):
        """Scalar params keep ``is_list=False`` (regression guard)."""
        blur = registry.get("GaussianBlur")
        assert blur is not None
        for p in blur.parameters.values():
            assert p.is_list is False


class TestQualityCheckCategory:
    """`_discover_analyzers` routes QualityCheck subclasses to a dedicated category.

    Per spec §1278–1283, ``QualityCheck`` subclasses (e.g.
    ``ExpectedVsDetectedCount``, ``ReplicateAgreement``) must land under
    the ``"quality_check"`` category so the QC tab's add-check dropdown
    has something to render. Without the explicit branch, the
    fall-through default would mis-route them into ``"Filter"``.

    Per spec §1419–1428, ``OperationRegistry._extract_parameters`` skips
    the inherited ``agg_func`` parameter on QC subclasses that opt out
    via ``_exposes_agg_func: ClassVar[bool] = False`` (the default for
    every v1 check) so the param form doesn't surface an unused dropdown.
    Backward-compat: analyzers without the attribute (``EdgeCorrector``,
    ``LogGrowthModel``) keep their ``agg_func`` parameter exposed.
    """

    @pytest.fixture(scope="class")
    def registry(self):
        reg = OperationRegistry()
        reg.discover()
        return reg

    def test_quality_check_subclasses_get_quality_check_category(self, registry):
        """``ExpectedVsDetectedCount`` and ``ReplicateAgreement`` register here."""
        qc_ops = registry.get_by_category("quality_check")
        qc_names = {info.name for info in qc_ops}
        assert "ExpectedVsDetectedCount" in qc_names
        assert "ReplicateAgreement" in qc_names
        # Each registered op must carry the matching category attribute.
        for info in qc_ops:
            assert info.category == "quality_check"

    def test_quality_check_classes_excluded_from_filter_or_model_categories(
            self, registry
    ):
        """QC classes must not leak into ``"Filter"`` or ``"Model"`` buckets."""
        filter_names = {info.name for info in registry.get_by_category("Filter")}
        model_names = {info.name for info in registry.get_by_category("Model")}
        assert "ExpectedVsDetectedCount" not in filter_names
        assert "ExpectedVsDetectedCount" not in model_names
        assert "ReplicateAgreement" not in filter_names
        assert "ReplicateAgreement" not in model_names

    def test_quality_check_base_class_itself_not_registered(self, registry):
        """The abstract ``QualityCheck`` ABC is excluded from every category."""
        all_ops = registry.get_all()
        assert "QualityCheck" not in all_ops
        for category in registry.get_categories():
            names = {info.name for info in registry.get_by_category(category)}
            assert "QualityCheck" not in names

    def test_quality_check_params_omit_agg_func_when_exposes_agg_func_is_false(
            self, registry
    ):
        """``_exposes_agg_func=False`` filters ``agg_func`` out of params."""
        info = registry.get("ExpectedVsDetectedCount")
        assert info is not None
        assert "agg_func" not in info.parameters

    def test_non_quality_check_analyzers_still_expose_agg_func(self, registry):
        """Backward-compat: analyzers without the flag keep ``agg_func``."""
        info = registry.get("EdgeCorrector")
        assert info is not None
        assert "agg_func" in info.parameters


class TestEdgeCorrectionCategory:
    def test_edge_corrector_is_edge_category(self):
        reg = OperationRegistry()
        reg.discover()
        info = reg.get("EdgeCorrector")
        assert info is not None
        assert info.category == "Edge Correction"
        filter_names = {i.name for i in reg.get_by_category("Filter")}
        model_names = {i.name for i in reg.get_by_category("Model")}
        assert "EdgeCorrector" not in filter_names
        assert "EdgeCorrector" not in model_names
        assert "EdgeCorrector" in {i.name for i in reg.get_by_category("Edge Correction")}
