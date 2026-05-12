"""Unit tests for OperationRegistry and ParamInfo."""

import pytest

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
        assert "CLAHE" in enhancer_names

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
        """ManualPointDetector and ManualSelector advertise the mixin marker."""
        det = registry.get("ManualPointDetector")
        assert det is not None
        assert det.is_point_pickable is True
        assert det.point_picker_param == "centers"

        sel = registry.get("ManualSelector")
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
            ("LinearSoftplus", "on", False),
            ("LinearSoftplus", "groupby", True),
            ("LinearSoftplus", "time_label", False),
            ("DoubleSoftplus", "on", False),
            ("DoubleSoftplus", "groupby", True),
            ("DoubleSoftplus", "time_label", False),
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
