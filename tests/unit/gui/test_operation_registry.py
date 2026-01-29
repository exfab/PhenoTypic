"""Unit tests for OperationRegistry and ParamInfo."""

import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageEnhancer, ObjectDetector
from phenotypic.enhance import GaussianBlur, CLAHE
from phenotypic.detect import OtsuDetector
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
            assert "center_detector" in info.parameters
            param = info.parameters["center_detector"]
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
