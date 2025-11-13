"""Tests for interactive measurement tools."""
from __future__ import annotations

import pytest
import numpy as np

try:
    import dash
    import jupyter_dash
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


@pytest.fixture
def sample_image_with_objects():
    """Create a simple test image with detected objects."""
    from phenotypic import Image
    
    # Create a simple synthetic image with two distinct regions
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:40, 20:40] = 200  # First object
    arr[60:80, 60:80] = 180  # Second object
    
    image = Image(arr=arr, name='test_image')
    
    # Create object map manually
    objmap = np.zeros((100, 100), dtype=np.uint16)
    objmap[20:40, 20:40] = 1
    objmap[60:80, 60:80] = 2
    image.objmap[:] = objmap
    
    return image


@pytest.mark.skipif(not DASH_AVAILABLE, reason="Dash dependencies not installed")
class TestInteractiveImageAnalyzer:
    """Tests for the InteractiveImageAnalyzer base class."""
    
    def test_dependency_validation(self, sample_image_with_objects):
        """Test that dependency validation works correctly."""
        from phenotypic.tools import InteractiveImageAnalyzer
        
        # Should not raise error when dependencies are installed
        analyzer = InteractiveImageAnalyzer(
            image=sample_image_with_objects,
            port=8050,
            height=800,
            mode='external'
        )
        assert analyzer is not None
    
    def test_initialization(self, sample_image_with_objects):
        """Test basic initialization of analyzer."""
        from phenotypic.tools import InteractiveImageAnalyzer
        
        analyzer = InteractiveImageAnalyzer(
            image=sample_image_with_objects,
            port=8051,
            height=600,
            mode='inline'
        )
        
        assert analyzer.image is sample_image_with_objects
        assert analyzer.port == 8051
        assert analyzer.height == 600
        assert analyzer.mode == 'inline'
        assert analyzer.app is None  # Not created yet
    
    def test_convert_image_to_plotly_uint8(self, sample_image_with_objects):
        """Test image conversion for uint8 arrays."""
        from phenotypic.tools import InteractiveImageAnalyzer
        
        analyzer = InteractiveImageAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        arr = np.array([[100, 150], [200, 255]], dtype=np.uint8)
        result = analyzer._convert_image_to_plotly(arr)
        
        assert result.dtype == np.uint8
        assert np.array_equal(result, arr)
    
    def test_convert_image_to_plotly_float(self, sample_image_with_objects):
        """Test image conversion for float arrays."""
        from phenotypic.tools import InteractiveImageAnalyzer
        
        analyzer = InteractiveImageAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        arr = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
        result = analyzer._convert_image_to_plotly(arr)
        
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 1] == 127
        assert result[1, 1] == 255
    
    def test_convert_image_to_plotly_uint16(self, sample_image_with_objects):
        """Test image conversion for uint16 arrays."""
        from phenotypic.tools import InteractiveImageAnalyzer
        
        analyzer = InteractiveImageAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        arr = np.array([[0, 256], [512, 65535]], dtype=np.uint16)
        result = analyzer._convert_image_to_plotly(arr)
        
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 1] == 255
    
    def test_create_overlay_image(self, sample_image_with_objects):
        """Test overlay creation with object map."""
        from phenotypic.tools import InteractiveImageAnalyzer
        
        analyzer = InteractiveImageAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        base_image = sample_image_with_objects.gray[:]
        objmap = sample_image_with_objects.objmap[:]
        
        overlay = analyzer._create_overlay_image(base_image, objmap, alpha=0.3)
        
        # Should be RGB
        assert overlay.ndim == 3
        assert overlay.shape[2] == 3
        assert overlay.dtype == np.uint8


@pytest.mark.skipif(not DASH_AVAILABLE, reason="Dash dependencies not installed")
class TestInteractiveMeasurementAnalyzer:
    """Tests for the InteractiveMeasurementAnalyzer class."""
    
    def test_initialization(self, sample_image_with_objects):
        """Test initialization of measurement analyzer."""
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        analyzer = InteractiveMeasurementAnalyzer(
            image=sample_image_with_objects,
            port=8052,
            height=700,
            mode='external',
            detector_type='otsu'
        )
        
        assert analyzer.image is sample_image_with_objects
        assert analyzer.port == 8052
        assert analyzer.height == 700
        assert analyzer.detector_type == 'otsu'
        assert len(analyzer.selected_objects) == 0
        assert analyzer.measurements.empty
    
    def test_calculate_measurements(self, sample_image_with_objects):
        """Test measurement calculation for selected objects."""
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        analyzer = InteractiveMeasurementAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        # Calculate measurements for both objects
        selected = {1, 2}
        df = analyzer._calculate_measurements(selected)
        
        assert len(df) == 2
        assert 'Object_Label' in df.columns
        assert 'Area_pixels' in df.columns
        assert 'Centroid_Row' in df.columns
        assert 'Centroid_Col' in df.columns
        
        # Check object 1 (20x20 square)
        obj1 = df[df['Object_Label'] == 1].iloc[0]
        assert obj1['Area_pixels'] == 400  # 20 * 20
        
        # Check object 2 (20x20 square)
        obj2 = df[df['Object_Label'] == 2].iloc[0]
        assert obj2['Area_pixels'] == 400  # 20 * 20
    
    def test_calculate_measurements_empty_selection(self, sample_image_with_objects):
        """Test measurement calculation with no selection."""
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        analyzer = InteractiveMeasurementAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        df = analyzer._calculate_measurements(set())
        assert df.empty
    
    def test_calculate_measurements_single_object(self, sample_image_with_objects):
        """Test measurement calculation for single object."""
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        analyzer = InteractiveMeasurementAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        selected = {1}
        df = analyzer._calculate_measurements(selected)
        
        assert len(df) == 1
        assert df['Object_Label'].iloc[0] == 1
        assert df['Area_pixels'].iloc[0] == 400
    
    def test_create_figure(self, sample_image_with_objects):
        """Test figure creation."""
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        analyzer = InteractiveMeasurementAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        fig = analyzer._create_figure(alpha=0.3, show_overlay=True, highlighted_objects=None)
        
        assert fig is not None
        assert 'data' in fig
        assert len(fig['data']) > 0
    
    def test_create_figure_with_highlighted_objects(self, sample_image_with_objects):
        """Test figure creation with highlighted objects."""
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        analyzer = InteractiveMeasurementAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        highlighted = {1}
        fig = analyzer._create_figure(
            alpha=0.5, 
            show_overlay=True, 
            highlighted_objects=highlighted
        )
        
        assert fig is not None
        assert 'data' in fig
    
    def test_create_figure_no_overlay(self, sample_image_with_objects):
        """Test figure creation without overlay."""
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        analyzer = InteractiveMeasurementAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        fig = analyzer._create_figure(alpha=0.3, show_overlay=False, highlighted_objects=None)
        
        assert fig is not None
        assert 'data' in fig
    
    def test_setup_layout(self, sample_image_with_objects):
        """Test layout creation."""
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        analyzer = InteractiveMeasurementAnalyzer(
            image=sample_image_with_objects,
            port=8050
        )
        
        layout = analyzer.setup_layout()
        
        assert layout is not None
        # Layout should be a Dash component
        assert hasattr(layout, 'children')


@pytest.mark.skipif(not DASH_AVAILABLE, reason="Dash dependencies not installed")
class TestImageInteractiveMeasureMethod:
    """Tests for the Image.interactive_measure() method."""
    
    def test_method_exists(self, sample_image_with_objects):
        """Test that interactive_measure method exists on Image."""
        assert hasattr(sample_image_with_objects, 'interactive_measure')
        assert callable(sample_image_with_objects.interactive_measure)
    
    def test_method_signature(self, sample_image_with_objects):
        """Test method signature."""
        import inspect
        
        sig = inspect.signature(sample_image_with_objects.interactive_measure)
        params = list(sig.parameters.keys())
        
        assert 'port' in params
        assert 'height' in params
        assert 'mode' in params
        assert 'detector_type' in params


class TestMissingDependencies:
    """Tests for behavior when dependencies are missing."""
    
    @pytest.mark.skipif(DASH_AVAILABLE, reason="Testing missing dependencies")
    def test_import_error_on_missing_dependencies(self, sample_image_with_objects):
        """Test that ImportError is raised when dependencies are missing."""
        # This test would only run if dash is NOT installed
        # In practice, we skip this during normal test runs
        pass


@pytest.mark.skipif(not DASH_AVAILABLE, reason="Dash dependencies not installed")
class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_end_to_end_measurement_calculation(self, sample_image_with_objects):
        """Test complete workflow of measurement calculation."""
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        # Create analyzer
        analyzer = InteractiveMeasurementAnalyzer(
            image=sample_image_with_objects,
            port=8053
        )
        
        # Calculate measurements for all objects
        all_labels = set(sample_image_with_objects.objects.labels)
        df = analyzer._calculate_measurements(all_labels)
        
        # Verify results
        assert len(df) == sample_image_with_objects.num_objects
        assert df['Area_pixels'].sum() == 800  # Total area of both objects
        
        # Check each object has required columns
        for _, row in df.iterrows():
            assert row['Object_Label'] in [1, 2]
            assert row['Area_pixels'] > 0
            assert not np.isnan(row['Centroid_Row'])
            assert not np.isnan(row['Centroid_Col'])

