import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from phenotypic import Image
from phenotypic.core._image_pipeline import ImagePipeline
from phenotypic.abc_ import ImageOperation

# Dummy operations
class OpA(ImageOperation):
    def __init__(self, val_a: int = 1):
        self.val_a = val_a
    def _operate(self, image):
        return image

class OpB(ImageOperation):
    def __init__(self, val_b: float = 2.5, toggle: bool = False):
        self.val_b = val_b
        self.toggle = toggle
    def _operate(self, image):
        return image

@pytest.fixture
def pipe():
    return ImagePipeline(ops=[OpA(), OpB()])

@pytest.fixture
def image():
    return Image(arr=np.zeros((10, 10), dtype=np.uint8))

def test_pipeline_widget_creation(pipe):
    """Test that pipeline widgets are created recursively."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")

    # Initialize widgets (show=False to avoid display in test)
    widget = pipe.widget(show=False)
    
    assert widget is not None
    # Pipe params
    assert 'benchmark' in pipe._param_widgets
    assert 'verbose' in pipe._param_widgets
    
    # Check if accordion for ops exists
    # The structure is VBox([params..., VBox([Accordion...])])
    # Let's just check if child widgets were created for operations
    
    # Since _create_widgets recursively calls widget(), OpA and OpB should have widgets now
    op_a = pipe._ops['OpA']
    op_b = pipe._ops['OpB']
    
    assert hasattr(op_a, '_ui') and op_a._ui is not None
    assert 'val_a' in op_a._param_widgets
    
    assert hasattr(op_b, '_ui') and op_b._ui is not None
    assert 'val_b' in op_b._param_widgets
    assert 'toggle' in op_b._param_widgets

def test_recursive_widget_structure(pipe):
    """Test nested pipelines."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")

    parent_pipe = ImagePipeline(ops=[pipe]) # Pipe inside pipe
    widget = parent_pipe.widget(show=False)
    
    # Parent pipe should have widget for 'ImagePipeline' (the child pipe)
    child_pipe_op = parent_pipe._ops['ImagePipeline']
    assert child_pipe_op is pipe
    assert child_pipe_op._ui is not None
    
    # And grandchild ops
    op_a = child_pipe_op._ops['OpA']
    assert op_a._ui is not None

def test_pipeline_viz_update(pipe, image):
    """Test visualization update on pipeline."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")

    pipe.widget(image=image, show=False)
    
    # Mock apply
    with patch.object(ImagePipeline, 'apply', wraps=pipe.apply) as mock_apply:
        with patch('matplotlib.pyplot.show'):
            pipe._on_update_view_click(None)
            
            assert mock_apply.called
            args, _ = mock_apply.call_args
            assert args[0] is not image # Should be a copy

