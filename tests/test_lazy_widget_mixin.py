import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from phenotypic.abc_._image_operation import ImageOperation
from phenotypic import Image

# Define a dummy operation class for testing
class DummyOp(ImageOperation):
    def __init__(self, param1: int = 10, param2: bool = True, param3: str = "test"):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    def _operate(self, image):
        # Dummy operation that modifies the image slightly so we can detect it
        image.metadata['processed'] = True
        return image

@pytest.fixture
def op():
    return DummyOp()

@pytest.fixture
def image():
    # Create a simple dummy image
    return Image(arr=np.zeros((10, 10), dtype=np.uint8))

def test_missing_dependency(op):
    """Test that ImportError is raised when ipywidgets is missing."""
    with patch.dict(sys.modules, {'ipywidgets': None, 'IPython': None}):
        with pytest.raises(ImportError, match="packages are required"):
            op.widget()

def test_widget_creation(op):
    """Test that widgets are created correctly based on type hints."""
    # Ensure ipywidgets is available (should be in dev env)
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")

    widget = op.widget(image=None)
    
    assert widget is not None
    assert op._ui is not None
    
    # Check if widgets were created for parameters
    assert 'param1' in op._param_widgets
    assert 'param2' in op._param_widgets
    assert 'param3' in op._param_widgets
    
    assert isinstance(op._param_widgets['param1'], ipywidgets.IntText)
    assert isinstance(op._param_widgets['param2'], ipywidgets.Checkbox)
    assert isinstance(op._param_widgets['param3'], ipywidgets.Text)
    
    # Check default values
    assert op._param_widgets['param1'].value == 10
    assert op._param_widgets['param2'].value == True
    assert op._param_widgets['param3'].value == "test"

def test_param_update(op):
    """Test that updating widget updates the instance attribute."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")

    op.widget()
    
    # Change widget value
    op._param_widgets['param1'].value = 20
    assert op.param1 == 20
    
    op._param_widgets['param2'].value = False
    assert op.param2 == False

def test_visualization_setup(op, image):
    """Test that visualization widgets are created when image is provided."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")

    op.widget(image=image)
    
    assert op._image_ref is image
    assert op._view_dropdown is not None
    assert op._update_button is not None
    assert op._output_widget is not None

def test_visualization_update(op, image):
    """Test the update view logic."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")

    op.widget(image=image)
    
    # Mock the apply method to verify it's called on a copy
    with patch.object(DummyOp, 'apply', wraps=op.apply) as mock_apply:
        with patch('matplotlib.pyplot.show'): # Suppress plt.show
             # Trigger the button click
            op._on_update_view_click(None)
            
            assert mock_apply.called
            # Verify it was called with a different image object (the copy)
            args, _ = mock_apply.call_args
            passed_image = args[0]
            assert passed_image is not image
            # Check that the copy was actually processed
            assert passed_image.metadata.get('processed') == True

