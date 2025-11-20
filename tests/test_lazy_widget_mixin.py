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

def test_union_widget_creation():
    """Test that Union types (e.g. int | None) are handled correctly."""
    from typing import Union
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
    
    # Define a local class using the mixin
    class UnionOp(ImageOperation):
        def __init__(self, param: Union[int, None] = None):
            self.param = param
        def _operate(self, image): return image

    op = UnionOp()

    op.widget()
    
    assert 'param' in op._param_widgets
    w = op._param_widgets['param']
    
    # It should be our custom UnionWidget (VBox subclass)
    assert isinstance(w, ipywidgets.VBox)
    # Since the class is local, we check by name or attribute
    assert type(w).__name__ == 'UnionWidget'
    assert hasattr(w, 'selector')
    
    # Initial state (None)
    assert w.selector.value == 'None'
    assert w.value is None
    
    # Change to int
    w.selector.value = 'int'
    # Should default to 0 for int
    assert w.value == 0
    assert isinstance(w.inner_widget, ipywidgets.IntText)
    
    # Update inner value
    w.inner_widget.value = 42
    assert w.value == 42
    assert op.param == 42
    
    # Switch back to None
    w.selector.value = 'None'
    assert w.value is None
    assert op.param is None

def test_union_widget_init_with_value():
    """Test initializing UnionWidget where the value matches the first option."""
    from typing import Union
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
        
    class UnionOp(ImageOperation):
        def __init__(self, param: Union[int, None] = None):
            self.param = param
        def _operate(self, image): return image

    # This was crashing because 'int' might be first option and value=5 matches it,
    # triggering logic that assumed inner_widget existed.
    op = UnionOp(param=5)
    op.widget()
    
    w = op._param_widgets['param']
    assert w.selector.value == 'int'
    assert w.value == 5
    assert isinstance(w.inner_widget, ipywidgets.IntText)
    assert w.inner_widget.value == 5

def test_union_widget_with_generics():
    """Test that Union types containing generics (e.g. List[int]) do not crash."""
    from typing import Union, List
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
        
    class GenericUnionOp(ImageOperation):
        def __init__(self, param: Union[int, List[int], None] = None):
            self.param = param
        def _operate(self, image): return image
        
    # Initialize with a list value
    op = GenericUnionOp(param=[1, 2, 3])
    
    # This should not raise TypeError
    op.widget()
    
    w = op._param_widgets['param']
    
    assert w.value == [1, 2, 3]

def test_union_widget_with_literal():
    """Test that Union types containing Literals (e.g. Union[int, Literal['auto'], None]) do not crash."""
    from typing import Union, Literal
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
        
    class LiteralUnionOp(ImageOperation):
        def __init__(self, param: Union[int, Literal['auto'], None] = None):
            self.param = param
        def _operate(self, image): return image
        
    # Initialize with the literal value
    op = LiteralUnionOp(param='auto')
    
    # This should not raise TypeError: typing.Literal cannot be used with isinstance()
    op.widget()
    
    w = op._param_widgets['param']
    
    # It should select the literal type option (which probably has a messy name like 'Literal' or similar, depending on mapping)
    # We just verify it holds the value correctly and didn't crash
    assert w.value == 'auto'

def test_pipeline_param_introspection_skipped():
    """Test that ImagePipeline does not generate widgets for its constructor parameters."""
    # Mock ImagePipeline because importing the real one pulls in too many dependencies for this unit test
    class ImagePipeline(ImageOperation):
        def __init__(self, ops=None, meas=None, benchmark=False, verbose=False):
            self.ops = ops
            self.meas = meas
            self.benchmark = benchmark
            self.verbose = verbose
        def _operate(self, image): return image
    
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
        
    # Instantiate
    pipeline = ImagePipeline()
    
    # Call widget
    pipeline.widget()
    
    # Verify _param_widgets is empty (ops/meas/etc ignored)
    assert len(pipeline._param_widgets) == 0
