import pytest
import numpy as np
from phenotypic import Image
from phenotypic.abc_ import ImageOperation


# Test classes with different docstring formats
class GoogleStyleOp(ImageOperation):
    """Test operation with Google-style docstring.
    
    Attributes:
        param_a (int): First parameter description.
        param_b (float): Second parameter with detailed
            description that spans multiple lines.
        param_c (bool): Third parameter.
    """
    def __init__(self, param_a: int = 10, param_b: float = 2.5, param_c: bool = True):
        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c
    
    def _operate(self, image):
        return image


class NumPyStyleOp(ImageOperation):
    """Test operation with NumPy-style docstring.
    
    Parameters
    ----------
    param_x : int
        First parameter description.
    param_y : float
        Second parameter description.
    """
    def __init__(self, param_x: int = 5, param_y: float = 1.0):
        self.param_x = param_x
        self.param_y = param_y
    
    def _operate(self, image):
        return image


class SphinxStyleOp(ImageOperation):
    """Test operation with Sphinx-style docstring.
    
    :param alpha: Alpha parameter description.
    :param beta: Beta parameter with longer description.
    """
    def __init__(self, alpha: int = 3, beta: str = "test"):
        self.alpha = alpha
        self.beta = beta
    
    def _operate(self, image):
        return image


class NoDocstringOp(ImageOperation):
    def __init__(self, value: int = 1):
        self.value = value
    
    def _operate(self, image):
        return image


@pytest.fixture
def image():
    return Image(arr=np.zeros((10, 10), dtype=np.uint8))


def test_google_style_docstring_parsing():
    """Test parsing Google-style docstrings."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
    
    op = GoogleStyleOp()
    doc_params = op._parse_docstring()
    
    assert 'param_a' in doc_params
    assert 'param_b' in doc_params
    assert 'param_c' in doc_params
    
    assert 'First parameter description' in doc_params['param_a']
    assert 'Second parameter' in doc_params['param_b']
    assert 'multiple lines' in doc_params['param_b']
    assert 'Third parameter' in doc_params['param_c']


def test_numpy_style_docstring_parsing():
    """Test parsing NumPy-style docstrings."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
    
    op = NumPyStyleOp()
    doc_params = op._parse_docstring()
    
    assert 'param_x' in doc_params
    assert 'param_y' in doc_params
    
    assert 'First parameter description' in doc_params['param_x']
    assert 'Second parameter description' in doc_params['param_y']


def test_sphinx_style_docstring_parsing():
    """Test parsing Sphinx-style docstrings."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
    
    op = SphinxStyleOp()
    doc_params = op._parse_docstring()
    
    assert 'alpha' in doc_params
    assert 'beta' in doc_params
    
    assert 'Alpha parameter description' in doc_params['alpha']
    assert 'Beta parameter' in doc_params['beta']


def test_no_docstring_graceful_fallback():
    """Test that operations without docstrings work normally."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
    
    op = NoDocstringOp()
    doc_params = op._parse_docstring()
    
    assert doc_params == {}


def test_widget_with_help_text():
    """Test that help text is displayed in widgets."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
    
    op = GoogleStyleOp()
    widget = op.widget(show=False)
    
    assert widget is not None
    assert 'param_a' in op._param_widgets
    assert 'param_b' in op._param_widgets
    assert 'param_c' in op._param_widgets


def test_widget_without_docstring():
    """Test that widgets work without docstrings."""
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets not installed")
    
    op = NoDocstringOp()
    widget = op.widget(show=False)
    
    assert widget is not None
    assert 'value' in op._param_widgets

