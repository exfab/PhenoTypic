import numpy as np
import pytest

from phenotypic.abc_ import GridFinder
from phenotypic.grid import CenteredAutoGridFinder, CenteredAutoGridFinderFallbackWarning


def test_is_gridfinder_and_constructs_keyword_only():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    assert isinstance(f, GridFinder)
    assert (f.nrows, f.ncols) == (8, 12)
    with pytest.raises(Exception):
        CenteredAutoGridFinder(8, 12)  # positional construction is rejected


def test_warning_class_is_userwarning():
    assert issubclass(CenteredAutoGridFinderFallbackWarning, UserWarning)
