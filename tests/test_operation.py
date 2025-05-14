import pickle, pytest

from .test_fixtures import _public


@pytest.mark.parametrize("qualname,obj", _public)
def test_picklable(qualname, obj):
    pickle.dumps(obj)            # will fail fast on the first bad object