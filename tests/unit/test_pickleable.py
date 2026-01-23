import pickle
import pytest

from unit.test_fixtures import _public

# Filter out CLI objects that aren't meant to be pickled
_pickleable_public = [
    (qualname, obj) for qualname, obj in _public
    if (
            ("phenotypic.phenotypicCLI" not in qualname)
            and ("phenotypic._cli" not in qualname)
            and ("phenotypic.tools_" not in qualname)
    )
]


@pytest.mark.parametrize("qualname,obj", _pickleable_public)
def test_picklable(qualname, obj):
    pickle.dumps(obj)  # will fail fast on the first bad object
