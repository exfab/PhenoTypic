import pickle
import pytest

from unit.test_fixtures import _public

# Module-level slow marker: this is a coverage probe that walks the entire
# phenotypic public namespace. The walk is heavy and the matrix doesn't move
# under typical PRs, so we run it on the nightly full lane only.
pytestmark = pytest.mark.slow

# Filter out CLI objects that aren't meant to be serialized
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
