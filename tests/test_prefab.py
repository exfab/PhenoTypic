import pytest
from .resources.TestHelper import walk_package_for_class, timeit
import phenotypic
from phenotypic.abc_ import PrefabPipeline

__prefab_classes = walk_package_for_class(phenotypic, PrefabPipeline)


@pytest.mark.parametrize("qualname,obj", __prefab_classes)
@timeit
def test_prefabs(qualname, obj):
    pass
