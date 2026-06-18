"""Back-compat shim tests for ``phenotypic.detect._filamentous_fungi``.

The shim re-exports symbols from ``phenotypic.sdk_.branch_pathfinding``
to preserve legacy import paths. These tests assert the re-exported
symbols are the same Python objects as their counterparts in the new
location so monkey-patches and ``isinstance`` checks keep working.
"""

from phenotypic.detect import _filamentous_fungi as shim
from phenotypic.sdk_ import branch_pathfinding as new


def test_public_function_identity():
    assert shim.run_multisource_dijkstra is new.run_multisource_dijkstra
    assert shim.assemble_composite_cost is new.assemble_composite_cost


def test_dataclass_identity():
    assert shim.DijkstraResult is new.DijkstraResult
    assert shim.FragmentPath is new.FragmentPath


def test_private_helper_identity():
    assert shim._apply_border_penalty_inplace is new._apply_border_penalty_inplace
    assert shim._compute_screening_envelope is new._compute_screening_envelope


def test_all_matches_new_surface():
    assert set(shim.__all__) == set(new.__all__)
