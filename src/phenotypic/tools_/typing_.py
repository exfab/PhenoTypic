from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Tuple, Any, Dict

if TYPE_CHECKING:
    from phenotypic.abc_ import ImageOperation

FootprintShape = Literal["disk", "square", "diamond"]

DetectMode = Literal["gray", "red", "green", "blue", "MinRGB", "LabL", "LabA", "LabB", "HsvS", "HsvV"]

GridSearchSaveData = List[
    Literal["rgb", "gray", "detect_mat", "objmap", "objmask", "map2rgb"]
]

GridSearchConfig = List[Tuple["ImageOperation", Dict[str, List[Any]]]]
