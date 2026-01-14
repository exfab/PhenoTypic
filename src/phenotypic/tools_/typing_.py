from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Tuple, Any, Dict, Union

if TYPE_CHECKING:
    from phenotypic.abc_ import ImageOperation

FootprintShape = Literal["disk", "square", "diamond"]

GridSearchSaveData = List[
    Literal["rgb", "gray", "enh_gray", "objmap", "objmask", "map2rgb"]
]

GridSearchConfig = List[Tuple["ImageOperation", Dict[str, List[Any]]]]
