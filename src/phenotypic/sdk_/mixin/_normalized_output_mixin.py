"""Mixin supplying the ``norm`` output-range policy as an appended pydantic field."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, model_validator
from skimage.exposure import rescale_intensity

from phenotypic.sdk_.typing_ import NormOut


class NormalizedOutputMixin(BaseModel):
    """Adds a ``norm`` field controlling how an operation's output is range-guarded.

    ``detect_mat`` is contractually [0, 1]. ``norm`` selects how an operation
    upholds that contract:

    - ``"clip"`` (default) saturates out-of-range values. It is the identity for
      in-range pixels, so absolute intensity is preserved and ``detect_mat``
      stays comparable across a batch of plates.
    - ``"rescale"`` linearly remaps the full observed range onto [0, 1]. Ordering
      survives, absolute scale does not: a single specular highlight sets the max.
    - ``None`` passes values through untouched. Required inside a Generalized
      Anscombe Transform region (where the signal is deliberately not in [0, 1])
      and by ``CompositeEnhance`` on non-normalized maps.

    The field is **appended** to the end of the subclass's field order rather than
    frontloaded, so an operation's own parameters keep their natural position in
    ``model_json_schema()`` and ``to_json()``.

    Note:
        Replaces the ``clip: bool`` field removed in 0.18.0. A bool cannot express
        ``"rescale"``, and the attribute name ``clip`` is claimed by
        :class:`NormControlMixin`, which duck-types on it.
    """

    norm: NormOut = "clip"

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Move ``norm`` to the end of the subclass's field order."""
        super().__pydantic_init_subclass__(**kwargs)
        fields = cls.__pydantic_fields__
        if "norm" in fields and list(fields)[-1] != "norm":
            fields["norm"] = fields.pop("norm")
            cls.model_rebuild(force=True)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_clip(cls, data: Any) -> Any:
        """Turn the 0.17.x ``clip`` key into an actionable migration error.

        ``BaseOperation`` sets ``extra="forbid"``, so without this the user sees
        pydantic's opaque "Extra inputs are not permitted".
        """
        if isinstance(data, dict) and "clip" in data:
            raise ValueError(
                f"{cls.__name__}: `clip` was replaced by `norm` in 0.18.0. "
                f"Use norm='clip' (was clip=True) or norm=None (was clip=False)."
            )
        return data

    def _apply_norm(self, arr: np.ndarray) -> np.ndarray:
        """Apply the configured output-range policy to *arr*."""
        match self.norm:
            case "clip":
                return np.clip(arr, 0.0, 1.0)
            case "rescale":
                return rescale_intensity(arr, out_range=(0.0, 1.0))
            case None:
                return arr
        raise ValueError(f"Unknown norm policy: {self.norm!r}")
