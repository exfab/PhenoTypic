"""The path-configured ground-truth mask loader (supervised-scorers; DEFERRED §1).

``GroundTruthMasks`` resolves per-image annotation masks from a configurable source,
mirroring the path-config round-trip discipline of
:class:`phenotypic.analysis.ExpectedVsDetectedCount` (its unified ``metadata`` field): a
**serializable** ``gt_masks_source`` path survives ``model_dump`` / ``tuning_spec.json``,
while the resolved frames/arrays live in an **excluded** in-memory cache. A loader
constructed without a source path cannot resolve anything on reload and honestly reports
``modality() == "none"`` (abstain) — the supervised scorer then degrades like the
reference-free gate does.

The source determines the **modality**:

* a **directory** of per-image masks → ``"mask"`` (the object-overlap path);
* a ``.csv`` / ``.parquet`` file of per-image counts → ``"count"`` (the count-MAE path,
  consumed by chunk B's ``SupervisedScorer`` via ``ExpectedVsDetectedCount``);
* ``None`` or an empty/absent directory → ``"none"`` (abstain).

Filename match is **directory + image stem**: ``"plate1.png"`` (or bare ``"plate1"``)
resolves to ``gt_masks_source/plate1.{npy,tif,png}`` in that **suffix-priority order**;
a missing stem yields ``None``.

# TODO(DEFERRED-WORK §1): validate against real annotated plates. v1 ships the
# path-config + modality + abstain machinery and reads masks structurally; the
# numeric correctness of resolved masks vs. a real annotated calibration set is
# deferred until such a set exists. No numeric-vs-real-GT test runs here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, TypeAlias

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    model_validator,
)

#: The ground-truth source modalities — a type-only closed set (no documentation
#: surface beyond this loader): ``"mask"`` (a directory of per-image masks),
#: ``"count"`` (a CSV/Parquet of per-image expected counts), ``"none"`` (abstain).
Modality = Literal["mask", "count", "none"]

#: How the pixels of a **mask-modality** GT source are interpreted — a
#: serializable closed set (it round-trips through ``model_dump`` /
#: ``tuning_spec.json`` like ``gt_masks_source`` does):
#:
#: * ``"instance"`` — the on-disk mask is already an **integer instance-label**
#:   array (each colony a distinct positive label); the matcher consumes it
#:   directly.
#: * ``"binary"`` — the on-disk mask is a **binary foreground** mask (any
#:   non-zero pixel is colony); per-cell instances are derived from the grid at
#:   scoring time (:mod:`._supervised`).
#:
#: Orthogonal to :data:`Modality`: ``gt_format`` only *interprets* a ``"mask"``
#: source — it has no bearing on the ``"count"`` / ``"none"`` modalities.
GTFormat: TypeAlias = Literal["instance", "binary"]

#: A loaded GT mask array — integer instance labels **or** a binary/boolean
#: foreground mask. The loader preserves the on-disk dtype (it no longer forces
#: ``bool``), so instance labels survive to the matcher and only the
#: :class:`~._supervised.SupervisedScorer` decides how to interpret them per
#: :data:`GTFormat`.
GTMaskArray: TypeAlias = "npt.NDArray[np.integer[Any]] | npt.NDArray[np.bool_]"

#: Mask file suffixes, in resolution priority order: a binary ``.npy`` array wins
#: over a ``.tif``, which wins over a ``.png`` — the first existing one is used.
_MASK_SUFFIX_PRIORITY: tuple[str, ...] = (".npy", ".tif", ".png")

#: Count-source suffixes (a per-image expected-count table, not per-image masks).
_COUNT_SUFFIXES: frozenset[str] = frozenset({".csv", ".parquet"})


class GroundTruthMasks(BaseModel):
    """Resolve per-image ground-truth masks from a serializable source path.

    Mirrors :class:`phenotypic.analysis.ExpectedVsDetectedCount`'s unified
    ``metadata`` path round-trip: ``gt_masks_source`` is the JSON-serializable
    handle (a directory of masks or a count table); the resolved masks are cached
    in an **excluded** private attribute, never serialized, and re-resolved on
    reload. A loader with ``gt_masks_source=None`` abstains
    (``modality() == "none"``).

    Args:
        gt_masks_source: Path to the ground-truth source — a **directory** of
            per-image masks (``modality "mask"``), a ``.csv``/``.parquet`` count
            table (``modality "count"``), or ``None`` to abstain
            (``modality "none"``). Captured as a string in the serialized surface
            so the source round-trips through ``tuning_spec.json``.
        gt_format: How a **mask**-modality source's pixels are interpreted —
            ``"instance"`` (an integer instance-label array, consumed directly by
            the matcher) or ``"binary"`` (a foreground mask, default; per-cell
            instances are derived from the grid by the supervised scorer).
            Orthogonal to the modality and round-trips through
            ``tuning_spec.json``. Ignored by the ``"count"`` / ``"none"``
            modalities.

    Examples:
        Construct from a directory of masks and round-trip the source path and
        format. The loader **preserves the on-disk dtype** — a binary mask loads
        as ``bool``, an integer-labeled mask stays integer:

        >>> import json, tempfile, numpy as np
        >>> from pathlib import Path
        >>> from phenotypic.tune.score._gt_loader import GroundTruthMasks
        >>> tmp = Path(tempfile.mkdtemp())
        >>> _ = np.save(tmp / "plate1.npy", np.array([[True, False]]))
        >>> _ = np.save(tmp / "labels.npy", np.array([[1, 0], [0, 2]]))
        >>> gt = GroundTruthMasks(gt_masks_source=tmp, gt_format="instance")
        >>> gt.modality()
        'mask'
        >>> sorted(gt.available_names())
        ['labels', 'plate1']
        >>> bool(gt.masks_for("plate1.png").dtype == bool)  # binary stays bool
        True
        >>> int(gt.masks_for("labels").max())  # integer labels preserved
        2
        >>> reloaded = GroundTruthMasks.model_validate_json(gt.model_dump_json())
        >>> str(reloaded.gt_masks_source) == str(tmp)
        True
        >>> reloaded.gt_format
        'instance'

        A sourceless loader abstains (and defaults ``gt_format`` to binary):

        >>> GroundTruthMasks(gt_masks_source=None).modality()
        'none'
        >>> GroundTruthMasks(gt_masks_source=None).gt_format
        'binary'
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gt_masks_source: Optional[Path] = None
    gt_format: GTFormat = "binary"

    #: In-memory cache of resolved masks (image stem → integer/boolean array).
    #: Excluded from serialization (arrays are not JSON-native) and rebuilt lazily
    #: on demand from :attr:`gt_masks_source`, so a reloaded loader re-resolves.
    _cache: dict[str, GTMaskArray] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _capture_source(cls, data: Any) -> Any:
        """Normalize an empty ``gt_masks_source`` to ``None`` (capture step).

        Mirrors ``ExpectedVsDetectedCount``'s unified ``metadata`` field: the
        serializable handle is the source *path*. Pydantic natively coerces a
        ``str`` (e.g. from ``tuning_spec.json``) into the ``Path`` field in both
        Python and JSON modes, so this validator only canonicalizes an empty
        string to ``None`` (abstain) — there is no resolved array to inject, since
        masks are read lazily by :meth:`masks_for`.

        Args:
            data: The raw constructor input. Normally a kwargs mapping; pydantic
                may hand a non-dict (a revalidated model), which passes through.

        Returns:
            The input mapping with an empty ``gt_masks_source`` canonicalized to
            ``None``.
        """
        if not isinstance(data, dict):
            return data
        raw = data.get("gt_masks_source")
        if isinstance(raw, str) and raw.strip() == "":
            data = dict(data)
            data["gt_masks_source"] = None
        return data

    def modality(self) -> Modality:
        """Report the ground-truth modality this source provides.

        Returns:
            ``"mask"`` when ``gt_masks_source`` is a directory holding at least one
            recognized mask file; ``"count"`` when it is a ``.csv``/``.parquet``
            count table; ``"none"`` when no source is configured, the directory is
            empty/absent, or the path is otherwise unusable (abstain).
        """
        source = self.gt_masks_source
        if source is None:
            return "none"
        if source.is_dir():
            return "mask" if self.available_names() else "none"
        if source.is_file() and source.suffix.lower() in _COUNT_SUFFIXES:
            return "count"
        return "none"

    def available_names(self) -> frozenset[str]:
        """The image stems with a resolvable mask in the source directory.

        Returns:
            The set of stems (e.g. ``"plate1"``) for which a mask file exists at
            any recognized suffix. Empty when the source is not a mask directory.
        """
        source = self.gt_masks_source
        if source is None or not source.is_dir():
            return frozenset()
        stems = {
            entry.stem
            for entry in source.iterdir()
            if entry.is_file() and entry.suffix.lower() in _MASK_SUFFIX_PRIORITY
        }
        return frozenset(stems)

    def masks_for(self, image_name: str) -> Optional[GTMaskArray]:
        """Resolve the ground-truth mask for one image by its stem.

        Strips any extension from ``image_name`` (so ``"plate1.png"`` and bare
        ``"plate1"`` match the same stem) and looks for
        ``gt_masks_source/<stem>.{npy,tif,png}`` in :data:`_MASK_SUFFIX_PRIORITY`
        order, returning the first that exists **with its on-disk dtype
        preserved** — integer instance labels stay integer, a binary mask stays
        boolean. How those values are interpreted (instance vs. binary) is the
        :attr:`gt_format` decision, applied by the supervised scorer, not here.
        The result is cached in the excluded :attr:`_cache`.

        Args:
            image_name: The image's name or filename; only its stem is matched.

        Returns:
            The ground-truth mask (integer labels or a boolean foreground mask,
            dtype preserved from disk), or ``None`` when no source directory is
            configured or no mask file exists for the stem.
        """
        source = self.gt_masks_source
        if source is None or not source.is_dir():
            return None
        stem = Path(image_name).stem
        if stem in self._cache:
            return self._cache[stem]
        for suffix in _MASK_SUFFIX_PRIORITY:
            candidate = source / f"{stem}{suffix}"
            if candidate.exists():
                mask = self._read_mask(candidate)
                self._cache[stem] = mask
                return mask
        return None

    @staticmethod
    def _read_mask(path: Path) -> GTMaskArray:
        """Read a mask file (``.npy`` / ``.tif`` / ``.png``), preserving dtype.

        The on-disk dtype is **not** coerced: an integer instance-label array
        stays integer (so per-object labels survive to the matcher) and a binary
        mask stays boolean. The :attr:`gt_format` flag — not this reader — decides
        whether the values are read as instance labels or as a binary foreground.

        Args:
            path: An existing mask file with a recognized suffix.

        Returns:
            The mask as loaded from disk: integer labels stay integer, a boolean
            mask stays boolean (non-zero entries are foreground either way).
        """
        if path.suffix.lower() == ".npy":
            return np.load(path)
        # Image masks (.tif/.png): read via skimage, guarded for headless/cross-
        # platform installs where the imageio backend may be unavailable.
        from skimage import io as skio

        return np.asarray(skio.imread(path))
