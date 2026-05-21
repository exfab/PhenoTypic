"""Scenario execution + golden I/O for the pydantic-migration harness.

This module is the shared engine behind both
``scripts/capture_migration_goldens.py`` (which *writes* goldens) and
``tests/migration/test_equivalence.py`` (which *re-runs* scenarios and
*compares* against goldens). Keeping the run/save/load/compare logic in
one place guarantees the capture pass and the equivalence pass treat
every scenario identically.

A scenario produces one of two golden shapes:

* :class:`ImageGolden` -- the ``detect_mat`` / ``objmask`` / ``objmap``
  arrays of a resulting ``Image`` / ``GridImage``, persisted as ``.npz``.
* :class:`FrameGolden` -- a measurement / analysis ``DataFrame``,
  persisted as parquet.

The ``RNG_SEED`` constant fixes ``numpy``'s legacy global RNG before any
stochastic operation runs, so GMM-based detectors (``InoculumDetector``)
and ``WatershedDetector`` are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.migration._inputs import load_frozen_input
from tests.migration._scenarios import (
    INVOKE_ANALYZE,
    INVOKE_APPLY_DF,
    INVOKE_APPLY_IMAGE,
    INVOKE_MEASURE,
    Scenario,
    resolve_class,
)

# Directory holding the captured golden artifacts.
GOLDENS_DIR = Path(__file__).parent / "_goldens"

# Fixed seed for numpy's legacy global RNG -- applied before every
# scenario so stochastic operations capture/replay identically.
RNG_SEED = 20240520


def golden_path(scenario: Scenario) -> Path:
    """Return the golden artifact path for a scenario.

    Args:
        scenario: The scenario whose golden path to compute.

    Returns:
        ``<scenario_id>.npz`` for image scenarios, ``.parquet`` for
        frame scenarios, ``.meta.json`` for structural-only scenarios.
    """
    if scenario.structural_only:
        suffix = ".meta.json"
    elif scenario.invocation == INVOKE_APPLY_IMAGE:
        suffix = ".npz"
    else:
        suffix = ".parquet"
    return GOLDENS_DIR / f"{scenario.scenario_id}{suffix}"


@dataclass
class ImageGolden:
    """The captured array state of an operated ``Image``/``GridImage``.

    Only the components an operation may legitimately modify are
    captured (see :func:`phenotypic`-mirroring
    :func:`tests.migration._scenarios.components_for`): an enhancer's
    golden holds only ``detect_mat``; a detector's holds only
    ``objmask`` + ``objmap``. This keeps the committed fixtures small
    and makes the equivalence assertion a tighter contract.

    Attributes:
        arrays: Mapping from component name (``"detect_mat"`` /
            ``"objmask"`` / ``"objmap"``) to the captured array. Keys
            are exactly the components relevant to the scenario.
    """

    arrays: dict[str, np.ndarray]

    @property
    def summary(self) -> str:
        """A short human-readable description of the captured arrays."""
        parts = []
        for name, arr in self.arrays.items():
            if name == "detect_mat":
                parts.append(f"detect_mat{arr.shape}{arr.dtype}")
            else:
                parts.append(
                    f"{name} nnz={int(np.count_nonzero(arr))}"
                )
        return ", ".join(parts)

    def save(self, path: Path) -> None:
        """Persist this golden to a compressed ``.npz`` file.

        Args:
            path: Destination ``.npz`` path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays)

    @classmethod
    def load(cls, path: Path) -> "ImageGolden":
        """Load an image golden from a ``.npz`` file.

        Args:
            path: Source ``.npz`` path.

        Returns:
            The reconstructed :class:`ImageGolden` carrying whichever
            components the artifact stored.
        """
        with np.load(path) as data:
            arrays = {
                key: np.ascontiguousarray(data[key])
                for key in data.files
            }
        return cls(arrays=arrays)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a parquet-stable, comparison-ready copy of a frame.

    Three normalizations are applied so the captured golden and a
    re-computed result compare faithfully -- crucially, *all three are
    applied identically to both sides*, so no measurement value is ever
    altered, only its storage presentation:

    * **Index materialization** -- some grid measurers return a frame
      whose row index is a *named* (e.g. ``Grid_RowMajorIdx``) or
      categorical index. Parquet round-trips would silently replace it
      with a default ``RangeIndex``. ``reset_index`` turns any
      non-default index into ordinary columns so its values survive and
      are still verified.
    * **Categorical de-wrapping** -- pyarrow does not round-trip a
      ``CategoricalDtype`` that originated from an index, so the saved
      golden's column comes back as the plain underlying dtype while a
      freshly computed frame still carries ``CategoricalDtype``. Casting
      every categorical column to its ``.categories`` dtype on *both*
      sides removes that storage-only mismatch; the category *values*
      are still compared exactly.
    * **String column labels** -- some measurers label columns with enum
      members; parquet requires string labels.

    Args:
        frame: The raw result DataFrame.

    Returns:
        A copy with a default ``RangeIndex``, no categorical columns,
        and string column labels.
    """
    out = frame.copy()
    # A frame already on a default RangeIndex needs no index column.
    if not (
        isinstance(out.index, pd.RangeIndex) and out.index.name is None
    ):
        out = out.reset_index()
    # De-wrap categoricals to their underlying value dtype.
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(out[col].cat.categories.dtype)
    out.columns = [str(c) for c in out.columns]
    return out


@dataclass
class FrameGolden:
    """The captured result ``DataFrame`` of a measurer/analyzer.

    Attributes:
        frame: The result DataFrame.
    """

    frame: pd.DataFrame

    @property
    def summary(self) -> str:
        """A short human-readable description of the captured frame."""
        return (
            f"frame {self.frame.shape[0]}x{self.frame.shape[1]}"
        )

    def save(self, path: Path) -> None:
        """Persist this golden to a parquet file.

        The frame is :func:`normalize_frame`-d first so any named or
        categorical index is materialized into columns and survives the
        parquet round-trip.

        Args:
            path: Destination ``.parquet`` path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        normalize_frame(self.frame).to_parquet(path, index=False)

    @classmethod
    def load(cls, path: Path) -> "FrameGolden":
        """Load a frame golden from a parquet file.

        Args:
            path: Source ``.parquet`` path.

        Returns:
            The reconstructed :class:`FrameGolden`.
        """
        return cls(frame=pd.read_parquet(path))


def _seed_rng() -> None:
    """Seed numpy's legacy global RNG for reproducible stochastic ops."""
    np.random.seed(RNG_SEED)


def construct(scenario: Scenario) -> Any:
    """Construct the operation/analyzer instance for a scenario.

    All construction is keyword-only, so this works identically on the
    pre-migration code and on the migrated pydantic models.

    Args:
        scenario: The scenario to construct.

    Returns:
        The constructed operation or analyzer instance.
    """
    cls = resolve_class(scenario)
    return cls(**scenario.resolve_kwargs())


# Accessors for each capturable image component.
_COMPONENT_ACCESSORS = {
    "detect_mat": lambda img: img.detect_mat[:],
    "objmask": lambda img: img.objmask[:],
    "objmap": lambda img: img.objmap[:],
}


def _normalize_image_result(
    result: Any, components: tuple[str, ...]
) -> ImageGolden:
    """Extract the relevant array components from an image result.

    Args:
        result: The ``Image`` / ``GridImage`` returned by ``apply``.
        components: The component names to capture (a subset of
            ``detect_mat`` / ``objmask`` / ``objmap``).

    Returns:
        An :class:`ImageGolden` holding exactly the requested
        components.
    """
    arrays = {
        name: np.ascontiguousarray(_COMPONENT_ACCESSORS[name](result))
        for name in components
    }
    return ImageGolden(arrays=arrays)


def run_scenario(scenario: Scenario) -> Any:
    """Execute a scenario and return its golden-shaped result.

    Seeds the global RNG first (so stochastic operations are
    reproducible), reconstructs the matching frozen input, constructs
    the operation/analyzer keyword-only, and invokes it via the kind
    fixed by ``scenario.invocation``.

    Args:
        scenario: The scenario to execute.

    Returns:
        An :class:`ImageGolden` for image operations, or a
        :class:`FrameGolden` for measurers / post-measurement
        transforms / analyzers.

    Raises:
        ValueError: If the scenario's invocation kind is unknown.
        RuntimeError: For ``structural_only`` scenarios, which must not
            be executed (they have no runnable golden).
    """
    if scenario.structural_only:
        raise RuntimeError(
            f"Scenario {scenario.scenario_id!r} is structural-only; "
            "it must not be executed."
        )

    _seed_rng()
    instance = construct(scenario)
    frozen = load_frozen_input(scenario.category)

    if scenario.invocation == INVOKE_APPLY_IMAGE:
        result = instance.apply(frozen, inplace=True)
        return _normalize_image_result(result, scenario.components)

    if scenario.invocation == INVOKE_MEASURE:
        frame = instance.measure(frozen)
        return FrameGolden(frame=pd.DataFrame(frame))

    if scenario.invocation == INVOKE_APPLY_DF:
        frame = instance.apply(frozen)
        return FrameGolden(frame=pd.DataFrame(frame))

    if scenario.invocation == INVOKE_ANALYZE:
        frame = instance.analyze(frozen)
        return FrameGolden(frame=pd.DataFrame(frame))

    raise ValueError(
        f"Unknown invocation kind {scenario.invocation!r} for "
        f"scenario {scenario.scenario_id!r}."
    )


def load_golden(scenario: Scenario) -> Any:
    """Load the persisted golden for a scenario.

    Args:
        scenario: The scenario whose golden to load.

    Returns:
        An :class:`ImageGolden` or :class:`FrameGolden`.

    Raises:
        FileNotFoundError: If the golden artifact is missing.
        ValueError: If the scenario's invocation kind is unknown.
    """
    path = golden_path(scenario)
    if not path.exists():
        raise FileNotFoundError(
            f"Golden for {scenario.scenario_id!r} missing: {path}. "
            "Run scripts/capture_migration_goldens.py first."
        )
    if scenario.invocation == INVOKE_APPLY_IMAGE:
        return ImageGolden.load(path)
    if scenario.invocation in (
        INVOKE_MEASURE,
        INVOKE_APPLY_DF,
        INVOKE_ANALYZE,
    ):
        return FrameGolden.load(path)
    raise ValueError(
        f"Unknown invocation kind {scenario.invocation!r} for "
        f"scenario {scenario.scenario_id!r}."
    )
