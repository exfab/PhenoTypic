"""A labelled synthetic time series for fitting-window examples.

Nothing here is real data: four conditions named ``Condition A`` to
``Condition D``, three numbered replicate groups each, and radii drawn from a
formula with a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from phenotypic.schema import CULTURE, IMAGE, OBJECT, PLATE, SAMPLE, SIZE

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["make_synthetic_window_series"]

#: The condition column. A user label, not a schema member: the join emits any
#: metadata label as ``Metadata_<Label>``.
_CONDITION = "Metadata_Condition"

#: Frames per plate, hours between frames, replicate groups per condition and
#: colonies per group.
_N_FRAMES = 30
_FRAME_SPACING_HOURS = 6.0
_N_GROUPS = 3
_N_COLONIES = 4


@dataclass(frozen=True)
class _ConditionSpec:
    """One synthetic condition: a growth curve and a replicate-spread shape.

    Colony ``k`` of replicate group ``g`` at frame ``f`` (hour ``t``) has radius
    ``radius(t) * exp(spread(f) * z[g, k] + jitter * e[g, k, f])``: ``z`` is
    drawn once per colony and ``e`` anew at every frame.
    """

    name: str
    r0: float
    v: float
    lag: float
    spread_floor: float
    spread_early: float
    settle_frames: float
    jitter: float
    split_frame: float | None = None
    split_size: float = 0.0
    split_width: float = 0.8

    def radius(self, hours: np.ndarray) -> np.ndarray:
        """The noise-free radius, ``r0 + v * softplus(t - lag)``, px."""
        return self.r0 + self.v * np.logaddexp(0.0, hours - self.lag)

    def spread(self, frames: np.ndarray) -> np.ndarray:
        """The persistent replicate spread at each frame (log scale)."""
        out = self.spread_floor + self.spread_early * np.exp(
            -(frames - 1) / self.settle_frames
        )
        if self.split_frame is not None:
            out = out + self.split_size / (
                1.0 + np.exp(-(frames - self.split_frame) / self.split_width)
            )
        return out


#: A and B share one growth curve and differ only in how fast their replicates
#: settle. C settles like A, then its replicates split apart late. D's
#: agreement jumps from frame to frame.
_CONDITIONS: tuple[_ConditionSpec, ...] = (
    _ConditionSpec(
        "Condition A", r0=8.0, v=0.25, lag=30.0,
        spread_floor=0.04, spread_early=0.20, settle_frames=1.5, jitter=0.004,
    ),
    _ConditionSpec(
        "Condition B", r0=8.0, v=0.25, lag=30.0,
        spread_floor=0.04, spread_early=0.20, settle_frames=4.5, jitter=0.004,
    ),
    _ConditionSpec(
        "Condition C", r0=8.0, v=0.18, lag=48.0,
        spread_floor=0.04, spread_early=0.20, settle_frames=1.5, jitter=0.004,
        split_frame=22.0, split_size=0.12,
    ),
    _ConditionSpec(
        "Condition D", r0=8.0, v=0.20, lag=36.0,
        spread_floor=0.06, spread_early=0.20, settle_frames=2.0, jitter=0.20,
    ),
)


def make_synthetic_window_series(seed: int = 7) -> pd.DataFrame:
    """Build a labelled synthetic colony-radius time series, one row per colony per image.

    Four conditions of three replicate plates each, four colonies per plate,
    imaged every 6 hours for 30 images (0 to 174 h). The conditions are built to
    show what a fitting window is and is not:

    * **A and B share one growth curve** (8 px, then 0.25 px/h after a 30 h
      lag) but B's replicates take longer to come together, so a window opens
      later for B. A window is where replicates agree, not where a colony grows.
    * **C** settles like A, then its replicates split apart late, so its window
      closes early.
    * **D**'s agreement jumps from frame to frame, so in a run with the others
      no window can be chosen for it.

    The table has no frame index: ``AddFrameIndex`` derives one from each
    plate's image names, which sort into capture order.

    Args:
        seed: Seed for :func:`numpy.random.default_rng`. The default, 7, is the
            series the fitting-window examples are written against.

    Returns:
        A frame with columns ``Metadata_Condition`` (``Condition A`` to
        ``Condition D``), ``Metadata_BioReplicate`` (1 to 3),
        ``Metadata_PlateID`` (one plate per condition and replicate group,
        e.g. ``A1``), ``Metadata_ImageName`` (``<plate>_<hour>h``),
        ``Metadata_Time`` (hours), ``Object_Label`` (1 to 4 on each image) and
        ``Size_RobustMeanRadius`` (px). 4 × 3 × 4 × 30 = 1,440 rows.

    Examples:
        >>> from phenotypic.data import make_synthetic_window_series
        >>> from phenotypic.schema import IMAGE
        >>> table = make_synthetic_window_series()
        >>> len(table), table["Metadata_Condition"].nunique()
        (1440, 4)
        >>> table[str(IMAGE.IMAGE_NAME)].iloc[:2].tolist()
        ['A1_000h', 'A1_000h']
        >>> bool((table["Size_RobustMeanRadius"] > 0).all())
        True
    """
    import pandas as pd

    frames = np.arange(1, _N_FRAMES + 1)
    hours = _FRAME_SPACING_HOURS * (frames - 1)
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for spec in _CONDITIONS:
        persistent = rng.standard_normal((_N_GROUPS, _N_COLONIES, 1))
        per_frame = rng.standard_normal((_N_GROUPS, _N_COLONIES, frames.size))
        log_noise = spec.spread(frames) * persistent + spec.jitter * per_frame
        radius = spec.radius(hours) * np.exp(log_noise)

        group, colony, frame = np.meshgrid(
            np.arange(1, _N_GROUPS + 1),
            np.arange(_N_COLONIES),
            np.arange(frames.size),
            indexing="ij",
        )
        letter = spec.name.rsplit(" ", 1)[-1]
        plate = np.array([f"{letter}{g}" for g in group.ravel()], dtype=object)
        image_hours = hours[frame.ravel()]
        parts.append(
            pd.DataFrame(
                {
                    _CONDITION: spec.name,
                    str(SAMPLE.BIO_REPLICATE): group.ravel(),
                    str(PLATE.PLATE_ID): plate,
                    str(IMAGE.IMAGE_NAME): [
                        f"{p}_{h:03.0f}h" for p, h in zip(plate, image_hours)
                    ],
                    str(CULTURE.TIME): image_hours,
                    str(OBJECT.LABEL): colony.ravel() + 1,
                    str(SIZE.ROBUST_MEAN_RADIUS): radius.ravel(),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(
        [_CONDITION, str(SAMPLE.BIO_REPLICATE), str(CULTURE.TIME), str(OBJECT.LABEL)],
        kind="stable",
        ignore_index=True,
    )
