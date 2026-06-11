"""4.5p2 E1 — the QC under-detection gaming regression (qc §7 mandatory lock).

The QC count objective must reward *faithful* detection over *under*-detection:
a pipeline that detects the expected colony count must score **strictly higher**
than one that detects far fewer for the SAME layout. ``_threshold_anchored`` is
monotone-decreasing in the count-divergence metric, so a faithful frame
(``metric == 0`` → score ``1.0``) must beat an under-detecting frame
(``metric > 0`` → score ``< 1.0``). This test locks that property so a future
scorer change cannot accidentally make under-detection pay.
"""
from __future__ import annotations

import pandas as pd

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import QCScorer


def _layout(n: int, name: str = "p1") -> pd.DataFrame:
    """A layout frame declaring ``n`` expected objects for plate ``name``."""
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


def _detected(n: int, name: str = "p1") -> pd.DataFrame:
    """A measurement frame with ``n`` detected objects for plate ``name``."""
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


def test_under_detect_scores_strictly_higher_cost():
    # SAME layout (96 expected); a faithful frame detects all 96, an
    # under-detecting one detects far fewer (24). Under cost, faithful detection
    # must score STRICTLY LOWER (better) than under-detection.
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96), groupby=["Metadata_ImageName"]
        )
    )
    faithful = scorer.score_image(None, _detected(96))["Count"]
    under_detect = scorer.score_image(None, _detected(24))["Count"]
    assert faithful < under_detect
