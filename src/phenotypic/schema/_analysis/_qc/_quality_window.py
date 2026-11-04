"""Measurement info container for the growth-curve fitting-window QC check."""

from ..._base._measurement_info import Entry
from ..._base._units import UNITS
from ..._base._tiers import QualityInfo


class QUALITY_WINDOW(QualityInfo):
    """Measurement info for the fitting-window QC check.

    Documents the per-condition columns the fitting-window check emits beyond
    the generic ``Metric``/``Flag``/``Status`` trio. The check reads a
    condition's replicate coefficient of variation over time and proposes the
    frame span over which replicates agree well enough for a growth model to be
    fitted; these columns say what it proposed, and when it proposed nothing,
    which guard stopped it.

    Every value is constant across a condition's rows: the check's verdict is
    about the condition, not about any one colony.
    """

    @classmethod
    def metric_family(cls) -> str:
        return "QC_Window"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.UNITLESS

    OUTCOME = Entry(
        "Outcome",
        "Which exit produced this condition's result, as a closed set of "
        "strings. ``found`` — a window was proposed, and ``Start``/``End`` "
        "give it. ``too_few_frames`` — the condition has fewer frames with a "
        "defined mean CV than the shortest series the rule will assess, so "
        "there was nothing to read (``NFramesCounted`` says how few; zero "
        "means no replicate group ever reached the minimum replicate count). "
        "``too_few_evaluable_frames`` — the series is long enough, but fewer "
        "than four of its interior frames yield a finite curvature once the "
        "excluded end margins are removed. ``too_noisy`` — the run-wide "
        "abrupt-change bar sits inside this condition's own curvature noise, "
        "so the test could not tell an event from jitter, and the rule "
        "abstains rather than guessing (``Bar`` and ``NGroups`` are the "
        "context). The bar is pooled over the run's conditions, so a "
        "condition analyzed alone never gets this outcome at the default "
        "settings. ``no_smoothed_value`` — the smoothed trend is undefined at "
        "every frame. ``abrupt_event_at_start`` — the window would be empty: "
        "the last frame it could end at is at or before the frame it would "
        "open at. The name is the reference's; in practice this is reached "
        "when the condition's best agreement comes in its last evaluable "
        "frames, so no event can follow the start. Only ``found`` can pass; "
        "every other value fails.",
        unit=UNITS.NOT_A_QUANTITY,
    )
    START = Entry(
        "Start",
        "First frame of the proposed window, as a value of the check's "
        "frame-index column, inclusive. Null unless ``Outcome`` is ``found``.",
        unit=UNITS.FRAME,
    )
    END = Entry(
        "End",
        "Last frame of the proposed window, as a value of the check's "
        "frame-index column, inclusive. Null unless ``Outcome`` is ``found``.",
        unit=UNITS.FRAME,
    )
    REASON = Entry(
        "Reason",
        "The rule's own sentence describing why no window was proposed, kept "
        "verbatim (e.g. ``only 6 frames``, ``bar is 1.1x this curve's own "
        "noise (<3): not assessable``). Empty when ``Outcome`` is ``found``. "
        "``Outcome`` is the machine-readable form; this is the human-readable "
        "one, and it is never parsed.",
        unit=UNITS.NOT_A_QUANTITY,
    )
    BEST_CV = Entry(
        "BestCV",
        "Smallest per-frame mean replicate CV anywhere in the condition's "
        "series — how close a condition with no window came to agreeing. "
        "Carried because every windowless condition shares the same failing "
        "metric and so cannot be ranked by it; null when no frame counted.",
    )
    BAR = Entry(
        "Bar",
        "The abrupt-change threshold on curvature, in CV units per frame "
        "squared. **One run-level value, identical on every row of every "
        "condition**: it is pooled over all conditions in the run by design, "
        "so that each is judged against the same absolute standard. A "
        "consequence worth knowing: adding or removing a condition can move "
        "another condition's window. Null when no condition contributed a "
        "finite curvature.",
    )
    N_GROUPS = Entry(
        "NGroups",
        "Median number of replicate groups standing behind each counted "
        "frame's mean CV — the sample size the condition's verdict rests on. "
        "Rounded to the nearest whole group, halves to even, so counted-group "
        "counts of 3 and 4 report 4. Null when no frame counted.",
    )
    N_FRAMES_COUNTED = Entry(
        "NFramesCounted",
        "Frames at which at least one replicate group reached the minimum "
        "replicate count and had a usable mean, so a mean CV is defined. "
        "Frames with no defined mean CV are absent from the series, not zero.",
        unit=UNITS.FRAME,
    )
