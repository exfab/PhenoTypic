"""Measurement info container for the generic QualityCheck output columns."""

from ..._base._measurement_info import Entry, _info_row, _render_info_table
from ..._base._tiers import QualityInfo


class QUALITY_CHECK(QualityInfo):
    """Generic QC output columns emitted by every QualityCheck subclass.

    The metric-family convention used elsewhere in the codebase would produce
    column names ``QC_Flag``, ``QC_Metric``, ``QC_Status`` — but the
    actual emitted columns include each subclass's ``name`` between the
    metric family and the label (``QC_Count_Flag``, ``QC_SE_Flag``). This
    enum overrides :meth:`append_rst_to_doc` to substitute the subclass's
    ``name`` into the RST column header, so each concrete subclass's
    docstring documents its real emitted columns.
    """

    FLAG = Entry(
        "Flag",
        "True when the metric crosses fail_threshold in the bad direction; "
        "eligible for curation.",
    )
    METRIC = Entry(
        "Metric",
        "Headline metric in the check's own units; the bad direction is set "
        "by the check's _HIGHER_IS_BAD flag. Drives Status.",
    )
    STATUS = Entry("Status", "Categorical: pass | warn | fail.")

    @classmethod
    def metric_family(cls) -> str:
        return "QC"

    @classmethod
    def header(cls, member: "QUALITY_CHECK", check_name: str | None = None) -> str:
        """Return the column a quality check writes for *member*.

        The single formatter behind ``QualityCheck.output_header`` (what
        ``analyze()`` writes), the check docstrings and the Measurements
        reference, so the three cannot disagree.

        Args:
            member: A ``QUALITY_CHECK`` member.
            check_name: The check's ``name`` (``ICC``). ``None`` renders the
                ``<name>`` placeholder used for the generic form.

        Returns:
            The header, e.g. ``QC_ICC_Metric``.
        """
        slug = check_name if check_name is not None else "<name>"
        return f"QC_{slug}_{member.label}"

    @classmethod
    def append_rst_to_doc(  # type: ignore[override]
        cls,
        doc: str | object,
        *,
        check_name: str | None = None,
    ) -> str:
        """Append an RST table with optional per-check name substitution.

        Args:
            doc: Existing class docstring to splice into. If a string, it
                is treated as the docstring itself; otherwise its
                ``__doc__`` attribute is used.
            check_name: When provided, column names render as
                ``QC_<check_name>_<Label>`` (e.g. ``QC_Count_Flag``).
                When omitted (called on the base ``QualityCheck`` class
                or for general docs), column names render as
                ``QC_<name>_<Label>`` with ``<name>`` as a literal
                placeholder, signalling the per-subclass substitution.

        Returns:
            Docstring with an appended RST table of output columns.
        """
        slug = check_name if check_name is not None else "<name>"
        rows = [_info_row(cls.header(m, check_name), m) for m in cls]
        table = _render_info_table(rows, title=f"QC_{slug}", name_header="Name")
        base = doc if isinstance(doc, str) else (doc.__doc__ or "")
        return base + "\n\n" + table
